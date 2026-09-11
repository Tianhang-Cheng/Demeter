import math
import torch.nn as nn
import torch

from pointcept.models.losses import build_criteria
from .builder import MODELS, build_model
import torch.nn.functional as F

@MODELS.register_module()
class DefaultCustom(nn.Module):
    """Semantic head plus a scalar organ-boundary head.

    ``dist_head`` is a bare linear layer, so its output is a logit.
    ``dist_loss="bce"`` reads it as one and reports ``sigmoid(logit)``;
    ``dist_loss="mse"`` reproduces the released recipe, which clips that logit to
    [0, 1] and regresses it against the target with MSE -- the clip zeroes the
    gradient wherever the logit leaves [0, 1].
    """

    def __init__(self, backbone=None, criteria=None, dist_loss="mse",
                 dist_loss_weight=15.0, dist_pos_weight=None, dist_prior=None,
                 offset_loss_weight=0.0, offset_direction_weight=0.0):
        super().__init__()
        if dist_loss not in ("mse", "bce"):
            raise ValueError(f"dist_loss must be 'mse' or 'bce', got {dist_loss!r}")
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.dist_loss = dist_loss
        self.dist_loss_weight = dist_loss_weight
        self.dist_pos_weight = dist_pos_weight
        # Weight for the optional offset head (PointGroup-style organ centroids).
        self.offset_loss_weight = offset_loss_weight
        # PointGroup pairs the L1 with a direction term; measured offset error here
        # is directional (predicted magnitude already matches the target's), so the
        # L1 alone leaves the part that decides whether organs separate.
        self.offset_direction_weight = offset_direction_weight
        if dist_prior is not None:
            self.init_boundary_prior(dist_prior)

    def init_boundary_prior(self, prior):
        """Start the boundary logit at `prior`, the mean of the training target.

        AdamW moves a parameter by roughly the learning rate per step, so over the
        ~1600 steps of a 100-epoch run this bias can travel about 0.4 in logit
        space -- far short of the logit(0.112) = -2.07 that the target's own mean
        needs. Regressing the clipped logit with MSE worked in probability space
        and never had to cross that distance; cross-entropy does, so seed it.
        """
        if not 0.0 < prior < 1.0:
            raise ValueError(f"dist_prior must be in (0, 1), got {prior}")
        linear = [m for m in getattr(self.backbone, "dist_head", nn.Identity()).modules()
                  if isinstance(m, nn.Linear) and m.bias is not None]
        if not linear:
            raise ValueError("dist_prior needs a backbone dist_head ending in a Linear with bias")
        nn.init.constant_(linear[-1].bias, math.log(prior / (1.0 - prior)))

    def boundary_loss(self, logit, target):
        """Loss for the boundary head, against the truncated inverse-distance target."""
        if self.dist_loss == "bce":
            weight = (None if self.dist_pos_weight is None else
                      torch.as_tensor(self.dist_pos_weight, device=logit.device,
                                      dtype=logit.dtype))
            # The target is soft (T / max(d, T) in (0, 1]), which cross-entropy accepts.
            return F.binary_cross_entropy_with_logits(
                logit, target, pos_weight=weight) * self.dist_loss_weight
        return torch.mean(torch.square(torch.clip(logit, 0, 1) - target)) * self.dist_loss_weight

    def centroid_loss(self, offset, input_dict):
        """L1 between the predicted and true vector to the organ centroid.

        Averaged over the points that belong to an organ, so points the annotation
        leaves out contribute nothing rather than pulling the head towards zero.
        """
        instance = input_dict["instance"].reshape(-1)
        target = input_dict["instance_centroid"] - input_dict["coord"]
        mask = (instance != -1).float()
        total = torch.sum(mask) + 1e-8
        error = torch.sum(torch.abs(offset - target), dim=-1)
        loss = (torch.sum(error * mask) / total) * self.offset_loss_weight
        if self.offset_direction_weight:
            unit_target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
            unit_offset = offset / (torch.norm(offset, dim=-1, keepdim=True) + 1e-8)
            cosine = -torch.sum(unit_target * unit_offset, dim=-1)
            loss = loss + (torch.sum(cosine * mask) / total) * self.offset_direction_weight
        return loss

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        predicted = self.backbone(input_dict)
        seg_logits, dist_logit = predicted[0], predicted[1]
        offset = predicted[2] if len(predicted) > 2 else None
        dist_logit = dist_logit[:, 0]
        dist = (torch.sigmoid(dist_logit) if self.dist_loss == "bce"
                else torch.clip(dist_logit, 0, 1))

        if "segment" not in input_dict.keys():
            # test
            result = dict(seg_logits=seg_logits, dist=dist[:, None])
            if offset is not None:
                result["offset_pred"] = offset
            return result

        gt = input_dict["segment"]
        segment = gt[:, 0].long()
        gt_distance = gt[:, 1].float()
        loss = self.criteria(seg_logits, segment) + self.boundary_loss(dist_logit, gt_distance)
        if offset is not None and self.offset_loss_weight:
            loss = loss + self.centroid_loss(offset, input_dict)
        if self.training:
            return dict(loss=loss)
        # eval
        return dict(loss=loss, seg_logits=seg_logits)


# @MODELS.register_module()
# class DefaultCustom2(nn.Module):
#     def __init__(self, backbone=None, criteria=None):
#         super().__init__()
#         self.backbone = build_model(backbone)
#         self.criteria = build_criteria(criteria)

#     def forward(self, input_dict):
#         if "condition" in input_dict.keys():
#             # PPT (https://arxiv.org/abs/2308.09718)
#             # currently, only support one batch one condition
#             input_dict["condition"] = input_dict["condition"][0]
#         seg_logits, bias_pred = self.backbone(input_dict)

#         # print(torch.max(dist), torch.min(dist))
#         # if torch.max(dist) > 1.0:
#         #     # raise ValueError('dist max > 1.0')
#         #     print('dist max > 1.0')
#         #     exit()

#         if "segment" in input_dict.keys():
#             gt = input_dict["segment"]
#             # segment = gt[:, 0].long()
#             segment = gt.long()
#             # print(segment[0:10])
#             # print(gt_distance[0:10])

#             # print(input_dict.keys())
#             # print(dist.shape)
#             # print(gt_distance.shape)
#             # print(input_dict["segment"].shape)
#             # print(seg_logits.shape)
#             # assert False

#         # print(input_dict.keys())

#         data_dict = input_dict

#         if self.training:
#             instance = data_dict["instance"]
#             instance_centroid = data_dict["instance_centroid"]
#             coord = data_dict["coord"]

#             mask = (instance != -1).float()
#             bias_gt = instance_centroid - coord
#             bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
#             bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

#             bias_pred_norm = bias_pred / (
#                 torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
#             )
#             bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
#             cosine_similarity = 1.0-(bias_pred_norm * bias_gt_norm).sum(-1)
#             bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
#                 torch.sum(mask) + 1e-8
#             )
#             # 

#         if self.training:
#             dist_loss = bias_l1_loss + bias_cosine_loss
#             loss = self.criteria(seg_logits, segment)
#             # print('bias_l1_loss', bias_l1_loss.item(), 'bias_cosine_loss', bias_cosine_loss.item(), 'loss', loss.item())
#             return dict(loss=loss + dist_loss)
#         # eval
#         elif "segment" in input_dict.keys():
#             dist_loss =  bias_l1_loss + bias_cosine_loss
#             loss = self.criteria(seg_logits, segment)
#             return dict(loss=loss + dist_loss, seg_logits=seg_logits)
#         # test
#         else:
#             return dict(seg_logits=seg_logits, dist=bias_pred)



@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
