import torch.nn as nn
import torch

from pointcept.models.losses import build_criteria
from .builder import MODELS, build_model
import torch.nn.functional as F

@MODELS.register_module()
class DefaultCustom(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits, dist = self.backbone(input_dict)

        dist = torch.clip(dist, 0, 1)

        # print(torch.max(dist), torch.min(dist))
        # if torch.max(dist) > 1.0:
        #     # raise ValueError('dist max > 1.0')
        #     print('dist max > 1.0')
        #     exit()

        if "segment" in input_dict.keys():
            gt = input_dict["segment"]
            segment = gt[:, 0].long()
            # gt_distance = gt[:, 1].float() 
            gt_distance = gt[:, 1:].float()  # ELYSIA
            # print(segment[0:10])
            # print(gt_distance[0:10])

            # print(input_dict.keys())
            # print(dist.shape)
            # print(gt_distance.shape)
            # print(input_dict["segment"].shape)
            # print(seg_logits.shape)
            # exit()
        

        if self.training:
            dist_loss = torch.mean(torch.square(dist[:,0] - gt_distance[:,0])) * 15.0
            # dist_loss1 = torch.mean(torch.abs(dist[:,0] - gt_distance[:,0])) * 15.0 # ELYSIA
            # dist_loss2 = torch.mean(torch.abs(dist[:,1] - gt_distance[:,1])) * 15.0 # ELYSIA
            # dist_loss = torch.mean(torch.abs(dist[:,0] - gt_distance)) * dist_weight
            # dist_loss = F.binary_cross_entropy_with_logits(dist[:,0], gt_distance) * dist_weight
            loss = self.criteria(seg_logits, segment)
            print('loss = {:.4f}, dist_loss = {:.4f} \n'.format(loss.item(), dist_loss.item()))
            # print('loss = {:.4f}, ll = {:.4f}, non_ll = {:.4f}'.format(loss.item(), dist_loss1.item(), dist_loss2.item()))
            return dict(loss=loss + dist_loss) # ELYSIA
            # return dict(loss=loss + dist_loss1 + dist_loss2)
        # eval
        elif "segment" in input_dict.keys():
            dist_loss = torch.mean(torch.square(dist[:,0]  - gt_distance[:,0])) * 15.0
            # dist_loss1 = torch.mean(torch.abs(dist[:,0] - gt_distance[:,0])) * 15.0 # ELYSIA
            # dist_loss2 = torch.mean(torch.abs(dist[:,1] - gt_distance[:,1])) * 15.0 # ELYSIA
            # dist_loss = torch.mean(torch.abs(dist[:,0] - gt_distance)) * dist_weight
            # dist_loss = F.binary_cross_entropy_with_logits(dist[:,0], gt_distance) * dist_weight
            loss = self.criteria(seg_logits, segment)
            return dict(loss=loss + dist_loss, seg_logits=seg_logits)
            # return dict(loss=loss + dist_loss1 + dist_loss2, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits, dist=dist)


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
