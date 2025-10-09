import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import torch
import glob

def get_chamfer_distance(gt_pcd, pred_pcd, normalize=True):
    """
    gt_pcd: ground truth point cloud, open3d.geometry.PointCloud
    pred_pcd: predicted point cloud, open3d.geometry.PointCloud
    """
    scale = 1.0
    if normalize:
        scale = np.linalg.norm(np.array(gt_pcd.get_max_bound()) - np.array(gt_pcd.get_min_bound()))

    gt_points = np.array(gt_pcd.points)
    pred_points = np.array(pred_pcd.points)

    gt_tree = cKDTree(np.array(gt_points))
    pred_tree = cKDTree(np.array(pred_points))

    pred_to_gt, _ = gt_tree.query(np.array(pred_points), k=1)
    pred_to_gt = np.mean(pred_to_gt)
    gt_to_pred, _ = pred_tree.query(np.array(gt_points), k=1)
    gt_to_pred = np.mean(gt_to_pred)
    # print('Baseline to GT:', baseline_to_gt)
    # print('GT to baseline:', gt_to_baseline)
    mean_cd = (pred_to_gt + gt_to_pred) / 2
    mean_cd = mean_cd / scale # normalize
    print('Chamfer distance:', mean_cd)
    return mean_cd

def intersection_and_union_gpu(output, target, semantic:int):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output).cuda()
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target).cuda()
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)

    area_intersection = ((output == semantic) & (target == semantic)).sum().reshape(1)
    area_output = (output == semantic).sum().reshape(1)
    area_target = (target == semantic).sum().reshape(1)
    area_union = area_output + area_target - area_intersection

    area_intersection = area_intersection.detach().cpu().numpy()
    area_union = area_union.detach().cpu().numpy()
    area_target = area_target.detach().cpu().numpy()
    return area_intersection, area_union, area_target