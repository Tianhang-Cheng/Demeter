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

def get_demeter_model_size(fit_folder, ignore_stem=False, ignore_surface=False):
    total_parameters_count = 0
    pths_path = glob.glob(fit_folder + '/*.pth')

    for pth_path in pths_path:
        model = torch.load(pth_path, weights_only=True) # is a dict

        # leaf: dict_keys(['quat', 's', 'dw', 'main_rotation', 'sub_rotation_left', 'sub_rotation_right'])
            # pca coeff: dw, main_rotation, sub_rotation_left, sub_rotation_right
            # non-pca coeff: quat, s
        # stem: dict_keys(['s', 'l', 'main_rotation', 'thickness', 'quat'])

        num = 0

        shape_pca_n_params = 8 # papaya is 8
        shape_pca_n_params = 9 # soybean is 8
        deform_pca_n_params = 18 # FIXME: check this, soybean is 12

        # stem
        if 's' in model.keys() and 'l' in model.keys():
            print('model {} is a curve'.format(pth_path))
            if ignore_stem:
                continue
            num += 1 # s (scale)
            num += 3 # quat (rotation)
            num += 1 # translation
            num += 2 # graph connection
            raise NotImplementedError
        # leaf
        else:
            print('model {} is a surface'.format(pth_path))
            if ignore_surface:
                continue
            num += 1 # s (scale)
            num += 3 # quat (rotation)
            num += 1 # translation
            num += 2 # graph connection
            num += shape_pca_n_params # shape pca
            for key in model.keys():
                if 'rot' in key:
                    if isinstance(model[key], torch.Tensor):
                        num += model[key].numel()
            # num += deform_pca_n_params
        total_parameters_count += num

    # calculate KB for the float32 parameters
    KB = total_parameters_count * 4 / 1024
    print('Total number of parameters: {}, Total size: {} KB'.format(total_parameters_count, KB))