import os
import glob
import open3d as o3d
import torch
import numpy as np
import pickle
import tqdm 
import copy
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from representation.primitive import CatmullRomSurface, CatmullRomCurve
from utils.rotation_pytorch3d import quaternion_to_matrix
from utils.color_print import *
from utils.constant import *
from utils.plot import generate_cylinder_along_curve_batch
from utils.graph import find_layers_and_paths, calculate_accumulation_values, max_geodesic_distance, load_class, load_parent
from utils.pcd import find_farthest_points_batch_torch, grid_to_mesh, find_nearest_pair
from utils.tool import polygon_area_torch
from utils.pca import NodePCA

def read_int_list(filepath):
    with open(filepath, "r") as f:
        # read all lines, strip whitespace, and convert to int
        return [int(line.strip()) for line in f if line.strip()]

def overfit(mesh_dir, species='soybean', PCD_MAX_NUM=30000, **kwargs):

    retrain = kwargs.get('retrain', False)
    retrain_stem = kwargs.get('retrain_stem', False)
    retrain_leaf = kwargs.get('retrain_leaf', False)
    blending_weight = 1

    meta_name = os.path.basename(mesh_dir)
    blue_print('Start overfitting species: {}, plant id: {}'.format(species, meta_name))

    ######## 1. Load point cloud data ########
    info_folder = os.path.join(mesh_dir, 'info')
    fit_folder = os.path.join(mesh_dir, 'fit')
    raw_folder = os.path.join(mesh_dir, 'raw')
    leaf_deform_folder = os.path.join(mesh_dir, 'leaf_template_deform')
    stem_deform_folder = os.path.join(mesh_dir, 'stem_template_deform')
    os.makedirs(fit_folder, exist_ok=True)
    os.makedirs(leaf_deform_folder, exist_ok=True)
    os.makedirs(stem_deform_folder, exist_ok=True)

    files = glob.glob(raw_folder + '/*.ply')
    files.sort()

    full_plant_path = os.path.join(mesh_dir, 'pcd.ply')
    assert os.path.exists(full_plant_path), 'full plant point cloud not found: {}'.format(full_plant_path)

    full_plant_o3d_pcd = o3d.io.read_point_cloud(full_plant_path)
    full_plant_np_pcd = np.asarray(full_plant_o3d_pcd.points)

    classes = load_class(os.path.join(info_folder, 'class.txt'))
    parents = load_parent(os.path.join(info_folder, 'parent.txt'))

    assert os.path.exists(os.path.join(info_folder, 'class.txt')), 'class.txt not found in {}'.format(info_folder)
    assert os.path.exists(os.path.join(info_folder, 'parent.txt')), 'parent.txt not found in {}'.format(info_folder)


    data_folder = f'sample_params/{species}'
    pca_leaf_2d = NodePCA(path=os.path.join(data_folder, '2d_leaf_pca.pth'))

    stem_pcd_paths = []
    leaf_pcd_paths = []
    stem_indices = []
    leaf_indices = []
    for index, file in enumerate(files):
        name = str(int(os.path.basename(file).split('.')[0]))
        if classes.get(name) == STEM_CLASS:
            stem_pcd_paths.append(file)
            stem_indices.append(name)
        elif classes.get(name) == LEAF_CLASS:
            leaf_pcd_paths.append(file)
            leaf_indices.append(name)
        elif classes.get(name) == FLOWER_CLASS:
            pass
        elif classes.get(name) == FRUIT_CLASS:
            pass
        else:
            raise ValueError('class not defined, {}:{}'.format(name, classes.get(name)))

    ######## 2. Surface fitting ########
    surfaces = edict()
    surface_points = edict()
    surface_points_sr = edict()
    surface_points_flat = edict()
    surface_roots = edict()
    surface_points_gt = o3d.geometry.PointCloud()
    for pcd_path in tqdm.tqdm(leaf_pcd_paths):

        print('fitting ', pcd_path)
        pcd_name = os.path.basename(pcd_path)
        pcd_index = int(pcd_name.split('.')[0])
        pcd = o3d.io.read_point_cloud(pcd_path)

        surface_points_gt += pcd # add to the ground truth point cloud for evaluation later

        root_path = os.path.join(info_folder, '{}_root.txt'.format(str(pcd_index).zfill(3)))
        root_point = np.loadtxt(root_path)
        zero = np.zeros(3)

        print('root point:', root_point)
        gt = np.asarray(pcd.points)
        gt = gt - root_point
        gt = torch.tensor(gt, dtype=torch.float, device='cuda')

        # _pcd = o3d.geometry.PointCloud()
        # _pcd.points = o3d.utility.Vector3dVector(pcd_points.detach().cpu().numpy())
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=zero)
        # o3d.visualization.draw_geometries([_pcd, axis], mesh_show_back_face=True)

        # random select 10000 points
        if gt.shape[0] > PCD_MAX_NUM:
            gt = gt[np.random.choice(gt.shape[0], PCD_MAX_NUM, replace=False), :]
        
        surf = CatmullRomSurface(species=species, shape_pca=pca_leaf_2d)

        fit_save_path = os.path.join(fit_folder, '{}_fit.pth'.format(str(pcd_index).zfill(3)))
        if os.path.exists(fit_save_path) and not retrain and not retrain_leaf:
            green_print('{} already exists'.format(pcd_name))
            surf.load(fit_save_path)
            PLOT_FLAG = False

        if retrain or retrain_leaf or (not os.path.exists(fit_save_path)):
            if species == 'soybean':
                surf.fit(gt, iters1=100, iters2=1500)
            elif species == 'maize':
                surf.fit(gt, iters1=1000, iters2=2000)
            else:
                surf.fit(gt, iters1=500, iters2=2000) # 3000
                
            surf.save(fit_save_path)
            PLOT_FLAG = True

        # save for 3d pca
        with torch.no_grad():
            inner_points, flat = surf.evaluate(w=1, super_resolution_rate=1, mean_shape=True)[0:2] # (43, 45, 3)

            # leaf_main_vein = inner_points[43//2]
            # self_axis = calculate_local_axes(inner_points, mode='surface', p0=leaf_main_vein[0], p1=leaf_main_vein[-1])[0]
            # self_axis = torch.tensor(self_axis, dtype=torch.float, device='cuda')
            # # skeleton_rot = inner_points @ self_axis.T # the data that we need for PCA (shape-free)
            # skeleton_rot = inner_points # the data that we need for PCA (shape-free)

        # pcd_ = o3d.geometry.PointCloud()
        # pcd2_ = o3d.geometry.PointCloud()
        # pcd3_ = o3d.geometry.PointCloud()
        # pcd_.points = o3d.utility.Vector3dVector(skeleton_rot.detach().cpu().numpy().reshape(-1,3))
        # pcd2_.points = o3d.utility.Vector3dVector(inner_points.detach().cpu().numpy().reshape(-1,3))
        # pcd3_.points = o3d.utility.Vector3dVector(flat.detach().cpu().numpy().reshape(-1,3))
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=zero)
        # o3d.visualization.draw_geometries([pcd_,  pcd2_, pcd3_, axis], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([pcd_, pcd3_, axis], mesh_show_back_face=True)
        torch.save(inner_points, os.path.join(leaf_deform_folder, pcd_name.replace('.ply', '.pth')))

        # visualize
        super_resolution_rate = 1
        # super_resolution_rate = 1
        with torch.no_grad():
            inner_points, inner_points_flat, inner_points_sr =\
                    surf.evaluate(w=blending_weight, super_resolution_rate=super_resolution_rate)

        # a,b,c = surf.invert(inner_points)
        # inner_points_ = surf.evaluate(main_rotation=a, sub_rotation_l=b, sub_rotation_r=c)[0]

        # deform_coeff = pca_leaf_3d.encode(inner_points.reshape(1, -1))
        # shape_coeff = surf.dw # (k, 1)
        # surf_p_new = coeff_to_leaf(surf, shape_coeff, deform_coeff, super_resolution_rate=12)

        rot = quaternion_to_matrix(surf.quat)
        gt_rotated = gt @ rot.T
        s = surf.s.item()
        gt_rotated_viz = gt_rotated.detach().cpu().numpy() * s
        # randomly pick 1000 points for visualization
        _n = gt_rotated_viz.shape[0]
        gt_rotated_viz = gt_rotated_viz[np.random.choice(_n, min(1000, _n), replace=False)]
        inner_points_viz = inner_points.detach().cpu().numpy() # using super resolution
        inner_points_sr_viz = inner_points_sr.detach().cpu().numpy()
        # inner_points_sr_pca_viz = surf_p_new.detach().cpu().numpy()
        inner_points_flat_viz = inner_points_flat.detach().cpu().numpy()

        if PLOT_FLAG:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(inner_points_viz[..., 0], inner_points_viz[..., 1], inner_points_viz[..., 2], color='g', rstride=1, cstride=1, alpha=1, label='curved leaf')
            # ax.plot_surface(inner_points_flat_viz[..., 0], inner_points_flat_viz[..., 1], inner_points_flat_viz[..., 2], color='orange', rstride=1, cstride=1, alpha=0.5, label='flat leaf')
            ax.scatter(gt_rotated_viz[:, 0], gt_rotated_viz[:, 1], gt_rotated_viz[:, 2], c='b', s=10, marker='o', label='input pcd')
            ax.scatter(zero[0], zero[1], zero[2], c='r', s=100, marker='x', label='root point')
            # set view
            # ax.view_init(elev=75, azim=30)
            plt.legend()
            # set axis limits
            x_min, y_min, z_min = inner_points_flat_viz.reshape(-1, 3).min(axis=0)
            x_max, y_max, z_max = inner_points_flat_viz.reshape(-1, 3).max(axis=0)
            x_mean = (x_max + x_min) / 2
            y_mean = (y_max + y_min) / 2
            z_mean = (z_max + z_min) / 2
            max_range = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) / 2
            max_range_scale = max_range
            ax.set_xlim(x_mean - max_range_scale, x_mean + max_range_scale)
            ax.set_ylim(y_mean - max_range_scale, y_mean + max_range_scale)
            ax.set_zlim(z_mean - max_range_scale, z_mean + max_range_scale)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set camera position
            ax.view_init(elev=45, azim=30)
            plt.savefig(os.path.join(fit_folder, '{}_fit.png'.format(str(pcd_index).zfill(3))), dpi=300)
            # plt.show(block=True)
            plt.close()

        area = polygon_area_torch(inner_points)
        area_flat = polygon_area_torch(inner_points_flat)

        init_area = area_flat.sum()
        final_area = area.sum()
        if PLOT_FLAG:
            area_info = torch.cat([area, area_flat], dim=-1) # (h, w, 2)
            # save area info as np array
            # area_info_path = pcd_path.replace(raw_folder, info_folder).replace('.ply', '_area.pth'.format(surf_res[0], surf_res[1]))
            # np.savetxt(area_info_path, area_info.detach().cpu().numpy())
            # green_print('area info saved to {}'.format(area_info_path))
        print(f'Initial area: {init_area}, Final area: {final_area}, error %: {(final_area - init_area) / init_area * 100}')

        surfaces[str(pcd_index)] = surf
        surface_roots[str(pcd_index)] = root_point

        realworld_p = inner_points_viz / s
        inv_rot = np.linalg.inv(rot.detach().cpu().numpy())
        h, w = realworld_p.shape[:2]
        realworld_p = (inv_rot @ realworld_p.reshape(-1, 3).T).T + root_point
        surface_points[str(pcd_index)] = realworld_p

        realworld_p_sr = inner_points_sr_viz / s
        realworld_p_sr = (inv_rot @ realworld_p_sr.reshape(-1, 3).T).T + root_point
        surface_points_sr[str(pcd_index)] = realworld_p_sr

        realworld_p_flat = inner_points_flat_viz / s
        h, w = realworld_p_flat.shape[:2]
        realworld_p_flat = (inv_rot @ realworld_p_flat.reshape(-1, 3).T).T + root_point
        surface_points_flat[str(pcd_index)] = realworld_p_flat

    ######## 3. Curve fitting ########

    curves = edict()
    curve_points = edict()
    curve_points_gt = o3d.geometry.PointCloud()
    curve_cps = edict()
    FLAG = 0

    for pcd_path in tqdm.tqdm(stem_pcd_paths):

        print('fitting ', pcd_path)
        pcd_name = os.path.basename(pcd_path)
        pcd_index = int(pcd_name.split('.')[0])
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_points_np = np.asarray(pcd.points)
        gt = torch.tensor(pcd_points_np, dtype=torch.float, device='cuda')

        curve_points_gt += pcd # add to the ground truth point cloud for evaluation later

        if gt.shape[0] > PCD_MAX_NUM:
            gt = gt[np.random.choice(gt.shape[0], PCD_MAX_NUM, replace=False), :]

        # load parent pcd and child pcd to constrain the end of the curve
        p0_end = None # only main stem has no parent
        p1_end = None
        endpoint_txt = os.path.join(fit_folder, '{}_endpoints.txt'.format(str(pcd_index).zfill(3)))

        if os.path.exists(endpoint_txt) and not retrain_stem:
            green_print('{} already exists'.format(pcd_name))
            _data = np.loadtxt(endpoint_txt)
            if _data.shape[0] == 2:
                p0_end, p1_end = _data
                p0_end = torch.tensor(p0_end, dtype=torch.float, device='cuda')
                p1_end = torch.tensor(p1_end, dtype=torch.float, device='cuda')
            else:
                p0_end = torch.tensor(_data, dtype=torch.float, device='cuda')
            max_dist = torch.norm(p0_end - p1_end).detach().cpu().numpy()
        else:
            root_path0 = pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_root0.txt')
            root_path1 = pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_root1.txt')

            if os.path.exists(root_path0) and os.path.exists(root_path1):
                p0_end = np.loadtxt(root_path0)
                p1_end = np.loadtxt(root_path1)
                p0_end = torch.tensor(p0_end, dtype=torch.float, device='cuda')
                p1_end = torch.tensor(p1_end, dtype=torch.float, device='cuda')
                max_dist = torch.norm(p0_end - p1_end).detach().cpu().numpy()
            else:
                # loop all leaf
                child_leaf_count = 0
                child_stem_count = 0
                child_id = None
                for k, v in parents.items():
                    if v == int(os.path.basename(pcd_path).split('.')[0]) and classes[k] == LEAF_CLASS:
                        child_leaf_count += 1
                        child_id = k
                    elif v == int(os.path.basename(pcd_path).split('.')[0]) and classes[k] == STEM_CLASS:
                        child_stem_count += 1
                if child_leaf_count == 1 and child_stem_count == 0:
                    ref = surface_roots[child_id]
                    # farthest_points = get_farthest_points(pcd_points_np, selected_point=p1_end, k=5, nn_k=10)
                    p_to_ref = np.linalg.norm(pcd_points_np - ref, axis=-1)
                    nearest_points_index = np.argsort(p_to_ref)[:4]
                    nearest_points = pcd_points_np[nearest_points_index]
                    p1_end = nearest_points.mean(axis=0)
                    p_to_p1_dist = np.linalg.norm(pcd_points_np - p1_end, axis=-1)
                    farthest_points_index = np.argsort(p_to_p1_dist)[-10:]
                    farthest_points = pcd_points_np[farthest_points_index]
                    p0_end = farthest_points.mean(axis=0)
                    max_dist = np.linalg.norm(p0_end - p1_end)

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(pcd_points_np[:, 0], pcd_points_np[:, 1], pcd_points_np[:, 2], c='b', label='input pcd', alpha=0.5, s=1)
                    # ax.scatter(farthest_points[:, 0], farthest_points[:, 1], farthest_points[:, 2], c='r', label='farthest points', alpha=0.5, s=20)
                    # ax.scatter(p1_end[0], p1_end[1], p1_end[2], c='g', label='p1_end', s=100)
                    # plt.show(block=True)

                    p0_end = torch.tensor(p0_end, dtype=torch.float, device='cuda')
                    p1_end = torch.tensor(p1_end, dtype=torch.float, device='cuda')

                else:
                
                    best_dist = 0
                    for k in [5, 10, 20, 30,]:
                        if gt.shape[0] <= k:
                            continue
                        max_dist, p0_end, p1_end = max_geodesic_distance(pcd_points_np, k=k)
                        if max_dist > best_dist:
                            best_dist = max_dist
                    _p0, _p1 = find_farthest_points_batch_torch(gt)
                    _p0 = _p0.detach().cpu().numpy()
                    _p1 = _p1.detach().cpu().numpy()
                    _dist =  np.linalg.norm(_p0 - _p1)
                    if _dist > best_dist:
                        p0_end, p1_end = _p0, _p1
                        max_dist = _dist

                    parent = parents[str(pcd_index)]
                    # p0 should closer to parent than p1
                    if parent != -1:
                        parent_pcd_path = os.path.join(raw_folder, '{}.ply'.format(str(parent).zfill(3)))
                        parent_pcd = o3d.io.read_point_cloud(parent_pcd_path)
                        parent_pcd_points = np.asarray(parent_pcd.points)
                        p0_end_to_parent_dist = np.linalg.norm(p0_end - parent_pcd_points, axis=-1)
                        p1_end_to_parent_dist = np.linalg.norm(p1_end - parent_pcd_points, axis=-1)
                        if np.min(p1_end_to_parent_dist) < np.min(p0_end_to_parent_dist):
                            p0_end, p1_end = p1_end, p0_end

                    p0_end = torch.tensor(p0_end, dtype=torch.float, device='cuda')
                    p1_end = torch.tensor(p1_end, dtype=torch.float, device='cuda')
                    # parent = parents[str(pcd_index)]
                    # if parent != -1:
                    #     parent_pcd_path = os.path.join(processed_folder, '{}.ply'.format(str(parent).zfill(3)))
                    #     parent_pcd = o3d.io.read_point_cloud(parent_pcd_path)
                    #     parent_pcd_points = np.asarray(parent_pcd.points)

                    #     fig = plt.figure()
                    #     ax = fig.add_subplot(111, projection='3d')
                    #     ax.scatter(pcd_points[:, 0].detach().cpu().numpy(), pcd_points[:, 1].detach().cpu().numpy(), pcd_points[:, 2].detach().cpu().numpy(), c='b', label='input pcd', alpha=0.5, s=1)
                    #     ax.scatter(parent_pcd_points[:, 0], parent_pcd_points[:, 1], parent_pcd_points[:, 2], c='black', label='parent pcd', alpha=0.5, s=1)
                    #     ax.scatter(p0_end[0].detach().cpu().numpy(), p0_end[1].detach().cpu().numpy(), p0_end[2].detach().cpu().numpy(), c='r', label='p0_end', s=100)
                    #     ax.scatter(p1_end[0].detach().cpu().numpy(), p1_end[1].detach().cpu().numpy(), p1_end[2].detach().cpu().numpy(), c='g', label='p1_end', s=100)
                    #     plt.legend()
                    #     plt.show(block=True)

                    if parent == -1:
                        # root point is at the origin
                        root_point = torch.tensor([0, 0, 0], dtype=torch.float, device='cuda')
                        if torch.norm(p0_end - root_point) < torch.norm(p1_end - root_point):
                            p0_end, p1_end = p0_end, p1_end
                        else:
                            p0_end, p1_end = p1_end, p0_end
            
            # save p0_end and p1_end
            np.savetxt(endpoint_txt, torch.stack([p0_end, p1_end]).detach().cpu().numpy())
            green_print('endpoints saved to {}'.format(endpoint_txt))

        # manually set strong constraint to small stem
        strong_constraint_path = os.path.join(info_folder, 'strong_constraint.txt')
        strong_constraint_list = []
        if os.path.exists(strong_constraint_path):
            strong_constraint_list = read_int_list(strong_constraint_path)
        force_straight = True if int(pcd_index) in strong_constraint_list else False

        fit_save_path = os.path.join(fit_folder, '{}_fit.pth'.format(str(pcd_index).zfill(3)))
        
        curve = CatmullRomCurve(alpha=0.5, p0_end=p0_end, p1_end=p1_end, force_straight=force_straight, force_smoothness=((parents[str(pcd_index)]!=-1) or species=='maize'))
        if os.path.exists(fit_save_path) and not retrain and not retrain_stem:
            green_print('{} already exists'.format(pcd_name))
            curve.load(fit_save_path)
            PLOT_FLAG = False
        else:
            curve.fit(gt, iters1=400, iters2=2500, input_scale=torch.tensor(max_dist, dtype=torch.float, device='cuda')) 
            curve.save(fit_save_path)
            PLOT_FLAG = True
        
        # save for 3d pca
        with torch.no_grad():
            # inner_points = torch.tensor(extract_curve_local_info(curve)['cp_w_local'])
            inner_points = curve.evaluate(num_points=2, w=1)[0]
        torch.save(inner_points, os.path.join(stem_deform_folder, pcd_name.replace('.ply', '.pth')))
        
        s = curve.s.item()
        if FLAG < 50:
            curved_cp, p_curve = curve.evaluate(num_points=2, w=blending_weight) # min(max(int(s * 5000), 10), 50)
            FLAG = FLAG + 1
        else:
            curved_cp, p_curve = curve.evaluate(num_points=2, w=blending_weight)
        
        quat = curve.quat
        rot = quaternion_to_matrix(quat)
        # curved_cp_viz = curved_cp.detach().cpu().numpy()
        # pcd_points_viz = ((pcd_points - curve.p0_end) @ rot.T).detach().cpu().numpy() * l
        # p_curve_viz = p_curve.detach().cpu().numpy()
        pcd_points_viz = gt.detach().cpu().numpy()
        p_curve_viz = ((p_curve / s) @ rot + curve.p0_end).detach().cpu().numpy()
        curved_cp_viz = ((curved_cp / s) @ rot + curve.p0_end).detach().cpu().numpy()

        # visualize curve fitting
        if PLOT_FLAG:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(p_curve_viz[:, 0], p_curve_viz[:, 1], p_curve_viz[:, 2], c='g', label='fitted curve')
            ax.scatter(pcd_points_viz[:, 0], pcd_points_viz[:, 1], pcd_points_viz[:, 2], c='b', label='input pcd', alpha=0.5, s=1)
            ax.scatter(curved_cp_viz[:, 0], curved_cp_viz[:, 1], curved_cp_viz[:, 2], c='r', label='control points', s=100)
            p_end = torch.stack([p0_end, p1_end], dim=0).detach().cpu().numpy()
            ax.scatter(p_end[:, 0], p_end[:, 1], p_end[:, 2], c='black', label='end points', s=300, marker='x')
            plt.legend()
            plt.title(f'index: {pcd_index}')
            plt.savefig(pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_fit.png'), dpi=300)
            # plt.show(block=True)
        
        curves[str(pcd_index)] = curve
        curve_points[str(pcd_index)] = p_curve_viz
        curve_cps[str(pcd_index)] = curved_cp_viz
    
    ######## 4. Automatic assembly ########
    rel_translations = edict()
    # parents_values = list(parents.values())
    
    index_to_names = [int(os.path.basename(file).split('.')[0]) for file in files]
    for i in range(len(index_to_names)):
        rel_translations[str(index_to_names[i])] = np.zeros(3).astype(np.float32)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for curve_p in curve_points.values():
    #     ax.scatter(curve_p[:, 0], curve_p[:, 1], curve_p[:, 2], label='fitted curve')
    # for surface_p in surface_points.values():
    #     ax.scatter(surface_p[:, 0], surface_p[:, 1], surface_p[:, 2], label='fitted surface')
    # ax.scatter(full_plant_np_pcd[:, 0], full_plant_np_pcd[:, 1], full_plant_np_pcd[:, 2], label='full plant', alpha=0.1, s=1)
    # plt.show()

    # for curve in curves.values():
    #     print(curve.thickness.item())

    layers, paths = find_layers_and_paths(parents)
    for index, layer in enumerate(layers.values()): 
        if index == 0:
            continue # skip root
        for node in layer:
            parent = parents[str(node)]
            parent_pcd_points = curve_points[str(parent)].reshape(-1, 3)
            parent_radius = curves[str(parent)].thickness.item()

            # find distance between parent and child
            if classes[str(node)] == 0: # leaf
                p_src = surface_points[str(node)]
                p_src = p_src.reshape(43,45,3)[:, 0] # use end of the leaf
            elif classes[str(node)] == 1: # stem
                p_src = curve_points[str(node)]
            else:
                continue

            p_src = p_src.reshape(-1, 3)
            p0, p1 = find_nearest_pair(parent_pcd_points, p_src)
            vec = p0 - p1
            len_vec = np.linalg.norm(vec)
            # rel_translations[str(node)] = vec
            rel_translations[str(node)] = p0 * (1 - parent_radius / len_vec) + p1 * (parent_radius / len_vec - 1)

    abs_translations = calculate_accumulation_values(parents, rel_translations)
    
    abs_translation_path = os.path.join(mesh_dir, 'info', 'abs_translations.pkl')
    with open(abs_translation_path, 'wb') as f:
        pickle.dump(abs_translations, f)
    green_print('abs_translations saved to {}'.format(abs_translation_path))

    ######## 5. Visualization ########
    """
    The parametric shape of each leaf/stem is disconnected with its parent branch.
    So we calulate the relative translation between each leaf/stem and its parent branch,
    then subtract the translation along the kinematic tree (absolute translation) to make them connected.
    """
    auto_connect = kwargs.get('auto_connect', False)
    viz_flat_leaf = False
    
    stem_meshs = []
    leaf_meshs = []
    _surface_points = surface_points_flat if viz_flat_leaf else surface_points
    for k, p in _surface_points.items():
        p = p.reshape(43, 45, 3)
        if auto_connect:
            p = p + abs_translations[k]
        mesh = grid_to_mesh(p)
        leaf_meshs.append(mesh)
        
    for k, p in curve_points.items():
        if auto_connect:
            p = p + abs_translations[k]
        thickness = curves[k].thickness.item()
        thickness = np.clip(thickness, a_min=0.0004, a_max=1e3) # FIXME: remove this line later
        p_cylinder = generate_cylinder_along_curve_batch(p, thickness, k=25)
        mesh = grid_to_mesh(p_cylinder)
        stem_meshs.append(mesh)

        _temp = o3d.geometry.PointCloud()
        _temp.points = o3d.utility.Vector3dVector(p_cylinder.reshape(-1, 3))
        # pcd_for_cd = pcd_for_cd + _temp

    # save the point cloud without auto connection
    for k, p in surface_points.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p.reshape(-1, 3))
        o3d.io.write_point_cloud(os.path.join(fit_folder, '{}.ply'.format(str(int(k)).zfill(3))), pcd)
    
    parent_id = [k for k, v in parents.items() if v == -1][0]
    for k, p in curve_points.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        o3d.io.write_point_cloud(os.path.join(fit_folder, '{}.ply'.format(str(int(k)).zfill(3))), pcd)

    a_viz = full_plant_np_pcd[np.random.choice(full_plant_np_pcd.shape[0], full_plant_np_pcd.shape[0] // 10, replace=False)]
    a_pcd = o3d.geometry.PointCloud()
    a_pcd.points = o3d.utility.Vector3dVector(a_viz)


    leaf_merged = copy.deepcopy(leaf_meshs[0])
    stem_merged = copy.deepcopy(stem_meshs[0])
    for i in range(1, len(leaf_meshs)):
        leaf_merged += leaf_meshs[i]
    for i in range(1, len(stem_meshs)):
        stem_merged += stem_meshs[i]
    leaf_merged.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(leaf_merged.vertices), 3))*0.7)
    stem_merged.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(stem_merged.vertices), 3))*0.7)

    visualize = kwargs.get('visualize', False)

    if visualize:
        a_pcd = o3d.geometry.PointCloud()
        a_pcd.points = o3d.utility.Vector3dVector(a_viz)
        a_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        if auto_connect:
            o3d.visualization.draw_geometries([leaf_merged, stem_merged], mesh_show_back_face=True)
        else:
            o3d.visualization.draw_geometries([surface_points_gt, curve_points_gt, leaf_merged, stem_merged], mesh_show_back_face=True)
            # o3d.visualization.draw_geometries([surface_points_gt, curve_points_gt], mesh_show_back_face=True)

    merged = leaf_merged + stem_merged
    
    bbox = merged.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    scale = 1.0 / max(extent) * 0.9

    leaf_merged.translate(-center)
    leaf_merged.scale(scale, center=np.zeros(3))
    stem_merged.translate(-center)
    stem_merged.scale(scale, center=np.zeros(3))
    
    leaf_save_path = os.path.join(f'/home/tianhang/data/{species}/fitted_mesh_viz', f'{meta_name}_leaf.ply')
    stem_save_path = os.path.join(f'/home/tianhang/data/{species}/fitted_mesh_viz', f'{meta_name}_stem.ply')
    os.makedirs(os.path.dirname(leaf_save_path), exist_ok=True)
    o3d.io.write_triangle_mesh(leaf_save_path, leaf_merged)
    o3d.io.write_triangle_mesh(stem_save_path, stem_merged)
    green_print('Done fitting {} to {}'.format(meta_name, os.path.dirname(leaf_save_path)))

if __name__ == '__main__':

    params = {
        'retrain': False,
        'retrain_stem': True,
        'retrain_leaf': False,
        'auto_connect': True,
        'visualize': True,
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, default='sample_point_cloud/val/7a', help='path to the mesh directory')
    parser.add_argument('--species', type=str, default='soybean', help='plant species')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the model')
    parser.add_argument('--retrain_stem', action='store_true', help='whether to retrain the stem model')
    parser.add_argument('--retrain_leaf', action='store_true', help='whether to retrain the leaf model')

    args = parser.parse_args()

    species = args.species
    mesh_dir = args.mesh_dir
    retrain = args.retrain
    retrain_stem = args.retrain_stem
    retrain_leaf = args.retrain_leaf

    new_params = {}
    for k, v in params.items():
        new_params[k] = getattr(args, k, v)
    new_params['retrain'] = retrain
    new_params['retrain_stem'] = True
    new_params['retrain_leaf'] = retrain_leaf
    overfit(mesh_dir, species, **new_params)