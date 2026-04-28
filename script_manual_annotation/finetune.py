import sys
sys.path.append('../Plant')

import os
import glob
import numpy as np
import open3d as o3d
import torch
import pickle
from representation.primitive import CatmullRomSurface, CatmullRomCurve
from utils.rotation_pytorch3d import quaternion_to_matrix
from utils.constant import *
from representation.build_graph import build_plant_graph
from utils.pcd import compute_frenet_serret_frame, visualize_frenet_frame
from utils.pca import NodePCA
from utils.metrics import get_chamfer_distance
from utils.plot import load_view_point
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import platform
import imageio.v2 as imageio
import copy
from utils.plot import draw_tree_with_colors
    
def _detach(x):
    return x.detach().cpu().numpy()

def finetune(meta_name, species='soybean', **kwargs):

    folder = f'/home/tianhang/data/{species}/pcd_processed'
    folder = os.path.join(folder, meta_name)

    raw_folder = os.path.join(folder, 'raw')
    os.makedirs(raw_folder, exist_ok=True)
    fit_folder = os.path.join(folder, 'fit')
    graph_folder = os.path.join(folder, 'graph')
    os.makedirs(graph_folder, exist_ok=True)
    seq_folder = os.path.join(folder, 'seq')
    os.makedirs(seq_folder, exist_ok=True)
    temp_pcd_path = os.path.join(folder, 'info', 'temp.ply')
    parent_txt = os.path.join(folder, 'info','parent.txt') 
    connect_txt = os.path.join(folder, 'info','connect.txt') # 0: not labeled, 1: labeled
    class_txt = os.path.join(folder, 'info','class.txt') # 0: leaf, 1: stem,  -1: not labeled
    abs_translation_pkl = os.path.join(folder, 'info','abs_translations.pkl')

    if os.path.exists(temp_pcd_path):
        os.remove(temp_pcd_path)
    files = glob.glob(raw_folder + '/*.ply')
    files.sort()
    files_dict = edict()
    for file in files:
        name = int(os.path.basename(file).split('.')[0])
        files_dict[str(name)] = file

    # load connect annotation
    connects = edict()
    assert os.path.exists(connect_txt)
    with open(connect_txt, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        name, is_labeled = line.strip().split(' ')
        connects[str(name)] = int(is_labeled)

    # load class annotation
    classes = edict()
    assert os.path.exists(class_txt)
    with open(class_txt, 'r') as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        name, _class = line.strip().split(' ')
        classes[str(name)] = int(_class)

    # load parent annotation
    parents = edict()
    UNIQUE_ROOT = 0
    assert os.path.exists(parent_txt)
    with open(parent_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        child, parent = line.strip().split('->')
        assert classes.get(str(int(parent))) == STEM_CLASS or int(parent) == -1, 'The parent should be stem or no parent'
        if int(parent) == -1:
            assert UNIQUE_ROOT == 0, 'There should be only one root'
            UNIQUE_ROOT += 1
        parents[str(int(child))] = int(parent)
    
    # draw_tree_with_colors(parents, c=classes)

    # classify path
    stem_pcd_paths = []
    leaf_pcd_paths = []
    for file in files:
        name = str(int(os.path.basename(file).split('.')[0]))
        if classes.get(name) == STEM_CLASS:
            stem_pcd_paths.append(file)
        elif classes.get(name) == LEAF_CLASS:
            leaf_pcd_paths.append(file)


    """
    read raw point cloud, put in "surface_points_gt"
    read fitted surface and curve, put in "surfaces" and "curves", and put their control points in "template_points"
    """
    surfaces = edict()
    curves = edict()
    template_points = edict()
    node_axis_along_parent_stem_all = edict() # store the local axis of the node along the parent stem for all points

    shape_pca = NodePCA()
    shape_species = species
    if species == 'ribes':
        shape_species = 'papaya'
    if species == 'tobacco' or species == 'maize' or species == 'rose':
        shape_species = 'soybean'
    shape_pca.load(f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{shape_species}_rotated_mask/skeleton_pca.pth')

    surface_points_gt = o3d.geometry.PointCloud()

    with torch.no_grad():

        for pcd_path in leaf_pcd_paths:

            pcd = o3d.io.read_point_cloud(pcd_path)
            surface_points_gt += pcd # add to the ground truth point cloud for evaluation later

            pcd_index = os.path.basename(pcd_path).split('.')[0]
            surf = CatmullRomSurface(species=species, shape_pca=shape_pca)
            fit_save_path = pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_fit.pth')
            surf.load(fit_save_path)
            inner_points, inner_points_flat, _ = surf.evaluate(w=1)
            inner_points_viz = _detach(inner_points)
            h, w = inner_points_viz.shape[:2]

            root_path = os.path.join(fit_folder, '{}_root.txt'.format(pcd_index))
            root_point = np.loadtxt(root_path)

            surfaces[str(int(pcd_index))] = surf

            s = surf.s.item()
            rot = _detach(quaternion_to_matrix(surf.quat))
            template_points[str(int(pcd_index))] = ((inner_points_viz / s) @ rot) + root_point
            del surf

        for pcd_path in stem_pcd_paths:
            pcd_index = os.path.basename(pcd_path).split('.')[0]
            curve = CatmullRomCurve()
            fit_save_path = pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_fit.pth')
            curve.load(fit_save_path)

            s = curve.s.item()
            rot = _detach(quaternion_to_matrix(curve.quat))
            num_points = 2
            cp, p_curve = curve.evaluate(num_points=num_points)
            cp = _detach(cp)
            p_curve = _detach(p_curve)

            endpoint_path = os.path.join(fit_folder, '{}_endpoints.txt'.format(pcd_index))
            endpoint = np.loadtxt(endpoint_path)[0]

            cp = (cp / s) @ rot + endpoint
            p_curve = (p_curve / s) @ rot + endpoint

            # compute_frenet_serret_frame
            # _axis_all = calculate_local_axes(cp, repeats=num_points-1, mode='default2')
            # _axis_all = _axis_all[0:p_curve.shape[0]]
            _axis_all = compute_frenet_serret_frame(cp, repeats=num_points-1)
            _axis_all = _axis_all[0:p_curve.shape[0]]

            # rot 45
            # random_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            # cp_rot = cp @ random_rot.T
            # # _axis_all_rot = calculate_local_axes(cp_rot, repeats=num_points-1, mode='default2')
            # # _axis_all_rot = _axis_all_rot[0:p_curve.shape[0]]
            # _axis_all_rot = compute_frenet_serret_frame(cp_rot)

            # _axis_all_direct_rot = _axis_all @ random_rot.T
            # # print('#')
            # # print(_axis_all_rot[0])
            # # print(_axis_all_direct_rot[0])

            # visualize_frenet_frame(cp, _axis_all, scale=np.linalg.norm(cp[1]-cp[0]))
            # visualize_frenet_frame(cp_rot, _axis_all_rot, scale=np.linalg.norm(cp_rot[1]-cp_rot[0]))

            # cp_straight = np.linspace(cp[0], cp[-1], 9)
            # axis_new = compute_frenet_serret_frame(cp_straight)
            # visualize_frenet_frame(cp_straight, axis_new, scale=np.linalg.norm(cp[1]-cp[0]))

            node_axis_along_parent_stem_all[str(int(pcd_index))] = _detach(_axis_all)

            curves[str(int(pcd_index))] = curve
            template_points[str(int(pcd_index))] = p_curve
            del curve

    """
    find a matrix that make main stem vertical (parent value == -1) for visualization
    """
    main_stem = [k for k, v in parents.items() if v == -1][0].zfill(3)
    main_stem_end_points_bottom = template_points[str(int(main_stem))][0]

    with open(abs_translation_pkl, 'rb') as f:
        abs_translation = pickle.load(f)

    connected_template_pcd = edict()
    for k, v in template_points.items():
        v = v - main_stem_end_points_bottom + abs_translation[str(int(k))] # only translate the point cloud
        # v = v - main_stem_end_points_bottom
        template_points[k] = v
        rotated_pcd = o3d.geometry.PointCloud()
        rotated_pcd.points = o3d.utility.Vector3dVector(template_points[k].reshape(-1, 3))
        connected_template_pcd[str(k)] = rotated_pcd

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # axis.translate([0, 0, 0])
    # t_aligned_pcds_items = [v for k, v in connected_template_pcd.items()]
    # o3d.visualization.draw_geometries([axis]+t_aligned_pcds_items)

    plant_graph = build_plant_graph(species, meta_name, parents, classes, curves, surfaces, fit_folder,
                                    main_stem_end_points_bottom, connected_template_pcd, node_axis_along_parent_stem_all, 
                                    retrain=kwargs.get('retrain', False),
                                    visualize=kwargs.get('visualize', False),
                                    return_graph=True,
                                    n_iter=300 if species=='soybean' else 300)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    axis.translate([0, 0, 0])

    # gt_pcd_path = os.path.join(f'/home/tianhang/data/{species}/pcd_clean_join', f'{meta_name}.ply')
    # gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    # o3d.visualization.draw_geometries([gt_pcd, surface_points_gt, axis], mesh_show_back_face=True)
    from utils.graph import load_parent, save_parent, load_class, save_class
    save_dir = os.path.join(f'/home/tianhang/code/Demeter/sample_params/{species}/{meta_name}')
    os.makedirs(save_dir, exist_ok=True)
    info_dir = os.path.join(save_dir, 'info')
    os.makedirs(info_dir, exist_ok=True)
    save_parent(parents, os.path.join(info_dir, 'parent.txt'))
    save_class(classes, os.path.join(info_dir, 'class.txt'))
    plant_graph.save(os.path.join(save_dir, f'graph.pkl'))

    _=1

    # index = 0
    # edit_id = 6
    # for t in np.linspace(1.5, 0.5, 30):
    #     p_final = plant_graph.generate(output_format='mesh', color='gray', align_global=True, 
    #                                    stem_blend_weight=2, leaf_shape_blend_weight=1, leaf_deform_blend_weight=1,
    #                                    stem_articulation_weight=1, all_the_same=True,
    #                                    edit_id=edit_id)
        
    #     # img_save_path = f'/home/tianhang/code/Plant/viz/edit/leaf_{edit_id}_deform/{index:03d}.png' # 7
    #     # img_save_path = f'/home/tianhang/code/Plant/viz/edit/leaf_{edit_id}_shape/{index:03d}.png' # 7
    #     # img_save_path = f'/home/tianhang/code/Plant/viz/edit/stem_{edit_id}_articulation/{index:03d}.png' # 0
    #     # img_save_path = f'/home/tianhang/code/Plant/viz/edit/stem_{edit_id}_deform/{index:03d}.png' # 3

    #     img_save_path = '/home/tianhang/code/Plant/viz/edit_all/stem2.png'

    #     o3d.visualization.draw_geometries([p_final, axis], mesh_show_back_face=True)

    #     os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
    #     load_view_point([p_final, axis], 
    #                     cam_json_path='/home/tianhang/code/Plant/cams/ScreenCamera_2025-10-01-23-55-10.json',
    #                     img_save_path=img_save_path,
    #                     show_window=False)
    #     index += 1

    # o3d.visualization.draw_geometries([p_final, axis], mesh_show_back_face=True)
    # _=1

if __name__ == "__main__":
    
    params = {
        'retrain': False,
        'visualize': True,
    }

    finetune('2_i', 'soybean', **params)   # leaf, stem done
    finetune('3_o', 'soybean', **params)   # leaf, stem done
    finetune('3_i', 'soybean', **params)   # leaf, stem done
    finetune('4_i', 'soybean', **params)   # leaf, stem done
    finetune('4_o', 'soybean', **params)   # leaf, stem done
    finetune('6_i', 'soybean', **params)   # leaf, stem done
    finetune('6_o', 'soybean', **params)   # leaf, stem done
    finetune('7_i', 'soybean', **params)   # leaf, stem done
    finetune('7_o', 'soybean', **params)   # leaf, stem done
    finetune('8_i', 'soybean', **params)   # leaf, stem done
    finetune('9_i', 'soybean', **params)
    finetune('10_i', 'soybean', **params)
    finetune('11_i', 'soybean', **params)
    finetune('13_i', 'soybean', **params)
    finetune('14_i', 'soybean', **params)
    finetune('15_i', 'soybean', **params)
    finetune('24_o', 'soybean', **params)  # leaf, stem done
    finetune('30_o', 'soybean', **params)
    finetune('34_i', 'soybean', **params)
    finetune('46_o', 'soybean', **params)
    finetune('47_o', 'soybean', **params)
    finetune('46_i', 'soybean', **params)
    finetune('48_i', 'soybean', **params)
    finetune('52_i', 'soybean', **params)
    finetune('53_i', 'soybean', **params)
    finetune('54_i', 'soybean', **params)
    finetune('101_o', 'soybean', **params)  # leaf, stem done
    finetune('154_o', 'soybean', **params)  # leaf, stem done
    finetune('160_i', 'soybean', **params)
    finetune('168_i', 'soybean', **params)
    finetune('169_o', 'soybean', **params)
    finetune('395_i', 'soybean', **params)  # leaf, stem done

    finetune('81_i', 'soybean', **params)  # leaf, stem done
    finetune('91_i', 'soybean', **params)  # leaf, stem done
    finetune('92_o', 'soybean', **params)  # leaf, stem done
    finetune('112_o', 'soybean', **params)  
    finetune('169_i', 'soybean', **params)

    finetune('14', 'ribes', **params)
    finetune('08', 'ribes', **params)

    finetune('1', 'tobacco', **params)
    finetune('2', 'tobacco', **params)
    finetune('3', 'tobacco', **params)

    finetune('0925da', 'maize', **params) 
    finetune('10008da', 'maize', **params)
    finetune('M01_0319_a', 'maize', **params)
    finetune('M01_0313_a', 'maize', **params)
    finetune('M01_0317_a', 'maize', **params)

    finetune('02', 'rose', **params)
    finetune('06', 'rose', **params)
    finetune('07', 'rose', **params)