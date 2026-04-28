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

def finetune(mesh_dir, species='soybean', **kwargs):

    raw_folder = os.path.join(mesh_dir, 'raw')
    os.makedirs(raw_folder, exist_ok=True)
    fit_folder = os.path.join(mesh_dir, 'fit')
    info_folder = os.path.join(mesh_dir, 'info')
    parent_txt = os.path.join(info_folder, 'parent.txt') 
    class_txt = os.path.join(info_folder, 'class.txt')
    abs_translation_pkl = os.path.join(info_folder, 'abs_translations.pkl')

    files = glob.glob(raw_folder + '/*.ply')
    files.sort()
    files_dict = edict()
    for file in files:
        name = int(os.path.basename(file).split('.')[0])
        files_dict[str(name)] = file

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

    shape_species = species
    if species == 'ribes':
        shape_species = 'papaya'
    if species == 'tobacco' or species == 'maize' or species == 'rose':
        shape_species = 'soybean'
    data_folder = f'sample_params/{shape_species}' #FIXME: may use other species' PCA model to represent the shape
    pca_leaf_2d = NodePCA(path=os.path.join(data_folder, '2d_leaf_pca.pth'))
    pca_stem_3d = NodePCA(path=os.path.join(data_folder, '3d_stem_pca.pth'))
    pca_leaf_3d = NodePCA(path=os.path.join(data_folder, '3d_leaf_pca.pth'))

    surface_points_gt = o3d.geometry.PointCloud()

    with torch.no_grad():

        for pcd_path in leaf_pcd_paths:

            pcd = o3d.io.read_point_cloud(pcd_path)
            surface_points_gt += pcd # add to the ground truth point cloud for evaluation later

            pcd_index = os.path.basename(pcd_path).split('.')[0]
            surf = CatmullRomSurface(species=species, shape_pca=pca_leaf_2d)
            fit_save_path = pcd_path.replace(raw_folder, fit_folder).replace('.ply', '_fit.pth')
            surf.load(fit_save_path)
            inner_points, inner_points_flat, _ = surf.evaluate(w=1)
            inner_points_viz = _detach(inner_points)
            h, w = inner_points_viz.shape[:2]

            root_path = os.path.join(info_folder, '{}_root.txt'.format(pcd_index))
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

            node_axis_along_parent_stem_all[str(int(pcd_index))] = _detach(_axis_all)

            curves[str(int(pcd_index))] = curve
            template_points[str(int(pcd_index))] = p_curve
            del curve

    """
    find a matrix that make main stem vertical (parent value == -1) for visualization
    """
    main_stem = [k for k, v in parents.items() if v == -1][0].zfill(3)
    main_stem_end_points_bottom = template_points[str(int(main_stem))][0]
    print('main_stem_end_points_bottom: ', main_stem_end_points_bottom)

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

    plant_graph = build_plant_graph(
        mesh_dir, species, parents, classes, curves, surfaces,
        pca_stem_3d=pca_stem_3d, pca_leaf_3d=pca_leaf_3d, pca_leaf_2d=pca_leaf_2d,
        main_stem_end_points_bottom=main_stem_end_points_bottom, 
        connected_template_pcd=connected_template_pcd, 
        node_axis_along_parent_stem_all=node_axis_along_parent_stem_all, 
        retrain=kwargs.get('retrain', False),
        visualize=kwargs.get('visualize', False),
        n_iter=300,
        full_plant_pcd_path=os.path.join(mesh_dir, 'pcd_y_aligned.ply'),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, default='sample_point_cloud/val/7a', help='path to the mesh directory')
    parser.add_argument('--species', type=str, default='soybean', help='plant species')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the model')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize the result')
    args = parser.parse_args()
    
    species = args.species
    mesh_dir = args.mesh_dir
    retrain = args.retrain
    visualize = args.visualize

    finetune(mesh_dir, species, retrain=True, visualize=True)