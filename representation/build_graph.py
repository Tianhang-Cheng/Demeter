from easydict import EasyDict as edict
import os
import numpy as np
import open3d as o3d
import torch
import copy
import glob

from utils.constant import *
from utils.pca import NodePCA
from utils.graph import find_layers_and_paths
from utils.rotation_pytorch3d import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion

from representation.primitive import CatmullRomSurface, CatmullRomCurve
from representation.graph import PlantGraphFixedTopology

def _detach(x):
    return x.detach().cpu().numpy()

def make_axis(R, R_align, t, size=0.005):
    _axis_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0,0,0])
    _axis_viz.rotate(R.T)
    _axis_viz.rotate(R_align)
    _axis_viz.translate(t)
    return _axis_viz

def build_plant_graph(species:str, meta_name:str, parents:edict, classes:edict, curves:edict, surfaces:edict, fit_folder:str,
                      pca_stem_3d, pca_leaf_3d, pca_leaf_2d,
                      main_stem_end_points_bottom, connected_template_pcd, node_axis_along_parent_stem_all, n_iter:int=300, lr=5e-3, **kwargs):
    """
    Build a plant graph from the given components, and fit to the original position from separate instance fits.
    """

    retrain = kwargs.get('retrain', False)
    
    # empty template
    empty_surface = CatmullRomSurface(species=species, shape_pca=pca_leaf_2d)
    empty_curve = CatmullRomCurve()

    layers, paths = find_layers_and_paths(parents)
    
    # read main stem end points
    main_stem = [int(k) for k, v in parents.items() if v == -1][0]


    leaf_3d_info_cp_local = edict() # just for viz
    leaf_3d_info_shape_coeff = edict()
    leaf_3d_info_deform_coeff = edict()
    leaf_3d_info_s = edict()
    leaf_3d_info_M_quat = edict()

    stem_3d_info_cp_local = edict() # just for viz
    stem_3d_info_thickness = edict()
    stem_3d_info_deform_coeff = edict()
    stem_3d_info_s = edict()
    stem_3d_info_M_quat = edict()

    node_length_along_parent_stem = edict()
    node_axis_along_parent_stem = edict()

    pcd_viz_stem = []
    pcd_viz_leaf = []

    local_axis_of_parent_sparse_viz = []
    local_axis_of_self_sparse_viz = []

    # loop through the layers to extract the 3d information in the local coordinate system
    for layer_idx in range(len(layers)):

        if layer_idx == 0:

            curve = curves[str(int(main_stem))]
            
            cp_local = curve.evaluate(num_points=2, w=1)[0]

            # store node information
            stem_3d_info_cp_local[str(int(main_stem))] = cp_local 
            stem_3d_info_deform_coeff[str(int(main_stem))] = pca_stem_3d.encode(cp_local.reshape(1, -1))
            stem_3d_info_s[str(int(main_stem))] = curve.s.item()
            stem_3d_info_thickness[str(int(main_stem))] = curve.thickness.item()
            # stem_3d_info_M_quat[str(int(main_stem))] = standardize_quaternion(matrix_to_quaternion(torch.from_numpy(curve_info['local_axis']).float().cuda()))
            # stem_3d_info_M_quat[str(int(main_stem))] = standardize_quaternion(matrix_to_quaternion(torch.from_numpy(np.eye(3)).float().cuda()))
            stem_3d_info_M_quat[str(int(main_stem))] = curve.quat
            node_length_along_parent_stem[str(int(main_stem))] = 1.0
            continue # skip the first layer, which is the root

        layer = layers[layer_idx]

        ############## show pcd of current parent and children
        cur_layer_child_connected_template_pcd = edict()
        for k in layer:
            if classes[str(k)] == FLOWER_CLASS or classes[str(k)] == FRUIT_CLASS:
                continue
            # if stem has no child, just skip it for a cleaner graph
            # if kwargs.get('skip_isolated_stem', False) and classes[str(k)] == STEM_CLASS and int(k) not in parents.values():
            #     print('Stem {} has no child, skip it'.format(k))
            #     continue
            cur_layer_child_connected_template_pcd[str(k)] = connected_template_pcd[str(k)]

        do_viz = False
        if do_viz:
            end_point_txt = os.path.join(fit_folder, '{}_endpoints.txt'.format(str(main_stem).zfill(3)))
            main_stem_end_points = np.loadtxt(end_point_txt)
            end_point_pcd = o3d.geometry.PointCloud()
            end_points_rot = main_stem_end_points - main_stem_end_points_bottom
            # end_points_rot = np.dot(end_points_rot, R_main_stem_post_alignment.T)
            end_point_pcd.points = o3d.utility.Vector3dVector(end_points_rot)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            axis.translate([0, 0, 0])
            selected_pcds_list = [v for k, v in cur_layer_child_connected_template_pcd.items()]
            o3d.visualization.draw_geometries([axis, connected_template_pcd[str(int(main_stem))]] + selected_pcds_list)

            t_aligned_pcds_list = [v for k, v in connected_template_pcd.items()]
            o3d.visualization.draw_geometries(t_aligned_pcds_list)

        ############## find relative coordinate information along the curve (stem)
        for k, v in cur_layer_child_connected_template_pcd.items():
            child_pcd = cur_layer_child_connected_template_pcd[k]
            _pc = np.asarray(child_pcd.points) # (n, 3)
            parent_id = parents[str(k)]
            parent_pcd = connected_template_pcd[str(parent_id)]
            _pp = np.asarray(parent_pcd.points)

            # parent is always stem, but child can be stem or leaf
            if classes[str(k)] == STEM_CLASS:
                _dist = np.linalg.norm(_pp - _pc[0], axis=1) # bottom of the stem
                _idx = np.argmin(_dist)
            elif classes[str(k)] == LEAF_CLASS:
                _dist = np.linalg.norm(_pp - _pc.reshape(43,45,3)[43//2, 0], axis=1)
                _idx = np.argmin(_dist)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(_pp[0, 0], _pp[0, 1], _pp[0, 2], c='r', s=100, marker='x')
            # ax.scatter(_pp[:, 0], _pp[:, 1], _pp[:, 2], c='r')
            # z = _pc.reshape(43,45,3)[43//2, 0]
            # ax.scatter(z[0], z[1], z[2], c='r', s=100, marker='x')
            # ax.scatter(_pc[:, 0], _pc[:, 1], _pc[:, 2], c='b')
            # plt.show()

            # find the length along the curve
            node_length_along_parent_stem[str(k)] = _idx / (len(_pp) - 1)
            node_axis_along_parent_stem[str(k)] = node_axis_along_parent_stem_all[str(parent_id)][_idx]

        # transfer child information into the local coordinate system
        for k, v in cur_layer_child_connected_template_pcd.items():

            local_axis_of_parent = node_axis_along_parent_stem[str(k)]
            llc = node_length_along_parent_stem[str(k)]

            # _pp = t_aligned_parent_pcd_test
            _pp = np.asarray(connected_template_pcd[str(parents[str(k)])].points)
            # local_axes_new = calculate_local_axes(_pp)
            local_axes_new = node_axis_along_parent_stem_all[str(parents[str(k)])]
            _idx = int(llc * (len(_pp) - 1))
            # _local_axis_new = local_axes_new[_idx] # if parent stem changes, the local axis should be updated
            # local_axis_of_parent_new = local_axis_of_parent
            
            # do_viz = False
            # if do_viz:
            #     _local_axis_viz = []
            #     for _ in range(len(local_axes_new)):
            #         if _ % 5 != 0:
            #             continue
            #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
            #         # axis.rotate()
            #         # _R = R_main_stem_post_alignment @ local_axes_new[_].T
            #         _R = local_axes_new[_].T
            #         # if np.linalg.det(_R) < 0:
            #         #     _R = -_R
            #         axis.rotate(_R)
            #         # axis.translate(np.dot(np.asarray(t_aligned_pcds[main_stem].points)[_], R_main_stem_post_alignment.T))
            #         axis.translate(np.asarray(connected_template_pcd[str(int(main_stem))].points)[_])
            #         _local_axis_viz.append(axis)
            #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            #     axis.translate([0, 0, 0])
            #     o3d.visualization.draw_geometries( _local_axis_viz + [axis])
            #     del  _local_axis_viz
            
            _axis_viz = make_axis(local_axis_of_parent, np.eye(3), np.dot(_pp[_idx], np.eye(3).T))
            local_axis_of_parent_sparse_viz.append(_axis_viz)
            
            if classes[k] == STEM_CLASS:

                curve = curves[str(k)]

                _pc_cano = curve.evaluate(num_points=2, w=1)[0]
                # _cpc = _cpc @ quaternion_to_matrix(curve.quat).detach().cpu().numpy()

                deform_coeff = pca_stem_3d.encode(_pc_cano.reshape(1, -1))
                # _pc_cano_recon = pca_stem_3d.decode(deform_coeff).reshape(9, 3)

                # _local_axis_of_self = calculate_local_axes(_cpc, mode='curve', p0=_cpc[0], p1=_cpc[-1])[0]
                _rot = _detach(quaternion_to_matrix(curve.quat))
                # cp_w_local = _cpc @ _rot # world coordinate

                # cp_w_local = cp_w_local @ _local_axis_of_self.T

                # store node information
                # M = _local_axis_of_self @ local_axis_of_parent.T
                M = _rot @ local_axis_of_parent.T
                assert np.abs(np.abs(np.linalg.det(M)) - 1) < 1e-3, 'The determinant of M should be 1, but got {}'.format(np.linalg.det(M))
                stem_3d_info_s[str(k)] = curve.s.item()
                stem_3d_info_thickness[str(k)] = curve.thickness.item()
                stem_3d_info_deform_coeff[str(k)] = _detach(deform_coeff)
                stem_3d_info_M_quat[str(k)] = matrix_to_quaternion(torch.from_numpy(M).float().cuda()) # only use the first two axis
                stem_3d_info_cp_local[str(k)] = _pc_cano # [8, 3]

                # _cpc = (cp_w_local @ M @ local_axis_of_parent)
                # _pc = compute_catmull_rom_curve(wrap_control_points(_cpc*l), alpha=0.5, num_points=max(int(l * 5000), num_points))

            elif classes[k] == LEAF_CLASS:

                surf = surfaces[str(k)]

                _rot = _detach(quaternion_to_matrix(surf.quat))
                _pc_cano = surf.evaluate()[0] # (W, H, 3)

                _a, _b, _c = empty_surface.invert(_pc_cano)
                _mean_shape = empty_surface.evaluate(main_rotation=_a, sub_rotation_l=_b, sub_rotation_r=_c, mean_shape=True)[0]

                deform_coeff = pca_leaf_3d.encode(_mean_shape.reshape(1, -1))
                shape_coeff = surf.dw # (k, 1)

                # a, b, c = empty_surface.invert(pca_leaf_3d.decode(deform_coeff).reshape(43, 45, 3))
                # _pc_cano_recon = empty_surface.evaluate(main_rotation=a, sub_rotation_l=b, sub_rotation_r=c, shape_coeff=shape_coeff)[0]
                
                ## viz
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # _pc_cano_viz = _detach(_pc_cano)
                # ax.scatter(_pc_cano_viz[:, :, 0], _pc_cano_viz[:, :, 1], _pc_cano_viz[:, :, 2], c='r')
                # _pc_cano_recon_viz = _detach(_pc_cano_recon)
                # ax.scatter(_pc_cano_recon_viz[:, :, 0], _pc_cano_recon_viz[:, :, 1], _pc_cano_recon_viz[:, :, 2], c='b')
                # plt.show()

                # store the 3d information of the leaf 
                M = _rot @ local_axis_of_parent.T 
                # assert np.abs(np.abs(np.linalg.det(M)) - 1) < 1e-3, 'The determinant of M should be 1, but got {}'.format(np.linalg.det(M))
                leaf_3d_info_s[str(k)] = surf.s.item()
                leaf_3d_info_shape_coeff[str(k)] = _detach(shape_coeff)
                leaf_3d_info_deform_coeff[str(k)] = _detach(deform_coeff)
                leaf_3d_info_cp_local[str(k)] = torch.from_numpy(_detach(_pc_cano)).float().cuda() # [392, 3]
                leaf_3d_info_M_quat[str(k)] = matrix_to_quaternion(torch.from_numpy(M).float().cuda())

    # viz all pcds
    # do_viz = False
    # if do_viz:
    #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    #     axis.translate([0, 0, 0])
    #     # o3d.visualization.draw_geometries( pcd_viz + local_axis_viz_sparse + [axis])
    #     o3d.visualization.draw_geometries(pcd_viz_leaf + pcd_viz_stem + local_axis_of_parent_sparse_viz + [axis] + local_axis_of_self_sparse_viz)
        # t_aligned_pcds_items = [v for k, v in t_aligned_pcds.items()]
        # o3d.visualization.draw_geometries( pcd_viz + local_axis_viz_sparse + t_aligned_pcds_items + [axis])

    # _pc = _pc / surf.s.item()
    # _pc = (np.linalg.inv(quaternion_to_matrix(surf.quat).detach().cpu().numpy()) @ _pc.T).T
    # _pc = (_pc + _pp[_idx]).reshape(-1, 3)
    # _pc = np.dot(_pc, R_main_stem_post_alignment.T)
    # _pcd_viz = o3d.geometry.PointCloud()
    # _pcd_viz.points = o3d.utility.Vector3dVector(_pc)
    # _pcd_viz.colors = o3d.utility.Vector3dVector(np.ones_like(_pc) * COLOR_DICT['leaf'] / 255)
    # o3d.visualization.draw_geometries([_pcd_viz])

    # do_viz = False
    # if do_viz:
    #     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    #     axis.translate([0, 0, 0])
    #     o3d.visualization.draw_geometries([axis] +  pcd_viz_stem + pcd_viz_leaf)
    
    # ground-truth raw point cloud
    full_plant_path = os.path.join(f'/home/tianhang/data/{species}/pcd_clean_join/{meta_name}.ply')
    full_plant = o3d.io.read_point_cloud(full_plant_path)
    full_plant.translate(-main_stem_end_points_bottom)

    instance_points_dict = edict()
    # instance_colors_dict = edict() # not used

    instance_points_viz = []
    instance_points_array = []

    for instance_fit in glob.glob(os.path.join(fit_folder, '*.ply')):
        _name = os.path.basename(instance_fit).split('.')[0]

        pcd = o3d.io.read_point_cloud(instance_fit)
        pcd.translate(-main_stem_end_points_bottom)
        instance_points_viz.append(pcd)
        points = np.asarray(pcd.points)
        instance_points_array.append(points)
        points = torch.from_numpy(points).float().cuda()

        # FIXME: do we need to read from raw folder for stem?
        # if classes[str(int(_name))] == STEM_CLASS:
        #     processed_file = os.path.join(raw_folder, os.path.basename(instance_fit))
        #     pcd2 = o3d.io.read_point_cloud(processed_file)
        #     pcd2.translate(-main_stem_end_points_bottom)
        #     pcd2.colors = o3d.utility.Vector3dVector([])
            
        #     points2 = np.asarray(pcd2.points)
        #     points2 = torch.from_numpy(points2).float().cuda()
        #     points = torch.cat([points, points2], dim=0)
        #     # points = points2

        #     # downsample
        #     # pcd2 = pcd2.voxel_down_sample(voxel_size=0.0025)
        #     instance_points_viz.append(pcd2)
            # o3d.visualization.draw_geometries([pcd, pcd2])
        
        # else:
        #     pcd = o3d.io.read_point_cloud(instance_fit)
        #     pcd.translate(-main_stem_end_points_bottom)
        #     instance_points_viz.append(pcd)
        #     points = np.asarray(pcd.points)
        #     points = torch.from_numpy(points).float().cuda()
        
        instance_points_dict[str(int(_name))] = points
        # instance_colors_dict[str(int(_name))] = np.asarray(pcd.colors)
    
    plant_graph = PlantGraphFixedTopology(layers, classes, parents, pca_stem_3d, pca_leaf_3d, pca_leaf_2d,
                                        stem_3d_info_cp_local, stem_3d_info_thickness,   stem_3d_info_deform_coeff, stem_3d_info_s, stem_3d_info_M_quat, 
                                        leaf_3d_info_cp_local, leaf_3d_info_shape_coeff, leaf_3d_info_deform_coeff, leaf_3d_info_s, leaf_3d_info_M_quat, 
                                        node_length_along_parent_stem, species=species).to('cuda')

    # with torch.no_grad():
    #     p_init = plant_graph.generate(output_format='mesh', color='orange', align_global=True)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # full_plant_rot = copy.deepcopy(full_plant).rotate(plant_graph.global_M_p.T, center=(0,0,0))
    # o3d.visualization.draw_geometries([p_init, full_plant, axis], mesh_show_back_face=True)

    model_path = kwargs.get('save_path', f'/home/tianhang/data/{species}/pcd_processed/{meta_name}/graph.pkl')

    if os.path.exists(model_path) and not retrain:
        plant_graph.load(model_path)
        print('The plant graph is loaded from {}'.format(model_path))
    
    if retrain and not os.path.exists(model_path):
        plant_graph.fit(instance_points_dict, n_iter=n_iter, lr=lr, mode='finetune')
        plant_graph.save(model_path)
        print('The plant graph is trained and saved to {}'.format(model_path))
    elif retrain and os.path.exists(model_path):
        plant_graph.fit(instance_points_dict, n_iter=n_iter, lr=lr/10, mode='finetune')
        plant_graph.save(model_path)
        print('The plant graph is retrained and saved to {}'.format(model_path))
    
    return_graph = kwargs.get('return_graph', False)
    if return_graph:
        return plant_graph

    # viz
    do_viz = kwargs.get('visualize', False)
    if do_viz:
        with torch.no_grad():
            p_final = plant_graph.generate(output_format='mesh', color='gray', align_global=True)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

        full_plant_rot = copy.deepcopy(full_plant).rotate(plant_graph.global_M, center=(0,0,0))
        # o3d.visualization.draw_geometries([p_final, full_plant_rot, axis], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([p_final,axis], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([p_final], mesh_show_back_face=True)
        # p_final.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(p_final.vertices), 3))*0.7)

    
    do_save = kwargs.get('save', False)
    if do_save:
        with torch.no_grad():
            stem_mesh, leaf_mesh = plant_graph.generate(output_format='seg_mesh', color='gray', align_global=True)
        leaf_save_path = os.path.join(f'/home/tianhang/data/{species}/fitted_mesh_viz', f'{meta_name}_leaf.ply')
        stem_save_path = os.path.join(f'/home/tianhang/data/{species}/fitted_mesh_viz', f'{meta_name}_stem.ply')
        os.makedirs(os.path.dirname(leaf_save_path), exist_ok=True)
        o3d.io.write_triangle_mesh(leaf_save_path, leaf_mesh)
        o3d.io.write_triangle_mesh(stem_save_path, stem_mesh)
        print('Done saving {} fitted mesh to {}'.format(meta_name, os.path.dirname(leaf_save_path)))