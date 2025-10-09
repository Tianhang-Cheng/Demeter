import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import torch
import open3d as o3d
import numpy as np
import os
import copy
import tqdm
import pickle

from utils.pcd import get_min_dist, segment_pcd_DBSCAN, find_farthest_points_batch_torch, compute_frenet_serret_frame
from utils.graph import get_guessed_parent, load_parent, save_class, save_parent, max_geodesic_distance, find_layers_and_paths, calculate_accumulation_values
from utils.constant import *
from utils.rotation_pytorch3d import quaternion_to_matrix
from representation.primitive import CatmullRomSurface, CatmullRomCurve
from representation.build_graph import build_plant_graph
from scipy.spatial import cKDTree
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from utils.pca import NodePCA

np.random.seed(5242)
color_map = np.random.rand(1000, 3)

semantics_color_map = np.array([
    [153,255,51], # green
    [139,69,19],
    [0,0,0],
    [0,0,0]
    ]) / 255.0

def _detach(x):
    return x.detach().cpu().numpy()

def reconstruction_3d(folder, species='soybean', **kwargs):

    plant_id = os.path.basename(folder)

    do_viz = kwargs.get('do_viz', False)
    retrain = kwargs.get('retrain', False)
    retrain_leaf = kwargs.get('retrain_leaf', False)
    retrain_stem = kwargs.get('retrain_stem', False)

    infer_dir = folder
    params_dir = os.path.join(folder, 'params') # save the fitted demeter parameters (graph file)
    os.makedirs(params_dir, exist_ok=True)
    info_dir = os.path.join(params_dir, 'info') # save the fitted demeter parameters (toplogy & semantics)
    os.makedirs(info_dir, exist_ok=True)

    points_path = os.path.join(infer_dir, 'points.pkl')
    instances_path = os.path.join(infer_dir, 'instances.pkl')
    semantics_path = os.path.join(infer_dir, 'semantics.pkl')

    data_folder = f'sample_params/{species}'
    pca_leaf_2d = NodePCA(path=os.path.join(data_folder, '2d_leaf_pca.pth'))
    pca_stem_3d = NodePCA(path=os.path.join(data_folder, '3d_stem_pca.pth'))
    pca_leaf_3d = NodePCA(path=os.path.join(data_folder, '3d_leaf_pca.pth'))

    transform_dict = pickle.load(open(os.path.join(folder, 'transform.pkl'), 'rb'))
    viz_rotation = transform_dict['rotation']
    viz_radius = transform_dict['radius']
    viz_center = transform_dict['bbox_center']

    # load input coordiantes
    model_path = os.path.join(folder, 'normalized_pcd.pth')
    model = torch.load(model_path, weights_only=True)
    coord = model['coord']
    color = model['color']
    raw_input_pcd = o3d.geometry.PointCloud()
    raw_input_pcd.points = o3d.utility.Vector3dVector(coord)
    raw_input_pcd.colors = o3d.utility.Vector3dVector(color)

    # load Point-Transformer prediction from cached files if exist
    if os.path.exists(points_path) and os.path.exists(instances_path) and os.path.exists(semantics_path):

        all_points2 = pickle.load(open(points_path, 'rb'))
        all_instances2 = pickle.load(open(instances_path, 'rb'))
        all_semantics2 = pickle.load(open(semantics_path, 'rb'))

    # otherwise do the full pipeline
    else:
        coord_tree = cKDTree(coord)

        # load gt thresholded
        sdfs_self_path = os.path.join(folder, 'sdfs_self.pkl')

        if os.path.exists(sdfs_self_path):
            with open(sdfs_self_path, 'rb') as f:
                sdfs_self = pickle.load(f)
            self_dist_array = np.concatenate([sdfs_self[str(name)] for name in sdfs_self.keys()], axis=0)
            self_dist_array = np.where(np.isinf(self_dist_array), 0, self_dist_array)
            self_dist_array = np.where(np.isnan(self_dist_array), 0, self_dist_array)
            gt_avg_self_dist = np.mean(np.max(self_dist_array, axis=-1)) * 2
            print(f"GT avg self dist: {gt_avg_self_dist}")

        # calculate the pred thresholded
        sdfs_all_path = os.path.join(folder, 'sdfs_all.pkl')

        if os.path.exists(sdfs_all_path):
            with open(sdfs_all_path, 'rb') as f:
                sdfs_all = pickle.load(f)
        else:
            sdfs_all = []
            batch_size = 10000
            for p in tqdm.tqdm(range(0, len(coord), batch_size)):
                sdfs_all.append(get_min_dist(coord[p:p+batch_size], coord, k=10, kdtree=coord_tree)[0])
            
            with open(sdfs_all_path, 'wb') as f:
                pickle.dump(sdfs_all, f)
        all_dist_array = np.concatenate(sdfs_all, axis=0)
        all_dist_array = np.where(np.isinf(all_dist_array), 0, all_dist_array)
        all_dist_array = np.where(np.isnan(all_dist_array), 0, all_dist_array)
        pred_avg_self_dist = np.mean(np.max(all_dist_array, axis=-1)) * 2
        print(f"Pred avg self dist: {pred_avg_self_dist}")

        # load pred semantics
        pred_path = os.path.join(folder, 'normalized_pcd_pred.npy')
        pred_semantics = np.load(pred_path)
        main_stem_mask = pred_semantics == 2
        # load pred dist
        pred_dist_path = os.path.join(folder, 'normalized_pcd_pred_dist.npy')
        pred_dist = np.load(pred_dist_path)[:, 0]
        # pred_dist = np.max(np.load(pred_dist_path), axis=-1)

        # pred_dist = torch.sigmoid(torch.tensor(pred_dist)).numpy()

        # mask = (pred_dist < 0.17) # non-boundary points are those with distance < T
        # pred_dist = pred_dist / np.max(pred_dist)
        if species == 'soybean':
            mask = pred_dist < 0.15 # soybean
        else:
            mask = pred_dist < 0.4

        coord_masked = coord[mask]
        assert len(coord_masked) > 0
        semantics_masked = pred_semantics[mask]

        pred_dist_color = np.repeat(pred_dist[:, None], 3, axis=1)
        pred_dist_color[~mask] = [1, 0, 0]

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        pcd_dist = o3d.geometry.PointCloud()
        pcd_dist.points = o3d.utility.Vector3dVector(coord)
        pcd_dist.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # if do_viz:
        # o3d.visualization.draw_geometries([pcd_dist ])

        pcd_dist = o3d.geometry.PointCloud()
        pcd_dist.points = o3d.utility.Vector3dVector(coord)
        pcd_dist.colors = o3d.utility.Vector3dVector(pred_dist_color)
        pcd_dist.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if do_viz:
            o3d.visualization.draw_geometries([pcd_dist ])

        inv_dist = 1 / (pred_dist + 1e-6)
        inv_dist = inv_dist / np.max(inv_dist)
        inv_dist_color = np.repeat(inv_dist[:, None], 3, axis=1)
        pcd_dist = o3d.geometry.PointCloud()
        pcd_dist.points = o3d.utility.Vector3dVector(coord)
        pcd_dist.colors = o3d.utility.Vector3dVector(inv_dist_color)
        # if do_viz:
        #     o3d.visualization.draw_geometries([pcd_dist ])

        # visualize the semantic segmentation
        pcd_sem = o3d.geometry.PointCloud()
        pcd_sem.points = o3d.utility.Vector3dVector(coord)
        pcd_sem.colors = o3d.utility.Vector3dVector(color_map[pred_semantics])
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # o3d.visualization.draw_geometries([pcd_sem, axis])
        if do_viz:
            o3d.visualization.draw_geometries([pcd_sem])


        # Perform # DBSCAN clustering
        instances_filtered, n_clusters, clustered_points, semantics_filtered = segment_pcd_DBSCAN(
            coord_masked, eps=pred_avg_self_dist*0.9, min_samples=8, semantic=semantics_masked,
        )

        print(f"Number of clusters found: {n_clusters}")
        cluster_viz = []
        clustered_points_concat = []
        clustered_instances_concat = []
        clustered_semantics_concat = []
        for i in range(0, n_clusters):

            _s = semantics_filtered[i]
            _p = clustered_points[i]
            _i = instances_filtered[i]
            _unique_semantics = np.unique(_s)

            if len(_unique_semantics) > 1:
                print(f"Cluster {i} contains multiple semantics: {_unique_semantics}")
                s_num_count = [np.sum(semantics_filtered[i] == s) for s in _unique_semantics]
                max_s = _unique_semantics[np.argmax(s_num_count)]
                print(f"Choosing the most frequent semantic: {max_s}")

                _s_mask = (_s == max_s)
                _s = _s[_s_mask]
                _p = _p[_s_mask]
                _i = _i[_s_mask]

            _main_stem_mask = _s == 2
            if np.sum(~_main_stem_mask) == 0:
                continue

            _p = _p[~_main_stem_mask]
            _i = _i[~_main_stem_mask]
            _s = _s[~_main_stem_mask]

            clustered_points_concat.append(_p)
            clustered_instances_concat.append(_i)
            clustered_semantics_concat.append(_s)

            _pcd = o3d.geometry.PointCloud()
            _pcd.points = o3d.utility.Vector3dVector(_p)
            _pcd.colors = o3d.utility.Vector3dVector(color_map[i] * np.ones((len(_p), 3)))
            cluster_viz.append(_pcd)

        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(coord[main_stem_mask])
        _pcd.colors = o3d.utility.Vector3dVector(color_map[n_clusters] * np.ones((len(_pcd.points), 3)))
        cluster_viz.append(_pcd)
        # o3d.visualization.draw_geometries(cluster_viz)

        # sometimes the main stem is not clustered, so we need to add it manually
        clustered_points_concat.append(coord[main_stem_mask])
        clustered_instances_concat.append(np.ones(len(coord[main_stem_mask])) * n_clusters)
        clustered_semantics_concat.append(pred_semantics[main_stem_mask])
        # if do_viz:
        #     o3d.visualization.draw_geometries(cluster_viz)

        # inpaint 
        clustered_points_concat = np.concatenate(clustered_points_concat, axis=0)
        clustered_instances_concat = np.concatenate(clustered_instances_concat, axis=0)
        clustered_semantics_concat = np.concatenate(clustered_semantics_concat, axis=0)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(clustered_points_concat)
        # pcd.colors = o3d.utility.Vector3dVector(color_map[clustered_instances_concat.astype(int)])
        # o3d.visualization.draw_geometries([pcd])

        # inpaint
        inpaint_with_semantic = True
        unique_semantics = np.unique(clustered_semantics_concat)

        all_instances = []
        all_points = []
        all_semantics = []

        if inpaint_with_semantic:
            for s in unique_semantics:

                cluster_point_select = clustered_points_concat[clustered_semantics_concat == s]
                ckdtree = cKDTree(cluster_point_select)

                full_point_select = coord[pred_semantics == s]
                _, idx = ckdtree.query(full_point_select)
                _instance = clustered_instances_concat[clustered_semantics_concat == s][idx]
                _semantic = np.ones(len(idx)) * s

                all_instances.append(_instance)
                all_points.append(full_point_select)
                all_semantics.append(_semantic)

        all_instances = np.concatenate(all_instances, axis=0).astype(int)
        all_points = np.concatenate(all_points, axis=0)
        all_semantics = np.concatenate(all_semantics, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(color_map[all_instances])

        if do_viz:
            o3d.visualization.draw_geometries([pcd])

        # seperate the clusters if they contain multiple clusters
        all_instances2 = {}
        all_points2 = {}
        all_semantics2 = {}

        all_unique_instances = np.unique(all_instances).tolist()

        def filter_threshold(s):
            if int(s) == 0:
                return 1000
            elif int(s) == 1 or int(s) == 2:
                return 100
            else:
                return 1e10

        for seq_id, i in enumerate(np.unique(all_instances)):

            # filter out the point
            mask = all_instances == i

            if i == 75:
                _ = 1

            cur_points = all_points[mask]
            cur_semantics = all_semantics[mask]
            cur_instances = all_instances[mask]

            print(f"Cluster {i} contains {len(cur_points)} points")

            n_clusters_ = 1
            if cur_semantics[0] != 2: # if not main stem
                instances_, n_clusters_, points_, semantics_ = segment_pcd_DBSCAN(
                cur_points, eps=pred_avg_self_dist, min_samples=10, semantic=cur_semantics # 1.05 is soybean
                )
                print('Point diff: ', len(cur_points) - np.sum([len(v) for k,v in points_.items()]))
        
            if n_clusters_ == 0:
                print('Cluster {} is empty after DBSCAN, skip'.format(i))
                continue

            elif n_clusters_ == 1:

                if len(cur_points) < filter_threshold(cur_semantics[0]) and cur_semantics[0] != 2:
                    print('Cluster {} is too small, skip'.format(i))
                    continue

                all_instances2[str(i)] = cur_instances
                all_points2[str(i)] = cur_points
                all_semantics2[str(i)] = cur_semantics

            elif n_clusters_ > 1:

                LEN = [len(points_[j]) for j in range(n_clusters_)]
                print(f"Cluster {i} contains multiple clusters: {LEN}")

                for j in range(n_clusters_):

                    if len(points_[j]) < filter_threshold(cur_semantics[0]):
                        continue
                    
                    new_id = np.max(all_unique_instances) + 1
                    all_instances2[str(new_id)] = np.ones(len(points_[j])) * np.max(all_unique_instances) + 1
                    all_points2[str(new_id)] = points_[j]
                    all_semantics2[str(new_id)] = semantics_[j]
                    all_unique_instances.append(np.max(all_unique_instances) + 1)
                    print('Adding new cluster {}'.format(np.max(all_unique_instances)))

        all_points2_viz = np.concatenate([v for k, v in all_points2.items()], axis=0)
        all_instances2_viz = np.concatenate([v for k, v in all_instances2.items()], axis=0).astype(int)
        all_semantics2_viz = np.concatenate([v for k, v in all_semantics2.items()], axis=0).astype(int)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points2_viz)
        pcd.colors = o3d.utility.Vector3dVector(color_map[all_instances2_viz.astype(int)])
        # if do_viz:
        #     o3d.visualization.draw_geometries([pcd])
        # o3d.visualization.draw_geometries([pcd])

        # Save the processed point cloud
        pickle.dump(all_points2, open(os.path.join(infer_dir, 'points.pkl'), 'wb'))
        pickle.dump(all_instances2, open(os.path.join(infer_dir, 'instances.pkl'), 'wb'))
        pickle.dump(all_semantics2, open(os.path.join(infer_dir, 'semantics.pkl'), 'wb'))
        o3d.io.write_point_cloud(os.path.join(infer_dir, 'segmentation.ply'), pcd)

    all_points2_viz = np.concatenate([v for k, v in all_points2.items()], axis=0)
    all_instances2_viz = np.concatenate([v for k, v in all_instances2.items()], axis=0).astype(int)
    all_semantics2_viz = np.concatenate([v for k, v in all_semantics2.items()], axis=0).astype(int)
    # all_instances2_viz = []
    # for seq_id, temp in enumerate(all_instances2):
    #     all_instances2_viz.append(np.ones(len(temp)) * seq_id)
    # all_instances2_viz = np.concatenate(all_instances2_viz, axis=0).astype(int)

    # find out the graph structure
    init_classes = {}
    root = None
    for key in all_points2.keys():
        semantics = int(all_semantics2[key][0])
        if semantics == 0:
            init_classes[key] = 0
        elif semantics == 1:
            init_classes[key] = 1
        elif semantics == 2:
            assert root is None
            root = key
            init_classes[key] = 1
        elif semantics == 3:
            init_classes[key] = 2
        elif semantics == 4:
            init_classes[key] = 3
    
    parent_txt = os.path.join(info_dir, 'parent.txt')
    class_txt = os.path.join(info_dir, 'class.txt')
    min_dist_txt = os.path.join(infer_dir, 'min_dist.npy')

    # update class every time
    classes = edict(init_classes)
    save_class(classes, class_txt)

    if os.path.exists(parent_txt) and os.path.exists(min_dist_txt) and not retrain:
        print('Loading existing parent file: {}'.format(parent_txt))
        parents = load_parent(parent_txt)
        min_dist_coords = np.load(min_dist_txt, allow_pickle=True).item()
    else:

        id_to_seq_map = {}
        seq_to_id_map = {}
        for i, key in enumerate(all_points2.keys()):
            id_to_seq_map[key] = i
            seq_to_id_map[i] = key
        seq_to_id_map[-1] = -1
        all_points2_seq = [v for k, v in all_points2.items()]
        root_seq = id_to_seq_map[root]
        init_classes_seq = {id_to_seq_map[k]: v for k, v in init_classes.items()}

        parents_seq = get_guessed_parent(all_points2_seq, root_seq, init_classes_seq)
        parents = {str(seq_to_id_map[int(k)]): int(seq_to_id_map[int(v)]) for k, v in parents_seq.items()}

        save_parent(parents, parent_txt)
        min_dist_coords = {}
        for key in all_points2.keys():
            if key != root:
                dist = get_min_dist(all_points2[key], ref_points=all_points2[str(parents[str(key)])])[0]
                min_index = np.argsort(dist)
                min_dist = dist[min_index]
                min_dist_coord = all_points2[key][min_index[0:5]].mean(axis=0)
                min_dist_coords[key] = min_dist_coord
        np.save(min_dist_txt, min_dist_coords)

    parents = {k: int(v) for k, v in parents.items()}
    # draw_tree_with_colors(parents, classes, save_path=os.path.join(plant_infer_dir, 'pred_graph.png'))
    layers, paths = find_layers_and_paths(parents)

    # fit each component
    fit_dir = os.path.join(folder, 'fit')
    os.makedirs(fit_dir, exist_ok=True)

    surfaces = edict()
    curves = edict()
    template_points = edict()
    node_axis_along_parent_stem_all = edict() # store the local axis of the node along the parent stem for all points

    surface_points = {}
    curve_points = {}

    # f = True

    # for key  in classes.keys():
    #     print('len(key): ', key, len(all_points2[key]))

    for key in tqdm.tqdm(classes.keys()):

        # if key == '116':
        #     f = False
        
        # if f:
        #     continue

        print('Fitting component {}'.format(key))
        
        fit_save_path = os.path.join(fit_dir, '{}.pth'.format(key))

        point_np = all_points2[key]
        point = torch.tensor(point_np).float().to('cuda')

        if classes[key] == LEAF_CLASS:


            # find the root point
            root_point = min_dist_coords[key]
            
            # fit the leaf
            point_norm = point - torch.from_numpy(root_point).float().to('cuda')

            surf = CatmullRomSurface(species=species, shape_pca=pca_leaf_2d)
            if os.path.exists(fit_save_path) and not retrain_leaf and not retrain:
                surf.load(fit_save_path)
                PLOT_FLAG = False
            else:

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(point_np[:, 0], point_np[:, 1], point_np[:, 2], c='r', marker='o')
                # ax.scatter(root_point[0], root_point[1], root_point[2], c='b', marker='x', s=100)
                # plt.show(block=True)

                surf.fit(point_norm, iters1=300, iters2=1200)
                surf.save(fit_save_path)
                PLOT_FLAG = True
            

            with torch.no_grad():
                inner_points, inner_points_flat, _  = surf.evaluate(w=1)

            surfaces[key] = surf
            rot = _detach(quaternion_to_matrix(surf.quat))
            s = surf.s.item()

            realworld_p = (_detach(inner_points) / s) @ rot + root_point
            
            surface_points[key] = realworld_p
            template_points[str(key)] = realworld_p
            
            # save the optimized point cloud for later graph fine-tuning
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(realworld_p.reshape(-1, 3))
            o3d.io.write_point_cloud(os.path.join(fit_dir, '{}.ply'.format(str(int(key)).zfill(3))), pcd)

            if PLOT_FLAG:
                zero = np.zeros(3)
                pcd_points_optimized = _detach(point_norm) @ rot.T
                pcd_points_optimized_viz = pcd_points_optimized * s
                # randomly pick 1000 points for visualization
                _n = pcd_points_optimized_viz.shape[0]
                pcd_points_optimized_viz = pcd_points_optimized_viz[np.random.choice(_n, min(1000, _n), replace=False)]
                inner_points_viz = inner_points.detach().cpu().numpy()
                inner_points_flat_viz = inner_points_flat.detach().cpu().numpy()

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(inner_points_viz[..., 0], inner_points_viz[..., 1], inner_points_viz[..., 2], color='g', rstride=1, cstride=1, alpha=1, label='curved leaf')
                # ax.plot_surface(inner_points_flat_viz[..., 0], inner_points_flat_viz[..., 1], inner_points_flat_viz[..., 2], color='orange', rstride=1, cstride=1, alpha=0.5, label='flat leaf')
                ax.scatter(pcd_points_optimized_viz[:, 0], pcd_points_optimized_viz[:, 1], pcd_points_optimized_viz[:, 2], c='b', s=10, marker='o', label='input pcd')
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
                # set camera position
                ax.view_init(elev=45, azim=30)
                plt.savefig(os.path.join(fit_dir, '{}.png'.format(key)), dpi=300)
                # plt.show(block=True)
                plt.close(fig)

        elif classes[key] == STEM_CLASS:

            # if stem has no child, just skip it for a cleaner graph
            if int(key) not in parents.values():
                print('Stem {} has no child, skip it'.format(key))
                continue

            endpoint_txt = os.path.join(fit_dir, '{}.txt'.format(key))
            if not os.path.exists(endpoint_txt) or retrain_stem and not retrain:
            
                # find root point
                best_dist = 0
                for k in [5, 10, 20, 30,]:
                    if point.shape[0] <= k:
                        continue
                    max_dist, p0_end, p1_end = max_geodesic_distance(point_np, k=k)
                    if max_dist > best_dist:
                        best_dist = max_dist
                _p0, _p1 = find_farthest_points_batch_torch(point)
                _p0 = _p0.detach().cpu().numpy()
                _p1 = _p1.detach().cpu().numpy()
                _dist =  np.linalg.norm(_p0 - _p1)
                if _dist > best_dist:
                    p0_end, p1_end = _p0, _p1
                    max_dist = _dist
                
                # find the order, p0 should be closer to the parent
                if parents[str(key)] != -1:
                    tree = cKDTree(all_points2[str(parents[str(key)])])
                    p0_to_parent = tree.query(p0_end)[0]
                    p1_to_parent = tree.query(p1_end)[0]
                    if p0_to_parent > p1_to_parent:
                        p0_end, p1_end = p1_end, p0_end
                else:
                    # if np.abs(p0_end[0]) > np.abs(p1_end[0]):
                    #     p0_end, p1_end = p1_end, p0_end
                    # visualize

                    rotation_click_path = os.path.join(folder, 'rotation_click.txt')
                    if os.path.exists(rotation_click_path):
                        a, b = np.loadtxt(rotation_click_path)
                    a = (a - viz_center) / viz_radius @ viz_rotation
                    b = (b - viz_center) / viz_radius @ viz_rotation

                    a_dist_p0 = np.linalg.norm(a - p0_end)
                    a_dist_p1 = np.linalg.norm(a - p1_end)
                    b_dist_p0 = np.linalg.norm(b - p0_end)
                    b_dist_p1 = np.linalg.norm(b - p1_end)

                    if a_dist_p0 + b_dist_p1 < a_dist_p1 + b_dist_p0:
                        # in this case, a is closer to p0, b is closer to p1, no need to do anything
                        pass
                    else:
                        p0_end, p1_end = p1_end, p0_end

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(point[:, 0].detach().cpu().numpy(), point[:, 1].detach().cpu().numpy(), point[:, 2].detach().cpu().numpy(), c='b', label='input pcd', alpha=0.5, s=1)
                    # ax.scatter(p0_end[0], p0_end[1], p0_end[2], c='r', label='p0_end', s=100)
                    # ax.scatter(p1_end[0], p1_end[1], p1_end[2], c='g', label='p1_end', s=100)
                    # a_viz = all_points2_viz[np.random.choice(all_points2_viz.shape[0], 10000, replace=False)]
                    # ax.scatter(a_viz[:, 0], a_viz[:, 1], a_viz[:, 2], c='black', label='full plant', alpha=0.1, s=1)
                    # ax.scatter(a[0], a[1], a[2], c='orange', label='click 1', s=200, marker='x')
                    # ax.scatter(b[0], b[1], b[2], c='purple', label='click 2', s=200, marker='x')
                    # plt.legend()
                    # plt.show(block=True)
                    # do_flip = input('do flip for main stem? (y/n): ')
                    # if do_flip == 'y':
                    #     p0_end, p1_end = p1_end, p0_end
                    #     print('flip the main stem')
                    # else:
                    #     print('no flip')

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(point_np[:, 0], point_np[:, 1], point_np[:, 2], c='r', marker='o')
                # ax.scatter(p0_end[0], p0_end[1], p0_end[2], c='b', marker='x', s=100)
                # ax.scatter(p1_end[0], p1_end[1], p1_end[2], c='g', marker='x', s=100)
                # plt.show(block=True)
                np.savetxt(endpoint_txt, np.stack([p0_end, p1_end, np.repeat(max_dist, 3)], axis=0), fmt='%.6f')
            else:
                p0_end, p1_end, max_dist = np.loadtxt(endpoint_txt)
                max_dist = max_dist[0]
            
            # fit the stem
            curve = CatmullRomCurve(p0_end=p0_end, p1_end=p1_end, force_smoothness=True)
            if os.path.exists(fit_save_path) and not retrain_stem and not retrain:
                curve.load(fit_save_path)
                PLOT_FLAG = False
            else:
                curve.fit(point, iters1=800, iters2=2000, input_scale=torch.tensor(max_dist, dtype=torch.float, device='cuda'))
                curve.save(fit_save_path)
                PLOT_FLAG = True

            with torch.no_grad():
                cp, p_curve = curve.evaluate(num_points=2, w=1)
             
            curves[key] = curve

            s = curve.s.item()
            rot = _detach(quaternion_to_matrix(curve.quat))
            cp = (_detach(cp) / s) @ rot + p0_end

            curve_points[key] = cp

            _axis_all = compute_frenet_serret_frame(cp)
            _axis_all = _axis_all[0:p_curve.shape[0]]

            node_axis_along_parent_stem_all[str(key)] = _detach(_axis_all)
            template_points[str(key)] = cp

            # save the optimized point cloud for later graph fine-tuning
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cp.reshape(-1, 3))
            o3d.io.write_point_cloud(os.path.join(fit_dir, '{}.ply'.format(str(int(key)).zfill(3))), pcd)

            if PLOT_FLAG:

                quat = curve.quat
                rot = quaternion_to_matrix(quat)
                # curved_cp_viz = curved_cp.detach().cpu().numpy()
                # pcd_points_viz = ((pcd_points - curve.p0_end) @ rot.T).detach().cpu().numpy() * l
                # p_curve_viz = p_curve.detach().cpu().numpy()
                pcd_points_viz = point.detach().cpu().numpy()

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(p_curve_viz[:, 0], p_curve_viz[:, 1], p_curve_viz[:, 2], c='g', label='fitted curve')
                # ax.scatter(pcd_points_viz[:, 0], pcd_points_viz[:, 1], pcd_points_viz[:, 2], c='b', label='input pcd', alpha=0.5, s=1)
                # ax.scatter(curved_cp_viz[:, 0], curved_cp_viz[:, 1], curved_cp_viz[:, 2], c='r', label='control points', s=100)
                # p_end = np.stack([p0_end, p1_end], axis=0)
                # ax.scatter(p_end[:, 0], p_end[:, 1], p_end[:, 2], c='black', label='end points', s=300, marker='x')
                # plt.legend()
                # # plt.savefig(os.path.join(fit_dir, '{}.png'.format(key)), dpi=300)
                # plt.show(block=True)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pcd_points_viz[:, 0], pcd_points_viz[:, 1], pcd_points_viz[:, 2], c='b', label='input pcd', alpha=0.5, s=1)
                ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c='r', label='control points', s=100)
                p_end = np.stack([p0_end, p1_end], axis=0)
                ax.scatter(p_end[:, 0], p_end[:, 1], p_end[:, 2], c='black', label='end points', s=300, marker='x')
                plt.legend()
                plt.savefig(os.path.join(fit_dir, '{}.png'.format(key)), dpi=300)
                # plt.show(block=True)
                plt.close(fig)
    
    # visualize the fitted and ground truth
    do_viz_initial_fit = False
    if do_viz_initial_fit:
        independed_plant_pcds = []
        for k, v in template_points.items():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(v.reshape(-1, 3))
            independed_plant_pcds.append(pcd)
        for k, v in all_points2.items():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(v.reshape(-1, 3))
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            independed_plant_pcds.append(pcd)
        o3d.visualization.draw_geometries(independed_plant_pcds, mesh_show_back_face=True)

    # force assemble the graph
    abs_translation_pkl = os.path.join(infer_dir, 'abs_translation.pkl')
    if os.path.exists(abs_translation_pkl) and False:
        with open(abs_translation_pkl, 'rb') as f:
            abs_translations = pickle.load(f)
    else:
        rel_translations = edict()
        for key in classes.keys():
            rel_translations[key] = np.zeros(3).astype(np.float32)
        
        for index, layer in enumerate(layers.values()):
            if index == 0:
                continue # skip root
            for node in layer:
                node = str(node)
                parent = str(parents[str(node)])
                parent_pcd_points = curve_points[parent].reshape(-1, 3)

                # find distance between parent and child
                if classes[node] == 0: # leaf
                    p_src = surface_points[node]
                elif classes[node] == 1: # stem
                    # if the stem has no child, just skip it for a cleaner graph
                    if int(node) not in parents.values():
                        continue
                    p_src = curve_points[node]
                else:
                    continue
                p_src = p_src.reshape(-1, 3)

                kdtree_Q = cKDTree(parent_pcd_points)
                distances, indices = kdtree_Q.query(p_src)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                p_nearest = p_src[min_distance_idx]
                q_nearest = parent_pcd_points[indices[min_distance_idx]]

                # rel_translations[node] = min_vec
                rel_translations[node] = q_nearest - p_nearest
        abs_translations = calculate_accumulation_values(parents, rel_translations)
        
        with open(abs_translation_pkl, 'wb') as f:
            pickle.dump(abs_translations, f)

    main_stem_end_points_bottom = curve_points[root][0]

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

    # visualize the aligned template points\
    do_viz_initial_aligned = False
    if do_viz_initial_aligned:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        axis.translate([0, 0, 0])
        t_aligned_pcds_items = [v for k, v in connected_template_pcd.items()]
        o3d.visualization.draw_geometries([axis]+t_aligned_pcds_items)
    

    # filter parents, remove stem that has no child
    new_parents = edict()
    for k, v in parents.items():
        if classes[k] == STEM_CLASS:
            if int(k) not in parents.values():
                continue
        new_parents[k] = v

    graph_save_path = os.path.join(params_dir, 'plant_graph.pth')
    plant_graph = build_plant_graph(species, plant_id, new_parents, classes, curves, surfaces,
                                    fit_folder=fit_dir,
                                    pca_stem_3d=pca_stem_3d,
                                    pca_leaf_3d=pca_leaf_3d,
                                    pca_leaf_2d=pca_leaf_2d,
                                    main_stem_end_points_bottom=main_stem_end_points_bottom,
                                    connected_template_pcd=connected_template_pcd,
                                    node_axis_along_parent_stem_all=node_axis_along_parent_stem_all,
                                    save_path=graph_save_path,
                                    n_iter=250,
                                    lr=1e-3,
                                    return_graph=True,
                                    retrain=retrain)
    plant_graph.save(graph_save_path)

    with torch.no_grad():
        predict_mesh = plant_graph.generate(output_format='mesh', color='blue')

    # visualize the plant
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([p_init, axis] + t_aligned_pcds_items, mesh_show_back_face=True)

    raw_input_pcd.translate(-main_stem_end_points_bottom)
    # o3d.visualization.draw_geometries( [raw_input_pcd] + [axis], mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([p_final, axis] , mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([p_final, axis, raw_input_pcd] , mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(os.path.join(infer_dir, 'predict.ply'), predict_mesh)
 
 
    gt_pcd = o3d.io.read_point_cloud(os.path.join(folder, 'pcd.ply'))
    gt_pcd_vertices = np.asarray(gt_pcd.points)
    gt_pcd_colors = np.asarray(gt_pcd.colors)

    new_vertices = (gt_pcd_vertices @ viz_rotation) / viz_radius
    gt_pcd_viz = o3d.geometry.PointCloud()
    gt_pcd_viz.points = o3d.utility.Vector3dVector(new_vertices)
    gt_pcd_viz.colors = o3d.utility.Vector3dVector(gt_pcd_colors)
    gt_pcd_viz.translate(-main_stem_end_points_bottom)

    o3d.visualization.draw_geometries([gt_pcd_viz, predict_mesh, axis], mesh_show_back_face=True)

    print('Saved plant graph to {}'.format(graph_save_path))
 

if __name__ == '__main__':


    kwargs = {
        'retrain': False,
        'retrain_leaf': False,
        'retrain_stem': False,
        'do_viz': True, # visualize intermediate steps
    }

    plant_id = '65_i'

    species = 'soybean'
    folder = f'sample_point_cloud/val/{plant_id}'

    reconstruction_3d(folder, species=species, **kwargs)
    # reconstruction_3d('27_o', species='soybean', **kwargs)
    # reconstruction_3d('15_i', species='soybean', **kwargs)

    # main('10008da', species='maize', do_viz=False)
    # main('M02_0325', species='maize', do_viz=False)
    # main('M07_0325', species='maize', do_viz=True)
    # main('0925da', species='maize', do_viz=False)
    # main('M02_0325', species='maize', do_viz=True)