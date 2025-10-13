import os
import numpy as np
import sys
import open3d as o3d
import time
import copy
import matplotlib.pyplot as plt
import scipy.optimize
import pickle
from skopt import gp_minimize

from maize_model import maize_plant, rotate2d
from reparameterize import maize_info_parametric
from soybean_model import soybean_plant
from reparameterize import soybean_info_parametric

def load_view_point(geoms, cam_json_path=None, img_save_path=None, h=1200, w=1200, point_size=1):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    ctr = vis.get_view_control()
    for geom in geoms:
        vis.add_geometry(geom)
    vis.get_render_option().mesh_show_back_face=True
    vis.get_render_option().point_size = point_size
    if cam_json_path is not None:
        param = o3d.io.read_pinhole_camera_parameters(cam_json_path)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.run()
    if img_save_path is not None:
        image = vis.capture_screen_float_buffer(False)
        print('Save image to {}'.format(img_save_path))
        plt.imsave(img_save_path, np.asarray(image), dpi = 100)
    vis.destroy_window()

sys.stdout = sys.stderr


PARAMETER = {}
PARAMETER['maize'] = {
    'LEAF_L_SCALE': [0.8, 1.2],
    'LEAF_ORDER_SHIFT': [-4.0, 4.0],
    'NODE_DIST_SCALE': [0.8, 1.2],
    'AZIMUTH_STD': [60, 60],
    'NODE_COUNT': [1, 18],
    'LEAF_L_STD': [0., 1.],
    'PET1_ANG_STD': [5., 5.],
    'NODE_DIST_STD': [0., 1.],
    'PLANT_DENSITY': [6.468, 6.468],
}
PARAMETER['soybean'] ={
    'LEAF_L_SCALE': [0.5, 1.25],
    'LEAF_W_TO_L': [0.6, 1.0],
    'PET_ANG_SCALE': [0.75, 4.0], 
    'NODE_DIST_SCALE': [0.75, 2.0], 
    'PET_LEN_SCALE': [0.5, 2.0],
    'AZIMUTH_STD': [0., 60.],
    'AZIMUTH_OFFSET': [0., 360.],
    'NODE_COUNT': [1, 14],
    
    'LEAF_L_STD': [0., 1.],
    'PET1_ANG_STD': [5., 5.],
    'NODE_DIST_STD': [0., 1.],
}
TO_FIT = {}
TO_FIT['maize'] = ['LEAF_L_SCALE', 'LEAF_ORDER_SHIFT', 'NODE_DIST_SCALE', 'NODE_COUNT']
TO_FIT['soybean'] = ['LEAF_L_SCALE', 'LEAF_W_TO_L', 'PET_ANG_SCALE', 'NODE_DIST_SCALE', 'PET_LEN_SCALE', 'NODE_COUNT', 
                     'AZIMUTH_OFFSET', 'AZIMUTH_STD'] 

def params2str(params, subset=None):
    subset = subset if subset is not None else params
    return ', '.join([('%s: %.2f' % (param_name, params[param_name])) for param_name in subset])

def soybean_plant_parametric(params, with_dummies=False):
    """Returns soybean plant with custom parameterization."""
    main_stem_info, branches_info = soybean_info_parametric(params, with_dummies=with_dummies)
    xyz, fac = soybean_plant(main_stem_info, branches_info)
    xyz /= 100.  # cm to m
    return xyz, fac
def maize_plant_parametric(params):
    """Returns maize plant with custom parameterization."""
    hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle = maize_info_parametric(params)
    xyz, fac = maize_plant(hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle)
    xyz = rotate2d(xyz, 90, dims=[0, 2])
    xyz /= 100.  # cm to m
    return xyz, fac

def fit_bo(target_pcd, method='bo', seed=1234, print_freq=20, verbose=True, species=None,
           bo_max_iter=150, bo_init_pts=100):
    """Fit the model parameters to target point cloud, using Bayesian optimization."""

    assert species in ['maize', 'soybean'], 'species should be either maize or soybean'
    
    # Initialize iteration counter
    iters = 0
    
    def params_from_vector(param_vector):
        """Parameter dictionary from subset vector."""
        params = params_init.copy()
        for i in range(len(param_subset)):
            params[param_subset[i]] = param_vector[i] * ranges[i][1]
        return params
    
    def f(param_vector):
        """Function to optimize."""
        nonlocal iters
        params = params_from_vector(param_vector)
        # params = {'LEAF_L_SCALE': 0.9581899535703184, 'LEAF_W_TO_L': 0.8, 'PET_ANG_SCALE': 1.6982513045250958, 'NODE_DIST_SCALE': 0.8101616448213436, 'PET_LEN_SCALE': 1.5497980305228836, 'AZIMUTH_STD': 60.0, 'NODE_COUNT': 13.234416231084584, 'LEAF_L_STD': 0.0, 'PET1_ANG_STD': 5.0, 'NODE_DIST_STD': 0.0, 'PLANT_DENSITY': 20.0}
                # params = {'LEAF_L_SCALE': 0.9486354382455569, 'LEAF_W_TO_L': 0.8, 'PET_ANG_SCALE': 1.6165775853087971, 'NODE_DIST_SCALE': 0.9574920962203928, 'PET_LEN_SCALE': 1.6424707458833445, 'AZIMUTH_STD': 60.0, 'NODE_COUNT': 13.013545939841624, 'LEAF_L_STD': 0.0, 'PET1_ANG_STD': 5.0, 'NODE_DIST_STD': 0.0, 'PLANT_DENSITY': 20.0}
        if iters % print_freq == 0:
            print('Iter %4d |' % iters, params2str(params, param_subset))
        if species == 'soybean':
            xyz, fac = soybean_plant_parametric(params)
        else:
            xyz, fac = maize_plant_parametric(params)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # chamfer distance
        loss = np.mean(np.array(pcd.compute_point_cloud_distance(target_pcd))) + \
               np.mean(np.array(target_pcd.compute_point_cloud_distance(pcd)))
        
        if iters % print_freq == 0:
            print('Query Loss:', loss)
            print('-' * 40)
        iters += 1

        return loss
    
    params_init = {k: v[0] for (k, v) in PARAMETER[species].items()}
    
    param_subset = TO_FIT[species]
    ranges = [PARAMETER[species][param_name] for param_name in param_subset]
    scaled_ranges = [(r[0] / r[1], 1) for r in ranges]
    # param_names, ranges = list(PARAMETER_RANGES.keys()), list(PARAMETER_RANGES.values())

    start = time.time()
 
        
    if method == 'grad':
        print([(r[0] + r[1]) / 2 for r in scaled_ranges])
        print(scaled_ranges)
        res = scipy.optimize.minimize(f, [(r[0] + r[1]) / 2 for r in scaled_ranges], 
                                      method='trust-constr', bounds=scaled_ranges, 
                                      options={'maxiter': bo_max_iter})
        # res = stochopy.optimize.minimize(f, scaled_ranges, method="pso", options={"maxiter": 100, "popsize": 10, "seed": 0})


        best_loss = res.fun
        best_params = params_from_vector(res.x)
        loss_history = [best_loss]
        params_history = [best_params]
        print(best_params)
        
    else:
        res = gp_minimize(f,                               # the function to minimize
                        scaled_ranges,                   # the bounds on each dimension of x
                        acq_func="EI",                   # the acquisition function
                        n_calls=bo_max_iter,           # the number of evaluations of f
                        n_initial_points=bo_init_pts,  # the number of random initialization points
                        initial_point_generator='sobol',    # the initial point generation strategy
                        noise='gaussian',                # the noise level (optional)
                        random_state=seed,               # the random seed
                        n_restarts_optimizer=10,
                        verbose=bool(verbose))          
            
        best_loss = res['fun']
        best_params = params_from_vector(res['x'])
        loss_history = res['func_vals']
        params_history = [params_from_vector(x_iter) for x_iter in res['x_iters']]
    print('total time:', time.time() - start, 's')
        
        
    return best_params, best_loss, loss_history, params_history


def reconstruct(species, pcd_path:str, save_path:str, bo_max_iter=150, bo_init_pts=100, retrain=False):
    """
    Reconstruct a plant from input point cloud.
    Parameters:
        species: 'maize' or 'soybean'
        pcd_path: path to input point cloud
        save_path: path to save the best parameters
        bo_max_iter: max iteration for bayesian optimization
        bo_init_pts: initial random points for bayesian optimization, should be less than bo_max_iter
    Returns:
        mesh: reconstructed mesh
    """
    assert species in ['maize', 'soybean'], 'species should be either maize or soybean'
    assert os.path.exists(pcd_path), f'pcd_path {pcd_path} does not exist'
    assert bo_init_pts < bo_max_iter, 'bo_init_pts should be less than bo_max_iter'
    assert save_path.endswith('.pkl'), 'save_path should be a .pkl file'

    # load input pcd
    target_pcd = o3d.io.read_point_cloud(pcd_path)

    if len(target_pcd.points) > 30000:
        random_index = np.random.choice(len(target_pcd.points), 30000, replace=False)
        target_pcd.points = o3d.utility.Vector3dVector(np.array(target_pcd.points)[random_index])
        if species == 'soybean':
            target_pcd.colors = o3d.utility.Vector3dVector(np.array(target_pcd.colors)[random_index])

    # calulate the scale of the target pcd
    scale = np.linalg.norm(np.array(target_pcd.get_max_bound()) - np.array(target_pcd.get_min_bound()))
    print('Scale of target pcd:', scale)

    target_pcd.scale(1/scale, center=(0, 0, 0))
    # our_plant.scale(1/scale, center=(0, 0, 0))

    # only get 1/3 of the target pcd randomly
    target_pcd_small = copy.deepcopy(target_pcd)
    # target_pcd_small.paint_uniform_color([0.5, 0.5, 0.5])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([target_pcd, our_plant, axis], mesh_show_back_face=True)

    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(save_path) or retrain:
        rep = 0
        iters = 0
        best_params, best_loss, best_loss_history, best_params_history = fit_bo(target_pcd, 
            method='bo', seed=0, print_freq=20, verbose=False, species=species,
            bo_max_iter=bo_max_iter, bo_init_pts=bo_init_pts)  # method='grad' or 'bo'
        print('Best params:', best_params)
        with open(save_path, 'wb') as f:
            pickle.dump(best_params, f)
    else:
        with open(save_path, 'rb') as f:
            best_params = pickle.load(f)
        print('Loaded best params:', best_params)
    
    # global_rotation = np.loadtxt(f'/home/tianhang/data/soybean_3d_infer/{plant_id}/global_M_p.txt')

    # visualize mesh
    if species == 'soybean':
        best_xyz, best_fac = soybean_plant_parametric(best_params)  ########### use this
    else:
        best_xyz, best_fac = maize_plant_parametric(best_params)  ########### use this
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(best_xyz)
    mesh.triangles = o3d.utility.Vector3iVector(best_fac)
    # mesh.rotate(global_rotation, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0., 0.])

    o3d.visualization.draw_geometries([mesh, target_pcd, axis], mesh_show_back_face=True)

    return mesh


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--point_path', type=str, default=None, help='path to input point cloud')
    parser.add_argument('--species', type=str, default='soybean', help='species: maize or soybean')
    args = parser.parse_args()

    point_path = args.point_path
    species = args.species

    if point_path is not None:
        save_path = os.path.join(os.path.dirname(point_path), 'l_sys', f'best_params.pkl')
        reconstruct(species, point_path, save_path, retrain=False)
        exit(0)

    # example usage
    # soybean
    pcd_path = '/home/tianhang/code/Demeter/sample_point_cloud/val/27_o/original_aligned.ply' # TODO: change to your path
    save_path = os.path.join(os.path.dirname(pcd_path), 'l_sys', f'best_params.pkl')
    reconstruct('soybean', pcd_path, save_path, retrain=False)

    # maize
    pcd_path = '/home/tianhang/code/Demeter/sample_point_cloud/val/10008da/original_aligned.ply' # TODO: change to your path
    save_path = os.path.join(os.path.dirname(pcd_path), 'l_sys', f'best_params.pkl')
    reconstruct('maize', pcd_path, save_path, retrain=False)