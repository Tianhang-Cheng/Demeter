import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import numpy as np
import matplotlib.pyplot as plt
from utils.color_print import *
import os
import open3d as o3d
import pickle
import torch

def interactive(pcd_temp, window_name=None, screen_width=2560, screen_height=1280):
    vis = o3d.visualization.VisualizerWithEditing()
    if window_name is not None:
        vis.create_window(window_name=window_name, width=screen_width, height=screen_height)
    else:
        vis.create_window(width=screen_width, height=screen_height)
    pcd = pcd_temp
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    selected_indices = vis.get_picked_points()
    selected_coords = np.asarray(pcd.points)[selected_indices] # (n,3)
    del pcd_temp
    return selected_indices, selected_coords

def rotation_matrix_from_axis_angle(axis, theta):
    """
    Construct a rotation matrix from axis and angle using Rodrigues' rotation formula.
    
    Parameters:
    axis (array-like): Unit vector representing rotation axis
    theta (float): Rotation angle in radians
    
    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    # Ensure axis is a unit vector
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    
    # Get components of axis vector
    ux, uy, uz = axis
    
    # Construct cross-product matrix K
    K = np.array([[0, -uz, uy],
                  [uz, 0, -ux],
                  [-uy, ux, 0]])
    
    # Construct rotation matrix using Rodrigues' formula
    # R = I + sin(θ)K + (1-cos(θ))K²
    I = np.eye(3)
    R = (I + 
         np.sin(theta) * K + 
         (1 - np.cos(theta)) * np.matmul(K, K))
    
    return R

def rotate_vector_to_target(P, target=[1, 0, 0]):
    """
    Rotate a unit vector P to align with a target vector using Rodrigues' rotation formula.
    
    Parameters:
    P (array-like): Input unit vector of length 3
    target (array-like): Target vector to align with, defaults to [1,0,0]
    
    Returns:
    numpy.ndarray: Rotated vector
    float: Rotation angle in radians
    numpy.ndarray: Rotation axis
    numpy.ndarray: 3x3 rotation matrix
    """
    # Convert inputs to numpy arrays and normalize them
    P = np.array(P, dtype=float)
    target = np.array(target, dtype=float)
    
    # Normalize vectors to ensure they're unit vectors
    P = P / np.linalg.norm(P)
    target = target / np.linalg.norm(target)
    
    # Calculate rotation angle (theta) using dot product
    cos_theta = np.dot(P, target)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # If vectors are already aligned or anti-aligned, return identity matrix
    if np.isclose(abs(cos_theta), 1, atol=1e-8):
        return P, theta, np.array([0, 0, 1]), np.eye(3)
    
    # Calculate rotation axis using cross product
    rotation_axis = np.cross(P, target)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Get rotation matrix
    R = rotation_matrix_from_axis_angle(rotation_axis, theta)
    
    # Apply rotation to original vector
    rotated = R @ P
    
    return rotated, theta, rotation_axis, R

def process_data(path: str):

    """
    Process point cloud data
    """

    print(f"Processing {path}")

    point_path = path
    assert os.path.exists(point_path), f"{point_path} does not exist!"

    # read data
    points_all = np.array(o3d.io.read_point_cloud(point_path).points)
    colors_all = np.array(o3d.io.read_point_cloud(point_path).colors)

    # center and scale
    points_all_center = points_all - points_all.mean(axis=0)
    radius = np.linalg.norm(points_all_center, axis=-1)
    radius = np.sort(radius)
    radius = radius[int(len(radius) * 0.95)] # find radius in 95% percentile
    
    geometries_points = points_all.astype(np.float32)
    geometries_colors = colors_all.astype(np.float32)

    rot_path = os.path.join(os.path.dirname(point_path), 'rotation.txt')

    if not os.path.exists(rot_path):

        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(geometries_points)
        pcd_temp.colors = o3d.utility.Vector3dVector(geometries_colors)
        blue_print("Please select two points to define the main axis (from bottom to top), use Ctrl + left click to select points. Then close the window.")
        selected_indices, selected_coords = interactive(pcd_temp)

        vector = selected_coords[1] - selected_coords[0]
        vector = vector / np.linalg.norm(vector)

        R = rotate_vector_to_target(vector)[-1]

        rotated_coords = points_all @ R.T
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(rotated_coords)
        pcd_new.colors = pcd_temp.colors
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_new, axis])

        rotation = (R.T).astype(np.float32)

        np.savetxt(rot_path, rotation)
        
        click_path = os.path.join(os.path.dirname(point_path), 'rotation_click.txt')
        np.savetxt(click_path, selected_coords)
    
    else:

        rotation = np.loadtxt(rot_path).astype(np.float32)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(geometries_points @ rotation)
    # # pcd.colors = o3d.utility.Vector3dVector(np.repeat(geometries_semantic_gt[:, None]/2, 3, axis=-1).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(geometries_colors)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=T, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis])

    bbox_center = geometries_points.mean(axis=0)
    geometries_points = (geometries_points - bbox_center)
    # bbox_radius = np.linalg.norm(geometries_points, axis=-1).max()
    # geometries_points = geometries_points / bbox_radius
    geometries_points = geometries_points @ rotation

    transform_dict = {'rotation': rotation, 'bbox_center': bbox_center, 'radius': radius}
    with open(os.path.join(os.path.dirname(point_path), 'transform.pkl'), 'wb') as f:
        pickle.dump(transform_dict, f)

    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(geometries_points / radius)
    pcd_final.colors = o3d.utility.Vector3dVector(geometries_colors)
    pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_final, axis])

    geometries_normals = np.array(pcd_final.normals)

    # save for training
    data = {}
    data['coord'] = geometries_points.astype(np.float32) / radius
    data['color'] = geometries_colors.astype(np.float32)
    data['normal'] = geometries_normals.astype(np.float32)
    
    sample_num = 300000
    sample_num = min(sample_num, len(data['coord']))
    sample_mask = np.random.choice(len(data['coord']), sample_num, replace=False)
    for key in data.keys():
        data[key] = data[key][sample_mask]
    data = {k: torch.tensor(v) for k, v in data.items()}
    
    data['scene_id'] = os.path.basename(os.path.dirname(point_path)) # can be any string, required by point-transformer

    # save as pth
    torch.save(data, os.path.join(os.path.dirname(point_path), 'normalized_pcd.pth'))
    print(f"Saved to {os.path.join(os.path.dirname(point_path), 'normalized_pcd.pth')}")

if __name__ == '__main__':

    point_id = '65_i'
    process_data(f'sample_point_cloud/val/{point_id}/pcd.ply')