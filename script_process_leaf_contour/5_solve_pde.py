
import argparse
import sys
from utils.pde import rasterize_polygon, relax_parallel, assign_boundary_value
import numpy as np
import os
import cv2
import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend for saving images
import matplotlib.pyplot as plt
import tqdm
import scipy.ndimage as ndimage
import time
import imageio.v2 as iio
from scipy.spatial import cKDTree

def sample_left_contour_heights(n):
    # Angles from bottom (3pi/2) to top (pi/2) equally spaced
    angles = np.linspace(3 * np.pi / 2,  np.pi / 2, n)
    
    # Compute y-coordinates
    y_values = np.sin(angles) * 0.5 + 0.5  # Scale to fit in [0, 1]
    x_values = np.cos(angles) * 0.5 + 0.5

    point = np.stack([x_values, y_values], axis=-1)
    return point

def sample_right_contour_heights(n):
    # Angles from bottom (3pi/2) to top (5pi/2) equally spaced
    angles = np.linspace(3 * np.pi / 2, 5 * np.pi / 2, n)
    
    # Compute y-coordinates
    y_values = np.sin(angles) * 0.5 + 0.5  # Scale to fit in [0, 1]
    x_values = np.cos(angles) * 0.5 + 0.5
    
    point = np.stack([x_values, y_values], axis=-1)
    # point = np.flip(point, axis=0) # flip to get the right contour
    return point

def horizontal_segment_lengths(y_values):
    # Compute the valid part inside the circle
    valid_mask = np.abs(y_values - 0.5) <= 0.5
    chord_lengths = np.zeros_like(y_values)
    chord_lengths[valid_mask] = 2 * np.sqrt(0.25 - (y_values[valid_mask] - 0.5) ** 2)
    return chord_lengths

def sample_left_contour(n):
    center = np.array([0.5, 0.5])
    radius = 0.5
    angles = np.linspace(3 * np.pi / 2, np.pi / 2, n)
    points = np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ))
    return points

def sample_right_contour(n):
    center = np.array([0.5, 0.5])
    radius = 0.5
    angles = np.linspace(3 * np.pi / 2, 5 * np.pi / 2, n)
    points = np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ))
    return points

def fill_hole(x): 
    boundary_mask_uint8 = x.astype(np.uint8) * 255

    # Find contours in the boundary mask
    contours, _ = cv2.findContours(boundary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask and fill the contours
    uv_mask = np.zeros_like(boundary_mask_uint8)
    cv2.drawContours(uv_mask, contours, -1, 255, cv2.FILLED)

    # Convert back to boolean if needed
    uv_mask = uv_mask > 0
    return uv_mask

def find_positions_in_matrix(matrix: np.ndarray, points: np.ndarray, mask: np.ndarray):
    """
    Finds the nearest positions of given 2D points in a (h, w, 2) matrix,
    considering only valid areas defined by the mask.

    Parameters:
        matrix (np.ndarray): A (h, w, 2) matrix where each [i, j] position stores a 2D coordinate.
        points (np.ndarray): An (n, 2) array of 2D points to locate in the matrix.
        mask (np.ndarray): A (h, w) boolean array where True indicates valid positions.

    Returns:
        np.ndarray: An (n, 2) array of indices in the matrix corresponding to the nearest valid positions.
    """
    h, w, _ = matrix.shape

    # Get valid indices and their corresponding coordinates
    valid_indices = np.argwhere(mask)  # (num_valid, 2)
    valid_coords = matrix[mask]  # (num_valid, 2)

    # Build a KDTree using only valid coordinates
    tree = cKDTree(valid_coords)

    # Query the tree to find nearest neighbor indices for each point
    _, nearest_valid_indices = tree.query(points)

    # Map back to the original (row, col) indices
    positions = valid_indices[nearest_valid_indices]

    return positions


def find_weighted_positions_in_matrix(matrix: np.ndarray, points: np.ndarray, mask: np.ndarray, k: int = 3):
    """
    Finds the nearest k positions of given 2D points in a (h, w, 2) matrix (only within valid mask areas)
    and computes a weighted position based on inverse-distance weighting.

    Parameters:
        matrix (np.ndarray): A (h, w, 2) matrix where each [i, j] position stores a 2D coordinate.
        points (np.ndarray): An (n, 2) array of 2D points to locate in the matrix.
        mask (np.ndarray): A (h, w) boolean array where True indicates valid positions.
        k (int): Number of nearest valid positions to consider (default is 3).

    Returns:
        np.ndarray: An (n, 2) array of weighted (interpolated) positions corresponding to the query points.
    """
    # Extract valid indices and corresponding coordinates
    valid_indices = np.argwhere(mask)  # (num_valid, 2)
    valid_coords = matrix[mask]  # (num_valid, 2)

    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(valid_coords)

    # Query k nearest valid neighbors for each point
    distances, nearest_valid_indices = tree.query(points, k=k)
    
    # Ensure distances and indices are 2D (useful if k=1)
    if k == 1:
        distances = distances[:, np.newaxis]
        nearest_valid_indices = nearest_valid_indices[:, np.newaxis]

    # Get the corresponding (row, col) indices of nearest neighbors
    nearest_positions = valid_indices[nearest_valid_indices]  # Shape: (n, k, 2)

    # Handle exact matches (zero distance): directly assign the corresponding valid index
    exact_match_mask = distances == 0  # Shape: (n, k)
    any_exact_match = np.any(exact_match_mask, axis=1)  # Shape: (n,)

    # Select the first exact match where available
    exact_match_indices = np.argmax(exact_match_mask, axis=1)  # Shape: (n,)
    exact_positions = nearest_positions[np.arange(points.shape[0]), exact_match_indices]  # Shape: (n, 2)

    # Compute inverse-distance weights for non-exact matches
    weights = np.where(distances > 0, 1.0 / (distances+1e-5), 0)  # Shape: (n, k)
    weighted_positions = np.sum(nearest_positions * weights[:, :, np.newaxis], axis=1) / np.sum(weights, axis=1, keepdims=True)

    # Combine exact matches and interpolated values
    final_positions = np.where(any_exact_match[:, np.newaxis], exact_positions, weighted_positions)

    return final_positions

def pose_to_leaf_uv(uv_map, pos):
    """
    Bilinear interpolation for UV coordinates from continuous positions.
    
    Args:
        uv_map: UV coordinate map of shape (H, W, 2)
        pos: Continuous positions of shape (..., 2) where pos[..., 0] is x, pos[..., 1] is y
    
    Returns:
        Interpolated UV coordinates of same shape as pos
    """
    h, w = uv_map.shape[:2]
    
    # Get continuous coordinates
    x = pos[..., 0]
    y = pos[..., 1]
    
    # Clamp to valid range
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    # Get integer parts (floor)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    
    # Get fractional parts
    dx = x - x0
    dy = y - y0
    
    # Get UV values at four corners
    uv_00 = uv_map[y0, x0]  # top-left
    uv_01 = uv_map[y1, x0]  # bottom-left
    uv_10 = uv_map[y0, x1]  # top-right
    uv_11 = uv_map[y1, x1]  # bottom-right
    
    # Expand dimensions for broadcasting
    dx = dx[..., np.newaxis]
    dy = dy[..., np.newaxis]
    
    # Bilinear interpolation
    uv_top = uv_00 * (1 - dx) + uv_10 * dx
    uv_bottom = uv_01 * (1 - dx) + uv_11 * dx
    uv_interpolated = uv_top * (1 - dy) + uv_bottom * dy

    # flip y axis to match image coordinate
    uv_interpolated = np.flip(uv_interpolated, axis=-1)  
    
    return uv_interpolated

def normalize_points(points, margin=0.0, center=True):
    """
    Normalize 2D points into [margin, 1-margin] x [margin, 1-margin] 
    with uniform scaling.

    Args:
        points (np.ndarray): shape (n_instance, n_points, 2), input 2D points.
        margin (float): margin from each boundary (default=0.0).
                        Example: 0.01 → [0.01, 0.99]
        center (bool): whether to center the shape inside the target box 
                       if aspect ratio < 1 (default=True).
    Returns:
        norm_points (np.ndarray): normalized points
        transform_params (dict): parameters needed for denormalization
    """ 
    
    # Apply initial alignment (same as your original code)
    aligned_points = points.copy()
    original_first_points = points[:, 0:1].copy()
    aligned_points -= original_first_points  # move the bottom point to the origin
    original_heights = np.abs(aligned_points[:, -1, 1])
    aligned_points /= original_heights[:, None, None]  # divide by height

    # Bounding box
    x_min, y_min = aligned_points[..., 0].min(), aligned_points[..., 1].min()
    x_max, y_max = aligned_points[..., 0].max(), aligned_points[..., 1].max()

    # Translate
    translation = (-x_min, -y_min)
    translated = aligned_points + translation

    # Uniform scale
    width = x_max - x_min
    height = y_max - y_min
    scale = (1 - 2 * margin) / max(width, height)
    scaled = translated * scale

    if center:
        # Compute available space
        box_w = scaled[..., 0].max()
        box_h = scaled[..., 1].max()
        offset_x = ((1 - 2 * margin) - box_w) / 2
        offset_y = ((1 - 2 * margin) - box_h) / 2
    else:
        offset_x = 0
        offset_y = 0

    # Final placement inside [margin, 1-margin]^2
    norm_points = scaled + np.array([margin + offset_x, margin + offset_y])
 
    # Sanity check
    y_max = norm_points[..., 1].max() 
    y_min = norm_points[..., 1].min()
    x_max = norm_points[..., 0].max()
    x_min = norm_points[..., 0].min()
    assert x_max <= 1.0 and x_min >= 0.0, f'Invalid x range: {x_min} to {x_max}'
    assert y_max <= 1.0 and y_min >= 0.0, f'Invalid y range: {y_min} to {y_max}'

    # Package transformation parameters for denormalization
    transform_params = {
        'original_first_points': original_first_points,
        'original_heights': original_heights,
        'translation': translation,
        'scale': scale,
        'offset': (offset_x, offset_y),
        'margin': margin
    }

    return norm_points, transform_params


def denormalize_points(norm_points, transform_params, idx=None):
    """
    Recover original points coordinates from normalized points.
    
    Args:
        norm_points (np.ndarray): normalized points from normalize_points()
        transform_params (dict): transformation parameters from normalize_points()
        
    Returns:
        points (np.ndarray): recovered original points
    """
    # Extract transformation parameters
    original_first_points = transform_params['original_first_points']
    original_heights = transform_params['original_heights']
    translation = transform_params['translation']
    scale = transform_params['scale']
    offset_x, offset_y = transform_params['offset']
    margin = transform_params['margin']
    
    # Reverse the transformations in opposite order
    
    # 1. Remove final placement offset and margin
    points = norm_points - np.array([margin + offset_x, margin + offset_y])
    
    # 2. Reverse uniform scaling
    points = points / scale
    
    # 3. Reverse translation
    points = points - translation
    
    if idx is None:

        # 4. Reverse height normalization
        points = points * original_heights[:, None, None]
        
        # 5. Reverse alignment (add back original first points)
        points = points + original_first_points
    
    else:

        # 4. Reverse height normalization
        points = points * original_heights[idx, None, None]
        
        # 5. Reverse alignment (add back original first points)
        points = points + original_first_points[idx]
    
    return points

def leaf_uv_to_image(uv, params, img, idx:int, bilinear: bool = True, size: tuple = None):
    """
    uv: [..., 2], uv on normalize leaf space
    params: transform params from normalize_points()
    size: (h, w) of the original image, if None, return a flat array
    """
    h, w = img.shape[0:2]
    # uv should be in [0, 1] x [0, 1]
    assert uv[..., 0].min() >= 0.0 and uv[..., 0].max() <= 1.0
    assert uv[..., 1].min() >= 0.0 and uv[..., 1].max() <= 1.0
    
    # map uv of normalized leaf to original image space
    uv_recover = denormalize_points(uv.reshape(1, -1, 2), params, idx=idx)[0] 
    
    if bilinear:
        # For bilinear interpolation, keep the floating point coordinates
        y_coords = uv_recover[..., 1]  # row coordinates
        x_coords = uv_recover[..., 0]  # column coordinates
        
        # Get the four corner points for bilinear interpolation
        x0 = np.floor(x_coords).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y_coords).astype(int)
        y1 = y0 + 1
        
        # Clip coordinates to valid image bounds
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        
        # Get fractional parts for interpolation weights
        wx = x_coords - np.floor(x_coords)
        wy = y_coords - np.floor(y_coords)
        
        # Handle edge cases where coordinates are exactly on boundaries
        # This prevents wx or wy from being used when x1 == x0 or y1 == y0
        wx = np.where(x1 == x0, 0, wx)
        wy = np.where(y1 == y0, 0, wy)
        
        # Get the four corner pixel values
        I00 = img[y0, x0]  # top-left
        I01 = img[y0, x1]  # top-right
        I10 = img[y1, x0]  # bottom-left
        I11 = img[y1, x1]  # bottom-right
        
        # Bilinear interpolation formula
        # First interpolate in x direction
        I0 = I00 * (1 - wx)[..., np.newaxis] + I01 * wx[..., np.newaxis]
        I1 = I10 * (1 - wx)[..., np.newaxis] + I11 * wx[..., np.newaxis]
        
        # Then interpolate in y direction
        recover_image = I0 * (1 - wy)[..., np.newaxis] + I1 * wy[..., np.newaxis]
        
    else:
        # Original nearest-neighbor sampling
        a = uv_recover[..., 1].astype(int)
        b = uv_recover[..., 0].astype(int)
        
        # clip to be valid indices
        a = np.clip(a, 0, h-1)
        b = np.clip(b, 0, w-1)
        # get color
        recover_image = img[a, b]
    
    if size is not None:
        recover_image = recover_image.reshape(size[0], size[1], -1)
        
    return recover_image


def add_texture_for_leaf(texture_on_template, l2t_solve_final, valid_mask):
    """
    Bilinear sampling of texture_on_template using l2t_solve_final coordinates.

    Args:
        texture_on_template (np.ndarray): (H, W, C) array, source texture.
        l2t_solve_final (np.ndarray): (h_viz, w_viz, 2) array, normalized coords in [0,1].
        valid_mask (np.ndarray): (h_viz, w_viz) boolean mask for valid pixels.

    Returns:
        np.ndarray: (h_viz, w_viz, C) retextured image.
    """
    h_viz, w_viz, _ = l2t_solve_final.shape
    leaf_retextured = np.ones((h_viz, w_viz, 3), dtype=np.float32) * 255

    # Map to texture coordinates
    _i = l2t_solve_final[..., 1] * (h_viz - 1)
    _j = l2t_solve_final[..., 0] * (w_viz - 1)

    x0 = np.floor(_i).astype(int)
    y0 = np.floor(_j).astype(int)
    x1 = np.clip(x0 + 1, 0, h_viz - 1)
    y1 = np.clip(y0 + 1, 0, w_viz - 1)

    dx = _i - x0
    dy = _j - y0

    # Gather values
    top_left     = texture_on_template[x0, y0] * (1 - dx)[..., None] * (1 - dy)[..., None]
    top_right    = texture_on_template[x0, y1] * (1 - dx)[..., None] * dy[..., None]
    bottom_left  = texture_on_template[x1, y0] * dx[..., None] * (1 - dy)[..., None]
    bottom_right = texture_on_template[x1, y1] * dx[..., None] * dy[..., None]

    sampled = top_left + top_right + bottom_left + bottom_right

    # Only assign to valid pixels
    leaf_retextured[valid_mask] = sampled[valid_mask]

    return leaf_retextured

def solve_pde(data_dir, bound_type='circle', resolution: int=512, retrain: bool = False):

    Bounds = ['square', 'circle']
    assert bound_type in Bounds, f'Invalid bound type: {bound_type}, should be one of {Bounds}'

    image_foler = os.path.abspath(data_dir)
    species = os.path.basename(os.path.normpath(image_foler))
    rot_folder = image_foler + '_rotated'
    os.makedirs(rot_folder, exist_ok=True)
    mask_folder = image_foler + '_rotated_mask'
    os.makedirs(mask_folder, exist_ok=True)
    curve_folder = image_foler
    edge_folder = os.path.join(mask_folder, 'edge')
    keypoint_folder = os.path.join(mask_folder, 'keypoints')
    
    solution_folder = os.path.join(mask_folder, 'solution', bound_type)
    os.makedirs(solution_folder, exist_ok=True)
    solution_viz_folder = os.path.join(mask_folder, 'solution_viz', bound_type)
    os.makedirs(solution_viz_folder, exist_ok=True)

    def read_mask(name):
        rot_mask_path = os.path.join(mask_folder, f'{name}.jpg')
        rot_mask = cv2.imread(rot_mask_path)
        return rot_mask

    def read_image(name):
        rot_img_path = os.path.join(rot_folder, f'{name}.jpg')
        rot_img = cv2.imread(rot_img_path)
        rot_img = np.flip(rot_img, axis=0)
        rot_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)
        return rot_img
    
    image_list = os.listdir(image_foler)
    image_list.sort()
    image_list = [image_name for image_name in image_list if 'jpg' in image_name]
    # image_list = image_list[split::total_splits]
    if species == 'geranium':
        bad_sample = ['008', '012','016','017']
        image_list = [image_name for image_name in image_list if not any(bad in image_name for bad in bad_sample)]

    image_list_filtered = []
    for index, image_name in enumerate(tqdm.tqdm(image_list)):
        name = image_name.split('.')[0]
        rot_mask_path = os.path.join(mask_folder, f'{name}.jpg')
        if not os.path.exists(rot_mask_path):
            print(f'No mask found for {name}, skipping...')
            continue
        image_list_filtered.append(name)

    point_path = os.path.join(mask_folder, 'info', 'uniform_points_all.npy')
    os.makedirs(os.path.dirname(point_path), exist_ok=True)
    if os.path.exists(point_path) and False:
        # print(f'Uniform points already exist at {point_path}, skipping...')
        uniform_points_all = np.load(point_path)
    else:
        # print(f'Uniform points not found at {point_path}, generating...')

        right_uniform_points_all = []
        left_uniform_points_all = []

        for index, image_name in enumerate(tqdm.tqdm(image_list_filtered)):

            name = image_name.split('.')[0]
            # if name != '00036':
            #     continue

            rot_keypoint_path = os.path.join(keypoint_folder, f'{name}.txt')
            keypoints = np.loadtxt(rot_keypoint_path)
            # get the highest and lowest points
            highest_point = keypoints[1]
            lowest_point = keypoints[0]

            # np.savetxt(os.path.join(mask_folder, f'{name}_left_uniform.txt'), left_uniform_points)
            # np.savetxt(os.path.join(mask_folder, f'{name}_right_uniform.txt'), right_uniform_points)

            left_uniform_points = np.loadtxt(os.path.join(edge_folder, f'{name}_left_uniform.txt'))
            right_uniform_points = np.loadtxt(os.path.join(edge_folder, f'{name}_right_uniform.txt'))

            left_uniform_points = np.flip(left_uniform_points, axis=1)
            right_uniform_points = np.flip(right_uniform_points, axis=1)

            left_uniform_points = np.flip(left_uniform_points, axis=0)
            right_uniform_points = np.flip(right_uniform_points, axis=0)

            if left_uniform_points[0, 1] > right_uniform_points[0, 1]:
                temp = left_uniform_points[0]
            else:
                temp = right_uniform_points[0]
            left_uniform_points[0] = temp
            right_uniform_points[0] = temp

            if left_uniform_points[-1, 1] < right_uniform_points[-1, 1]: # top, select the higher one
                temp = left_uniform_points[-1]
            else:
                temp = right_uniform_points[-1]
            left_uniform_points[-1] = temp
            right_uniform_points[-1] = temp

            h, w = read_image(name).shape[0:2]
            left_uniform_points[..., 1] = h - left_uniform_points[..., 1]
            right_uniform_points[..., 1] = h - right_uniform_points[..., 1]

            # plt.figure()
            # plt.scatter(left_uniform_points[:, 0], left_uniform_points[:, 1], s=1, c='r')
            # plt.scatter(right_uniform_points[:, 0], right_uniform_points[:, 1], s=1, c='b')
            # plt.xlim(0, w)
            # plt.ylim(0, h)
            # plt.imshow(np.flip(read_mask(name), axis=0), origin='lower')
            # plt.show(block=False)

            left_uniform_points_all.append(left_uniform_points)
            right_uniform_points_all.append(right_uniform_points) # (width, height)

        left_uniform_points_all = np.stack(left_uniform_points_all, axis=0) # (n_instance, n_points, 2)
        right_uniform_points_all = np.stack(right_uniform_points_all, axis=0) # (n_instance, n_points, 2)

        uniform_points_all = np.concatenate([left_uniform_points_all, right_uniform_points_all], axis=1) # (n_instance, 2*n_points, 2)

        # np.save(os.path.join(mask_folder, 'info', 'uniform_points_all.npy'), uniform_points_all)

    # normalize to [0, 1] x [0, 1] with margin 0.01
    norm, params = normalize_points(uniform_points_all, margin=0.01) #TODO: also update for other species except papaya
    # recover = denormalize_points(norm, params)
    # np.testing.assert_allclose(recover, uniform_points_all, atol=1e-5)

    n_instance, n_points, _ = uniform_points_all.shape
    n_points = n_points // 2

    for index, uniform_points_unit in enumerate(tqdm.tqdm(norm)):

        name = image_list_filtered[index]
        rot_img = read_image(name)
        h, w = rot_img.shape[0:2]

        h_viz = w_viz = resolution

        l_p = uniform_points_unit[:n_points] # (n_points, 2)
        r_p = uniform_points_unit[n_points:] # (n_points, 2)

        if species == 'papaya':
            # for papaya and geranium, the points are from bottom to top, no need
            pass
        elif species == 'soybean' or species == 'geranium' or species == 'maize' or species == 'ficus':
            # for soybean, the points are from top to bottom, need to flip
            l_p = l_p[::-1]
            r_p = r_p[::-1]
        else:
            raise NotImplementedError(f'Not implemented for species: {species}')

        # fig = plt.figure(figsize=(5, 5))
        # plt.scatter(l_p[:, 0], l_p[:, 1], s=1, c='r', label='Left contour')
        # plt.scatter(r_p[:, 0], r_p[:, 1], s=1, c='b', label='Right contour')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.legend()
        # plt.title(f'Uniform contour points of {name}')
        # plt.show()

        # plt.imshow(rot_img)
        # plt.show(block=False, origin='lower')

        l_seq = rasterize_polygon(l_p, h_viz, w_viz, scale=True)[1]
        r_seq = rasterize_polygon(r_p, h_viz, w_viz, scale=True)[1]

        if bound_type == 'square':
            # use rectangle to get the boundary value
            l_bound_value_leaf = assign_boundary_value(l_seq, left=True)
            r_bound_value_leaf = assign_boundary_value(r_seq, right=True)
        else:
            # use circle to get the boundary value
            l_bound_value_leaf = sample_left_contour(len(l_seq))
            r_bound_value_leaf = sample_right_contour(len(r_seq))

        if bound_type == 'circle':
            _u, _v = np.meshgrid(np.linspace(0, 1, w_viz), np.linspace(0, 1, h_viz))
            _uv = np.stack([_u, _v], axis=-1).reshape(h_viz, w_viz, 2)
            temp_shape_mask = np.sqrt((_uv[..., 0] - 0.5) ** 2 + (_uv[..., 1] - 0.5) ** 2) <= 0.5
        else:
            temp_shape_mask = np.ones((h_viz, w_viz), dtype=bool)

        # Assign boundary values to the solution space (leaf to template)
        temp = np.zeros((h_viz, w_viz, 2))
        boundary_mask = np.zeros((h_viz, w_viz), dtype=bool)
        for i in range(len(l_seq)):
            ind = l_seq[i]
            temp[ind[0], ind[1]] = l_bound_value_leaf[i]
            boundary_mask[ind[0], ind[1]] = True
        for i in range(len(r_seq)):
            ind = r_seq[i]
            temp[ind[0], ind[1]] = r_bound_value_leaf[i]
            boundary_mask[ind[0], ind[1]] = True
        l2t_solve = np.zeros((h_viz, w_viz, 3)) + 0.5
        l2t_solve[..., 0:2][boundary_mask] = temp[boundary_mask]

        # Define necessary variables for the iterative solution
        # (fill the hole of boundary mask)

        uv_mask = ndimage.binary_fill_holes(boundary_mask)
        update_mask_l2t = uv_mask & ~boundary_mask
        valid_mask_leaf = (uv_mask | boundary_mask)

        # UV reference of template
        u, v = np.meshgrid(np.linspace(0, 1, w_viz), np.linspace(0, 1, h_viz))
        uv_ref_template = np.zeros((h_viz, w_viz, 3))
        uv_ref_template[..., 0] = u
        uv_ref_template[..., 1] = v
        uv = np.stack([u, v], axis=-1)
        
        init_guess = np.concatenate([uv, np.zeros((h_viz, w_viz, 1))], axis=-1)
        l2t_solve[update_mask_l2t] = init_guess[update_mask_l2t]# using the uv coordinates as the initial guess

        # leaf rgb
        leaf_rgb_normed = leaf_uv_to_image(uv, params, rot_img, idx=index, size=(h_viz, w_viz)).astype(np.uint8)

        # Solve Laplace's equation by iteratively relaxing U and V.
        solution_path = os.path.join(solution_folder, f'{name}_{h_viz}.npy')
        if os.path.exists(solution_path) and not retrain:
            print(f'Solution already exists at {solution_path}, skipping...')
            l2t_solve_final = np.load(solution_path)
            U_l2t = l2t_solve_final[..., 0].copy()
            V_l2t = l2t_solve_final[..., 1].copy()
            # continue
        else:
            U_l2t = l2t_solve[..., 0].copy()
            V_l2t = l2t_solve[..., 1].copy()
            start_time = time.time()
            # iters, U_l2t, V_l2t = relax(U_l2t, V_l2t, update_mask_l2t, h_viz, w_viz) # 55.93 s
            iters, U_l2t, V_l2t = relax_parallel(U_l2t, V_l2t, update_mask_l2t, h_viz, w_viz, max_iter=100000) # 14.49 s
            end_time = time.time()
            print(f"L2T Converged after {iters} iterations in {end_time - start_time:.2f} seconds.")

            # UV solution of T2L
            l2t_solve_final = np.zeros((h_viz, w_viz, 3))
            l2t_solve_final[..., 0] = U_l2t
            l2t_solve_final[..., 1] = V_l2t
            l2t_solve_final[~valid_mask_leaf] = 0.5

            np.save(solution_path, l2t_solve_final)
    
        # plt.imshow(uv_mask, origin="lower")
        # plt.imshow(boundary_mask, origin="lower")
        # plt.imshow(update_mask_l2t)
        # plt.show()
        # plt.imshow(t2l_solve_final, origin="lower")
        # plt.imshow(l2t_solve, origin="lower")
        # plt.imshow(l2t_solve_final, origin="lower")
        # plt.imshow(init_guess, origin="lower")
        # plt.imshow(np.concatenate([uv, np.zeros([h_viz, w_viz, 1])], axis=-1), origin="lower")

        # Get UV solution of L2T from T2L by finding the nearest neighbor
        pos = find_weighted_positions_in_matrix(matrix=l2t_solve_final[..., 0:2], points=uv.reshape(-1, 2), mask=valid_mask_leaf, k=3)
        pos = pos.reshape(h_viz, w_viz, 2)
        t2l_solve_final = np.zeros((h_viz, w_viz, 3))
        t2l_solve_final[..., 0:2] = pos / np.array([h_viz-1, w_viz-1])

        # Save texture to template
        # First from template to leaf, then query the texture from original image
        t2l_uv = pose_to_leaf_uv(uv_map=uv, pos=pos) # uv[pos[..., 0].astype(int), pos[..., 1].astype(int)]
        texture_on_template = leaf_uv_to_image(t2l_uv, params, rot_img, idx=index, size=(h_viz, w_viz)).astype(np.uint8)
        texture_on_template[~temp_shape_mask] = 255
        # fig = plt.figure()
        # plt.imshow(texture_on_template, origin='lower')
        # plt.show(block=False)

        # Query texture on template and map to leaf
        leaf_retextured = add_texture_for_leaf(texture_on_template, l2t_solve_final, valid_mask_leaf).astype(np.uint8)

        do_retexture_viz = True
        if do_retexture_viz:
            src_species_list = ['papaya', 'geranium', 'soybean', 'maize', 'ficus']
            for src_species in src_species_list:
                texture_from_src_path = f'/home/tianhang/Desktop/texture/{src_species}.png'
                texture_from_src = cv2.imread(texture_from_src_path)
                texture_from_src = cv2.cvtColor(texture_from_src, cv2.COLOR_BGR2RGB)
                texture_from_src = np.flip(texture_from_src, axis=0)
                # plt.imshow(texture_from_src, origin='lower')

                leaf_retextured_from_src = add_texture_for_leaf(texture_from_src, l2t_solve_final, valid_mask_leaf).astype(np.uint8)
                plt.imshow(leaf_retextured_from_src.reshape(h_viz, w_viz, 3), origin='lower')
                plt.axis('off')
                plt.savefig(os.path.join(f'/home/tianhang/Desktop/texture/{src_species}_{species}.png'), bbox_inches='tight', pad_inches=0)
                # plt.show(block=False)
                plt.close()
                print(f'Retarget retextured from {src_species} texture to {species}.png')

        if bound_type == 'circle':
            texture_on_template[~temp_shape_mask] = 255
            uv_ref_template[~temp_shape_mask] = 0.5
            t2l_solve_final[~temp_shape_mask] = 0.5

        # skeleton joints on template

        eps = 1e-5
        viz_skeleton = False

        if bound_type == 'square':
            n_joint = 43
            
            u_joints, v_joints = np.meshgrid(np.linspace(eps, 1-eps, n_joint), np.linspace(0, 1, n_joint+2)[1:-1])
            # u_joints, v_joints = np.meshgrid(np.linspace(eps, 1-eps, n_joint), np.linspace(0, 1, n_joint))
            uv_joints = np.stack([u_joints, v_joints], axis=-1)
            pos_skeleton = find_weighted_positions_in_matrix(matrix=l2t_solve_final[..., 0:2], points=uv_joints.reshape(-1, 2), mask=valid_mask_leaf, k=3)
            leaf_skeleton_image = np.zeros((h_viz, w_viz, 3))
            pos_skeleton = pos_skeleton.reshape(n_joint, n_joint, 2)

            if viz_skeleton:
                fig = plt.figure()
                for layer_id, layer in enumerate(pos_skeleton):
                    plt.plot(layer[:, 1], layer[:, 0], c='r')
                    plt.scatter(layer[:, 1], layer[:, 0], s=5, c='white')

        if bound_type == 'circle':

            n_joint = 43
            u_joints, v_joints = np.meshgrid(np.linspace(eps, 1-eps, n_joint), np.linspace(0, 1, n_joint+2))

            # _height = np.linspace(0, 1, n_joint+2)
            # _length = horizontal_segment_lengths(_height)
            # u_joints = (u_joints - 0.5) * _length[:, None] + 0.5
            # uv_joints = np.stack([u_joints, v_joints], axis=-1)
            # plt.scatter(u_joints, v_joints, s=1)
            # plt.show()

            _lp = sample_left_contour_heights(n_joint+2) # (45, 2)
            _rp = sample_right_contour_heights(n_joint+2)
            _t = np.linspace(0, 1, n_joint) # (43)
            _gap = (_rp - _lp) # (45, 2)
            uv_joints = _lp[:, None] + _gap[:, None] * _t[None, :, None] # (45, 1, 2) + (45, 1, 2) * (1, 43, 1)
            # plt.scatter(uv_joints[..., 0], uv_joints[..., 1], s=1, c='r')
            # plt.scatter(_lp[:, 0], _lp[:, 1], s=1)
            # plt.scatter(_rp[:, 0], _rp[:, 1], s=1)
            # plt.show()

            pos_skeleton = find_weighted_positions_in_matrix(matrix=l2t_solve_final[..., 0:2], points=uv_joints.reshape(-1, 2), mask=valid_mask_leaf, k=3)

            leaf_skeleton_image = np.zeros((h_viz, w_viz, 3))
            pos_skeleton = pos_skeleton.reshape(n_joint+2, n_joint, 2)
            
            if viz_skeleton:
                fig = plt.figure()
                for layer_id, layer in enumerate(pos_skeleton):
                    plt.plot(layer[:, 1], layer[:, 0], c='r')
                    plt.scatter(layer[:, 1], layer[:, 0], s=5, c='white')
                # plt.scatter(u_joints, v_joints, s=5)
                plt.show(block=False)
            
            # Assign color to skeleton image


        collect_inner_points = True
        if collect_inner_points:
            inner_points = pos_skeleton.copy() / np.array([h_viz-1, w_viz-1])
            leaf_bottom = inner_points[0, n_joint//2]
            inner_points = inner_points - leaf_bottom # move the bottom point to the origin
            # plt.scatter(inner_points[:, :, 1], inner_points[:, :, 0], s=5, c='blue')
            # plt.scatter(inner_points[:, n_joint//2, 1], inner_points[:, n_joint//2, 0], s=2, c='r')
            save_folder = os.path.join(mask_folder, 'skeleton', bound_type)
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, '{}.npy'.format(name)), inner_points)
        
        # continue
        do_viz_skeleton = True
        if do_viz_skeleton:
            dpi = 300
            fig = plt.figure(figsize=(1200/dpi, 1200/dpi), dpi=dpi)

            # if canonical:
            #     pos_skeleton = uv_joints.copy()

            for layer_id, layer in enumerate(pos_skeleton):
                plt.plot(layer[0:(n_joint)//2+1, 1], layer[:(n_joint)//2+1, 0], c='purple', linewidth=1.5, alpha=0.7)
                plt.plot(layer[(n_joint)//2:, 1], layer[(n_joint)//2:, 0], c='deeppink', linewidth=1.5, alpha=0.7)
                plt.scatter(layer[:, 1], layer[:, 0], s=9, facecolors='none', edgecolors='gray')
            # for layer_id, layer in enumerate(pos_skeleton.transpose(1, 0, 2)):
            #     if layer_id == (n_joint//2):
            #         continue
            #     plt.plot(layer[:, 1], layer[:, 0], c='orange', linewidth=0.8)
            plt.plot(pos_skeleton[:, (n_joint)//2, 1], pos_skeleton[:, n_joint//2, 0], c='orange', linewidth=1.5, alpha=0.7)
            # plt.scatter(u_joints, v_joints, s=5) 
            # plt.plot(l_seq_np[:, 1], l_seq_np[:, 0], c='r', linewidth=2)
            # plt.plot(r_seq_np[:, 1], r_seq_np[:, 0], c='b', linewidth=2)

            # if not canonical:
            l_p_viz = pos_skeleton[:, 0]
            r_p_viz = pos_skeleton[:, -1]
            plt.plot(l_p_viz[:, 1], l_p_viz[:, 0], c='r', linewidth=1.5, alpha=0.7)
            plt.plot(r_p_viz[:, 1], r_p_viz[:, 0], c='b', linewidth=1.5, alpha=0.7)

            plt.xlim(0, w_viz)
            plt.ylim(0, h_viz)
            # else:
            #     plt.plot(l_bound_value_leaf[:, 1], l_bound_value_leaf[:, 0], c='r', linewidth=2)
            #     plt.plot(r_bound_value_leaf[:, 1], r_bound_value_leaf[:, 0], c='b', linewidth=2)

            plt.axis('off')
            # equal aspect ratio
            plt.gca().set_aspect('equal', adjustable='box')
            # plt.show(block=False)

            # plt.savefig(f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{species}_rotated_mask/figure/l2t_solve_final.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            os.makedirs(os.path.join(mask_folder, 'skeleton_viz'), exist_ok=True)
            plt.savefig(os.path.join(mask_folder, 'skeleton_viz', f'{name}_{resolution}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

        # for _v, _u in pos_skeleton.reshape(-1, 2):
        #     _v = int(_v)
        #     _u = int(_u)
        #     leaf_skeleton_image[_v-3:_v+3, _u-3: _u+3] = 255
        if viz_skeleton:
            leaf_skeleton_image = leaf_skeleton_image.astype(np.uint8)
            # plt.imshow(leaf_rgb_normed)
            plt.imshow(leaf_skeleton_image, alpha=0.8)
            # plt.xlim(0, w_viz)
            # plt.ylim(0, h_viz)
            plt.show(block=False)

        # texture_on_template[_v, _u] = 255
        # # skeleton joints on leaf
        # _coord = np.round(t2l_solve_final[..., :2] * np.array([h-1, w-1])).astype(np.int32).reshape(-1, 2)
        
        # visualize
        do_viz_solution = True
        if do_viz_solution:
            fig = plt.figure()

            ax1 = fig.add_subplot(232)
            ax1.imshow(l2t_solve_final)
            ax1.set_title('Leaf to Template')
            ax1.set_ylim(0, h_viz)
            ax1.set_xlim(0, w_viz)

            ax2 = fig.add_subplot(231)
            ax2.imshow(uv_ref_template)
            ax2.set_title('Uniform UV')
            ax2.set_ylim(0, h_viz)
            ax2.set_xlim(0, w_viz)

            ax3 = fig.add_subplot(234)
            ax3.imshow(leaf_rgb_normed)
            ax3.set_title('Original')
            ax3.set_ylim(0, h_viz)
            ax3.set_xlim(0, w_viz)

            ax4 = fig.add_subplot(233)
            ax4.imshow(t2l_solve_final)
            ax4.set_title('Template to Leaf')
            ax4.set_ylim(0, h_viz)
            ax4.set_xlim(0, w_viz)
            ax5 = fig.add_subplot(235)
            ax5.imshow(texture_on_template)
            ax5.set_title('Texture')
            ax5.set_ylim(0, h_viz)
            ax5.set_xlim(0, w_viz)

            ax5 = fig.add_subplot(236)
            ax5.imshow(leaf_retextured)
            ax5.set_title('Retextured')
            ax5.set_ylim(0, h_viz)
            ax5.set_xlim(0, w_viz)
            plt.tight_layout()

            # plt.show()
            
            plt.savefig(os.path.join(solution_viz_folder, f'{name}_{h_viz}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        # save texture_on_template
        texture_folder = os.path.join(mask_folder, 'texture', bound_type)
        os.makedirs(texture_folder, exist_ok=True)
        texture_on_template_flip = np.flip(texture_on_template, axis=0)
        cv2.imwrite(os.path.join(texture_folder, f'{index}_{h_viz}.png'), cv2.cvtColor(texture_on_template_flip, cv2.COLOR_RGB2BGR))

        _=1

        # leaf_retextured_figure = leaf_retextured.copy()
        # leaf_retextured_figure[~valid_mask_leaf] = 255
        # leaf_retextured_figure = leaf_retextured_figure.astype(np.uint8)
        # leaf_retextured_figure = np.flip(leaf_retextured_figure, axis=0)
        # cv2.imwrite('/home/tianhang/data/folio/Folio Leaf Dataset/Folio/papaya_rotated_mask/figure/leaf.png', leaf_retextured_figure)

        # texture_on_template_figure = texture_on_template.copy()
        # texture_on_template_figure[~temp_shape_mask] = 255
        # texture_on_template_figure = texture_on_template_figure.astype(np.uint8)
        # texture_on_template_figure = np.flip(texture_on_template_figure, axis=0)
        # cv2.imwrite('/home/tianhang/data/folio/Folio Leaf Dataset/Folio/papaya_rotated_mask/figure/template.png', texture_on_template_figure)

        # uv_ref_template_figure = uv_ref_template.copy() * 255
        # uv_ref_template_figure[~temp_shape_mask] = 255
        # uv_ref_template_figure = uv_ref_template_figure.astype(np.uint8)
        # uv_ref_template_figure = np.flip(uv_ref_template_figure, axis=0)
        # iio.imwrite('/home/tianhang/data/folio/Folio Leaf Dataset/Folio/papaya_rotated_mask/figure/uv_ref_template.png', uv_ref_template_figure)

        # l2t_solve_final_figure = l2t_solve_final.copy() * 255
        # l2t_solve_final_figure[~valid_mask_leaf] = 255
        # l2t_solve_final_figure = l2t_solve_final_figure.astype(np.uint8)
        # l2t_solve_final_figure = np.flip(l2t_solve_final_figure, axis=0)
        # iio.imwrite('/home/tianhang/data/folio/Folio Leaf Dataset/Folio/papaya_rotated_mask/figure/l2t_solve_final.png', l2t_solve_final_figure)

        _=1

def parse_args():
    parser = argparse.ArgumentParser(description='Solve PDE for leaf UV parameterization.')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    parser.add_argument(
        '--bound-type',
        default='square',
        choices=['square', 'circle'],
        help='Boundary shape for PDE.',
    )
    parser.add_argument('--resolution', type=int, default=1024, help='Solution grid resolution.')
    parser.add_argument('--retrain', action='store_true', help='Recompute even if cached.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    solve_pde(
        data_dir=args.data_dir,
        bound_type=args.bound_type,
        resolution=args.resolution,
        retrain=args.retrain,
    )