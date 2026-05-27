import argparse
import os
import cv2
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import tqdm
import imageio.v2 as iio
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import binary_erosion

def find_edge_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Given a 2D binary mask, find all pixels on the edge.
    A pixel is considered an edge if it has at least one 0 neighbor.
    
    Parameters:
        mask (np.ndarray): A 2D binary mask (values are 0 or 1).
    
    Returns:
        np.ndarray: A boolean array of the same shape as `mask`, where True indicates edge pixels.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])  # 8-connected neighborhood
    neighbor_sum = convolve(mask, kernel, mode='constant', cval=0)
    
    _k = 3
    return (mask == 1) & (neighbor_sum < 9-_k) # at least _k neighbors

def sort_boundary_and_classify_points(boundary_points, target_points):
    """
    Sort boundary points by height (y-coordinate) and classify target points as
    either left or right of the resulting polyline.
    
    Args:
        boundary_points: numpy array of shape (n, 2) with boundary points
        target_points: numpy array of shape (m, 2) with target points to classify
        
    Returns:
        sorted_boundary: numpy array of sorted boundary points
        left_points: numpy array of target points on left side of polyline
        right_points: numpy array of target points on right side of polyline
    """
    
    # Convert inputs to numpy arrays for easier handling
    B = np.array(boundary_points)
    A = np.array(target_points)

    B = np.flip(B, axis=1)  # Flip x and y coordinates for consistency
    A = np.flip(A, axis=1)  # Flip x and y coordinates for consistency
    
    # Sort boundary points by height (y-coordinate)
    sorted_indices = np.argsort(B[:, 1])
    sorted_boundary = B[sorted_indices]
    
    # Create extended polyline
    polyline = sorted_boundary.copy()
    
    # Add extension for lowest point (from B[0]B[1])
    first_vec = polyline[1] - polyline[0]
    first_vec = first_vec / np.linalg.norm(first_vec)  # Normalize
    extension_first = polyline[0] - first_vec * 1000  # Extend by a large amount
    
    # Add extension for highest point (from B[-2]B[-1])
    last_vec = polyline[-1] - polyline[-2]
    last_vec = last_vec / np.linalg.norm(last_vec)  # Normalize
    extension_last = polyline[-1] + last_vec * 1000  # Extend by a large amount
    
    # Full polyline with extensions
    full_polyline = np.vstack([extension_first, polyline, extension_last])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(rotated_image)
    # ax.scatter(full_polyline[:, 0], full_polyline[:, 1], c='r', s=1)
    # plt.show(block=False)
    
    # Classify target points as left or right of the polyline
    left_points = []
    right_points = []
    
    for point in A:
        if is_point_left_of_polyline(point, full_polyline):
            left_points.append(point)
        else:
            right_points.append(point)
    
    left_points = np.array(left_points)
    right_points = np.array(right_points)

    left_points = np.flip(left_points, axis=1)  # Flip back to original coordinates
    right_points = np.flip(right_points, axis=1)  # Flip back to original coordinates
    sorted_boundary = np.flip(sorted_boundary, axis=1)  # Flip back to original coordinates

    return sorted_boundary, np.array(left_points), np.array(right_points)

def is_point_left_of_polyline(point, polyline):
    """
    Determine if a point is to the left of a polyline.
    
    Args:
        point: numpy array [x, y] - the point to check
        polyline: numpy array of shape (n, 2) - the polyline vertices
        
    Returns:
        True if the point is to the left of the polyline, False otherwise
    """
    # Count how many times a horizontal ray from the point to the right
    # intersects with the polyline segments
    num_intersections = 0
    n = len(polyline)
    
    for i in range(n - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        
        # Check if the point is within the y-range of the segment
        if ((p1[1] <= point[1] and p2[1] > point[1]) or 
            (p2[1] <= point[1] and p1[1] > point[1])):
            
            # Calculate x-coordinate of intersection
            if p1[1] != p2[1]:  # Not a horizontal line
                x_intersect = p1[0] + (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                
                # If intersection is to the right of the point, count it
                if x_intersect > point[0]:
                    num_intersections += 1
    
    # If odd number of intersections, point is inside/left of polyline
    return num_intersections % 2 == 1


import networkx as nx
from scipy.spatial import KDTree

def build_path(coords, skip=1, k=100, lowest_point=None):
    points = coords[::skip]  # Downsample for efficiency

    # Build k-NN graph
    tree = KDTree(points)
    graph = nx.Graph()

    for i, p in enumerate(points):
        neighbors = tree.query(p, k+1)[1][1:]  # Exclude self
        for j in neighbors:
            dist = np.linalg.norm(points[i] - points[j])
            graph.add_edge(i, j, weight=dist)

    start_node = np.argmin(np.linalg.norm(points - lowest_point, axis=1))
    source_index = start_node

    distances = geodesic_distances(graph, source_index)

    _coord = coords[np.array(list(distances.keys()))]
    _dist = np.array(list(distances.values()))

    # sort _coord and _dist based on _dist, from smallest to largest
    sorted_indices = np.argsort(_dist)
    _coord = _coord[sorted_indices]
    _dist = _dist[sorted_indices]


    # color = _dist / np.max(_dist)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(rotated_image)
    # ax.scatter(_coord[:, 1], _coord[:, 0], c=color, s=1)
    # plt.show(block=False)

    return _coord, _dist

def geodesic_distances(graph, source_index):
    return nx.single_source_dijkstra_path_length(graph, source_index, weight='weight')
 
def find_closest_indices(arr, k):
    arr = arr/np.max(arr)  # Normalize the array to [0, 1]
    targets = np.linspace(0, 1, k + 1)  # Generate k+1 target values
    indices = np.searchsorted(arr, targets)  # Find insertion indices
    
    closest_indices = []
    for target, idx in zip(targets, indices):
        if idx == 0:
            closest_indices.append(0)
        elif idx == len(arr):
            closest_indices.append(len(arr) - 1)
        else:
            # Choose the closest of the two neighboring points
            left, right = arr[idx - 1], arr[idx]
            closest_indices.append(idx - 1 if abs(left - target) <= abs(right - target) else idx)
    
    return closest_indices

def shrink_mask(mask: np.ndarray, k=3) -> np.ndarray:
    """
    Shrink (erode) a binary mask by 1 pixel.

    Args:
        mask (np.ndarray): Binary mask of shape [h, w], values {0,1}.

    Returns:
        np.ndarray: Eroded mask of shape [h, w].
    """
    structure = np.ones((k, k), dtype=bool)  # 3x3 kernel
    eroded = binary_erosion(mask.astype(bool), structure=structure)
    return eroded.astype(np.uint8)

def find_contour(data_dir):
    image_foler = os.path.abspath(data_dir)
    rot_folder = image_foler + '_rotated'
    os.makedirs(rot_folder, exist_ok=True)
    mask_folder = image_foler + '_rotated_mask'
    os.makedirs(mask_folder, exist_ok=True)
    curve_folder = image_foler
    keypoint_folder = os.path.join(mask_folder, 'keypoints')
    os.makedirs(keypoint_folder, exist_ok=True)
    edge_folder = os.path.join(mask_folder, 'edge')
    os.makedirs(edge_folder, exist_ok=True)

    image_list = os.listdir(image_foler)
    image_list.sort()
    image_list = [image_name for image_name in image_list if 'jpg' in image_name]
    # image_list = image_list[split::total_splits]

    edge_discrete_viz_folder = os.path.join(mask_folder, 'edge_discrete_viz')
    os.makedirs(edge_discrete_viz_folder, exist_ok=True)
    edge_viz_folder = os.path.join(mask_folder, 'edge_viz')
    os.makedirs(edge_viz_folder, exist_ok=True)

    # temp = '/home/tianhang/data/folio/Folio Leaf Dataset/Folio/soybeans_rotated_mask/temp/filtered.txt'
    # with open(temp, 'r') as f:
    #     lines = f.readlines()
    #     lines = [line.strip() for line in lines]

    for index, image_name in enumerate(tqdm.tqdm(image_list)):

        name = image_name.split('.')[0]

        # if name not in lines:
        #     continue

        if not os.path.exists(os.path.join(mask_folder, image_name)):
            print(f'mask of {image_name} not found, skip')
            continue

        # if os.path.exists(os.path.join(mask_folder, f'{name}_left_uniform.txt')):
        #     print(f'{name} already processed, skip')
        #     continue

        rot_keypoint_path = os.path.join(keypoint_folder, f'{name}.txt')
        keypoints = np.loadtxt(rot_keypoint_path)
        assert not np.isnan(keypoints).any()

        # get the highest and lowest points
        highest_point = keypoints[1]
        lowest_point = keypoints[0]

        extra_rot_keypoint_path = os.path.join(keypoint_folder, f'{name}extra.txt')
        extra_keypoints = None
        if os.path.exists(extra_rot_keypoint_path):
            extra_keypoints = np.loadtxt(extra_rot_keypoint_path)
            if len(extra_keypoints) == 0:
                extra_keypoints = None
            elif len(extra_keypoints.shape)== 1:
                extra_keypoints = extra_keypoints[None]

        # read image and mask
        image_path = os.path.join(image_foler, image_name)
        mask_path = os.path.join(mask_folder, image_name)
  
        rotated_image_path  = os.path.join(rot_folder, image_name)
        # read
        rotated_image = cv2.imread(rotated_image_path)

        rotated_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        rotated_mask[0:10] = 0
        rotated_mask[-10:] = 0
        rotated_mask[:, 0:10] = 0
        rotated_mask[:, -10:] = 0

        rotated_mask[rotated_mask <= 128] = 0
        rotated_mask[rotated_mask > 128] = 1

        for _ in range(2):
            rotated_mask = shrink_mask(rotated_mask, k=5) # SAM produce some noise near the edge

        # plt.figure()
        # plt.imshow(rotated_mask - erode_mask)
        # plt.title('Eroded mask')
        # plt.show(block=False)
        
        edge = find_edge_pixels(rotated_mask)
        edge_coordinates = np.argwhere(edge) # (height, width) = edge.shape

        # plt.figure()
        # plt.imshow(rotated_image)
        # plt.imshow(edge, alpha=0.5)
        # plt.title('Edge pixels')
        # plt.show(block=False)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(rotated_mask)
        # plt.show(block=False)

        h, w = rotated_mask.shape

        if extra_keypoints is None:
            bound = lowest_point[1]
            left_edge_coordinates = edge_coordinates[edge_coordinates[:, 1] < bound]
            right_edge_coordinates = edge_coordinates[edge_coordinates[:, 1] > bound]
        else:
            bound_more = np.concatenate([lowest_point[None], extra_keypoints, highest_point[None]], axis=0)
            _, left_edge_coordinates, right_edge_coordinates = sort_boundary_and_classify_points(bound_more, edge_coordinates )

        # plt.figure()
        # plt.imshow(rotated_image)
        # plt.scatter(right_edge_coordinates[:, 1], right_edge_coordinates[:, 0], c='r', s=1)
        # plt.scatter(left_edge_coordinates[:, 1], left_edge_coordinates[:, 0], c='b', s=1)
        # plt.scatter(bound_more[:, 1], bound_more[:, 0], c='r', s=60)
        # plt.show(block=True)
 
        # Build k-NN graph
        
        right_edge_coords_sorted, right_dist = build_path(right_edge_coordinates, skip=1, k=5, lowest_point=lowest_point)
        left_edge_coords_sorted, left_dist = build_path(left_edge_coordinates, skip=1, k=5, lowest_point=lowest_point)


        # linear interpolation between the right_edge_coords_sorted[-1] and the highest_point
        # M = 200
        # left_extra_points0 = np.linspace(lowest_point, left_edge_coords_sorted[0], M)
        # left_extra_points1 = np.linspace(left_edge_coords_sorted[-1], highest_point, M)
        # right_extra_points0 = np.linspace(lowest_point, right_edge_coords_sorted[0], M)
        # right_extra_points1 = np.linspace(right_edge_coords_sorted[-1], highest_point, M)

        # right_edge_coords_sorted = np.concatenate([right_extra_points0, right_edge_coords_sorted, right_extra_points1], axis=0)
        # left_edge_coords_sorted = np.concatenate([left_extra_points0, left_edge_coords_sorted, left_extra_points1], axis=0)


        plt.figure()
        plt.imshow(rotated_image)
        # plot lines
        plt.scatter(left_edge_coords_sorted[:, 1], left_edge_coords_sorted[:, 0], c='r', s=1)
        # plt.plot(left_edge_coordinates[:, 1], left_edge_coordinates[:, 0], c='r')
        plt.plot(right_edge_coords_sorted[:, 1], right_edge_coords_sorted[:, 0], c='b')
        plt.savefig(os.path.join(edge_viz_folder, 'edge_{}.png'.format(image_name.split('.')[0])), dpi=300)
        # plt.show(block=False)
        plt.close()

        # save the edge coordinates
        # np.savetxt(os.path.join(mask_folder, f'{name}_left_edge.txt'), left_edge_coords_sorted)
        # np.savetxt(os.path.join(mask_folder, f'{name}_right_edge.txt'), right_edge_coords_sorted)

        ################## Uniform Sampling ################
        N = 120
        left_ind = find_closest_indices(left_dist, N-1)
        right_ind = find_closest_indices(right_dist, N-1)

        left_uniform_points = left_edge_coords_sorted[left_ind]
        right_uniform_points = right_edge_coords_sorted[right_ind]

        np.savetxt(os.path.join(edge_folder, f'{name}_left_uniform.txt'), left_uniform_points)
        np.savetxt(os.path.join(edge_folder, f'{name}_right_uniform.txt'), right_uniform_points)

        plt.figure()
        # rotated_image_viz change color from BGR to RGB
        rotated_image_viz = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rotated_image_viz)
        plt.scatter(left_uniform_points[:, 1], left_uniform_points[:, 0], c='r', s=2)
        plt.scatter(right_uniform_points[:, 1], right_uniform_points[:, 0], c='b', s=2)
        # set linewidth to 0.5
        plt.plot(left_uniform_points[:, 1], left_uniform_points[:, 0], c='r', linewidth=0.5)
        plt.plot(right_uniform_points[:, 1], right_uniform_points[:, 0], c='b', linewidth=0.5)
        plt.scatter(lowest_point[1], lowest_point[0], c='g', s=10)
        plt.scatter(highest_point[1], highest_point[0], c='y', s=10)
        plt.title('Uniform points')
        plt.savefig(os.path.join(edge_discrete_viz_folder, f'uniform_{name}.png'), dpi=300)
        # plt.show(block=False)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Find left/right leaf contours from rotated masks.')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    find_contour(args.data_dir)