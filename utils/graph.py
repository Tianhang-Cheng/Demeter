from collections import defaultdict, deque
import numpy as np
from easydict import EasyDict as edict
import os
from utils.pcd import get_min_dist
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import heapq
import tqdm
from typing import Dict, List, Tuple

def find_layers_and_paths(a: edict):
    n = len(a)
    layers = defaultdict(list)
    paths = defaultdict(list)
    
    # Step 1: Find the root node
    root = -1
    for k in a.keys():
        if a[k] == -1:
            root = int(k)
            break
    
    # Step 2: Initialize BFS
    queue = deque([(root, 0)])  # (node, layer)
    paths[root] = [root]
    
    while queue:
        node, layer = queue.popleft()
        layers[layer].append(node)
        
        # Step 3: Find children, record their paths, and add them to the queue
        for k in a.keys():
            if a[k] == int(node):
                k = int(k)
                # Path to this node is the parent's path plus this node
                paths[k] = paths[int(node)] + [k]
                queue.append((k, layer + 1))
    
    return layers, paths

def calculate_accumulation_values(a:edict, values:edict):
    _, paths = find_layers_and_paths(a)
    accumulation_values = edict()

    for node, path in paths.items():
        accumulation_values[str(node)] = sum(values[str(i)] for i in path)
    
    # sort by node index
    accumulation_values = dict(sorted(accumulation_values.items()))
    return accumulation_values


def generate_full_tree(n_layer=3, n_leaf_per_child=3, n_stem_per_child=2):
    # Initialize parent and color dictionaries
    parent = {0: -1}  # root has no parent, root is id=0
    color = {0: 'black'}  # root is black

    # Initialize the list of nodes, starting with root (id=0)
    current_layer = [0]
    node_id = 1  # Next available node id

    for layer in range(n_layer):
        next_layer = []
        for node in current_layer:
            if color[node] == 'black':
                # Each black node has 2 black children and 3 green children
                for _ in range(n_stem_per_child):  # Add black children
                    parent[node_id] = node
                    color[node_id] = 'black'
                    next_layer.append(node_id)
                    node_id += 1

                for _ in range(n_leaf_per_child):  # Add green children
                    parent[node_id] = node
                    color[node_id] = 'green'
                    next_layer.append(node_id)
                    node_id += 1

        # Move to the next layer of nodes
        current_layer = next_layer
    
    color_map = lambda x: 1 if x == 'black' else 0

    parents = edict({str(k): v for k, v in parent.items()})
    classes = edict({str(k): color_map(v) for k, v in color.items()})

    return parents, classes


# 1. Generate a KNN graph for the point cloud
def compute_knn_graph(points, k=5):
    kdtree = KDTree(points)
    distances, indices = kdtree.query(points, k=k+1)  # k+1 because the first result is the point itself

    # Exclude self-distances and create a sparse adjacency matrix
    rows, cols = np.repeat(np.arange(len(points)), k), indices[:, 1:].flatten()
    weights = distances[:, 1:].flatten()
    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(len(points), len(points)))
    
    return adjacency_matrix

# 2. Compute the maximum geodesic distance and return the endpoints
def max_geodesic_distance(points, k=30, max_points=600, return_idx=False):
    
    if len(points) > max_points:
        points = points[np.random.choice(points.shape[0], max_points, replace=False), :]
    k = min(k, len(points) - 1)

    adjacency_matrix = compute_knn_graph(points, k=k)
    dist_matrix, predecessors = dijkstra(adjacency_matrix, directed=False, return_predecessors=True)
    
    # Find the maximum distance and the corresponding points
    max_distance = np.max(dist_matrix[np.isfinite(dist_matrix)])
    max_indices = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)
    
    point1, point2 = points[max_indices[0]], points[max_indices[1]]
    
    if return_idx:
        return max_distance, point1, point2, max_indices

    return max_distance, point1, point2


from sklearn.neighbors import NearestNeighbors
def compute_knn_graph2(points, k=30):
    """Compute k-nearest neighbors graph as a sparse adjacency matrix."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Create sparse adjacency matrix
    n_points = points.shape[0]
    adjacency_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j, dist in zip(indices[i], distances[i]):
            adjacency_matrix[i, j] = dist
    
    return adjacency_matrix

def get_farthest_points(points, selected_point, k=5, nn_k=30, max_points=600):
    """
    Find the k farthest points from a selected point using geodesic distance.
    
    Args:
        points: numpy array of shape (n, d) containing point cloud coordinates
        selected_point: the selected point
        k: number of farthest points to return
        nn_k: number of nearest neighbors for graph construction
        max_points: maximum number of points to consider for computation
        
    Returns:
        indices: indices of the k farthest points
        distances: geodesic distances to the k farthest points
    """
    # Subsample if needed
    if len(points) > max_points:

        dist_to_selected = np.linalg.norm(points - selected_point, axis=1)
        selected_point_idx = np.argmin(dist_to_selected)

        # Always include the selected point in the subsample
        remaining_indices = np.array([i for i in range(len(points)) if i != selected_point_idx])
        sampled_indices = np.random.choice(remaining_indices, max_points - 1, replace=False)
        all_indices = np.append(sampled_indices, selected_point_idx)
        
        points = points[all_indices]
        # Map selected_point_idx to its new position in the subsampled array
        selected_point_idx = max_points - 1
    
    # Compute the KNN graph
    adjacency_matrix = compute_knn_graph2(points, k=nn_k)
    
    # Compute geodesic distances from the selected point to all other points
    distances = dijkstra(adjacency_matrix, directed=False, indices=selected_point_idx)
    
    # Get indices of k farthest points (excluding the selected point itself)
    farthest_indices = np.argsort(distances)[-k-1:-1][::-1]  # -1 to exclude self, reversed to get descending order
    # farthest_distances = distances[farthest_indices]
    # return farthest_indices, farthest_distances

    # Get indices of k farthest points (excluding the selected point itself)
    farthest_points = points[farthest_indices]
    return farthest_points

def load_parent(path: str):
    # load parent annotation
    parents = edict()
    UNIQUE_ROOT = 0
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        for line_id, line in enumerate(lines):
            # if empty line, skip
            if not line.strip():
                continue
            if '->' not in line:
                raise ValueError('The format of the parent file is wrong, please check line %d' % line_id)
            child, parent = line.strip().split('->')
            # assert classes.get(str(int(parent))) == STEM_CLASS or int(parent) == -1, 'The parent should be stem or no parent'
            if int(parent) == -1:
                assert UNIQUE_ROOT == 0, 'There should be only one root'
                UNIQUE_ROOT += 1
            parents[str(int(child))] = int(parent)
    return parents

def load_parent(path: str):
    # load parent annotation
    parents = edict()
    UNIQUE_ROOT = 0
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            child, parent = line.strip().split('->')
            # assert classes.get(str(int(parent))) == STEM_CLASS or int(parent) == -1, 'The parent should be stem or no parent'
            if int(parent) == -1:
                assert UNIQUE_ROOT == 0, 'There should be only one root'
                UNIQUE_ROOT += 1
            parents[str(int(child))] = int(parent)
    else:
        raise ValueError('Parent file not found: %s' % path)
    return parents

def save_parent(parents, path):
    with open(path, 'w') as f:
        for k, v in parents.items():
            f.write(f'{k} -> {v}\n')

def load_class(path):
    classes = edict()
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            name, _class = line.strip().split(' ')
            classes[str(name)] = int(_class)
    else:
        raise ValueError('Class file not found: %s' % path)
    return classes

def save_class(classes, path):
    with open(path, 'w') as f:
        for k, v in classes.items():
            f.write(f'{k} {v}\n')

class MinimalSpanningGraph:
    def __init__(self, num_nodes: int):
        """
        Initialize the graph with a given number of nodes.
        
        :param num_nodes: Total number of nodes in the graph
        """
        self.num_nodes = num_nodes
        self.graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    
    def add_edge(self, start: int, end: int, weight: float):
        """
        Add a directed edge to the graph.
        
        :param start: Starting node
        :param end: Ending node
        :param weight: Weight of the edge
        """
        self.graph[start].append((end, weight))
    
    def minimal_spanning_graph(self, root: int) -> List[Tuple[int, int, float]]:
        """
        Generate a minimal spanning graph using a modified Prim's algorithm.
        
        :param root: The root node to start the spanning graph from
        :return: List of edges in the minimal spanning graph (start, end, weight)
        """
        # Initialize distances and previous nodes
        distances = [float('inf')] * self.num_nodes
        distances[root] = 0
        previous = [None] * self.num_nodes
        
        # Priority queue to store (distance, node)
        pq = [(0, root)]
        
        # Set to keep track of visited nodes
        visited = set()
        
        # List to store the minimal spanning graph edges
        mst_edges = []
        
        # Keep track of unvisited nodes to ensure complete coverage
        unvisited = set(range(self.num_nodes))
        
        while pq or unvisited:
            # If priority queue is empty, find the next unvisited node with minimal connection
            if not pq:
                # Find the unvisited node with minimal connection to visited nodes
                min_dist = float('inf')
                next_node = None
                for node in unvisited:
                    # Check if we can find a minimal connection
                    for visited_node in visited:
                        # Check edges from visited nodes to this unvisited node
                        for potential_end, weight in self.graph[visited_node]:
                            if potential_end == node and weight < min_dist:
                                min_dist = weight
                                next_node = (visited_node, node, weight)
                
                if next_node:
                    start, node, weight = next_node
                    mst_edges.append((start, node, weight))
                    visited.add(node)
                    unvisited.remove(node)
                    continue
                else:
                    # If no connection found, just pick an unvisited node
                    node = unvisited.pop()
                    visited.add(node)
                    continue
            
            # Normal Prim's algorithm processing
            current_distance, current_node = heapq.heappop(pq)
            
            # Skip if already visited
            if current_node in visited:
                continue
            
            visited.add(current_node)
            unvisited.discard(current_node)
            
            # If this is not the root, add the edge to MST
            if current_node != root and previous[current_node] is not None:
                mst_edges.append((previous[current_node], current_node, 
                                  current_distance))
            
            # Explore neighbors
            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    # Relaxation step
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_distance, neighbor))
        
        return mst_edges


def get_guessed_parent(points, parent_id: int, classes: List[int]):
    """
    points: List, n_node point clouds
    classes: [n_node] List of classes for each point cloud
    """
    n_node = len(points)

    graph = MinimalSpanningGraph(n_node)

    kdtrees = []
    for i in range(n_node):
        kdtrees.append(KDTree(points[i]))
 
    for i in tqdm.tqdm(range(n_node), desc='Building Graph'):
        for j in range(n_node):
            if i >= j:
                continue
            dist = get_min_dist(points[i], kdtree=kdtrees[j])[0]
            # find the index
            min_index = np.argsort(dist)
            min_dist = dist[min_index]
            node_dist = np.percentile(min_dist, 1) # 1% percentile, approximating the minimum distance
            class_j = classes[j]
            class_i = classes[i]
            if class_j == 1:
                graph.add_edge(j, i, node_dist)
            if class_i == 1:
                graph.add_edge(i, j, node_dist)
    
    mst = graph.minimal_spanning_graph(root=parent_id)
    
    parents_pred = {}
    print("Minimal Spanning Graph Edges:")
    for start, end, weight in mst:
        print(f"{start} -> {end} (Weight: {weight})")
        parents_pred[str(end)] = start
    parents_pred[str(parent_id)] = -1

    parents_pred = dict(sorted(parents_pred.items(), key=lambda x: int(x[0])))
    
    # classes_filtered = {k: v for k, v in classes.items() if str(k) in parents_pred.keys()}
    # draw_tree_with_colors(parents_pred, c=classes)

    return parents_pred