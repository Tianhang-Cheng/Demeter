import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import trimesh
import os
import scipy
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

def visualize_frenet_frame(curve, axis, scale=1):
    """
    Visualize the curve and its Frenet-Serret frame
    
    Args:
    curve (torch.Tensor): Original curve
    tangent (torch.Tensor): Tangent vectors
    normal (torch.Tensor): Normal vectors
    binormal (torch.Tensor): Binormal vectors
    """

    tangent, normal, binormal = axis[:, 0], axis[:, 1], axis[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the curve
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'k-', linewidth=2, label='Curve')
    
    # Subsample for clearer visualization
    sample_stride = max(1, len(curve) // 10)
    
    # Plot coordinate frames
    for i in range(0, len(curve), sample_stride):
        # Origin point
        origin = curve[i]
        
        # Plot tangent (red)
        ax.quiver(origin[0], origin[1], origin[2], 
                  tangent[i, 0], tangent[i, 1], tangent[i, 2], 
                  color='r', length=scale, arrow_length_ratio=0.1)
        
        # Plot normal (green)
        ax.quiver(origin[0], origin[1], origin[2], 
                  normal[i, 0], normal[i, 1], normal[i, 2], 
                  color='g', length=scale, arrow_length_ratio=0.1)
        
        # Plot binormal (blue)
        ax.quiver(origin[0], origin[1], origin[2], 
                  binormal[i, 0], binormal[i, 1], binormal[i, 2], 
                  color='b', length=scale, arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Curve with Frenet-Serret Frame')
    # ax.set_xlim(-6,6)
    # ax.set_ylim(-6,6)
    # ax.set_zlim(0,12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def normalize_rotation_matrix(R):
    """Projects a 3x3 matrix onto the closest valid rotation matrix in SO(3)."""
    U, _, Vt = np.linalg.svd(R)  # Perform SVD
    R_normalized = U @ Vt  # Ensure orthogonality
    return R_normalized

def find_nearest_pair(M: np.ndarray, N: np.ndarray):
    tree = cKDTree(N)  # Construct a KDTree for N
    distances, indices = tree.query(M)  # Find nearest neighbor in N for each point in M
    min_idx = np.argmin(distances)  # Get the index of the minimum distance pair
    
    return M[min_idx], N[indices[min_idx]] 

def extract_curve_local_info(curve):
    # control points
    cp_cano = curve.evaluate(num_points=2, w=1)[0].detach().cpu().numpy()

    # calculate the local axis based on the points coordinates
    local_axis = calculate_local_axes(cp_cano, mode='curve', p0=cp_cano[0], p1=cp_cano[-1])[0]

    # project the control points to the local coordinate system
    cp_w_local = cp_cano @ local_axis.T

    # visualize the local axis
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # axis.translate([0, 0, 0])
    # axis_local = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # axis_local.rotate(local_axis.T)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cp_cano.reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd, axis, axis_local])
    # x = 1

    res = {'cp_w_local': cp_w_local,
           'cp_cano': cp_cano,
           'local_axis': local_axis}
    return res

def merge_pcd(files, output_path=None, delete_files=False):
    pcds = []
    for file in files:
        pcd = trimesh.load(file)
        pcds.append(pcd)

    if delete_files:
        for file_to_merge in files:
            # print(f'{file_to_merge} is selected')
            if file_to_merge == output_path:
                continue
            print(f"Deleting {file_to_merge}")
            os.remove(file_to_merge)
            if os.path.exists(file_to_merge.replace('raw', 'processed')):
                os.remove(file_to_merge.replace('raw', 'processed'))

    new_vertices = np.vstack([pcd.vertices for pcd in pcds])
    new_colors = np.vstack([pcd.colors if len(pcd.colors) > 0 else np.repeat(np.array([[0,0,0,255]]), repeats=len(pcd.vertices), axis=0) for pcd in pcds])
    # Merge two point clouds and keep color information
    if output_path is None:
        output_path = files[0].split('.')[-2] + '_merged.ply'
    merged_pcd = trimesh.PointCloud(vertices=new_vertices, colors=new_colors)
    merged_pcd.export(output_path)
    print(f"Saved merged point cloud to {output_path}")

def find_nearest(source, target, sample_num=1000, viz=False, return_dist=False):
    """
    find the nearest point in source to target
    """
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source).float().cuda()
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target).float().cuda()

    _n = source.shape[0]
    if sample_num == -1:
        pcd_subset = source
    else:
        pcd_subset = source[np.random.choice(_n, min(sample_num, _n), replace=False)]
    dist = torch.cdist(pcd_subset, target)
    dist = dist.min(dim=1).values

    if return_dist:
        return dist.min().item()

    # find the nearest 5 point to parent
    nearest_indices = torch.argsort(dist, dim=0)[:5]
    nearest_points = pcd_subset[nearest_indices.flatten()].mean(dim=0)

    if viz:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pcd_subset_viz = pcd_subset.detach().cpu().numpy()[::10]
        parent_pcd_points_viz = target.detach().cpu().numpy()[::10]
        nearest_points_viz = nearest_points.detach().cpu().numpy()
        ax.scatter(pcd_subset_viz[:, 0], pcd_subset_viz[:, 1], pcd_subset_viz[:, 2], c='b', label='child pcd')
        ax.scatter(parent_pcd_points_viz[:, 0], parent_pcd_points_viz[:, 1], parent_pcd_points_viz[:, 2], c='r', label='parent pcd')
        ax.scatter(nearest_points_viz[0], nearest_points_viz[1], nearest_points_viz[2], c='g', label='nearest point', s=100)
        plt.legend()
        plt.show(block=True)

    return nearest_points


def find_farthest_points_batch_torch(points, p0=None, p1=None):
    
    if p0 is not None and p1 is not None:
        return p0, p1
    
    # randomly select 1000 points for efficiency

    num_points = points.shape[0]

    if num_points > 1000:
        points = points[torch.randperm(num_points)[:1000]]
        num_points = len(points)

    max_dist = -1
    farthest_points = None

    batch_size = min(10, num_points // 3)
    
    if p0 is None:
        for i in range(0, num_points, batch_size):
            for j in range(i + batch_size, num_points, batch_size):
                # 获取当前批次的点
                batch1 = points[i:i+batch_size]
                batch2 = points
                
                # 计算批次之间的距离
                diff = batch1[:, None, :] - batch2[None, :, :]
                dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
                
                # 找到当前批次的最大距离及其对应的索引
                max_dist_in_batch = torch.max(dist)
                if max_dist_in_batch > max_dist:
                    max_dist = max_dist_in_batch
                    idx1, idx2 = torch.nonzero(dist == max_dist_in_batch, as_tuple=True)
                    farthest_points = (batch1[idx1[0]], batch2[idx2[0]])
    else:
        for i in range(0, num_points, batch_size):
            batch1 = points[i:i+batch_size]
            diff = batch1 - p0
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
            max_dist_in_batch = torch.max(dist)
            if max_dist_in_batch > max_dist:
                max_dist = max_dist_in_batch
                idx = torch.argmax(dist)
                farthest_points = (p0, batch1[idx])
    
    return farthest_points

from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra

def compute_maximum_manifold_distance(point_cloud, k=10, batch_size=1000):
    # Step 1: 构建KNN图
    tree = KDTree(point_cloud.cpu().numpy())  # KDTree 仅支持CPU计算
    distances, indices = tree.query(point_cloud.cpu().numpy(), k=k)
    
    # Step 2: 构建稀疏矩阵表示的邻接图
    n = point_cloud.size(0)
    graph = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(1, k):  # 跳过j=0，因为它是点自己
            graph[i, indices[i, j]] = distances[i, j]
    
    # 将邻接矩阵对称化
    graph = np.minimum(graph, graph.T)
    
    # Step 3: 使用Dijkstra算法计算所有点对的最短路径
    distances_matrix = dijkstra(graph, directed=False)
    
    # Step 4: 批量查找最大距离
    max_distance = 0
    max_pair = (0, 0)
    
    for i in range(0, n, batch_size):
        batch_distances = torch.tensor(distances_matrix[i:i+batch_size], device=point_cloud.device)
        batch_max_dist, indices = torch.max(batch_distances, dim=1)
        batch_max_dist, idx = torch.max(batch_max_dist, dim=0)
        
        if batch_max_dist.item() > max_distance:
            max_distance = batch_max_dist.item()
            max_pair = (i + idx.item(), indices[idx].item())
    
    return max_pair, max_distance

def grid_to_mesh(grid, color='gray'):
    
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()

    assert grid.ndim == 3
    m, n, _ = grid.shape
    
    # 将 grid array 展平为一个顶点列表
    vertices = grid.reshape(-1, 3)
    
    # 生成三角形面
    faces = []
    for i in range(m - 1):
        for j in range(n - 1):
            # 当前单元格的四个顶点索引
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            
            # 两个三角形
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    
    faces = np.array(faces)
    
    # 使用 Open3D 构建三角网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # set color
    if isinstance(color, str):
        if color == 'red':
            mesh.paint_uniform_color([0.9, 0.5, 0.5])
        elif color == 'green':
            # mesh.paint_uniform_color([0.5, 0.9, 0.5])
            mesh.paint_uniform_color([0.1, 1, 0.1])
        elif color == 'blue':
            mesh.paint_uniform_color([0.5, 0.5, 0.9])
        elif color == 'gray':
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
        elif color == 'orange':
            mesh.paint_uniform_color(np.array([255, 165, 0]).astype(np.float32) / 255)
    else:
        mesh.paint_uniform_color(color)
    
    # 计算法线
    mesh.compute_vertex_normals()
    
    return mesh

def gram_schmidt_batch(matrices):
    # Assume matrices have shape [b, 3, 3] where each matrix's columns are the input vectors.
    
    # Normalize the first vector (first column of each matrix)
    u = matrices[..., 0]
    u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-9)
    
    # Make the second vector orthogonal to the first
    v = matrices[..., 1] - torch.sum(matrices[..., 1] * u, dim=1, keepdim=True) * u
    v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-9)
    
    # Compute the third vector as the cross product of u and v
    w = torch.linalg.cross(u, v, dim=1)
    w = w / (torch.norm(w, dim=1, keepdim=True) + 1e-9)
    
    # Stack the orthogonal vectors as columns to form the batch of orthogonal matrices
    orthogonal_matrices = torch.stack([u, v, w], dim=-1)
    
    return orthogonal_matrices

def gram_schmidt(matrices):
    # Normalize the first vector in the batch
    u = matrices[:, 0] / (np.linalg.norm(matrices[:, 0], axis=1, keepdims=True) + 1e-9)
    
    # Make the second vector orthogonal to the first
    v = matrices[:, 1] - np.sum(matrices[:, 1] * u, axis=1, keepdims=True) * u
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    
    # Compute the third vector as the cross product of u and v
    w = np.cross(u, v)
    w /= (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
    
    # Stack the orthogonal vectors to form the batch of matrices
    orthogonal_matrices = np.stack([u, v, w], axis=1)
    return orthogonal_matrices


def project_point_to_plane(plane_coeff, point):
    """
    point: [n, 3]
    """
    # Plane coefficients (a, b, c, d)
    a, b, c, d = plane_coeff
    # Point coordinates (x_m, y_m, z_m)
    x_m, y_m, z_m = point[..., 0], point[..., 1], point[..., 2]
    
    # Normal vector of the plane
    normal = np.array([a, b, c])
    
    # Distance from the point to the plane
    numerator = a * x_m + b * y_m + c * z_m + d
    denominator = np.linalg.norm(normal)
    distance = numerator / denominator
    
    # Projection of the point onto the plane
    projection = np.stack([x_m, y_m, z_m], axis=-1) - distance[:, None] * (normal / denominator)
    
    return projection, distance

from scipy.optimize import minimize
def find_optimal_plane(P, pa, pb, eps=1e-10):
    """
    Find a plane that contains points pa and pb and minimizes L2 distances to points in P.
    
    Args:
        P: numpy array of shape (n, 3) containing n 3D points
        pa: numpy array of shape (3,) representing first point that must lie on plane
        pb: numpy array of shape (3,) representing second point that must lie on plane
        eps: small number for numerical stability checks
    
    Returns:
        params: dictionary containing plane parameters
    """
    # Convert inputs to numpy arrays and ensure they're float64
    P = np.array(P, dtype=np.float64)
    pa = np.array(pa, dtype=np.float64)
    pb = np.array(pb, dtype=np.float64)
    
    # Check if pa and pb are the same point
    if np.allclose(pa, pb, rtol=eps, atol=eps):
        raise ValueError("Points pa and pb must be different")
    
    # 1. Get the direction vector of line papb
    papb = pb - pa
    papb_norm = np.linalg.norm(papb)
    if papb_norm < eps:
        raise ValueError("Points pa and pb are too close")
    
    # Normalize papb vector
    papb = papb / papb_norm
    
    # 2. Create an orthonormal basis with papb as one vector
    # Find a vector not parallel to papb
    if abs(papb[0]) < abs(papb[1]):
        v = np.array([1., 0., 0.])
    else:
        v = np.array([0., 1., 0.])
    
    # Use Gram-Schmidt to get first perpendicular vector
    v1 = v - np.dot(v, papb) * papb
    v1 = v1 / np.linalg.norm(v1)
    
    # Get second perpendicular vector using cross product
    v2 = np.cross(papb, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # 3. The plane normal must be in the space spanned by v1 and v2
    # Let's say normal = cos(θ)v1 + sin(θ)v2
    # This ensures the normal is unit length and perpendicular to papb
    
    # 4. For each point p in P, its distance to the plane is:
    # dist = |n·(p - pa)| = |cos(θ)v1·(p - pa) + sin(θ)v2·(p - pa)|
    
    # Set up the optimization problem
    P_centered = P - pa
    
    # Compute the projections once
    proj_v1 = np.dot(P_centered, v1)
    proj_v2 = np.dot(P_centered, v2)
    
    # Find θ that minimizes the sum of squared distances
    # This can be done analytically by finding the eigenvector 
    # corresponding to the smallest eigenvalue of the 2x2 matrix:
    # [sum(proj_v1²)    sum(proj_v1·proj_v2)]
    # [sum(proj_v1·proj_v2)    sum(proj_v2²)   ]
    
    M = np.array([[np.sum(proj_v1**2), np.sum(proj_v1 * proj_v2)],
                  [np.sum(proj_v1 * proj_v2), np.sum(proj_v2**2)]])
    
    # Find eigenvector corresponding to smallest eigenvalue
    eigenvals, eigenvecs = np.linalg.eigh(M)
    x = eigenvecs[:, 0]  # Column corresponding to smallest eigenvalue
    
    # Construct the normal vector
    normal = x[0] * v1 + x[1] * v2
    normal = normal / np.linalg.norm(normal)
    
    # Compute d in the plane equation ax + by + cz + d = 0
    d = -np.dot(normal, pa)
    
    # Create the plane equation string
    a, b, c = normal
    equation_str = f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"
    
    # Verify the solution
    error_papb = abs(np.dot(normal, pb - pa))
    if error_papb > eps:
        print(f"Warning: Solution may be numerically unstable. Error: {error_papb}")
    
    # Compute average L2 distance to verify the solution
    distances = np.abs(np.dot(P - pa, normal))
    avg_distance = np.mean(distances)

    equation_coeff = [a, b, c, d]
    
    return {
        'normal': normal,  # [a, b, c]
        'd': d,           # d in ax + by + cz + d = 0
        'equation_str': equation_str,
        'error': error_papb,
        'avg_distance': avg_distance,
        'equation_coeff': equation_coeff
    }

def compute_frenet_serret_frame(curve):
    """
    Compute Frenet-Serret frame with robust handling for degenerate cases
    
    Args:
    curve (torch.Tensor): Input curve of shape [n, 3]
    
    Returns:
    tuple: (tangent, normal, binormal) each of shape [n, 3]
    """
    # Ensure input is a torch tensor
    if not isinstance(curve, torch.Tensor):
        curve = torch.tensor(curve, dtype=torch.float32)
    
    # Compute first derivative (tangent)
    derivatives = torch.zeros_like(curve)
    derivatives[:-1] = curve[1:] - curve[:-1]
    derivatives[-1] = derivatives[-2]  # Last point extrapolation
    
    # Normalize tangent vector
    tangent = derivatives / (torch.norm(derivatives, dim=1, keepdim=True) + 1e-8)
    
    # Robust method for selecting an initial reference vector
    def get_reference_vector(tan):
        # Try different reference vectors to avoid parallel vectors
        candidates = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0])
        ]
        
        for ref in candidates:
            ref = ref.to(tan.device).float()
            # Check if reference is not parallel to tangent
            if torch.abs(torch.dot(tan, ref)) < 0.9:
                return ref
        
        # Fallback: use cross product with a random vector
        return torch.cross(tan, torch.randn_like(tan))
    
    # Compute normal and binormal
    normal = torch.zeros_like(curve)
    binormal = torch.zeros_like(curve)
    
    for i in range(len(curve)):
        # Get a reference vector not parallel to tangent
        ref = get_reference_vector(tangent[i])
        
        # Compute normal using Gram-Schmidt process
        n = ref - torch.dot(ref, tangent[i]) * tangent[i]
        n = n / (torch.norm(n) + 1e-8)
        normal[i] = n
        
        # Compute binormal as cross product
        b = torch.cross(tangent[i], n)
        b = b / (torch.norm(b) + 1e-8)
        binormal[i] = b
    
    # return tangent, normal, binormal
    
    axis = torch.stack([tangent, normal, binormal], dim=1) # [n, 3, 3]
    return axis


def compute_frenet_serret_frame(r, epsilon=1e-6, repeats=1):
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, dtype=torch.float32)

    n = r.shape[0]
    device = r.device

    # Compute tangent vectors T
    T = torch.zeros_like(r)
    if n > 2:
        T[1:-1] = (r[2:] - r[:-2]) / 2.0  # Central difference for internal points
    # Forward difference for the first point
    if n > 1:
        T[0] = r[1] - r[0]
    # Backward difference for the last point
    T[-1] = r[-1] - r[-2]

    # Normalize tangents
    T_norm = torch.norm(T, dim=1, keepdim=True)
    T = T / (T_norm + epsilon)

    # Compute derivative of T to get normal direction vectors
    dT = torch.zeros_like(T)
    if n > 2:
        dT[1:-1] = (T[2:] - T[:-2]) / 2.0  # Central difference for internal points
    if n > 1:
        # Forward difference for the first point
        dT[0] = T[1] - T[0]
        # Backward difference for the last point
        dT[-1] = T[-1] - T[-2]

    dT_norm = torch.norm(dT, dim=1, keepdim=True)
    mask = (dT_norm.squeeze(1) < epsilon)

    # Handle cases where dT is near zero by finding a perpendicular vector to T
    if mask.any():
        default_dir = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(T)
        proj = (default_dir * T).sum(dim=1, keepdim=True) * T
        perp = default_dir - proj
        perp_norm = torch.norm(perp, dim=1, keepdim=True)
        mask_perp_zero = (perp_norm.squeeze(1) < epsilon)
        if mask_perp_zero.any():
            default_dir2 = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(T)
            proj2 = (default_dir2 * T).sum(dim=1, keepdim=True) * T
            perp2 = default_dir2 - proj2
            perp2_norm = torch.norm(perp2, dim=1, keepdim=True)
            perp = torch.where(mask_perp_zero.unsqueeze(1), perp2, perp)
            perp_norm = torch.where(mask_perp_zero.unsqueeze(1), perp2_norm, perp_norm)
        perp_normalized = perp / (perp_norm + epsilon)
        # Replace dT where the norm is too small
        dT_normalized = torch.where(mask.unsqueeze(1), perp_normalized, dT / (dT_norm + epsilon))
    else:
        dT_normalized = dT / (dT_norm + epsilon)

    N = dT_normalized

    # Ensure N is orthogonal to T
    N = N - (T * (N * T).sum(dim=1, keepdim=True))
    N_norm = torch.norm(N, dim=1, keepdim=True)
    N = N / (N_norm + epsilon)

    # Compute binormal vectors B
    B = torch.cross(T, N, dim=1)
    B_norm = torch.norm(B, dim=1, keepdim=True)
    B = B / (B_norm + epsilon)

    # return T, N, B
    axis = torch.stack([T, N, B], dim=1) # [n, 3, 3]

    if not (torch.abs(torch.abs(torch.linalg.det(axis)) - 1) < 1e-3).all():
        axis = gram_schmidt_batch(axis) # FIXME
    
    if repeats > 1:
        axis = torch.repeat_interleave(axis, repeats, dim=0) # (n * num_points, 3, 3)
    return axis

def calculate_local_axes_torch(p, mode, repeats=1):
    """
    Input:
        cp: (n, 3) 
        p: (m, 3)
    Calculate the local axes from cp, then repeat it to the length of p
    """
    assert mode == 'default', 'Only default mode is supported for now'


    """
    Input:
        p: (n, 3)
    Output:
        local_axes: (n, 3, 3), [normals, tangents, binormals]
    """
    assert isinstance(p, torch.Tensor)
    p = p.reshape(-1, 3)

    dp = p[1:] - p[:-1]
    dp = dp / (torch.linalg.norm(dp, dim=1, keepdims=True) + 1e-9)
    
    local_axes = compute_rotation_matrices(dp[:-1], dp[1:])# (n-1, 3, 3)
    local_axes = torch.permute(local_axes, (0, 2, 1)) # (n-1, 3, 3)
    R0 = generate_right_hand_system(dp[0])# define the coordinate system, can be any right-hand system that rotates around dp[0] (y-axis)
    local_axes = torch.cat([R0[None], local_axes], dim=0) # (n, 3, 3)
    local_axes = compute_cumulative_matrices_torch(local_axes)
    local_axes = torch.cat([local_axes, local_axes[-1:]], dim=0)

    assert repeats is not None
    local_axes = torch.repeat_interleave(local_axes, repeats, dim=0) # (n * num_points, 3, 3)

    return local_axes

def right_handed_rotation_matrix(a, b):
    # Normalize input vectors
    a = a / torch.norm(a)
    b = b / torch.norm(b)
    
    # Compute the rotation axis (cross product of a and b)
    axis = torch.cross(a, b)
    
    # Check if we need to flip the axis to maintain right-handedness
    if torch.dot(axis, torch.cross(a, b)) < 0:
        axis = -axis
    
    # Normalize the axis
    axis = axis / torch.norm(axis)
    
    # Compute the angle between vectors
    angle = torch.acos(torch.clamp(torch.dot(a, b), -1.0, 1.0))
    
    # Compute the skew-symmetric matrix
    K = torch.zeros(3, 3)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1], axis[0]
    
    # Compute the rotation matrix using Rodrigues' formula
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)

    print('det(R):', torch.linalg.det(R).item())
    
    return R

def compute_rotation_matrices(a, b):
    n = a.shape[0]
    
    # Normalize input vectors
    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    
    # Compute rotation axis (cross product of a and b)
    axis = torch.cross(a_norm, b_norm, dim=1)
    axis_norm = torch.nn.functional.normalize(axis, dim=1)
    
    # Compute rotation angle
    cos_theta = torch.sum(a_norm * b_norm, dim=1)
    sin_theta = torch.norm(axis, dim=1)
    
    # Compute rotation matrices using Rodrigues' rotation formula
    K = torch.zeros(n, 3, 3, device=a.device)
    K[:, 0, 1] = -axis_norm[:, 2]
    K[:, 0, 2] = axis_norm[:, 1]
    K[:, 1, 0] = axis_norm[:, 2]
    K[:, 1, 2] = -axis_norm[:, 0]
    K[:, 2, 0] = -axis_norm[:, 1]
    K[:, 2, 1] = axis_norm[:, 0]
    
    I = torch.eye(3, device=a.device).unsqueeze(0).repeat(n, 1, 1)
    cos_theta = cos_theta.view(n, 1, 1)
    sin_theta = sin_theta.view(n, 1, 1)
    
    R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    
    # Ensure right-handedness
    det = torch.det(R)
    R[det < 0] *= -1
    
    return R

def compute_cumulative_matrices(a):
    n = a.shape[0]
    b = np.zeros_like(a)
    
    b[0] = a[0]
    
    for i in range(1, n):
        b[i] = b[i-1] @ a[i]
    return b

def compute_cumulative_matrices_torch(a):
    """
    a: [n, 3, 3]
    """
    n = a.shape[0]
    b = []
    
    b.append(a[0])
    
    for i in range(1, n):
        b.append(torch.matmul(b[i-1], a[i]))
    
    b = torch.stack(b, dim=0)

    return b
 

def generate_right_hand_system(b):

    # rotation matrix around y
    # theta = np.random.rand() * 2 * np.pi
    theta = np.pi / 4 # FIXME: hard-coded
    Ry = torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32).cuda()
    
    a = torch.tensor([0, 1, 0], dtype=torch.float32).cuda()

    R = compute_rotation_matrices(a[None], b[None])[0].T # (3, 3)

    return Ry @ R

def calculate_local_axes(p, mode, repeats=1, p0=None, p1=None):
    """
    Input:
        p: (n, 3)
        p0: (3,)
        p1: (3,)
        p_ref: (m, 3)
    Output:
        local_axes: (n, 3, 3), [normals, tangents, binormals]
    """
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    p = p.reshape(-1, 3)

    if p0 is None:
        p0 = p[0]
    if p1 is None:
        p1 = p[-1]
    if isinstance(p0, torch.Tensor):
        p0 = p0.detach().cpu().numpy()
    if isinstance(p1, torch.Tensor):
        p1 = p1.detach().cpu().numpy()
    
    if mode == 'debug':

        near_tangent = p1 - p0
        near_tangent = near_tangent / (np.linalg.norm(near_tangent) + 1e-9) # X
        near_tangent = near_tangent[None]

        near_tangent2 = p1 - p[p.shape[0]//2]
        near_tangent2 = near_tangent2 / (np.linalg.norm(near_tangent2) + 1e-9) # Y'
        near_tangent2 = near_tangent2[None]

        normal = np.cross(near_tangent, near_tangent2)[0]
        normal = normal / (np.linalg.norm(normal) + 1e-9) # Z

        near_tangent2 = np.cross(normal, near_tangent) # Y
        near_tangent2 = near_tangent2 / (np.linalg.norm(near_tangent2) + 1e-9)

        local_axes = np.stack([near_tangent, near_tangent2, normal[None]], axis=-2) # (1, 3, 3)
        if np.min(np.linalg.det(local_axes)) < 0:
            raise ValueError
            local_axes[:, 2] = -local_axes[:, 2]
        local_axes = np.repeat(local_axes, p.shape[0], axis=0) # (n, 3, 3)
    
    elif mode == 'default':

        dp = p[1:] - p[:-1]
        dp = dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-9)
        
        dp = torch.from_numpy(dp).float().cuda()
        local_axes = compute_rotation_matrices(dp[:-1], dp[1:]).detach().cpu().numpy() # (n-1, 3, 3)
        local_axes = np.transpose(local_axes, (0, 2, 1)) # (n-1, 3, 3)
        R0 = generate_right_hand_system(dp[0]).detach().cpu().numpy() # define the coordinate system, can be any right-hand system that rotates around dp[0] (y-axis)
        local_axes = np.concatenate([R0[None], local_axes], axis=0) # (n, 3, 3)
        local_axes = compute_cumulative_matrices(local_axes)
        local_axes = np.concatenate([local_axes, local_axes[-1:]], axis=0)

        assert repeats is not None
        local_axes = np.repeat(local_axes, repeats, axis=0) # (n * num_points, 3, 3)

        return local_axes

    elif mode == 'default2':

        # acceleration based

        d = p[1] - p[0]
        _p = np.concatenate([(p[0] - 0.01 * d)[None],p])
        tangents = _p[1:] - _p[:-1]
        tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9) # Green, X

        delta_tangents = (tangents[1:] - tangents[:-1]) # for numerical stability, Y'
        delta_tangents += np.array([0, 1e-4, 1e-4])
        normals = delta_tangents / (np.linalg.norm(delta_tangents, axis=1, keepdims=True) + 1e-9) # Y'
        binormals = np.cross(tangents[:-1], normals) # Z
        binormals = binormals / (np.linalg.norm(binormals, axis=1, keepdims=True) + 1e-9)

        normals = np.cross(binormals, tangents[1:]) # Y
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

        local_axes = np.stack([tangents[1:], normals, binormals], axis=1)
        # make local_axes right-handed by flipping the binormals
        if np.min(np.linalg.det(local_axes)) < 0:
            local_axes[:, 2] = -local_axes[:, 2]
        local_axes = local_axes[1:] # (n-2, 3, 3), the first one is not stable
        local_axes = np.concatenate([local_axes[0:1], local_axes[0:1], local_axes], axis=0) # (n, 3, 3)
        # np.linalg.det(local_axes)
    
    elif mode == 'curve':

        plane_info = find_optimal_plane(p, p0, p1)
        plane_coeff = plane_info['equation_coeff']

        near_tangent = p1 - p0
        near_tangent = near_tangent / (np.linalg.norm(near_tangent) + 1e-9) # X
        near_tangent = near_tangent[None]

        projected, distance = project_point_to_plane(plane_coeff, p) # p[p.shape[0]//2][None]

        projected_mean = np.mean(projected, axis=0) # (3,)
        projected_mean = projected_mean / (np.linalg.norm(projected_mean) + 1e-9)

        plane_normal = np.cross(near_tangent, projected_mean)[0] # Z
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)
        binormals = plane_normal[None]

        in_plane_vec = np.cross(plane_normal, near_tangent) # Y
        in_plane_vec = in_plane_vec / (np.linalg.norm(in_plane_vec) + 1e-9)

        local_axes = np.stack([near_tangent, in_plane_vec, binormals], axis=-2) # (1, 3, 3)
        if np.min(np.linalg.det(local_axes)) < 0:
            local_axes[:, 2] = -local_axes[:, 2]
        if np.abs(np.abs(np.linalg.det(local_axes)) - 1) > 1e-3:
            print('Warning: local_axes is not right-handed')
            local_axes = gram_schmidt(local_axes)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # draw arrow with unit length
            # ax.quiver(0, 0, 0, near_tangent[0, 0], near_tangent[0, 1], near_tangent[0, 2], color='r', length=0.01)
            # ax.quiver(0, 0, 0, in_plane_vec[0, 0], in_plane_vec[0, 1], in_plane_vec[0, 2], color='g', length=0.01)
            # ax.quiver(0, 0, 0, binormals[0, 0], binormals[0, 1], binormals[0, 2], color='b', length=0.01)
            # plt.show(block=True)
        local_axes = np.repeat(local_axes, p.shape[0], axis=0) # (n, 3, 3)

    elif mode == 'surface':

        plane_info = find_optimal_plane(p, p0, p1)
        plane_coeff = plane_info['equation_coeff']
        plane_normal = plane_info['normal']

        projected, distance = project_point_to_plane(plane_coeff, p) # p[p.shape[0]//2][None]

        # projected_reshape = projected.reshape(resolution[0], resolution[1]+1, 3)
        # projected_reshape_left = projected_reshape[0:resolution[0]//2]
        # projected_reshape_left_mean = np.mean(projected_reshape_left, axis=(0,1))
        # projected_reshape_left_mean = projected_reshape_left_mean / (np.linalg.norm(projected_reshape_left_mean) + 1e-9)
 
        # if distance[1] > 0:
        #     plane_normal = -plane_normal

        # find how many points are on the positive side of the plane
        pos_distance_square = (distance[distance > 0]) ** 2
        neg_distance_square = (distance[distance < 0]) ** 2
        if pos_distance_square.sum() < neg_distance_square.sum():
            plane_normal = -plane_normal

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p[..., 0], p[..., 1], distance, c='b')
        # ax.plot(p[..., 0], p[..., 1], p[..., 2], c='r')
        # plt.show(block=True)
 
        near_tangent = p1 - p0
        near_tangent = near_tangent / (np.linalg.norm(near_tangent) + 1e-9) # X
        near_tangent = near_tangent[None]

        # plane_normal = np.cross(near_tangent, projected_reshape_left_mean)[0]

        # _tangents = projected.mean(0) - p0 # Y'
        # _tangents = _tangents / (np.linalg.norm(_tangents) + 1e-9)
        # binormals = np.cross(near_tangent, _tangents) # Z
        # binormals = binormals / (np.linalg.norm(binormals) + 1e-9)
        # in_plane_vec = np.cross(binormals, near_tangent) # Y

        binormals = plane_normal[None] # Z
        in_plane_vec = np.cross(binormals, near_tangent) # Y
        in_plane_vec = in_plane_vec / (np.linalg.norm(in_plane_vec) + 1e-9)

        x_range = (np.min(p[..., 0]), np.max(p[..., 0]))
        y_range = (np.min(p[..., 1]), np.max(p[..., 1]))
        z_range = (np.min(p[..., 2]), np.max(p[..., 2]))
        x_mean = np.mean(x_range)
        y_mean = np.mean(y_range)
        z_mean = np.mean(z_range)
        max_range = max([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]) *0.5

        _x = np.linspace(x_range[0], x_range[1], 20)
        _y = np.linspace(y_range[0], y_range[1], 20)
        _x, _y = np.meshgrid(_x, _y)
        # plane: ax + by + cz + d = 0
        # z = (-ax - by - d) / c
        _z = (-plane_coeff[0] * _x - plane_coeff[1] * _y - plane_coeff[3]) / (plane_coeff[2] + 1e-9)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p[..., 0], p[..., 1], p[..., 2], c='b', label='leaf points')
        # # ax.scatter(projected[0], projected[1], projected[2], c='r')
        # ax.scatter(p0[0], p0[1], p0[2], c='r', s=100, marker='x', label='main vein end')
        # ax.scatter(p1[0], p1[1], p1[2], c='r', s=100, marker='x', label='main vein end')
        # ax.plot_surface(_x, _y, _z, alpha=0.5)
        # length = 0.3
        # # length = 0.005
        # ax.quiver(p0[0], p0[1], p0[2], near_tangent[0, 0], near_tangent[0, 1], near_tangent[0, 2], color='r', length=length, label='vein axis')
        # ax.quiver(p0[0], p0[1], p0[2], in_plane_vec[0, 0], in_plane_vec[0, 1], in_plane_vec[0, 2], color='g', length=length,label='in-plane axis')
        # ax.quiver(p0[0], p0[1], p0[2], binormals[0, 0], binormals[0, 1], binormals[0, 2], color='b', length=length, label='surface normal axis')
        # # ax.quiver(p0[0], p0[1], p0[2], plane_normal[0], plane_normal[1], plane_normal[2], color='pink', length=length)
        # ax.set_xlim(x_mean - max_range, x_mean + max_range)
        # ax.set_ylim(y_mean - max_range, y_mean + max_range)
        # ax.set_zlim(z_mean - max_range, z_mean + max_range)
        # plt.legend()
        # plt.show(block=True)

        local_axes = np.stack([near_tangent, in_plane_vec, binormals], axis=-2) # (1, 3, 3)
        if np.min(np.linalg.det(local_axes)) < 0:
            raise ValueError
            local_axes[:, 2] = -local_axes[:, 2]
        local_axes = np.repeat(local_axes, p.shape[0], axis=0) # (n, 3, 3)
    
    return local_axes

def viz_p(p):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    # ax.plot(p[..., 0], p[..., 1], p[..., 2], c='b')
    ax.scatter(p[..., 0], p[..., 1], p[..., 2], c='b')
    plt.show(block=True)


def grid_to_normals(xyz):
    """
    Computes the normal of each grid point for a 3D grid.
    
    Args:
        xyz: Tensor of shape (..., m, n, 3) representing the grid points in 3D space.
    
    Returns:
        normals: Tensor of shape (..., m, n, 3) representing the normal at each grid point.
    """
    # Compute gradients in the x and y directions using finite differences
    grad_x = xyz[..., 1:, :-1, :] - xyz[..., :-1, :-1, :]
    grad_y = xyz[..., :-1, 1:, :] - xyz[..., :-1, :-1, :]

    if isinstance(xyz, torch.Tensor):
        normals = torch.cross(grad_x, grad_y, dim=-1) # Compute the cross product of the gradients to get the normal
        normals = torch.nn.functional.normalize(normals, dim=-1) # Normalize the normal vectors
    elif isinstance(xyz, np.ndarray):
        normals = np.cross(grad_x, grad_y, axis=-1)
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9)
    
    return normals

# test_p = np.linspace(0, 1, 20)
# test_p = np.stack([test_p, np.zeros(20), np.zeros(20)], axis=1)
# axises = calculate_local_axes(test_p, debug=True)
# vs = []
# for idx, r in enumerate(axises):
#     axis_ = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     axis_.rotate(r.T)
#     axis_.translate(test_p[idx])
#     vs.append(axis_)
# test_pcd = o3d.geometry.PointCloud()
# test_pcd.points = o3d.utility.Vector3dVector(test_p)
# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# axis.translate([0, 0, 0])
# o3d.visualization.draw_geometries([test_pcd, axis] + vs)

def get_min_dist(query_points, ref_points=None, k=1, kdtree=None):
    if kdtree is None:
        assert ref_points is not None
        kdtree = scipy.spatial.cKDTree(ref_points)
    distances, indices = kdtree.query(query_points, k=k)
    return distances, indices

def segment_pcd_DBSCAN(points, eps=0.3, min_samples=10, semantic=None):
    """
    Segment a plant point cloud into different parts using DBSCAN.
    
    Parameters:
    points: numpy array of shape [n, 3] containing point cloud coordinates
    eps: float, maximum distance between two samples for them to be considered neighbors
    min_samples: int, minimum number of points to form a dense region
    
    Returns:
    labels: cluster labels for each point (-1 represents noise)
    n_clusters: number of clusters found
    clustered_points: dictionary containing points for each cluster
    """
    # Normalize the point cloud data
    # scaler = StandardScaler()
    # points_normalized = scaler.fit_transform(points)
    points_normalized = points
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points_normalized)
    
    # Get number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Organize points by cluster
    clustered_points = {}
    # for i in range(-1, n_clusters):  # Include -1 for noise points
    #     cluster_mask = (labels == i)
    #     clustered_points[i] = points[cluster_mask]
    
    labels_filtered = {}
    semantics_filtered = {}
    for i in range(0, n_clusters):
        cluster_mask = (labels == i)
        clustered_points[i] = points[cluster_mask]
        labels_filtered[i] = labels[cluster_mask]
        if semantic is not None:
            semantics_filtered[i] = semantic[cluster_mask]
        assert len(clustered_points[i]) == len(labels_filtered[i])
    
    return labels_filtered, n_clusters, clustered_points, semantics_filtered
