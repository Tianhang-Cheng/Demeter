import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from utils.rotation_pytorch3d import quaternion_to_matrix, matrix_to_quaternion
from utils.rotation_custom import cartesian_to_spherical, spherical_to_cartesian
from utils.color_print import *
from utils.tool import polygon_area_torch
from utils.graph import max_geodesic_distance
import math

ZERO0 = torch.zeros((1, 3), device='cuda')
ZERO = torch.zeros((20, 1, 3), device='cuda')
ZERO1 = torch.zeros((21, 1, 3), device='cuda')
pi = np.pi

def laplacian_smoothness_weighted(pts):
    """
    pts: [H, W, 3] tensor of control points in leaf space
    Returns: scalar smoothness loss, weighted by edge length
    """
    H, W, _ = pts.shape
    
    # Horizontal neighbors
    dx = pts[:, 1:, :] - pts[:, :-1, :]
    dx_len = torch.norm(dx, dim=-1, keepdim=True) + 1e-8
    dx_norm = dx / dx_len
    dx_diff = dx_norm[:, 1:, :] - dx_norm[:, :-1, :]
    loss_x = (dx_diff ** 2).sum(dim=-1) / (dx_len[:, 1:, 0] + dx_len[:, :-1, 0])
    
    # Vertical neighbors
    dy = pts[1:, :, :] - pts[:-1, :, :]
    dy_len = torch.norm(dy, dim=-1, keepdim=True) + 1e-8
    dy_norm = dy / dy_len
    dy_diff = dy_norm[1:, :, :] - dy_norm[:-1, :, :]
    loss_y = (dy_diff ** 2).sum(dim=-1) / (dy_len[1:, :, 0] + dy_len[:-1, :, 0])
    
    return (loss_x.mean() + loss_y.mean())

def curve_smoothness_loss(points: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness loss for a sequence of 3D points.
    Encourages the curve to be as straight as possible.

    Args:
        points: Tensor of shape [N, 3], sequence of 3D points.

    Returns:
        loss: scalar tensor, smoothness loss.
    """
    d1 = points[1:] - points[:-1]               # [N-1, 3]
    d1_norm = torch.norm(d1, dim=1, keepdim=True) + 1e-8
    d1_unit = d1 / d1_norm

    d2 = d1_unit[1:] - d1_unit[:-1]             # [N-2, 3]
    loss = (d2 ** 2).sum(dim=1).mean()
    return loss

def find_z_rotation_matrix(p0):
    """
    Find a rotation matrix around the Z-axis that transforms point P0(x0, y0, z0)
    to point P1(0, y1, z0) where y1 <= 0, placing the point in the YZ plane.
    
    Parameters:
    p0 (tuple or list): The coordinates (x0, y0, z0) of the initial point
    
    Returns:
    numpy.ndarray: 3x3 rotation matrix for the Z-axis rotation
    """
    x0, y0, z0 = p0
    
    # Special case: If the point is already in the YZ plane (x0 = 0)
    if abs(x0) < 1e-10:  # Using a small epsilon for floating-point comparison
        # If y0 is already non-positive, no rotation needed
        if y0 <= 0:
            return np.eye(3)  # Return identity matrix
        # If y0 is positive, rotate by 180 degrees around Z
        else:
            theta = math.pi
            cos_theta = -1
            sin_theta = 0
    else:
        # Calculate the required rotation angle
        r = math.sqrt(x0**2 + y0**2)  # Distance from Z-axis
        
        # Find the angle to rotate to make x = 0 and y ≤ 0
        if y0 == 0 and x0 > 0:
            # Point is on positive X-axis, rotate 90 degrees counterclockwise
            theta = math.pi/2
        elif y0 == 0 and x0 < 0:
            # Point is on negative X-axis, rotate 90 degrees clockwise
            theta = -math.pi/2
        else:
            # Get the current angle in XY plane
            current_angle = math.atan2(y0, x0)
            
            # Target angle is -π/2 (to get to negative Y-axis)
            target_angle = -math.pi/2
            
            # Compute the rotation angle
            theta = target_angle - current_angle
            
            # Normalize the angle to be between -π and π
            if theta > math.pi:
                theta -= 2*math.pi
            elif theta < -math.pi:
                theta += 2*math.pi
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
    
    # Create the rotation matrix around Z-axis
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix, theta

def constrain_theta_range(angles, low=0, high=np.pi):
    """
    angles: [..., 2],  (theta, phi)
    """
    # return angles
    theta, phi = angles[..., 0], angles[..., 1]
    # flip_mask = theta < 0
    # theta = torch.where(flip_mask, -theta, theta)
    # phi = torch.where(flip_mask, phi + np.pi, phi)
    theta = torch.clip(theta, low, high)
    angles_new = torch.stack([theta, phi], dim=-1)
    return angles_new

def calculate_t(P0, P1, alpha):
    """Calculates the parameter t_i based on the distance between points and the alpha value."""
    return np.linalg.norm(P1 - P0) ** alpha

def compute_t_values(control_points, alpha):
    """Computes the non-uniform parameter values t for the control points."""
    # Calculate distances between consecutive control points
    # distances = np.linalg.norm(np.diff(control_points, axis=0), axis=1) ** alpha
    distances = torch.linalg.norm(torch.diff(control_points, dim=0), dim=-1) ** alpha
    
    # Prepend a 0 for t0 and use cumulative sum to get t1, t2, ..., tn
    # t = np.concatenate([[0], np.cumsum(distances)])
    t = torch.cat([torch.zeros([1,*distances.shape[1:]], dtype=distances.dtype, device='cuda'), torch.cumsum(distances, dim=0)], dim=0)
    return t

def catmull_rom_spline(P0, P1, P2, P3, t0, t1, t2, t3, num_points=100):
    """Computes points on a Catmull-Rom spline with non-uniform parameterization."""
    def basis_function(t, t0, t1, t2, t3, P0, P1, P2, P3):
        if len(P0.shape) == 2:
            t_ = torch.linspace(0, 1, num_points).cuda()[:, None]
            t = (t1[None] + (t2 - t1)[None] * t_)[..., None] # [num_points, batch, 1]
            P0 = P0[None]
            P1 = P1[None]
            P2 = P2[None]
            P3 = P3[None]
            t0 = t0[None, :, None]
            t1 = t1[None, :, None]
            t2 = t2[None, :, None]
            t3 = t3[None, :, None]
        else:
            t = torch.linspace(0, 1, num_points)[:, None].cuda() # [num_points, 1]
            t = t1 + (t2 - t1) * t

        A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
        A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
        A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2

        return C
    
    return basis_function(t1, t0, t1, t2, t3, P0, P1, P2, P3)

def compute_catmull_rom_curve_slow(control_points, alpha=0.5, num_points=100):
    """
    Computes a Catmull-Rom spline curve for a set of control points using non-uniform parameterization.
    control_points: [n, 3] or [n, batch, 3]
    """
    curve = []
    n = len(control_points)
    
    # Compute parameter values t_i without loop
    t = compute_t_values(control_points, alpha)

    # Compute each spline segment
    for i in range(1, n - 2):
        P0, P1, P2, P3 = control_points[i-1], control_points[i], control_points[i+1], control_points[i+2]
        segment = catmull_rom_spline(P0, P1, P2, P3, t[i-1], t[i], t[i+1], t[i+2], num_points)
        if i != 1:
            curve.append(segment[1:])
        else:
            curve.append(segment)

    # Concatenate all segments
    return torch.vstack(curve)

def compute_catmull_rom_curve(control_points, alpha=0.5, num_points=100):
    """
    Computes a Catmull-Rom spline curve for a set of control points using non-uniform parameterization.
    control_points: [n, 3]
    """
    assert len(control_points.shape) == 2
    
    curve = []

    n = len(control_points)
    
    t = compute_t_values(control_points, alpha)
    
    # for i in range(1, n - 2):
    #     print(i)
    #     P0_ref, P1_ref, P2_ref, P3_ref = control_points[i-1], control_points[i], control_points[i+1], control_points[i+2]
    #     segment = catmull_rom_spline(P0_ref, P1_ref, P2_ref, P3_ref, t[i-1], t[i], t[i+1], t[i+2], num_points)
    #     if i != 1:
    #         # print(segment[1:].shape)
    #         curve.append(segment[1:])
    #     else:
    #         # print(segment.shape)
    #         curve.append(segment)
    # curve_ref = torch.vstack(curve).clone().detach()

    P0_batch = torch.roll(control_points, shifts=1, dims=0)
    P1_batch = control_points
    P2_batch = torch.roll(control_points, shifts=-1, dims=0)
    P3_batch = torch.roll(control_points, shifts=-2, dims=0)

    t0 = torch.roll(t, shifts=1)
    t1 = t
    t2 = torch.roll(t, shifts=-1)
    t3 = torch.roll(t, shifts=-2)

    P0, P1, P2, P3 = P0_batch[1:-2], P1_batch[1:-2], P2_batch[1:-2], P3_batch[1:-2]
    t0, t1, t2, t3 = t0[1:-2], t1[1:-2], t2[1:-2], t3[1:-2]

    # Generate spline for each segment in a batch
    segments = catmull_rom_spline(P0, P1, P2, P3, t0, t1, t2, t3, num_points)
    
    # Concatenate all segments, excluding overlap
    segments = segments.permute(1, 0, 2) # [n_cp-3, seg_length ,dimension]
    curve = torch.cat([segment[1:] if i != 0 else segment for i, segment in enumerate(segments)])

    # Concatenate all segments
    return curve

def wrap_control_points(control_points, c0=None, c1=None):
    if c0 is None:
        c0 = control_points[0] - (control_points[1] - control_points[0])
    if c1 is None:
        c1 = control_points[-1] + (control_points[-1] - control_points[-2])
    if isinstance(c0, torch.Tensor):
        c0 = c0.detach()
    if isinstance(c1, torch.Tensor):
        c1 = c1.detach()
    if len(c0.shape) == len(control_points.shape) - 1:
        c0 = c0[None]
    if len(c1.shape) == len(control_points.shape) - 1:
        c1 = c1[None]
    return torch.cat([c0, control_points, c1], dim=0)

def get_radius(points, roubust=False):
    if roubust:
        center = torch.mean(points, dim=0)
        dist_to_center = torch.norm(points - center, dim=1)
        radius = torch.quantile(dist_to_center, 0.5) # median
    else:
        min_coords = torch.min(points, dim=0)[0] # (3,)
        max_coords = torch.max(points, dim=0)[0]
        # Compute the size of the bounding box
        bounding_box_size = max_coords - min_coords
        radius = torch.norm(bounding_box_size, p=2) / 2
    return radius

class CatmullRomCurve():
    def __init__(self, alpha=0.5, p0_end=None, p1_end=None, 
                 force_straight=False, 
                 force_smoothness=False):

        self.alpha = alpha
        self.thickness = None
        self.res = 24
        assert self.res > 2

        self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float, device='cuda'), requires_grad=False) # scale, articulation
        self.l = nn.Parameter(torch.tensor(1.0/(self.res-1), dtype=torch.float, device='cuda'), requires_grad=True) # length, shape parameter
        self.main_rotation = nn.Parameter(torch.zeros([self.res - 1, 2], device='cuda') , requires_grad=True) # deformation
        self.quat = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device='cuda'), requires_grad=True) # rot, articulation

        self.p0_end = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device='cuda'), requires_grad=False)
        self.p1_end = nn.Parameter(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device='cuda'), requires_grad=False)
        # self.p0_end_opt = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device='cuda'), requires_grad=True)

        if p0_end is not None:
            if not isinstance(p0_end, torch.Tensor):
                p0_end = torch.tensor(p0_end, dtype=torch.float, device='cuda')
            self.p0_end.data = p0_end.clone().detach()
        if p1_end is not None:
            if not isinstance(p1_end, torch.Tensor):
                p1_end = torch.tensor(p1_end, dtype=torch.float, device='cuda')
            self.p1_end.data = p1_end.clone().detach()
        self.find_end = True
        if p0_end is not None and p1_end is not None:
            self.find_end = False

        self.fitted = False
        self.force_straight = force_straight
        self.force_smoothness = force_smoothness
    
    def get_resolution(self):
        return self.res
    
    def fit(self, p, iters1=100, iters2=100, input_scale=None):

        self.fitted = True

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float, device='cuda')
        
        self.fit_R(p, iters=iters1, input_scale=input_scale)
        self.fit_all(p, iters=iters2)
    
    def fit_R(self, p, iters=100, input_scale=None):

        """
        init_l: the size of input pcd
        """

        self.fitted = True

        # self.p0_end_opt.data = self.p0_end.clone().detach()
        
        self.s.data = 1 / input_scale # init scale, so the input will be of unit size
        
        lr = 3e-3
        optimizer = torch.optim.AdamW([{'params': self.quat, 'lr': lr}])

        bar = trange(iters, desc='Fitting curve', leave=True)

        cp, fitted_points = self.evaluate()
        cp = cp.detach()

        for i in bar:
            optimizer.zero_grad()

            rot = quaternion_to_matrix(self.quat)
            s = torch.abs(self.s)
            # p_transformed = ((p-self.p0_end_opt) @ rot.T) * s
            # p_transformed = ((p-self.p0_end) @ rot.T) * s

            # input pcd should be close to the curve end points
            keypoint_loss = torch.mean(torch.square(((self.p1_end - self.p0_end) @ rot.T) * s - cp[-1]))
            # keypoint_loss = torch.mean(torch.square(((self.p0_end_opt - self.p0_end) @ rot.T) * s - cp[0])) + torch.mean(torch.square(((self.p1_end - self.p0_end) @ rot.T) * s - cp[-1]))
            total_loss = keypoint_loss
            if i % 10 == 0:
                bar.set_description(f'Total: {total_loss.item():.3e}, keypoint: {keypoint_loss.item():.3e}')
            total_loss.backward()
            optimizer.step()
        
        # p_transformed = ((p-self.p0_end) @ rot.T) * s
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p_transformed[:, 0].cpu().detach().numpy(), p_transformed[:, 1].cpu().detach().numpy(), p_transformed[:, 2].cpu().detach().numpy(), c='r')
        # ax.scatter(cp[:, 0].cpu().detach().numpy(), cp[:, 1].cpu().detach().numpy(), cp[:, 2].cpu().detach().numpy(), c='b')
        # plt.xlabel('x'); plt.ylabel('y')
        # # equal scale
        # max_range = torch.max(cp, dim=0)[0] - torch.min(cp, dim=0)[0]
        # max_range = max(max_range[0], max_range[1], max_range[2]).item() / 2
        # mid_x = (torch.max(cp, dim=0)[0][0] + torch.min(cp, dim=0)[0][0]).item() / 2
        # mid_y = (torch.max(cp, dim=0)[0][1] + torch.min(cp, dim=0)[0][1]).item() / 2
        # mid_z = (torch.max(cp, dim=0)[0][2] + torch.min(cp, dim=0)[0][2]).item() / 2
        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # plt.show()
        _=1

    def fit_all(self, p, iters=100):

        self.fitted = True
        
        d = self.p1_end - self.p0_end
        l_straight = torch.linalg.norm(d, ord=2)
        print('l = ', l_straight)

        if l_straight.item() < 0.01:
            self.force_straight = True

        thickness = nn.Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'), requires_grad=True)

        lr = 2e-3

        optimizer = torch.optim.AdamW([{'params': self.main_rotation, 'lr': lr},
                                    #   {'params': self.p0_end_opt, 'lr': lr},
                                    #   {'params': self.s, 'lr': lr * 10},
                                      {'params': self.l, 'lr': lr},
                                      {'params': thickness, 'lr': lr},
                                      {'params': self.quat, 'lr': lr * 0.5}
                                      ])

        bar = trange(iters, desc='Fitting curve', leave=True)

        with torch.no_grad():
            rot = quaternion_to_matrix(self.quat)
            s = torch.abs(self.s)
            # p_transformed = ((p-self.p0_end_opt) @ rot.T) * s
            p_transformed = ((p - self.p0_end) @ rot.T) * s

        for i in bar:
            optimizer.zero_grad()

            cp, fitted_points = self.evaluate(num_points=20)
            # cp = self.evaluate(num_points=50, return_cp=True)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(cp[:, 0].cpu().detach().numpy(), cp[:, 1].cpu().detach().numpy(), cp[:, 2].cpu().detach().numpy(), c='r')
            # # ax.scatter(p[:, 0].cpu().detach().numpy(), p[:, 1].cpu().detach().numpy(), p[:, 2].cpu().detach().numpy(), c='g')
            # ax.scatter(p_transformed[:, 0].cpu().detach().numpy(), p_transformed[:, 1].cpu().detach().numpy(), p_transformed[:, 2].cpu().detach().numpy(), c='b')
            # plt.xlabel('x'); plt.ylabel('y');
            # plt.show()
                       
            # self.visualize(data_points=points)
            distances_sq = torch.cdist(p_transformed, fitted_points.reshape(-1, 3), p=2)
            dist1 = distances_sq.min(dim=1)[0] # from p to curve
            dist2 = distances_sq.min(dim=0)[0] # from curve to p

            # end_loss = (torch.mean(torch.abs(cp[-1] - self.p1_end)) + torch.mean(torch.abs(cp[0] - self.p0_end))) * 1e-3 * MUL
            # keypoint_loss = torch.mean(torch.square(((self.p0_end_opt - self.p0_end) @ rot.T) * s - cp[0])) + torch.mean(torch.square(((self.p1_end - self.p0_end) @ rot.T) * s - cp[-1]))
            keypoint_loss = torch.mean(torch.square(((self.p1_end - self.p0_end) @ rot.T) * s - cp[-1])) * 0.5
            # end_loss = torch.mean(torch.square(cp[-1, 0]) + torch.square(cp[0, 0])) * 5e2 * MUL # control points should lie in z axis
            end_loss = torch.mean(torch.square(cp[-1]-torch.tensor([0,0,1]).float().cuda())) * 5 # control points should close to (0, 0, 1)

            # fit_loss = (torch.mean((dist1).square()) + torch.mean((dist2).square())) 

            if i < iters // 2:
                fit_loss = (torch.mean((dist1).square()) + torch.mean((dist2).square())) 
                # thickness.data = (dist1.mean() + dist2.mean()) * s * 1.5 # we hope the distance is close to thickness
            else:
                # fit_loss1 = (torch.mean((dist1).square()) + torch.mean((dist2).square())) 
                # fit_loss2 = (torch.mean((dist1 - thickness).square()) + torch.mean((dist2 - thickness).square())) # we hope the distance is close to thickness
                # fit_loss = fit_loss1 * 0.2 + fit_loss2
                # fit_loss = fit_loss2 + fit_loss1 * 0.5
                fit_loss1 = (torch.mean((dist1 - thickness).square()) + torch.mean((dist2 - thickness).square()))
                fit_loss2 = (torch.mean((dist1).square()) + torch.mean((dist2).square()))
                # fit_loss2 = thickness * 5e-3
                fit_loss = fit_loss1 * 0.8 + fit_loss2 * 0.2
            
            if self.force_smoothness:
                smooth_loss = curve_smoothness_loss(cp) * 2e-3
            else:
                smooth_loss = curve_smoothness_loss(cp) * 2e-4
            # smooth_loss = torch.tensor(0.0, device='cuda')

            # var_loss = torch.square((dist1 - torch.mean(dist1))).mean() + torch.square((dist2 - torch.mean(dist2))).mean() * 0.5
            # var_loss = torch.tensor(0.0, device='cuda')
            # total_loss = fit_loss + end_loss + var_loss + keypoint_loss
            # keypoint_loss *= 100
            # end_loss *= 100
            # fit_loss *= 100
            # if i < iters // 2:
            #     total_loss = fit_loss
            # else:
            #     total_loss = fit_loss + end_loss + keypoint_loss
            # total_loss = fit_loss + (end_loss + keypoint_loss) * i / iters
            total_loss = fit_loss + keypoint_loss + smooth_loss + end_loss
            if i % 10 == 0:
                bar.set_description(f'Total: {total_loss.item():.3e}, fit: {fit_loss.item():.3e}, end: {end_loss.item():.3e}, keypoint: {keypoint_loss.item():.3e}, smooth: {smooth_loss.item():.3e}, l: {self.l.item():.3e}, thickness: {thickness.item():.3e}')
            total_loss.backward()
            optimizer.step()

            if self.force_straight:
                init_l = 1.0 / (self.res-1)
                self.l.data = torch.clamp(self.l, init_l * 0.95 , init_l* 1.05)

            self.l.data = torch.abs(self.l.data)
        
        # after optimization, the cp is not on (0, 0, 1), we need to rotate it back


        # make the mean of cp lie in ZY plane, which means x should be 0
        cp_mean = torch.mean(cp, dim=0).detach().cpu().numpy()
        R, theta = find_z_rotation_matrix(cp_mean)
        # apply R to cp
        # cp_ = cp.detach().cpu().numpy()
        # cp_rot = (R @ cp_.T).T
        # with torch.no_grad():
        #     rotation_new = self.main_rotation.clone()
        #     rotation_new[:,1] += theta
        #     x = self.evaluate(rotation=rotation_new)[0]
        #     print(x.mean(0))

        self.main_rotation.data[:, 1] += theta
        self.quat.data = matrix_to_quaternion(torch.tensor(R, dtype=torch.float, device='cuda') @ rot)
        self.thickness = thickness.detach() / s

    def save(self, path):
        params = {'s': self.s,
                  'l': self.l,
                  'main_rotation': self.main_rotation,
                #   'p0_end_opt': self.p0_end_opt,
                  'thickness': self.thickness,
                  'quat': self.quat,}
        torch.save(params, path)
        green_print(f'Model saved to {path}')

    def load(self, path):
        params = torch.load(path, weights_only=True)
        self.s.data = params['s']
        self.l.data = params['l']
        self.main_rotation.data = params['main_rotation']
        # self.p0_end_opt.data = params['p0_end_opt']
        self.quat.data = params['quat']
        self.thickness = params['thickness']
        # green_print(f'Model loaded from {path}')
        self.fitted = True

    def evaluate(self, num_points=100, w=1, l=None, rotation=None, cp=None, return_cp=False):

        assert self.fitted or (cp is not None or rotation is not None), 'Model not fitted yet'
        
        if cp is None:

            if rotation is None:
                rotation = self.main_rotation * w # (n-1, 2)
            else:
                if not isinstance(rotation, torch.Tensor):
                    rotation = torch.tensor(rotation, dtype=torch.float, device='cuda')
                rotation = rotation.mean(0) + (rotation - rotation.mean(0)) * w
            
            if l is None:
                l = torch.abs(self.l)
            
            line_m_deform = spherical_to_cartesian(rotation, r=l) # (n-1, 3), make sure the total length is 1
            curved_cp = torch.cumsum(line_m_deform, dim=0) # (n-2, 3)
            curved_cp = torch.cat([ZERO0, curved_cp], dim=0)  # (n-1, 3)
        
        else:

            if not isinstance(cp, torch.Tensor):
                cp = torch.tensor(cp, dtype=torch.float, device='cuda')

            curved_cp = cp

        if return_cp:
            return curved_cp
        
        p = compute_catmull_rom_curve(wrap_control_points(curved_cp), alpha=self.alpha, num_points=num_points)
        return curved_cp, p

    def invert(self, p):
        """
        p [43, 45, 3]
        """
        assert p.shape == (self.res, 3)
        # invert
        line_m_deform_invert = torch.diff(p, dim=0)
        r_invert, rotation_invert = cartesian_to_spherical(line_m_deform_invert)
        return r_invert, rotation_invert

class CatmullRomSurface():
    def __init__(self, species: str, shape_pca=None):
        
        self.species = species

        pca_species = species
        if species == 'ribes':
            pca_species = 'papaya' # FIXME: we use papaya pca for ribes
        if species == 'rose' or species == 'pepper' or species == 'tobacco' or species == 'maize':
            pca_species = 'soybean'

        self.pca = shape_pca
        assert self.pca is not None, 'Please provide shape_pca'
        
        use_sx = False
        if species == 'maize' or species == 'tobacco':
            use_sx = True
        self.use_sx = use_sx
        
        if use_sx:
            sx_init = 1.0
            if species == 'maize':
                sx_init = 0.5
            self.sx = nn.Parameter(torch.tensor([sx_init], dtype=torch.float, device='cuda'), requires_grad=True)
        else:
            self.sx = torch.tensor([1.0], dtype=torch.float, device='cuda')

        self.mean = self.pca.data_mean.clone().detach().cuda()
        self.components = self.pca.components.clone().detach().cuda()
        self.dw = nn.Parameter(torch.zeros(len(self.pca.components), 1, device='cuda'), requires_grad=True)
        self.quat = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device='cuda'), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(1.0, dtype=torch.float, device='cuda'), requires_grad=True)

        self._theta_bias = torch.tensor([np.pi/2, 0], device='cuda', dtype=torch.float) # 45，43
        self.main_rotation = nn.Parameter(torch.zeros([44, 2], device='cuda') , requires_grad=True) # [0, 1] cartesian
        self.sub_rotation_left = nn.Parameter(torch.zeros([20, 43, 2], device='cuda') , requires_grad=True) # [0, 1]
        self.sub_rotation_right = nn.Parameter(torch.zeros([20, 43, 2], device='cuda') , requires_grad=True) # [0, 1]

        self.main_rotation.data[..., 0] = np.pi / 2
        self.sub_rotation_left.data[..., 0] = np.pi / 2
        self.sub_rotation_right.data[..., 0] = np.pi / 2

        # sigma = np.loadtxt('/home/tianhang/data/folio/Folio Leaf Dataset/Folio/papaya_rotated_mask/inner_pca_sigma.txt')
        # sigma = np.loadtxt(f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{pca_species}_rotated_mask/info/inner_pca_sigma.txt')
        sigma = np.loadtxt(f'/home/tianhang/data/folio/Folio Leaf Dataset/Folio/{pca_species}_rotated_mask/skeleton_pca_sigma.txt')
        self.shape_sigma = torch.tensor(sigma, dtype=torch.float, device='cuda')[:, None]

        self.fitted = False

        # temp variable
        self.a = None
        self.a_l = None
        self.a_r = None

    def evaluate(self, w=1, flat_only=False, main_rotation=None, sub_rotation_l=None, sub_rotation_r=None, shape_coeff=None,
                 super_resolution_rate=1, fast_forward=False, mean_shape=False, return_2d_shape=False, return_intermidiate=False):

        # assert w >= 0 and w <= 1, 'w should be in [0, 1]'
        # show warining if w is out of range
        if w < 0 or w > 1:
            print('Warning: w should be in [0, 1], but now it is ', w)

        # fitted
        flag1 = self.fitted
        # return the rotation status of the mean shape
        flag2 = return_intermidiate and mean_shape and not self.fitted 
        # get 3d deformation using given rotation for the mean shape or given shape
        flag3 = main_rotation is not None and sub_rotation_l is not None and sub_rotation_r is not None and (mean_shape or shape_coeff is not None) 
        assert flag1 or flag2 or flag3, 'Model not fitted yet'

        if mean_shape:
            feat = self.mean
        else:
            if shape_coeff is None:
                dw = self.dw
            else:
                dw = shape_coeff
            feat = self.mean + (torch.clamp(dw, -5, 5) * self.shape_sigma * self.components).sum(dim=0)

        feat_reshape = feat.reshape(45, 43, 2)
        scale = torch.concatenate([self.sx, torch.tensor([1.0]).float().cuda()], dim=0) # (2,)
        scale = torch.clamp(scale, min=0.05, max=10.0)
        feat_reshape = feat_reshape * scale[None, None, :] # (45, 43, 2)
        feat = feat_reshape.reshape(-1, 2) # (45*43, 2)

        if return_2d_shape or flat_only:
            return feat 
        
        if main_rotation is None:
            main_rotation = self.main_rotation
        
        feat_reshape = feat.reshape(45, 43, 2)
        feat_reshape = torch.cat([feat_reshape, torch.zeros((45, 43, 1), device='cuda')], dim=-1) # make it 3d
        feat_reshape = feat_reshape.transpose(1, 0) # (43, 45, 3)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.scatter(feat_reshape[..., 0].cpu().detach().numpy().reshape(-1), feat_reshape[..., 1].cpu().detach().numpy().reshape(-1))
        # # ax.plot(feat_reshape[:, 43//2, 0].cpu().detach().numpy(), feat_reshape[:, 43//2, 1].cpu().detach().numpy())
        # plt.show(block=True)

        main_skeleton = feat_reshape[43//2] # 45,3
        r, a = cartesian_to_spherical(torch.diff(main_skeleton, dim=0))
        # line_m_deform = spherical_to_cartesian(a + (main_rotation - self._theta_bias) * w, r=r)
        line_m_deform = spherical_to_cartesian(a + constrain_theta_range((main_rotation - self._theta_bias) * w, low=-pi/2, high=pi/2), r=r)
        # elif not isinstance(main_rotation, torch.Tensor):
        #     assert w == 1
        #     main_rotation = torch.tensor(main_rotation, dtype=torch.float, device='cuda')
        #     line_m_deform = spherical_to_cartesian(main_rotation, r=r)
        
        abs_p_m = torch.cumsum(line_m_deform, dim=0)
        abs_p_m = torch.cat([torch.zeros((1, 3), device='cuda'), abs_p_m], dim=0)

        if sub_rotation_l is None:
            sub_rotation_l = self.sub_rotation_left
        if sub_rotation_r is None:
            sub_rotation_r = self.sub_rotation_right
        
        left_points = feat_reshape[0:43//2]
        right_points = feat_reshape[43//2+1:]
        left_points = left_points.flip(0) # flip 1st dim, so we goes from right to middle

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(left_points[..., 0].cpu().detach().numpy().reshape(-1), left_points[..., 1].cpu().detach().numpy().reshape(-1), left_points[..., 2].cpu().detach().numpy().reshape(-1), c='r', marker='o')
        # ax.scatter(left_points[0, ..., 0].cpu().detach().numpy().reshape(-1), left_points[0,..., 1].cpu().detach().numpy().reshape(-1), left_points[0,..., 2].cpu().detach().numpy().reshape(-1), c='b', marker='o')
        # ax.scatter(right_points[0, ..., 0].cpu().detach().numpy().reshape(-1), right_points[0,..., 1].cpu().detach().numpy().reshape(-1), right_points[0,..., 2].cpu().detach().numpy().reshape(-1), c='g', marker='o')
        # plt.show()

        rel_p_l = (left_points - main_skeleton[None])[:, 1:-1] # (43, 21, 3) discard the up and down point of leaf
        line_l = torch.diff(rel_p_l, dim=0) # (43, 20, 3)
        r_l, a_l = cartesian_to_spherical(line_l) # (n, num_points-2)
        line_l_deform = spherical_to_cartesian(a_l + constrain_theta_range((sub_rotation_l - self._theta_bias) * w, low=-pi/2, high=pi/2), r=r_l)  # line_l_deform = line_l

        rel_p_r = (right_points - main_skeleton[None])[:, 1:-1]
        line_r = torch.diff(rel_p_r, dim=0) # (n, num_points-2, 3)
        r_r, a_r = cartesian_to_spherical(line_r) # (n, num_points-2)
        line_r_deform = spherical_to_cartesian(a_r + constrain_theta_range((sub_rotation_r - self._theta_bias) * w, low=-pi/2, high=pi/2), r=r_r) # (n, num_points-2, 3)

        if return_intermidiate:
            # return a, r, a_l, r_l, a_r, r_r
            return a, a_l, a_r

        # line_l_deform = line_l
        # torch.cumsum(line_l_deform, dim=0) - torch.cumsum(line_l, dim=0)
        #                                      torch.cumsum(torch.cat([rel_p_l[0:1], line_l]), dim=0) = rel_p_l

        # rel_p_l_deform = torch.cumsum(line_l_deform, dim=0)
        # rel_p_l_deform = torch.cat([ZERO, rel_p_l_deform, ZERO], dim=1) # (n, num_points, 3)
        # abs_p_l_deform = rel_p_l_deform + abs_p_m # (n, num_points, 3)

        rel_p_l_deform = torch.cumsum(torch.cat([rel_p_l[0:1], line_l_deform]), dim=0) # (21, 43, 3)
        rel_p_l_deform = torch.cat([ZERO1, rel_p_l_deform, ZERO1], dim=1) # (21, 45, 3)
        abs_p_l_deform = rel_p_l_deform + abs_p_m[None] # (21, 45, 3)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.scatter(abs_p_l_deform[:, :, 0].cpu().detach().numpy().reshape(-1), abs_p_l_deform[:, :, 1].cpu().detach().numpy().reshape(-1))
        # plt.show()

        rel_p_r_deform = torch.cumsum(torch.cat([rel_p_r[0:1], line_r_deform]), dim=0)
        rel_p_r_deform = torch.cat([ZERO1, rel_p_r_deform, ZERO1], dim=1)
        abs_p_r_deform = rel_p_r_deform + abs_p_m[None] # (n, num_points, 3)

        # combine all points
        p_deform = torch.cat([abs_p_l_deform.flip(0), abs_p_m[None], abs_p_r_deform], dim=0)

        # fig = plt.figure()catmull_
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p_deform[:, :, 0].cpu().detach().numpy(), p_deform[:, :, 1].cpu().detach().numpy(), p_deform[:, :, 2].cpu().detach().numpy())
        # # ax.scatter(p_deform[30:32, :, 0].cpu().detach().numpy(), p_deform[30:32, :, 1].cpu().detach().numpy(), c='r')
        # ax.scatter(feat_reshape[..., 0].cpu().detach().numpy(), feat_reshape[..., 1].cpu().detach().numpy(), feat_reshape[..., 2].cpu().detach().numpy(), c='r')
        # ax.scatter(self.mean.reshape(-1,2)[:, 0].cpu().detach().numpy(), self.mean.reshape(-1,2)[:, 1].cpu().detach().numpy(), 0*self.mean.reshape(-1,2)[:, 1].cpu().detach().numpy(), c='g')
        # plt.show(block=True)

        if fast_forward:
            return p_deform
        
        if super_resolution_rate > 1:
            # p_deform (w, h, 3)
            p_deform_sr = compute_catmull_rom_curve_slow(wrap_control_points(p_deform.permute(1,0,2)), num_points=(super_resolution_rate+1)) # (h, w, 3)
            p_deform_sr = p_deform_sr.permute(1,0,2) # (w, h, 3)
            p_deform_sr_0 = p_deform_sr[0, 0]
            p_deform_sr_1 = p_deform_sr[0, -1]
            p_deform_sr = compute_catmull_rom_curve_slow(wrap_control_points(p_deform_sr[:, 1:-1]), num_points=(super_resolution_rate+1))
            _zero = torch.zeros([p_deform_sr.shape[0], 1, 3], device='cuda', dtype=torch.float)
            p_deform_sr = torch.cat([_zero + p_deform_sr_0, p_deform_sr, _zero + p_deform_sr_1], dim=1)
        else:
            p_deform_sr = p_deform


        return p_deform, feat_reshape, p_deform_sr 

    def invert(self, p):
        """
        p [43, 45, 3]
        """

        assert p.shape == (43, 45, 3), 'p should be of shape (43, 45, 3)'

        if self.a is None:
            with torch.no_grad():
                self.a, self.a_l, self.a_r = self.evaluate(return_intermidiate=True, mean_shape=True)

        # invert
        abs_p_m_invert = p[43//2]
        abs_p_l_deform_invert = p[0:43//2].flip(0)
        abs_p_r_deform_invert = p[43//2+1:]

        rel_p_r_deform_invert = (abs_p_r_deform_invert - abs_p_m_invert[None])
        rel_p_l_deform_invert = (abs_p_l_deform_invert - abs_p_m_invert[None])

        line_r_deform_invert = torch.diff(rel_p_r_deform_invert[:, 1:-1], dim=0)
        line_l_deform_invert = torch.diff(rel_p_l_deform_invert[:, 1:-1], dim=0)
        line_m_deform_invert = torch.diff(abs_p_m_invert, dim=0)

        r_r_invert, a_r_invert = cartesian_to_spherical(line_r_deform_invert)
        r_l_invert, a_l_invert = cartesian_to_spherical(line_l_deform_invert)
        r_invert, a_invert = cartesian_to_spherical(line_m_deform_invert)

        sub_rotation_r_invert = a_r_invert - self.a_r + self._theta_bias
        sub_rotation_l_invert = a_l_invert - self.a_l + self._theta_bias
        main_rotation_invert = a_invert - self.a + self._theta_bias

        return main_rotation_invert, sub_rotation_l_invert, sub_rotation_r_invert

    def fit(self, p, iters1=100, iters2=100, verbose=True, **kwargs):
    
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float, device='cuda') # (num_points, 3)

        self.fitted = True
        self.fit_R(p, iters=iters1, verbose=verbose, **kwargs)
        self.fit_all(p, iters=iters2, verbose=verbose, **kwargs)

    def fit_R(self, p, iters=100, verbose=True, **kwargs):
        if self.use_sx:
            optimizer = torch.optim.AdamW([{'params': self.quat, 'lr': 5e-2},
                                           {'params': self.sx, 'lr': 5e-2}])
        else:
            optimizer = torch.optim.AdamW([{'params': self.quat, 'lr': 5e-1}])
        # optimizer = torch.optim.AdamW([{'params': self.quat, 'lr': 5e-1}])

        if verbose:
            bar = trange(iters, desc='Fitting R', leave=True)
        else:
            bar = range(iters)
        
        robust_training = kwargs.get('robust_training', False)

        if self.species == 'maize':
            s1_curved, pa, pb, idx = max_geodesic_distance(p.detach().cpu().numpy(), return_idx=True, max_points=5000, k=100) # use geodesic distance to estimate the scale of the input pcd
            # pab = torch.from_numpy(np.stack([pa, pb], axis=0)).float().cuda()
            # s1 = torch.tensor(np.linalg.norm(pa-pb), dtype=torch.float, device='cuda')
            s1 = torch.tensor(s1_curved, dtype=torch.float, device='cuda')
        else:
            s1 = get_radius(p, roubust=robust_training)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(p[:, 0].cpu().detach().numpy(), p[:, 1].cpu().detach().numpy(), p[:, 2].cpu().detach().numpy(), c='r')
        # ax.scatter(pa[0], pa[1], pa[2], c='g', s=100)
        # ax.scatter(pb[0], pb[1], pb[2], c='b', s=100)
        # plt.show(block=True)

        for i in bar:
            optimizer.zero_grad()
            rot = quaternion_to_matrix(self.quat)
            p_transformed = p @ rot.T
            p_flat = self.evaluate(flat_only=True).reshape(-1, 2)
            p_flat_grid = p_flat.reshape(45, 43, 2)
            p_flat = torch.cat([p_flat, torch.zeros((p_flat.shape[0], 1), device='cuda')], dim=1) # make it 3d
            # scale based on the radius and time
            if self.species == 'maize':
                s2 = torch.linalg.norm((p_flat_grid[-1, 43//2]-p_flat_grid[0, 43//2]))
            else:
                s2 = get_radius(p_flat, roubust=robust_training)
            s = s2 / s1 # s map input pcd to template
            self.s.data = s.detach()
            p_transformed = p_transformed * s.detach()

            # plt.figure()
            # p_flat_reshape = p_flat.reshape(-1,2 )
            # plt.scatter(p_flat_reshape[:, 0].cpu().detach().numpy(), p_flat_reshape[:, 1].cpu().detach().numpy(), c='g')
            # plt.scatter(p_flat[-1, 43//2,0].cpu().detach().numpy(), p_flat[-1, 43//2,1].cpu().detach().numpy(), c='b')
            # plt.show()

            distances = torch.norm(p_transformed[:, None] - p_flat[None], dim=-1)
            if robust_training:
                fit_loss1 = torch.mean(torch.abs(distances.min(dim=1)[0]))
                fit_loss2 = torch.mean(torch.abs(distances.min(dim=0)[0]))
            else:
                fit_loss1 = torch.mean(torch.square(distances.min(dim=1)[0]))
                fit_loss2 = torch.mean(torch.square(distances.min(dim=0)[0]))
            fit_loss = fit_loss1 + fit_loss2

            leaf_tip_loss = torch.tensor(0.0, device='cuda')
            if self.species == 'maize':
                leaf_tip_a = p_flat.reshape(45, 43, 3)[0, 43//2] # leaf root
                leaf_tip_b = p_flat.reshape(45, 43, 3)[-1, 43//2]
                dist_aa = torch.norm(leaf_tip_a - p_transformed[idx[0]])
                dist_ab = torch.norm(leaf_tip_a - p_transformed[idx[1]])
                if dist_aa < dist_ab:
                    # pa maps to leaf root
                    leaf_tip_loss = torch.norm(leaf_tip_b - p_transformed[idx[1]]) * 5
                    self.idx_b = idx[1]
                    self.idx_a = idx[0]
                else:
                    # pb maps to leaf root
                    leaf_tip_loss = torch.norm(leaf_tip_b - p_transformed[idx[0]]) * 5
                    self.idx_a = idx[1]
                    self.idx_b = idx[0]
                total_loss = leaf_tip_loss
            else:
                total_loss = fit_loss

            # if i==0 and self.species == 'maize':
                # plt.figure()
                # ax = plt.subplot(111, projection='3d')
                # ax.scatter(p_transformed[:, 0].cpu().detach().numpy(), p_transformed[:, 1].cpu().detach().numpy(), p_transformed[:, 2].cpu().detach().numpy(), c='r')
                # ax.scatter(p_flat[:, 0].cpu().detach().numpy(), p_flat[:, 1].cpu().detach().numpy(), p_flat[:, 2].cpu().detach().numpy(), c='g')
                # ax.scatter(p_transformed[self.idx_a, 0].cpu().detach().numpy(), p_transformed[self.idx_a, 1].cpu().detach().numpy(), p_transformed[self.idx_a, 2].cpu().detach().numpy(), c='b', s=100)
                # ax.scatter(p_transformed[self.idx_b, 0].cpu().detach().numpy(), p_transformed[self.idx_b, 1].cpu().detach().numpy(), p_transformed[self.idx_b, 2].cpu().detach().numpy(), c='yellow', s=100)
                # plt.show(block=True)

            if i % 10 == 0 and verbose:
                bar.set_description(f'Fit: {fit_loss.item()}, Tip: {leaf_tip_loss.item()}, s: {s.item()}')

            total_loss.backward()
            optimizer.step()
        print('s = ', self.s.item())

        # self.faces = create_grid_faces(43, 45).to(device='cuda')
        # self.adjacency = create_face_adjacency(self.faces).to(device='cuda')

    def fit_all(self, p, iters=100, verbose=True, **kwargs) -> None:
        """
        p: (n, 3) tensor
        """
        # robust_training = kwargs.get('robust_training', False)

        thickness = nn.Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'), requires_grad=True)

        area_weight = 5e2
        if self.species == 'maize':
            normal_weight = 5e-3
        else:
            normal_weight = 2e-2
        mean_weight = 0
        fit_weight = 5
        t = 1.0

        params = [ 
            {'params': self.dw, 'lr': 3e-3*t},
            {'params': self.main_rotation, 'lr': 2e-3*t},
            {'params': self.sub_rotation_left, 'lr': 2e-3*t},
            {'params': self.sub_rotation_right, 'lr': 2e-3*t},
            {'params': self.s, 'lr': 2e-4*t},
            {'params': self.quat, 'lr': 2e-4*t},
            {'params': thickness, 'lr': 2e-3*t}
            ]
        if self.use_sx:
            params.append({'params': self.sx, 'lr': 2e-3*t})

        optimizer = torch.optim.AdamW(params)

        if verbose:
            bar = trange(iters, desc='Fitting', leave=True)
        else:
            bar = range(iters)
        
        area_loss = torch.tensor(0.0, device='cuda')
        normal_loss = torch.tensor(0.0, device='cuda')

        for i in bar:
            optimizer.zero_grad()
            rot = quaternion_to_matrix(self.quat)
            p_transformed = (p @ rot.T) * self.s

            p_curved, p_flat, _ = self.evaluate()

            distances_sq = torch.cdist(p_transformed, p_curved.reshape(-1, 3), p=2)
            dist0 = distances_sq.min(dim=0)[0]
            dist1 = distances_sq.min(dim=1)[0]
            fit_loss = (torch.mean(dist0 ** 2) + torch.mean(dist1 ** 2)) * fit_weight
            
            if normal_weight > 0:
                normal_loss = laplacian_smoothness_weighted(p_curved) * normal_weight
            if area_weight > 0:
                area = polygon_area_torch(p_curved)
                area_flat = polygon_area_torch(p_flat)
                area_loss = torch.square(area - area_flat).sum() * area_weight   # 1e2

            mean_loss = torch.square(self.dw).mean() * mean_weight
            
            p_end = p_curved.reshape(43,45,3)[43//2, -1]

            # p_temp = p_curved.reshape(-1,3)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(p_temp[:, 0].cpu().detach().numpy(), p_temp[:, 1].cpu().detach().numpy(), p_temp[:, 2].cpu().detach().numpy(), c='r', s=1)
            # ax.scatter(p_end[0].cpu().detach().numpy(), 
            #            p_end[ 1].cpu().detach().numpy(),
            #             p_end[2].cpu().detach().numpy(), c='g', s=100)
            # ax.scatter(p_transformed[:, 0].cpu().detach().numpy(), p_transformed[:, 1].cpu().detach().numpy(), p_transformed[:, 2].cpu().detach().numpy(), c='b', s=1)
            # ax.scatter(p_transformed[self.idx][0].cpu().detach().numpy(), 
            #            p_transformed[self.idx][ 1].cpu().detach().numpy(),
            #             p_transformed[self.idx][2].cpu().detach().numpy(), c='yellow', s=100)
            # plt.show(block=False)

            if i % 5 == 0:
                end_loss = torch.mean(torch.square(p_end[[0,2]])) * 1e1 # make x and z of p_end close to 0
            else:
                end_loss = torch.tensor(0.0, device='cuda')
            
            if self.species == 'maize':
                tip_loss = 0.5 * torch.norm(p_end - p_transformed[self.idx_b]) ** 2
            else:
                tip_loss = torch.tensor(0.0, device='cuda')

            if i < iters // 2 or self.species == 'maize' or self.species == 'tobacco':
                total_loss = fit_loss + area_loss + normal_loss + mean_loss + end_loss + tip_loss
            else:
                fit_loss2 = (torch.mean((dist0 - thickness)** 2)  + torch.mean((dist1 - thickness)**2)) * fit_weight
                fit_loss = fit_loss * 0.2 + fit_loss2 * 0.8
                total_loss = fit_loss + area_loss + normal_loss + mean_loss + end_loss + tip_loss

            if i % 10 == 0 and verbose:
                loss_text = f'Fit: {fit_loss.item():.3e}, Area: {area_loss.item():.3e}, Normal: {normal_loss.item():.3e}, Mean: {mean_loss.item():.3e}, End: {end_loss.item():.3e}, Thickness: {thickness.item():.3e}, Tip: {tip_loss.item():.3e}'
                bar.set_description(loss_text)
            
            total_loss.backward()
            optimizer.step()

            # constrain the range of rotation so that theta is in [0, pi] (but we don't constrain to [0, pi/2] because the leaf can be folded)
            self.main_rotation.data = constrain_theta_range(self.main_rotation)
            self.sub_rotation_left.data = constrain_theta_range(self.sub_rotation_left)
            self.sub_rotation_right.data = constrain_theta_range(self.sub_rotation_right)

            # with torch.no_grad():
            #     leaf_main_vein = p_curved.reshape(43,45,3)[43//2]
            #     self_axis = calculate_local_axes(p_curved, mode='surface', p0=leaf_main_vein[0], p1=leaf_main_vein[-1])[0]
            #     self_axis = torch.tensor(self_axis, dtype=torch.float, device='cuda')
            #     # skeleton_rot = inner_points @ self_axis.T # the data that we need for PCA (shape-free)
            #     self.quat.data = matrix_to_quaternion(self_axis @ rot)

            # if robust_training:
            #     sigma_range = self.shape_sigma * 3 # 3 sigma
            #     sigma_range = torch.tensor(sigma_range[:, None], device='cuda', dtype=torch.float)
            #     # self.dw.data = torch.clamp(self.dw, -1, 1)
            #     self.dw.data = torch.clamp(self.dw, -sigma_range, sigma_range)

        print('Fitting done!')

    def save(self, path):
        params = {
            'quat': self.quat,
            's': self.s,
            'dw': self.dw,
            'main_rotation': self.main_rotation,
            'sub_rotation_left': self.sub_rotation_left,
            'sub_rotation_right': self.sub_rotation_right,
        }
        if self.use_sx:
            params['sx'] = self.sx
        torch.save(params, path)
        green_print(f'Model saved to {path}')
    
    def load(self, path):
        params = torch.load(path, weights_only=True)
        self.quat.data = params['quat']
        self.s.data = params['s']
        self.dw.data = params['dw']
        self.main_rotation.data = params['main_rotation']
        self.sub_rotation_left.data = params['sub_rotation_left']
        self.sub_rotation_right.data = params['sub_rotation_right']
        if self.use_sx:
            self.sx.data = params['sx']

        # green_print(f'Model loaded from {path}')
        self.fitted = True