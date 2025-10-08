import torch
import torch.nn.functional as F
import numpy as np

def spherical_to_cartesian(angles, r=1.0):
    """
    将球坐标转换为笛卡尔坐标
    :param theta: 形状为 (N,) 的张量，表示俯仰角（单位：弧度), 0 <= theta <= pi
    :param phi: 形状为 (N,) 的张量，表示方位角（单位：弧度）, 0 <= phi <= 2 * pi
    :param r: 半径，默认为单位球 r = 1
    :return: 形状为 (N, 3) 的张量，表示笛卡尔坐标 (x, y, z)
    """
    if isinstance(angles, torch.Tensor):
        theta, phi = angles[..., 0], angles[..., 1]
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        
        # 合并为形状为 (N, 3) 的张量
        return torch.stack((x, y, z), dim=-1)
    
    else:
        theta, phi = angles[..., 0], angles[..., 1]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # 合并为形状为 (N, 3) 的张量
        return np.stack((x, y, z), axis=-1)

def cartesian_to_spherical(xyz):
    """
    将笛卡尔坐标转换为球坐标
    :param xyz: 形状为 (N, 3) 的张量，表示笛卡尔坐标 (x, y, z)
    :return: 两个张量 theta 和 phi，分别表示俯仰角和方位角, 0 <= theta <= pi, 0 <= phi <= 2 * pi
    """
    if isinstance(xyz, torch.Tensor):
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-8  # 计算半径 r
        theta = torch.acos(z / r)  # 计算俯仰角 theta
        phi = torch.atan2(y, x)  # 计算方位角 phi
        angles = torch.stack([theta, phi], dim=-1) # (N, 2)
        return r, angles
    
    else:
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        # phi = np.arctan(y / x)
        angles = np.stack([theta, phi], axis=-1)
        return r, angles

def rotation_6d_to_matrix(M_vec):
    """
    M_vec: [2, 3]
    """
    
    # TODO: find if there is a better way to recover the matrix

    # if isinstance(M_vec, np.ndarray):
    #     M_vec = torch.tensor(M_vec).float()
    assert isinstance(M_vec, torch.Tensor), 'M_vec should be a tensor'
    assert len(M_vec.shape) == 2, 'M_vec should have shape (2, 3), but got {}'.format(M_vec.shape)

    a1 = M_vec[0]
    a2 = M_vec[1]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    M = torch.stack((b1, b2, b3), dim=-2)
    assert torch.abs(torch.abs(torch.linalg.det(M)) - 1) < 1e-2, 'The determinant of M should be 1, but got {}'.format(torch.abs(torch.linalg.det(M)))
    return M