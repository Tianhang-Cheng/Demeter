import torch

def polygon_area_torch(points):
    """
    使用 PyTorch 计算多边形的面积。
    假设输入 points 是形状为 [m, n, 3] 的点云。
    """
    # 获取四边形的顶点
    p1 = points[:-1, :-1]  # 左上
    p2 = points[1:, :-1]   # 左下
    p3 = points[:-1, 1:]   # 右上
    p4 = points[1:, 1:]    # 右下

    # 分割四边形为两个三角形 (p1, p2, p3) 和 (p2, p4, p3)
    # 计算第一个三角形的法向量
    v1 = p2 - p1
    v2 = p3 - p2
    cross1 = torch.cross(v1, v2, dim=2)  # 叉积得到法向量
    area1 = torch.norm(cross1, dim=2) / 2  # 面积是法向量模长的一半

    # 计算第二个三角形的法向量
    v3 = p4 - p2
    v4 = p3 - p2
    cross2 = torch.cross(v3, v4, dim=2)  # 叉积得到法向量
    area2 = torch.norm(cross2, dim=2) / 2  # 面积是法向量模长的一半

    # 总面积为两个三角形面积之和
    total_area = area1 + area2

    return total_area