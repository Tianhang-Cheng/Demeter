import numpy as np
import open3d as o3d

def vis_mesh(xyz, facets):
    verts = o3d.utility.Vector3dVector(xyz)
    tris = o3d.utility.Vector3iVector(facets)
    mesh = o3d.geometry.TriangleMesh(vertices=verts, triangles=tris)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.2, 0.4, 0.2])
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh
    
def rotate2d(xyz, ang, dims):
    """Clockwise rotation."""
    ang *= np.pi / 180
    c, s = np.cos(ang), np.sin(ang)
    R = np.array(((c, -s), (s, c)))
    xyz[:, dims] = xyz[:, dims] @ R
    return xyz

def agg(xyzs, facs):
    """Aggregate collection of XYZ point sets and face sets."""
    all_xyzs = xyzs[0]
    all_facs = facs[0]
    for i in range(1, len(facs)):
        all_facs = np.append(all_facs, facs[i] + len(all_xyzs), axis=0)
        all_xyzs = np.append(all_xyzs, xyzs[i], axis=0)
    return all_xyzs, all_facs