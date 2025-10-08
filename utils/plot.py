import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from easydict import EasyDict as edict
import open3d as o3d

def draw_tree_with_colors(array: edict, c: edict = None, save_path: str = None, early_return: bool = False):

    if c is not None:
        assert array.keys() == c.keys(), "The keys of the array and c must be the same, diff is {}".format(set(array.keys()) ^ set(c.keys()))

    my_dpi = 96
    fig = plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)

    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Add edges based on the array
    for idx, (child, parent) in enumerate(array.items()):
        child = int(child)
        parent = int(parent)
        if parent != -1:
            G.add_edge(parent, child)
        else:
            root = child

    # Determine the color of each node based on the c array
    nodes = np.array(G.nodes)
    node_colors = []
    for node in nodes:
        if c is None:
            node_colors.append('gray')
        else:
            if c[str(node)] == 0:
                node_colors.append('green')
            elif c[str(node)] == 1:
                node_colors.append('brown')
            elif c[str(node)] == 2:
                node_colors.append('pink') # flower
            else:
                node_colors.append('darkgreen') # fruit
    
    if early_return:
        return G

    # Draw the graph
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=500, node_color=node_colors, font_size=10)
    
    # Highlight the root node
    nx.draw_networkx_nodes(G, pos, nodelist=[root], node_color='lightblue', node_size=700)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Save the tree plot to {save_path}")
    plt.show(block=True)
    return G

def generate_cylinder_along_curve_batch(curve_points, r, k=10):
    if isinstance(curve_points, torch.Tensor):
        curve_points = curve_points.detach().cpu().numpy()
    if isinstance(r, torch.Tensor):
        r = r.detach().cpu().numpy()

    n = curve_points.shape[0]
    tangents = np.zeros_like(curve_points)
    tangents[1:-1] = curve_points[2:] - curve_points[:-2]
    tangents[0] = curve_points[1] - curve_points[0]
    tangents[-1] = curve_points[-1] - curve_points[-2]
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    # Initialize frame
    arbitrary = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(arbitrary, tangents[0])), 1.0, atol=1e-3):
        arbitrary = np.array([0, 1, 0])
    n1 = np.cross(tangents[0], arbitrary)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(tangents[0], n1)

    normals1, normals2 = [n1], [n2]

    # Propagate frame
    for i in range(1, n):
        v = tangents[i]
        # Project previous normal onto new tangent plane
        n1_proj = normals1[-1] - np.dot(normals1[-1], v) * v
        n1_proj /= np.linalg.norm(n1_proj)
        n2_proj = np.cross(v, n1_proj)
        n2_proj /= np.linalg.norm(n2_proj)

        normals1.append(n1_proj)
        normals2.append(n2_proj)

    normals1 = np.stack(normals1)
    normals2 = np.stack(normals2)

    # Circle points
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    circles = (curve_points[:, np.newaxis, :]
               + r * (cos_t[np.newaxis, :, None] * normals1[:, None, :]
                      + sin_t[np.newaxis, :, None] * normals2[:, None, :]))
    circles = np.concatenate([circles, circles[:, 0:1]], axis=1)
    return circles




# def load_view_point(geoms, cam_json_path=None, img_save_path=None, h=1200, w=1200, point_size=1, light_on=True):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=w, height=h)
#     ctr = vis.get_view_control()
#     for geom in geoms:
#         vis.add_geometry(geom)
#     vis.get_render_option().mesh_show_back_face=True
#     vis.get_render_option().point_size = point_size
#     vis.get_render_option().light_on = light_on
#     if cam_json_path is not None:
#         param = o3d.io.read_pinhole_camera_parameters(cam_json_path)
#         ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
#     vis.run()
#     if img_save_path is not None:
#         image = vis.capture_screen_float_buffer(False)
#         print('Save image to {}'.format(img_save_path))
#         plt.imsave(img_save_path, np.asarray(image), dpi = 100)
#     vis.destroy_window()

def add_normal_arrows(geom, scale=0.05, color=[1, 0, 0]):
    """
    Create line/arrow geometries that represent vertex normals.

    Parameters
    ----------
    geom : o3d.geometry.TriangleMesh or o3d.geometry.PointCloud
        The input geometry with normals.
    scale : float
        Arrow length scaling.
    color : list of 3 floats
        RGB color for normal arrows.
    """
    if isinstance(geom, o3d.geometry.TriangleMesh) and not geom.has_vertex_normals():
        geom.compute_vertex_normals()
    if isinstance(geom, o3d.geometry.PointCloud) and not geom.has_normals():
        geom.estimate_normals()

    vertices = np.asarray(geom.vertices if hasattr(geom, "vertices") else geom.points)
    normals = np.asarray(geom.vertex_normals if hasattr(geom, "vertex_normals") else geom.normals)

    lines = []
    points = []
    for v, n in zip(vertices, normals):
        start = v
        end = v + n * scale
        points.append(start)
        points.append(end)
        lines.append([len(points)-2, len(points)-1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)
    return line_set

def load_view_point(geoms, cam_json_path=None, img_save_path=None, h=1200, w=1200, 
                   point_size=1, light_on=True, show_window=True, line_width=1.0,
                   show_normal_color=False, show_normals=False, normal_scale=0.05):
    """
    Load and visualize geometries with optional camera parameters and image saving.
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=True)
    ctr = vis.get_view_control()
    
    for geom in geoms:
        if show_normal_color and hasattr(geom, "has_vertex_normals") and geom.has_vertex_normals():
            # Assign normals as colors (convert from [-1,1] to [0,1])
            normals = np.asarray(geom.vertex_normals)
            colors = (normals + 1) / 2.0
            geom.vertex_colors = o3d.utility.Vector3dVector(colors)
        if show_normals:
            arrow_lines = add_normal_arrows(geom, scale=normal_scale)
            vis.add_geometry(arrow_lines)
        vis.add_geometry(geom)
    
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size = point_size
    opt.light_on = light_on
    opt.line_width = line_width
    
    if cam_json_path is not None:
        param = o3d.io.read_pinhole_camera_parameters(cam_json_path)
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    
    vis.poll_events()
    vis.update_renderer()
    
    if img_save_path is not None:
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(img_save_path, np.asarray(image), dpi=200)
    
    if show_window:
        vis.run()
    else:
        print("Image captured. Window closing immediately.")
    
    vis.destroy_window()
