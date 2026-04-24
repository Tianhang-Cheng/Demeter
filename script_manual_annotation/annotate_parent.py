import argparse
import glob
import open3d as o3d
import tqdm
import numpy as np
import os
import trimesh
import matplotlib.pyplot as plt
from utils.plot import generate_cylinder_along_curve_batch
from utils.constant import *
from utils.pcd import find_nearest, merge_pcd, grid_to_mesh
from utils.color_print import *
from utils.graph import load_parent, save_parent, load_class, save_class
from easydict import EasyDict as edict
import platform
from collections import defaultdict
from representation.primitive import CatmullRomCurve
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Interactive topology annotation for segmented plant point clouds.'
    )
    parser.add_argument('--meta-name', required=True, help='Sample folder name under data root.')
    parser.add_argument('--data-root', default='sample_point_cloud/val', help='Override data root directly.')
    parser.add_argument('--screen-width', type=int, default=900, help='Open3D window width.')
    parser.add_argument('--screen-height', type=int, default=900, help='Open3D window height.')
    parser.add_argument('--skip-post-merge', action='store_true', help='Skip post-merge stage.')
    return parser.parse_args()

def resolve_data_root(override_root=None):
    if override_root is not None:
        return override_root
    if platform.system() == 'Windows':
        return 'G:/pcd_processed'
    if platform.system() == 'Linux':
        return '/home/tianhang/data/pcd_processed'
    raise RuntimeError(f'Unsupported platform: {platform.system()}')

def nearest_distance(points, selected_coord):
    if cKDTree is not None:
        return cKDTree(points).query(selected_coord)[0]
    dist = np.linalg.norm(points - selected_coord, axis=1)
    return np.min(dist)


def find_loop(parents_dict):
    visited = set()
    rec_stack = set()

    def dfs(node, path):
        if node in rec_stack:
            return path[path.index(node):]
        if node in visited:
            return None
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        if node in parents_dict and parents_dict[node] != -1:
            loop_path = dfs(parents_dict[node], path)
            if loop_path:
                return loop_path
        rec_stack.remove(node)
        path.pop()
        return None

    for node in parents_dict:
        if node not in visited:
            result = dfs(node, [])
            if result:
                return result
    return None


def find_components(parents_dict):
    nodes = set(parents_dict.keys()).union(set(parents_dict.values()))
    if -1 in nodes:
        nodes.remove(-1)
    reverse_map = defaultdict(list)
    for child, parent in parents_dict.items():
        if parent != -1:
            reverse_map[parent].append(child)

    visited = set()
    components = []

    def dfs(node, component):
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        parent = parents_dict.get(node, -1)
        if parent != -1:
            dfs(parent, component)
        for nxt in reverse_map.get(node, []):
            dfs(nxt, component)

    for node in nodes:
        if node not in visited:
            current = set()
            dfs(node, current)
            if current:
                components.append(sorted(current))
    return components


def print_parent_graph_diagnostics(parents):
    if len(parents) == 0:
        red_print('parent graph is empty')
        return
    parent_int = {int(k): int(v) for k, v in parents.items()}
    loop_path = find_loop(parent_int)
    if loop_path is not None:
        red_print(f'Loop detected in parent graph: {loop_path}')
        red_print('Please fix loop in parent.txt before continuing annotation.')
        return
    components = find_components(parent_int)
    if len(components) > 1:
        red_print(f'Parent graph has {len(components)} disconnected components.')
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5]):
            red_print(f'component {i}: size={len(comp)}, nodes={comp[:15]}')
    else:
        green_print('Parent graph connectivity check passed (single component, no loop).')

def create_temp(ctx, files, file_a=None, file_b=None, hide_others=False, p_highlight=None,
                show_flower=False, show_fruit=False, show_stem=False,
                random_color=True, class_color=False, classes=None):
    """
    p_highlight: (1, 3) np.array
    """
    objects = []
    vertices = edict()

    file_a_name = str(int(os.path.basename(file_a).split('.')[0]) if file_a is not None else None)
    file_b_name = str(int(os.path.basename(file_b).split('.')[0]) if file_b is not None else None)
 
    for _, file in enumerate(files):
        cur_name = str(int(os.path.basename(file).split('.')[0]))

        pcd = trimesh.load(file)
        
        # if show_flower or show_fruit:
        #     # move mesh to center
        #     center = np.mean(pcd.vertices, axis=0)
        #     pcd.vertices -= center
        #     # scale mesh
        #     pcd.vertices *= 0.95
        #     pcd.vertices += center
        
        if random_color:
            pcd.visual.vertex_colors = np.tile(np.random.randint(100, 200, 3), (pcd.vertices.shape[0], 1))
        
        if file_a is not None and cur_name == file_a_name:
            pcd.visual.vertex_colors = np.tile(np.array([30, 30, 30], dtype=np.uint8), (pcd.vertices.shape[0], 1))
        elif file_b is not None and cur_name == file_b_name:
            # set orange
            pcd.visual.vertex_colors = np.tile(np.array([0, 255, 0], dtype=np.uint8), (pcd.vertices.shape[0], 1))
        else:
            if hide_others:
                continue

        if class_color:
            name = str(int(os.path.basename(file).split('.')[0]))
            if classes.get(name) == LEAF_CLASS:
                pcd.visual.vertex_colors = np.tile(np.array([0, 255, 0], dtype=np.uint8), (pcd.vertices.shape[0], 1))
            elif classes.get(name) == STEM_CLASS:
                pcd.visual.vertex_colors = np.tile(np.array([0, 0, 255], dtype=np.uint8), (pcd.vertices.shape[0], 1))
            elif classes.get(name) == FLOWER_CLASS:
                # pink
                pcd.visual.vertex_colors = np.tile(np.array([255,192,203], dtype=np.uint8), (pcd.vertices.shape[0], 1))
            elif classes.get(name) == FRUIT_CLASS:
                # orange
                pcd.visual.vertex_colors = np.tile(np.array([255,165,0], dtype=np.uint8), (pcd.vertices.shape[0], 1))
            else:
                # gray
                pcd.visual.vertex_colors = np.tile(np.array([128, 128, 128], dtype=np.uint8), (pcd.vertices.shape[0], 1))
        
        if show_flower:
            if classes.get(cur_name) == FLOWER_CLASS:
                pcd.visual.vertex_colors = np.tile(np.array([255,192,203, 255], dtype=np.uint8), (pcd.vertices.shape[0], 1)) * 0.5
            else:
                if hide_others:
                    continue
        
        if show_fruit:
            if classes.get(cur_name) == FRUIT_CLASS:
                pcd.visual.vertex_colors = np.tile(np.array([255,165,0, 255], dtype=np.uint8), (pcd.vertices.shape[0], 1)) * 0.5
            else:
                if hide_others:
                    continue
            
        if show_stem:
            if classes.get(cur_name) == STEM_CLASS:
                pcd.visual.vertex_colors = np.tile(np.array([0, 0, 255], dtype=np.uint8), (pcd.vertices.shape[0], 1))
            else:
                if hide_others:
                    continue

        objects.append(pcd)
        vertices[cur_name] = np.asarray(pcd.vertices)
    # use trimsh to concatenate point clouds
    merged = None
    for obj in objects:
        if merged is None:
            merged = obj
        else:
            merged = merged + obj
    if p_highlight is not None:
        if len(p_highlight.shape) == 1:
            p_highlight = p_highlight[None, :] # (1, 3)
        # Render highlighted points as spheres so they are easier to see.
        if merged is not None and hasattr(merged, "bounds"):
            extent = np.linalg.norm(merged.bounds[1] - merged.bounds[0])
            highlight_radius = max(extent * 0.01, 1e-3)
        else:
            highlight_radius = 1e-2
        sphere_template = trimesh.creation.icosphere(radius=highlight_radius, subdivisions=2)
        sphere_vertices = sphere_template.vertices
        highlight_points = np.concatenate([sphere_vertices + p for p in p_highlight], axis=0)
        p_highlight_pcd = trimesh.points.PointCloud(highlight_points)
        p_highlight_pcd.visual.vertex_colors = np.tile(
            np.array([0, 0, 255], dtype=np.uint8), (highlight_points.shape[0], 1)
        )
        merged = p_highlight_pcd if merged is None else merged + p_highlight_pcd
    merged.export(ctx.temp_pcd_path)
    return vertices

def interactive(ctx, window_name=None, pcd_path=None, screen_width=None, screen_height=None):
    if pcd_path is None:
        pcd_path = ctx.temp_pcd_path
    if screen_width is None:
        screen_width = ctx.args.screen_width
    if screen_height is None:
        screen_height = ctx.args.screen_height
    vis = o3d.visualization.VisualizerWithEditing()
    if window_name is not None:
        vis.create_window(window_name=window_name, width=screen_width, height=screen_height)
    else:
        vis.create_window(width=screen_width, height=screen_height)
    pcd = o3d.io.read_point_cloud(pcd_path)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    selected_indices = vis.get_picked_points()
    selected_coords = np.asarray(pcd.points)[selected_indices] # (n,3)
    return selected_indices, selected_coords

def visualize(files, labels=None):
    objects = []
    for i, file in enumerate(files):
        pcd = o3d.io.read_point_cloud(file)
        objects.append(pcd)
        if labels is not None:
            # 计算点云的边界框
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            
            # 创建文本对象
            text = o3d.t.geometry.TriangleMesh.create_text(str(labels[i]), depth=0.1).to_legacy()
            
            # 根据点云的尺寸调整文本大小
            scale_factor = bbox.get_extent().max() * 0.1  # 假设文本大小与点云类似
            text.scale(scale_factor, center=text.get_center())
            
            # 将文本平移到点云对象旁边
            text.translate(center + [scale_factor, 0, 0])  # 偏移位置
            
            # 给文本对象涂上颜色
            text.paint_uniform_color([0.1, 0.1, 0.1])
            
            objects.append(text)
    # set camera near the center of the point clouds
    center = np.mean([np.asarray(obj.get_center()) for obj in objects], axis=0)
    o3d.visualization.draw_geometries(objects, window_name='point clouds', front=[0, 0, -1], lookat=center, up=[0, 1, 0], zoom=0.5)

#-----------------------------------------------------------------------------------------------

def setup_phase(args):
    """Paths, rename raw/*.ply, flags; returns shared annotation context (edict)."""
    ctx = edict()
    ctx.args = args
    meta_name = args.meta_name
    folder = os.path.join(resolve_data_root(args.data_root), meta_name)
    ctx.meta_name = meta_name
    ctx.folder = folder
    ctx.info_folder = os.path.join(folder, 'info')
    ctx.raw_folder = os.path.join(folder, 'raw')
    ctx.temp_folder = os.path.join(folder, 'temp')
    os.makedirs(ctx.info_folder, exist_ok=True)
    os.makedirs(ctx.raw_folder, exist_ok=True)
    os.makedirs(ctx.temp_folder, exist_ok=True)
    ctx.temp_pcd_path = os.path.join(folder, 'info', 'temp.ply')
    ctx.parent_txt = os.path.join(folder, 'info', 'parent.txt')
    ctx.connect_txt = os.path.join(folder, 'info', 'connect.txt')
    ctx.class_txt = os.path.join(folder, 'info', 'class.txt')
    ctx.added_txt = os.path.join(folder, 'info', 'added.txt')
    ctx.done_txt = os.path.join(folder, 'info', 'done.txt')
    ctx.done_rename_txt = os.path.join(folder, 'info', 'done_rename.txt')
    if os.path.exists(ctx.temp_pcd_path):
        os.remove(ctx.temp_pcd_path)
    ctx.files = sorted(glob.glob(ctx.raw_folder + '/*.ply'))

    if (not os.path.exists(ctx.done_rename_txt)) and (not os.path.exists(ctx.done_txt)):
        for _, file in enumerate(ctx.files):
            name = os.path.basename(file)
            if (len(name.split('.')[0]) == 3 and all([c.isdigit() for c in name.split('.')[0]])):
                continue
            if all([c.isdigit() for c in name.split('.')[0]]):
                new_file = os.path.join(ctx.raw_folder, name.split('.')[0].zfill(3) + '.ply')
            else:
                index = len(glob.glob(ctx.raw_folder + '/*.ply'))
                _file_path = os.path.join(ctx.raw_folder, str(index).zfill(3) + '.ply')
                while os.path.exists(_file_path):
                    index += 1
                    _file_path = os.path.join(ctx.raw_folder, str(index).zfill(3) + '.ply')
                new_file = os.path.join(ctx.raw_folder, str(index).zfill(3) + '.ply')
            if file != new_file:
                print(f'{file} -> {new_file}')
                os.rename(file, new_file)
                with open(ctx.done_rename_txt, 'w') as f:
                    pass
        green_print('Rename done')
    else:
        green_print('Rename has been done before')

    ctx.files = sorted(glob.glob(ctx.raw_folder + '/*.ply'))
    ctx.n_files = len(ctx.files)
    ctx.merge_done_txt = os.path.join(ctx.info_folder, 'merge_done.txt')
    ctx.EASY_LABEL = True
    ctx.BAD_CONNECT_LIST = []
    ctx.EASY_CONNECT = True
    return ctx

def annotation_phase(ctx):
    """Interactive class/parent/connectivity labeling and parent graph completion."""
    files = ctx.files
    raw_folder = ctx.raw_folder
    info_folder = ctx.info_folder
    parent_txt = ctx.parent_txt
    connect_txt = ctx.connect_txt
    class_txt = ctx.class_txt
    added_txt = ctx.added_txt
    merge_done_txt = ctx.merge_done_txt
    n_files = ctx.n_files
    EASY_LABEL = ctx.EASY_LABEL
    BAD_CONNECT_LIST = ctx.BAD_CONNECT_LIST
    EASY_CONNECT = ctx.EASY_CONNECT

    if not os.path.exists(merge_done_txt):
    
        did_merge = False
    
        while True:
    
            blue_print('merge point clouds')
    
            files = glob.glob(raw_folder + '/*.ply')
            files.sort()
            create_temp(ctx, files, random_color=True)
            selected_coords = interactive(ctx, window_name='select point clouds to merge')[1] # (n,3)
    
            n_select = len(selected_coords)
    
            with open(merge_done_txt, 'w') as f:
                pass
    
            if n_select > 0:
                dists = np.zeros((len(files), n_select))
    
                for i, file in enumerate(files):
                    pcd = o3d.io.read_point_cloud(file)
                    pcd = np.asarray(pcd.points)
                    # calculate the distance between each point cloud and selected points
                    dist = np.linalg.norm(pcd[:, None] - selected_coords[None], axis=-1)
                    dists[i] = np.min(dist, axis=0)
    
                viz = []
                files_to_merge = []
                merge_index = np.argmin(dists, axis=0)
                merge_index = np.unique(merge_index)
                for count, idx in enumerate(merge_index):
                    print(f'{files[idx]} is selected')
                    files_to_merge.append(files[idx])
                    pcd = o3d.io.read_point_cloud(files[idx])
                    # assign with random color
                    pcd.paint_uniform_color(np.random.rand(3))
                    viz.append(pcd)
                o3d.visualization.draw_geometries(viz)
    
                did_merge = True
    
                merge_pcd(files_to_merge, os.path.join(os.path.dirname(info_folder), files_to_merge[0]), delete_files=True)
            
            elif n_select == 0 and did_merge:
                print('merge done, please re-run the script')
                exit()
            
            elif n_select == 0:
                print('no point cloud is selected,  end merging')
                break
    
    ####### 3. Annotate missing connectivity
    # EXIT_FLAG = False # if we add a new point cloud, we should re-run the script
    # if not os.path.exists(added_txt):
    #     while True:
    
    #         blue_print('add missing connectivity')
    #         # create a empty file added_txt
    #         if not os.path.exists(added_txt):
    #             with open(added_txt, 'w') as f:
    #                 pass
    
    #         create_temp(ctx, files, file_a=files[-1])
    #         selected_indices, selected_coords = interactive(temp_pcd_path)
    
            
    #         if len(selected_indices) == 0:
    #             blue_print('no more connectivity to annotate') 
    #             break
    #         EXIT_FLAG = True
    
    #         # create a line between two points
    #         curve = CatmullRomCurve()
    #         dist = np.linalg.norm(selected_coords[0] - selected_coords[1])
    #         # t = np.linspace(0, 1, int(dist * 10000))
    #         num_points = max(int(dist * 10000), 20)
    #         print('create a line with {} points'.format(num_points))
    #         # new_line = selected_coords[0] + t[:, None] * (selected_coords[1] - selected_coords[0])
    #         new_line = curve.evaluate(cp=selected_coords, num_points=num_points, canonical=True)[1].detach().cpu().numpy()
    #         new_pcd = trimesh.points.PointCloud(new_line)
    #         last_name = os.path.basename(files[-1])
    #         last_index = int(last_name.split('.')[0])
    #         added_index = last_index + 1
    #         added_name = str(added_index).zfill(3) + '.ply'
    #         added_file = os.path.join(raw_folder, added_name)
    #         new_pcd.export(added_file)
    
    #         print(f'{added_file} is added')
    #         with open(added_txt, 'a') as f:
    #             f.write(f'{added_index}\n')
    
    #         # update files
    #         files.append(added_file)
    #         n_files += 1
    # if EXIT_FLAG:
    #     green_print('exit annotation, please re-run the script')
    #     exit()
    
    ####### 4. Annotate parent
    parents = load_parent(parent_txt)
    print_parent_graph_diagnostics(parents)
    
    # select main stem
    has_main_stem = (len([int(k) for k, v in parents.items() if v == -1]) == 1)
    if not has_main_stem:
        vertices = create_temp(ctx, files)
        blue_print(f'annotate main stem')
        selected_indices, selected_coords = interactive(ctx, window_name='select main stem')
        assert len(selected_indices) == 1
        # calculate distance to all objects
        distances = edict()
        for vertex_key, vertex in vertices.items():
            distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
        # find the closest object
        parent_sorted = sorted(distances.items(), key=lambda x: x[1])
        main_stem = str(parent_sorted[0][0])
        parents[main_stem] = -1
        with open(parent_txt, 'a') as f:
            f.write(f'{main_stem} -> {-1}\n')
    
    # load class annotation
    classes = load_class(class_txt)
    
    
    # select flowers
    done_flower_txt = os.path.join(info_folder, 'done_flowers.txt')
    if not os.path.exists(done_flower_txt):
        
        while True:
            vertices = create_temp(ctx, files, classes=classes, show_flower=True, show_fruit=True, random_color=True)
            blue_print(f'annotate flowers')
            selected_indices, selected_coords = interactive(ctx, window_name='select flowers')
            if len(selected_indices) == 0:
                blue_print('no more flowers to annotate')
                break
            assert len(selected_indices) == 1
    
            distances = edict()
            for vertex_key, vertex in vertices.items():
                distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
            # find the closest object
            select_index_sorted = sorted(distances.items(), key=lambda x: x[1])
    
            name = str(select_index_sorted[0][0])
            classes[name] = 2
            with open(class_txt, 'a') as f:
                f.write(f'{name} {2}\n')
    
        with open(done_flower_txt, 'w') as f:
            pass
    
    # select fruits
    done_fruit_txt = os.path.join(info_folder, 'done_fruits.txt')
    if not os.path.exists(done_fruit_txt):
    
        while True:
            vertices = create_temp(ctx, files, classes=classes, show_flower=True, show_fruit=True, random_color=True)
            blue_print(f'annotate fruits')
            selected_indices, selected_coords = interactive(ctx, window_name='select fruits')
            if len(selected_indices) == 0:
                blue_print('no more fruits to annotate')
                break
            assert len(selected_indices) == 1
    
            distances = edict()
            for vertex_key, vertex in vertices.items():
                distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
            # find the closest object
            select_index_sorted = sorted(distances.items(), key=lambda x: x[1])
    
            name = str(select_index_sorted[0][0])
            classes[name] = 3
            with open(class_txt, 'a') as f:
                f.write(f'{name} {3}\n')
    
        with open(done_fruit_txt, 'w') as f:
            pass
    
    # select stems
    done_stem_txt = os.path.join(info_folder, 'done_stem.txt')
    if not os.path.exists(done_stem_txt):
    
        while True:
            vertices = create_temp(ctx, files, classes=classes, show_stem=True, show_fruit=True, show_flower=True, )
            blue_print(f'annotate isolated stems')
            selected_indices, selected_coords = interactive(ctx, window_name='select isolated stems')
            if len(selected_indices) == 0:
                blue_print('no more stems to annotate')
                break
            assert len(selected_indices) == 1
    
            distances = edict()
            for vertex_key, vertex in vertices.items():
                distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
            # find the closest object
            select_index_sorted = sorted(distances.items(), key=lambda x: x[1])
    
            name = str(select_index_sorted[0][0])
            classes[name] = 1
            with open(class_txt, 'a') as f:
                f.write(f'{name} {1}\n')
    
        with open(done_stem_txt, 'w') as f:
            pass
    
    # select leaves
    # done_leaf_txt = os.path.join(info_folder, 'done_leaf.txt')
    # if not os.path.exists(done_leaf_txt):
    
    #     while True:
    #         vertices = create_temp(ctx, files, classes=classes, show_stem=True, show_fruit=True, show_flower=True, )
    #         blue_print(f'annotate leaves')
    #         selected_indices, selected_coords = interactive(ctx, window_name='select leaves')
    #         if len(selected_indices) == 0:
    #             blue_print('no more leaves to annotate')
    #             break
    #         assert len(selected_indices) == 1
    
    #         distances = edict()
    #         for vertex_key, vertex in vertices.items():
    #             distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
    #         # find the closest object
    #         select_index_sorted = sorted(distances.items(), key=lambda x: x[1])
    
    #         name = str(select_index_sorted[0][0])
    #         classes[name] = 0
    #         with open(class_txt, 'a') as f:
    #             f.write(f'{name} {0}\n')
    
    #     with open(done_leaf_txt, 'w') as f:
    #         pass
    
    #load connect annotation
    connects = edict()
    if os.path.exists(connect_txt):
        with open(connect_txt, 'r') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            name, is_labeled = line.strip().split(' ')
            connects[str(name)] = int(is_labeled)
    
    for i in range(n_files):
        file = files[i]
        print(f'processing {file}')
        name = str(int(os.path.basename(file).split('.')[0]))
        del i
    
        # skip if already labeled and connected
        if (connects.get(name) == CONNECTED and classes.get(name) is not None):
            continue
    
        # if the point cloud is leaf, we skip connectivity annotation
        if classes.get(name) == LEAF_CLASS:
            with open(connect_txt, 'a') as f:
                f.write(f'{name} 1\n')
            continue
    
        if connects.get(name) == CONNECTED:
            green_print(f'no need to fix connectivity for mesh {name}')
            continue
        # else:
        #     blue_print(f'could fix connectivity for mesh {name}')
        
        selected_indices = []
        if not EASY_CONNECT or name in BAD_CONNECT_LIST:
            # find children
            children = [k for k, v in parents.items() if v == int(name)]
            file_b = None
            if len(children) > 0:
                file_b = os.path.join(raw_folder, children[0].zfill(3) + '.ply')
            vertices = create_temp(ctx, files, file, file_b)
            selected_indices, selected_coords = interactive(ctx, window_name='select connectivity')
        # fix connectivity if not labeled or labeled as stem
        if connects.get(name) is None:
            print(f'connectivity for mesh {name}')
            if len(selected_indices) == 0:
                # print('mesh {}: no need to fix connectivity'.format(name))
                with open(connect_txt, 'a') as f:
                    f.write(f'{name} 1\n')
                continue
    
            # create a line between two points
            if len(selected_indices) > 0:
                curve = CatmullRomCurve()
                dist = np.linalg.norm(selected_coords[0] - selected_coords[1])
                # t = np.linspace(0, 1, int(dist * 10000))
                num_points = max(int(dist * 10000), 20)
                print('create a line with {} points'.format(num_points))
                # new_line = selected_coords[0] + t[:, None] * (selected_coords[1] - selected_coords[0])
                new_line = curve.evaluate(cp=selected_coords, num_points=num_points, canonical=True)[1].detach().cpu().numpy()
                old_pcd = trimesh.load(file)
                # copy the old pcd to backup folder
                # create a new pcd with the line
                new_pcd = trimesh.points.PointCloud(new_line)
                merged_pcd = old_pcd + new_pcd
                merged_pcd.export(raw_folder)
                print(f'mesh {name} is fixed')
            with open(connect_txt, 'a') as f:
                f.write(f'{name} 1\n')
    
    all_names = [int(os.path.basename(file).split('.')[0]) for file in files]
    all_fruit = [int(k) for k, v in classes.items() if v == FRUIT_CLASS]
    all_flower = [int(k) for k, v in classes.items() if v == FLOWER_CLASS]
    
    vertices_all = create_temp(ctx, files=files)
    
    while not ctx.args.skip_post_merge:
    
        parents = load_parent(parent_txt)
        keys_to_delete = []
        for k, v in parents.items():
            file_k = os.path.join(raw_folder, '{}.ply'.format(str(k).zfill(3)))
            file_v = os.path.join(raw_folder, '{}.ply'.format(str(v).zfill(3)))
            
            if not os.path.exists(file_k) or not os.path.exists(file_v):
                if v != -1:
                    keys_to_delete.append(k)
                    red_print(f'{k} -> {v} does not exist file for both sides, delete it')
        for k in keys_to_delete:
            del parents[k]
        # if k in classes:
        #     del classes[k]
        save_parent(parents, parent_txt)
        parents = load_parent(parent_txt)
        # save_class(classes, class_txt)
        # classes = load_class(class_txt)
    
        finish_annotation = False
    
        for file in tqdm.tqdm(files):
            name = str(int(os.path.basename(file).split('.')[0]))
    
            # label parent for file i
            if parents.get(name) is not None:
                assert int(name) in all_names, 'file {} does not exist'.format(name)
                if file == files[-1]:
                    finish_annotation = True
                continue
        
            # guess a parent
            inpossible_parents = [int(k) for k, v in parents.items() if v == int(name)] # no circular parent
            possible_parents = set(all_names) - set(inpossible_parents) - set([int(name)]) - set(all_flower) - set(all_fruit)
            # find the closest object to the current object
            min_dist = 1e10
            min_dist_path = None
            min_dist_name = None
    
            all_dist = {}
            cur_pcd_points = np.array(o3d.io.read_point_cloud(file).points)
            for possible_parent in possible_parents:
                _parent_path = os.path.join(raw_folder, str(possible_parent).zfill(3) + '.ply')
                _parent_pcd = vertices_all[str(possible_parent)]
                _dist = find_nearest(cur_pcd_points, _parent_pcd, sample_num=2000, return_dist=True)
                all_dist[possible_parent] = _dist
                if _dist < min_dist:
                    min_dist = _dist
                    min_dist_path = _parent_path
                    min_dist_name = int(possible_parent)
            
            for _name in all_names:
                if _name == int(name):
                    continue
                _dist = find_nearest(cur_pcd_points, vertices_all[str(_name)], sample_num=2000, return_dist=True)
                all_dist[_name] = _dist
            
            # rank the possible parents based on the distance, from small to large
            sorted_dist = sorted(all_dist.items(), key=lambda x: x[1])
            # get first 10 possible parents
            # sorted_dist = sorted_dist[:30]
            sorted_files = [os.path.join(raw_folder, str(k).zfill(3) + '.ply') for k, v in sorted_dist] + [file]
            vertices = create_temp(ctx, files=sorted_files, file_a=file, file_b=min_dist_path)
            blue_print(f'annotate parent for mesh {file}, guess parent {min_dist_name}')
    
            selected_indices, selected_coords = interactive(ctx, window_name='select parent')
    
            viz_parent = False
    
            if len(selected_indices) == 0:
                print('mesh {} has parent {}'.format(name, min_dist_name))
                parent = min_dist_name
                
            elif len(selected_indices) == 1:
                # calculate distance to all objects
                distances = edict()
                for vertex_key, vertex in vertices.items():
                    distances[vertex_key] = nearest_distance(vertex, selected_coords[0])
                # find the closest object
                parent_sorted = sorted(distances.items(), key=lambda x: x[1])
                # if the closest object is itself, select the second closest object
                parent = int(parent_sorted[0][0]) if parent_sorted[0][0] != name else int(parent_sorted[1][0])
                viz_parent = True
            elif len(selected_indices) > 1:
                # add connection
                curve = CatmullRomCurve()
                dist = np.linalg.norm(selected_coords[0] - selected_coords[1])
                # t = np.linspace(0, 1, int(dist * 10000))
                num_points = max(int(dist * 10000), 20)
                print('create a line with {} points'.format(num_points))
                # new_line = selected_coords[0] + t[:, None] * (selected_coords[1] - selected_coords[0])
                new_line = curve.evaluate(cp=selected_coords, num_points=num_points, canonical=True)[1].detach().cpu().numpy()
                new_pcd = trimesh.points.PointCloud(new_line)
                files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
                last_name = os.path.basename(files[-1])
                last_index = int(last_name.split('.')[0])
                added_index = last_index + 1
                added_name = str(added_index).zfill(3) + '.ply'
                added_file = os.path.join(raw_folder, added_name)
                new_pcd.export(added_file)
    
                print(f'{added_file} is added')
                with open(added_txt, 'a') as f:
                    f.write(f'{added_index}\n')
        
                mode = input('[0] Add to parent. [1] Add to current. [2] Add new \n')
                mode = int(mode)
    
                if mode == 0:
    
                    print(f'add {added_file} to {min_dist_path}')
                    parent_path = min_dist_path
                    merge_pcd([parent_path, added_file], parent_path, delete_files=True)
                
                elif mode == 1:
    
                    print(f'add {added_file} to {file}')
                    merge_pcd([file, added_file], file, delete_files=True)
                    vertices_all = create_temp(ctx, files=files)
                    all_names = [int(os.path.basename(x).split('.')[0]) for x in files]
                    break
                
                elif mode == 2:
    
                    print(f'add {added_file} as a new mesh')
                    files.append(added_file)
                    vertices_all = create_temp(ctx, files=files)
                    all_names = [int(os.path.basename(x).split('.')[0]) for x in files]
                    break
                
            parents[name] = parent
    
            no_new_annotation = False
            
            if viz_parent:
                green_print(f'visualize parent (orange) {parent} for current (red) {file}')
                parent_file = os.path.join(raw_folder, str(parent).zfill(3) + '.ply')
                create_temp(ctx, sorted_files, file, parent_file)
                interactive(ctx, window_name='visualize parent')
    
            with open(parent_txt, 'a') as f:
                f.write(f'{name} -> {parent}\n')
           
        if finish_annotation:
            break
    
    # check_parents = True
    # if check_parents:
    #     for i in range(n_files):
    #         if i < 91:
    #             continue
    #         file = files[i]
    #         name = str(int(os.path.basename(file).split('.')[0]))
    #         del i
    #         parent = parents.get(name)
    #         green_print(f'visualize parent (orange) {parent} for current (red) {file}')
    #         parent_file = os.path.join(raw_folder, str(parent).zfill(3) + '.ply')
    #         create_temp(ctx, files, file, parent_file)
    #         interactive(temp_pcd_path)
    #     green_print('parent annotation is correct')
    #     exit()
    
    ctx.parents = parents
    ctx.classes = classes
    ctx.n_files = len(ctx.files)

def validation_phase(ctx):
    """Infer missing classes, assert parent/class consistency."""
    parents = ctx.parents
    classes = ctx.classes
    files = ctx.files
    raw_folder = ctx.raw_folder
    class_txt = ctx.class_txt
    EASY_LABEL = ctx.EASY_LABEL
    ####### 2. Fix connectivity issues and visualize all point clouds with different colors
    for file in files:
        print(f'processing {file}')
        name = str(int(os.path.basename(file).split('.')[0]))
    
        # set input type
        if classes.get(name) is None:
    
            has_children = [int(k) for k, v in parents.items() if v == int(name)]
            has_parent = parents.get(name) != -1
    
            if classes.get(str(int(name))) is None:
                if has_children:
                    _class = 1
                else:
                    if EASY_LABEL:
                        _class = 0
                    else:
                        _class = input('input class for {}. [0] leaf. [1] stem. [2] flower. [3] fruit\n'.format(name))
                        _class = int(_class)
            else:
                continue
    
            classes[name] = _class
            with open(class_txt, 'a') as f:
                f.write(f'{name} {_class}\n')
    
            if _class == 0:
                print(f'mesh {name} is labeled as leaf')
            elif _class == 1:
                print(f'mesh {name} is labeled as stem')
            elif _class == 2:
                print(f'mesh {name} is labeled as flower')
            elif _class == 3:
                print(f'mesh {name} is labeled as fruit')
            else:
                raise ValueError('Invalid class')
    
    
    # check validity of parent annotation
    for k, v in parents.items():
        if v == -1:
            continue
        assert os.path.exists(os.path.join(raw_folder, str(k).zfill(3) + '.ply')), f'{k} does not exist'
        assert classes.get(str(k)) is not None, f'class of {k} is not labeled'
        assert classes.get(str(v)) is not None, f'class of {v} is not labeled'
        if classes.get(str(v)) != STEM_CLASS:
            raise ValueError(f'{v} should be stem')
    
    # draw tree
    assert parents.keys() == classes.keys(), 'The keys of parents and classes should be the same, different keys are {}'.format(set(parents.keys()) ^ set(classes.keys()))
    
    if platform.system() == 'Linux' and False:
        from utils.plot import draw_tree_with_colors
        draw_tree_with_colors(parents, c=classes)
    else:
        print('Draw tree is only supported in Linux')
    
    # create_temp(ctx, files, class_color=True, classes=classes)
    # selected_coords = interactive(ctx, window_name='class color')[1] # (n,3)
    # mesh_path = os.path.join('/home/tianhang/data/mesh_clean_join_centered', meta_name + '.ply')
    # mesh = o3d.io.read_triangle_mesh(mesh_path)
    

def postprocess_phase(ctx):
    """Post-merge graph/meshes, topology preview geometry, leaf root points."""
    parents = ctx.parents
    classes = ctx.classes
    files = ctx.files
    raw_folder = ctx.raw_folder
    info_folder = ctx.info_folder
    folder = ctx.folder
    parent_txt = ctx.parent_txt
    class_txt = ctx.class_txt
    merge_done_txt = ctx.merge_done_txt
    meta_name = ctx.meta_name
    n_files = ctx.n_files
    # post merge
    update_txt = os.path.join(info_folder, 'update.txt')
    
    # save parent and class
    if not os.path.exists(update_txt):
        save_parent(parents, path=parent_txt.replace('.txt', '_backup.txt'))
        save_class(classes, path=class_txt.replace('.txt', '_backup.txt'))
    
    while True:
    
        blue_print('merge point clouds and graph')
    
        files = glob.glob(raw_folder + '/*.ply')
        files.sort()
        create_temp(ctx, files, random_color=True)
        selected_coords = interactive(ctx, window_name='post merge')[1] # (n,3)
    
        n_select = len(selected_coords)
    
        with open(merge_done_txt, 'w') as f:
            pass
    
        if n_select > 0:
            dists = np.zeros((len(files), n_select))
    
            for i, file in enumerate(files):
                pcd = o3d.io.read_point_cloud(file)
                pcd = np.asarray(pcd.points)
                # calculate the distance between each point cloud and selected points
                dist = np.linalg.norm(pcd[:, None] - selected_coords[None], axis=-1)
                dists[i] = np.min(dist, axis=0)
    
            viz = []
            files_to_merge = []
            merge_index = np.argmin(dists, axis=0)
            for count, idx in enumerate(merge_index):
                print(f'{files[idx]} is selected')
                files_to_merge.append(files[idx])
                pcd = o3d.io.read_point_cloud(files[idx])
                # assign with random color
                pcd.paint_uniform_color(np.random.rand(3))
                viz.append(pcd)
            o3d.visualization.draw_geometries(viz)
    
            assert len(merge_index) == 2, 'only two point clouds can be selected to merge at a time'
    
            main_merge_id = int(os.path.basename(files_to_merge[0]).split('.')[0])
            wait_to_merge_id = int(os.path.basename(files_to_merge[1]).split('.')[0])
            assert (parents.get(str(wait_to_merge_id)) == main_merge_id), f'{wait_to_merge_id} should -> {main_merge_id}'
            with open(update_txt, 'a') as f:
                f.write(f'{wait_to_merge_id}\n')
                f.write(f'{main_merge_id}\n')
            for k, v in parents.items():
                if v == wait_to_merge_id:
                    parents[k] = main_merge_id
                    print(f'{k} is re-point to {main_merge_id}')
            del parents[str(wait_to_merge_id)]
    
            delete_k = []
            for k, v in classes.items():
                if k == str(wait_to_merge_id):
                    # remove the item
                    delete_k.append(k)
            for k in delete_k:
                del classes[k]
                print(f'{k} is removed')
    
            save_parent(parents, path=parent_txt)
            print('parent is updated')
            parents = load_parent(parent_txt)
    
            save_class(classes, path=class_txt)
            print('class is updated')
            classes = load_class(class_txt)
    
            did_merge = True
    
            merge_pcd(files_to_merge, output_path=files_to_merge[0], delete_files=True)
            # merge_pcd(files_to_merge_raw, output_path=files_to_merge_raw[0], delete_files=True)
        
        else:
    
            print('no point cloud is selected,  end merging')
            break
    
    if ctx.args.skip_post_merge:
        blue_print('skip post merge stage by --skip-post-merge')
    
    print_parent_graph_diagnostics(parents)
    
    geometries = []
    geometries_input_pcd = []
    geometries_mesh = []
    files = glob.glob(raw_folder + '/*.ply')
    for file in files:
        pcd = o3d.io.read_point_cloud(file)
        # downsample
        
        # seg random color
        # color = np.random.randint(150, 200, 3)
        # pcd.paint_uniform_color(color / 255)
    
        idx = int(os.path.basename(file).split('.')[0])
        parent_id = parents.get(str(idx))
        if parent_id != -1:
            parent_file = os.path.join(raw_folder, str(parent_id).zfill(3) + '.ply')
            parent_center = np.mean(np.asarray(o3d.io.read_point_cloud(parent_file).points), axis=0)
    
        center = np.mean(np.asarray(pcd.points), axis=0)
    
        # pcd = pcd.voxel_down_sample(voxel_size=0.004)
        # pcd.paint_uniform_color([0.8, 0.8, 0.8])
    
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        ball.compute_vertex_normals()
        ball_center = np.mean(np.asarray(ball.vertices), axis=0)
        ball.translate(center - ball_center)
        if classes.get(str(idx)) == LEAF_CLASS:
            ball.paint_uniform_color([0.2, 0.9, 0.2])
        elif classes.get(str(idx)) == STEM_CLASS or classes.get(str(idx)) == FRUIT_CLASS:
            ball.paint_uniform_color([0.2, 0.2, 0.2])
        elif classes.get(str(idx)) == FLOWER_CLASS:
            ball.paint_uniform_color([0.9, 0.2, 0.2])
    
        if parent_id == -1:
            ball.paint_uniform_color([0.5, 0.5, 0.5])
    
        # arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0005, cone_radius=0.001, cylinder_height=0.01, cone_height=0.005)
        # arrow.compute_vertex_normals()
        # arrow.translate(center)
        if parent_id != -1:
            vec = parent_center - center
            vec_len = np.linalg.norm(vec)
            t = np.linspace(0, 1, max(int(vec_len * 2500), 50))
            line_p = center + t[:, None] * vec # (n, 3)
            cylinder = generate_cylinder_along_curve_batch(line_p, r=0.0005, k=12)
            cylinder_mesh = grid_to_mesh(cylinder)
            
            if classes.get(str(idx)) == LEAF_CLASS:
                cylinder_mesh.paint_uniform_color([0.2, 0.9, 0.2])
            elif classes.get(str(idx)) == STEM_CLASS or classes.get(str(idx)) == FRUIT_CLASS:
                cylinder_mesh.paint_uniform_color([0.2, 0.2, 0.2])
            elif classes.get(str(idx)) == FLOWER_CLASS:
                cylinder_mesh.paint_uniform_color([0.9, 0.2, 0.2])
    
            geometries.append(cylinder_mesh)
            geometries_mesh.append(cylinder_mesh)
    
        geometries.append(pcd)
        geometries.append(ball)
    
        geometries_input_pcd.append(pcd)
        geometries_mesh.append(ball)
    
    # o3d.visualization.draw_geometries(geometries_mesh, window_name='topology', mesh_show_back_face=True)
    
    # for idx, x in enumerate(geometries_mesh):
    #     o3d.io.write_triangle_mesh(os.path.join(folder, 'viz', '{}_mesh.ply'.format(idx)), x)
    # for idx, x in enumerate(geometries_pcd):
    #     o3d.io.write_point_cloud(os.path.join(folder, 'viz', '{}_pcd.ply'.format(idx)), x)
    
    ####### 6. annotate root for each leaf
    leaf_paths = []
    processed_files = glob.glob(raw_folder + '/*.ply')
    processed_files.sort()
    for i, file in tqdm.tqdm(enumerate(processed_files)):
        name = str(int(os.path.basename(file).split('.')[0]))
        if classes.get(name) == LEAF_CLASS:
            leaf_paths.append(file)
        else:
            continue
        root_path = file.replace(raw_folder, info_folder).replace('.ply', '_root.txt')
        if not os.path.exists(root_path):
            blue_print(f'annotate root for leaf {name}')
            parent = int(parents[name])
            parent_file = os.path.join(raw_folder, str(parent).zfill(3) + '.ply')
    
            # automatically select the center of the leaf
            child_pcd = o3d.io.read_point_cloud(file)
            child_pcd = np.asarray(child_pcd.points)
            parent_pcd = o3d.io.read_point_cloud(parent_file)
            parent_pcd = np.asarray(parent_pcd.points)
            root_point = find_nearest(child_pcd, parent_pcd, sample_num=3000).detach().cpu().numpy()
    
            create_temp(ctx, files, file, parent_file, hide_others=True, p_highlight=root_point)
            selected_indices, selected_coords = interactive(ctx, window_name='select leaf root point')
            if len(selected_indices) > 0:
                # human annotation
                root_point = np.mean(selected_coords, axis=0)
            np.savetxt(root_path, root_point)
            print(f'root point for leaf {i} is annotated')


def main():
    ctx = setup_phase(parse_args())
    annotation_phase(ctx)
    validation_phase(ctx)
    postprocess_phase(ctx)


if __name__ == '__main__':
    main()
