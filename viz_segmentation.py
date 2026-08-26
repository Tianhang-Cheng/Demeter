import os
import glob
import json
import argparse
import colorsys
import numpy as np
import open3d as o3d

from decode import decode_params


def distinct_colors(n):
    """Return n visually distinct RGB colors (golden-angle sweep in HSV)."""
    cols = []
    for i in range(n):
        hue = (i * 0.6180339887498949) % 1.0
        cols.append(colorsys.hsv_to_rgb(hue, 0.85, 0.95))
    return cols


def load_raw_pcd(raw_folder, color_by_segment=False):
    """Merge all per-organ .ply segments in raw_folder into one colored point cloud."""
    paths = sorted(glob.glob(os.path.join(raw_folder, '*.ply')))
    assert paths, f'no .ply files found in {raw_folder}'
    palette = distinct_colors(len(paths)) if color_by_segment else None
    pts, cols = [], []
    for i, f in enumerate(paths):
        p = o3d.io.read_point_cloud(f)
        pts.append(np.asarray(p.points))
        if color_by_segment:
            cols.append(np.tile(palette[i], (len(p.points), 1)))
        else:
            cols.append(np.asarray(p.colors) if p.has_colors() else np.full((len(p.points), 3), 0.5))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols, axis=0))
    return pcd


def viz_segmentation(sample_name, data_folder, species, show_segmentation_color=False):
    """Show the fitted mesh (canonical) together with the segmented pcd (raw->canonical)."""
    inst_dir = os.path.join(data_folder, species, 'instances', sample_name)
    transform_path = os.path.join(inst_dir, 'transform.json')
    if not os.path.exists(transform_path):
        raise SystemExit(
            f'No transform.json for {sample_name}. Run build_transforms.py first to generate it.'
        )
    with open(transform_path) as f:
        tf = json.load(f)
    T = np.asarray(tf['T_raw2canonical'])

    raw_pcd = load_raw_pcd(
        os.path.join(inst_dir, 'raw'), color_by_segment=show_segmentation_color
    )
    raw_pcd.transform(T)  # raw scan frame -> canonical

    mesh = decode_params(data_folder, sample_name, species, return_mesh=True, align_global=True)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    print(f'{sample_name}: raw points {len(raw_pcd.points)} | mesh verts {len(mesh.vertices)}')
    o3d.visualization.draw_geometries([raw_pcd, mesh, axis], mesh_show_back_face=True)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize a single instance: the fitted parametric mesh together with the '
                    'segmented point cloud, both aligned in the canonical frame.'
    )
    parser.add_argument('--data_folder', default='sample_params',
                        help='path to the folder containing the sample parameters '
                             '(default: sample_params)')
    parser.add_argument('--species', default='soybean',
                        help='species name (default: soybean)')
    parser.add_argument('--sample_name', required=True,
                        help='instance name, e.g. 24_o, 2_i')
    parser.add_argument('--show_segmentation_color', action='store_true',
                        help='color each per-organ segment with a distinct color '
                             'instead of the original scan colors')
    args = parser.parse_args()

    viz_segmentation(args.sample_name, args.data_folder, args.species,
                 show_segmentation_color=args.show_segmentation_color)


if __name__ == '__main__':
    main()
