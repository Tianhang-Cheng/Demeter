import os
import glob
import argparse
import colorsys
import numpy as np
import open3d as o3d

from decode import (
    decode_params,
    raw_to_canonical_transform,
)


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


def resolve_viz_root(data_folder, species):
    """Return the instance root for batch visualization."""
    return os.path.join(data_folder, species, 'instances')


def resolve_requested_root(data_folder, species, instance_root):
    """Resolve a user-supplied root against the species folder when needed."""
    if not instance_root:
        return resolve_viz_root(data_folder, species)
    if os.path.isabs(instance_root):
        return instance_root
    species_root = os.path.join(data_folder, species)
    candidate = os.path.join(species_root, instance_root)
    return candidate if os.path.isdir(candidate) else instance_root


def viz_segmentation(sample_name, data_folder, species, show_segmentation_color=False,
                     instance_folder=None):
    """Show the fitted mesh (canonical) together with the segmented pcd (raw->canonical)."""
    inst_dir = instance_folder or os.path.join(data_folder, species, 'instances', sample_name)
    raw_folder = os.path.join(inst_dir, 'raw')
    # Derive the transform from the fitted graph instead of trusting an older
    # transform.json.  The graph uses row vectors (p @ M), whereas Open3D uses
    # column-vector homogeneous transforms, hence R=M and t=-M@root_bottom.
    T = raw_to_canonical_transform(inst_dir)

    raw_pcd = load_raw_pcd(raw_folder, color_by_segment=show_segmentation_color)
    raw_pcd.transform(T)  # raw scan frame -> canonical

    mesh = decode_params(
        data_folder,
        sample_name,
        species,
        return_mesh=True,
        align_global=True,
        instance_folder=inst_dir,
    )

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    print(f'{sample_name}: raw points {len(raw_pcd.points)} | mesh verts {len(mesh.vertices)}')
    o3d.visualization.draw_geometries([raw_pcd, mesh, axis], mesh_show_back_face=True)


def batch_viz_segmentation(data_folder, species, show_segmentation_color=False, instance_root=None):
    root = resolve_requested_root(data_folder, species, instance_root)
    sample_names = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )
    if not sample_names:
        raise SystemExit(f'No instance folders found under {root}.')

    for sample_name in sample_names:
        print(f'Visualizing {species}/{sample_name}')
        inst_dir = os.path.join(root, sample_name)
        viz_segmentation(
            sample_name,
            data_folder,
            species,
            show_segmentation_color=show_segmentation_color,
            instance_folder=inst_dir,
        )


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
    parser.add_argument('--sample_name',
                        help='instance name, e.g. 24_o, 2_i')
    parser.add_argument('--batch', action='store_true',
                        help='visualize every instance under the selected root')
    parser.add_argument('--instance_root',
                        help='override the instance root, e.g. instances')
    parser.add_argument('--show_segmentation_color', action='store_true',
                        help='color each per-organ segment with a distinct color '
                             'instead of the original scan colors')
    args = parser.parse_args()

    if args.batch:
        batch_viz_segmentation(
            args.data_folder,
            args.species,
            show_segmentation_color=args.show_segmentation_color,
            instance_root=args.instance_root,
        )
    else:
        if not args.sample_name:
            parser.error('--sample_name is required unless --batch is set')
        viz_segmentation(
            args.sample_name,
            args.data_folder,
            args.species,
            show_segmentation_color=args.show_segmentation_color,
        )


if __name__ == '__main__':
    main()
