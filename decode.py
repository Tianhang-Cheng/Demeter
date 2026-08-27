import os
import json
import numpy as np
import open3d as o3d
import torch
from utils.constant import *
from utils.device import set_device, get_device
from representation.graph import PlantGraphFixedTopology
from utils.pca import NodePCA
from utils.graph import load_parent, load_class
from utils.rotation_pytorch3d import quaternion_to_matrix


def raw_to_canonical_transform(instance_folder: str) -> np.ndarray:
    """Return the rigid transform from a segmented scan to graph canonical space.

    ``build_transforms.py`` precomputes this matrix (rotation from the fitted
    main-stem quaternion plus the bottom translation) and stores it in
    ``transform.json``.  Older archives lack the rotation, in which case it is
    reconstructed from the fitted graph instead.
    """
    transform_path = os.path.join(instance_folder, 'transform.json')
    with open(transform_path) as f:
        data = json.load(f)

    if 'main_stem_quat' in data:
        return np.asarray(data['T_raw2canonical'], dtype=np.float64)

    return _transform_from_graph(instance_folder, data)


def _transform_from_graph(instance_folder: str, data: dict) -> np.ndarray:
    """Reconstruct the raw->canonical transform from the fitted graph.

    Fallback for ``transform.json`` files that predate the rotation field: the
    main-stem quaternion lives in ``graph.pkl`` and is combined with the bottom
    translation already stored in the JSON.
    """
    main_stem = str(data['main_stem'])
    bottom = np.asarray(data['main_stem_end_points_bottom'], dtype=np.float64)

    state = torch.load(
        os.path.join(instance_folder, 'graph.pkl'), map_location='cpu', weights_only=True
    )
    quat_key = f'M_quat_{main_stem}'
    if quat_key not in state:
        raise KeyError(f'Missing {quat_key} in {instance_folder}/graph.pkl.')
    rotation = quaternion_to_matrix(torch.as_tensor(state[quat_key])).cpu().numpy()

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ bottom
    return transform


def decode_params(data_folder:str, sample_name:str, species:str='soybean', return_mesh:bool=False,
                  align_global:bool=True, instance_folder=None, **kwargs):

    if instance_folder is None:
        instance_folder = os.path.join(data_folder, species, 'instances', sample_name)

    # load class annotation
    classes = load_class(os.path.join(instance_folder, 'info','class.txt'))

    # load parent annotation
    parents = load_parent(os.path.join(instance_folder, 'info','parent.txt'))

    # load PCA weights for stem and leaf
    pca_stem_3d = NodePCA(path=os.path.join(data_folder, species, '3d_stem_pca.pth'))
    pca_leaf_3d = NodePCA(path=os.path.join(data_folder, species, '3d_leaf_pca.pth'))
    pca_leaf_2d = NodePCA(path=os.path.join(data_folder, species, '2d_leaf_pca.pth'))

    # init plant graph
    plant_graph = PlantGraphFixedTopology(
        classes=classes, parents=parents, species=species,
        pca_leaf_3d=pca_leaf_3d, pca_stem_3d=pca_stem_3d, pca_leaf_2d=pca_leaf_2d
    )
    plant_graph.load(os.path.join(instance_folder, 'graph.pkl'))
    plant_graph.to(get_device())

    # draw graph structure
    if kwargs.get('draw_graph', False):
        plant_graph.draw_topology()

    with torch.no_grad():
        # align the plant to global X-axis to make it stand straight
        mesh = plant_graph.generate(output_format='mesh', color='gray', align_global=align_global)

    if return_mesh:
        return mesh

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    axis.translate([0, 0, 0])
    o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)

if __name__ == "__main__":

    # example usage 1
    # use argparse to parse the command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='sample_params', help='path to the folder containing the sample parameters')
    parser.add_argument('--sample_name', type=str, default=None, help='name of the sample to process')
    parser.add_argument('--species', type=str, default='soybean', help='species of the plant')
    parser.add_argument('--draw_graph', action='store_true', help='whether to draw the graph structure')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='device to use (auto, cpu, or cuda)')
    args = parser.parse_args()

    # Set device before any model loading
    set_device(args.device)
    print(f'Using device: {get_device()}')

    data_folder = args.data_folder
    sample_name = args.sample_name
    species = args.species

    if sample_name is not None:
        decode_params(data_folder, sample_name, species)
        exit(0)

    # example usage 2
    data_folder = 'sample_params'
    decode_params(data_folder, '24_o', 'soybean') # '3_o', '3_i', '4_o', '4_i', '6_o',  '8_i', '24_o', '101_o' etc.
    decode_params(data_folder, '1', 'tobacco')
    decode_params(data_folder, '02', 'rose')
    decode_params(data_folder, '10008da', 'maize')
    decode_params(data_folder, '08', 'ribes')
