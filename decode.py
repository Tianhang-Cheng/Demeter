import os
import numpy as np
import open3d as o3d
import torch
from utils.constant import *
from representation.graph import PlantGraphFixedTopology
from utils.pca import NodePCA
from utils.graph import load_parent, load_class
    
def infer(data_folder:str, sample_name:str, species:str='soybean', **kwargs):

    instance_folder = os.path.join(data_folder, species, sample_name)

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
    plant_graph.cuda()

    # draw graph structure
    plant_graph.draw_topology()

    with torch.no_grad():
        # align the plant to global X-axis to make it stand straight
        mesh = plant_graph.generate(output_format='mesh', color='gray', align_global=True)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    axis.translate([0, 0, 0])
    o3d.visualization.draw_geometries([mesh, axis], mesh_show_back_face=True)

if __name__ == "__main__":

    data_folder = 'sample_data'

    infer(data_folder, '24_o', 'soybean')

    # sample_names = ['3_o', '3_i', '4_o', '4_i', '6_o',  '8_i', '24_o', '101_o']
    # for sample_name in sample_names:
    #     print(f'Processing {sample_name} ...')
    #     infer(data_folder, sample_name, 'soybean')
    #     break
