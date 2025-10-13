import numpy as np
import sys
from utils import *
from constants import new_leaf, new_petiole


def soybean_leaf(L, W, ang):
    """
    Returns XYZ points and faces (index triplets) for a leaf. Angle allows for XY-rotation (z-axis).
    """
    xyz, fac = new_leaf()
    xyz[:, 0] *= L
    xyz[:, 1] *= W
    xyz = rotate2d(xyz, ang, dims=[0, 1])
    return xyz, fac

def soybean_petiole(L, W):
    """
    Returns XYZ points and faces (index triplets) for a petiole.
    """
    xyz, fac = new_petiole()
    xyz[:, 2] *= L
    xyz[:, :2] *= W
    xyz = rotate2d(xyz, 90, dims=[0, 2])
    return xyz, fac

def soybean_node(lat1L, lat1W, lat1ang, 
                 lat2L, lat2W, lat2ang, 
                 midL, midW, midang, 
                 pet1len, pet1ang, 
                 pet2len, pet2ang):
    """
    Returns XYZ points and faces (index triplets) for a node, which contains two petioles and a leaf trio.
    
    lat1L, lat1W, lat1ang: length, width, and z-angle of first lateral leaf (same for lat2 and mid)
    pet1len: length of parent petiole (petiole1)
    pet2len: length of petiole with middle leaf
    pet1ang: angle between petiole1 and parent stem
    pet2ang: angle between petiole2 and petiole1
    """
    mid_xyz, mid_fac = soybean_leaf(midL, midW, midang)
    mid_xyz[:, 0] += pet2len
    
    lat1_xyz, lat1_fac = soybean_leaf(lat1L, lat1W, lat1ang)
    lat2_xyz, lat2_fac = soybean_leaf(lat2L, lat2W, -lat2ang)  # rotate the other way from lat1
    
    pet2_xyz, pet2_fac = soybean_petiole(pet2len, 0.2)
    pet1_xyz, pet1_fac = soybean_petiole(pet1len, 0.3)
    
    all_xyz, all_fac = agg([mid_xyz, lat1_xyz, lat2_xyz, pet2_xyz], [mid_fac, lat1_fac, lat2_fac, pet2_fac])
    all_xyz = rotate2d(all_xyz, pet2ang, dims=[0, 2])
    all_xyz[:, 0] += pet1len
    all_xyz, all_fac = agg([all_xyz, pet1_xyz], [all_fac, pet1_fac])
    all_xyz = rotate2d(all_xyz, pet1ang, dims=[0, 2])
    
    return all_xyz, all_fac

def soybean_stem(stem_info):
    """
    Returns XYZ points and faces (index triplets) for a stem, which contains multiple nodes.
    """
    total_height = 0
    node_xyzs, node_facs = [], []
    for node_data in stem_info:
        node_xyz, node_fac = soybean_node(**node_data['node_info'])
        node_xyz = rotate2d(node_xyz, node_data['node_rotation'], dims=[1, 2])
        node_xyz = rotate2d(node_xyz, node_data['node_ang'], dims=[0, 1])
        total_height += node_data['node_height']
        node_xyz[:, 0] += total_height
        node_xyzs.append(node_xyz)
        node_facs.append(node_fac)
        
    stem_xyz, stem_fac = soybean_petiole(total_height, 0.5)
    all_xyz, all_fac = agg(node_xyzs + [stem_xyz], node_facs + [stem_fac])
    return all_xyz, all_fac

def soybean_plant(main_stem_info, branches_info):
    """
    Returns XYZ points and faces (index triplets) for a full plant, which contains a main stem and branches.
    """
    branch_xyzs, branch_facs = [], []
    for branch_data in branches_info:
        branch_xyz, branch_fac = soybean_stem(branch_data['branch_info'])
        branch_xyz = rotate2d(branch_xyz, branch_data['branch_ang'], dims=[0, 2])
        branch_xyz = rotate2d(branch_xyz, branch_data['branch_rotation'], dims=[1, 2])
        branch_xyz[:, 0] += branch_data['branch_height']
        branch_xyzs.append(branch_xyz)
        branch_facs.append(branch_fac)
        
    main_stem_xyz, main_stem_fac = soybean_stem(main_stem_info)
    all_xyz, all_fac = agg(branch_xyzs + [main_stem_xyz], branch_facs + [main_stem_fac])
    return all_xyz, all_fac


if __name__ == '__main__':
    # Example parameters
    LEAF_L = 3.985
    LEAF_W = 4
    PET1_ANG = 20.61
    PET2_ANG = 20
    PET1_LEN = 6
    PET2_LEN = 2
    BRANCH_NODE_ANG = 60
    BRANCH_ANG = 50
    NODE_DIST = 6.949

    default_node_info = {'lat1L': LEAF_L, 'lat1W': LEAF_W, 'lat1ang': 90, 
                         'lat2L': LEAF_L, 'lat2W': LEAF_W, 'lat2ang': 90, 
                         'midL': LEAF_L, 'midW': LEAF_W, 'midang': 0,
                         'pet1len': PET1_LEN, 'pet1ang': PET1_ANG,
                         'pet2len': PET2_LEN, 'pet2ang': PET2_ANG}
    default_branch_info = [{'node_info': default_node_info, 
                            'node_rotation': 0,
                            'node_ang': BRANCH_NODE_ANG,
                            'node_height': NODE_DIST},
                           {'node_info': default_node_info, 
                            'node_rotation': 0, 
                            'node_ang': -BRANCH_NODE_ANG,
                            'node_height': NODE_DIST} ,
                           {'node_info': default_node_info, 
                            'node_rotation': 0, 
                            'node_ang': 0,
                            'node_height': NODE_DIST}]
    branches_info = [{'branch_info': default_branch_info, 
                      'branch_rotation': i * 90, 
                      'branch_height': NODE_DIST * (i + 1), 
                      'branch_ang': BRANCH_ANG} 
                     for i in range(4)]
    
    main_stem_info = default_branch_info = [{'node_info': default_node_info, 'node_rotation': i * 150, 'node_ang': 0, 'node_height': NODE_DIST} for i in range(8)]
    xyz, fac = soybean_plant(main_stem_info, branches_info)

    mesh = vis_mesh(xyz, fac)
    