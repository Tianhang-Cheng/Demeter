import numpy as np
import sys
from utils import *
from maize_constants import new_leaf, new_stem
import scipy.io as sio

#, leaf_Mask
#hs: an array of height of leaves
#dS: diameter of stem
#Nt: Number of corn leaves
#LA: an ararry of leaf areas
#Azi: an array of azimuth angles of corn leaves
#leafAspectRatio: the ratio of leaf length and leaf width
#leaf_angle: leaf bending angle
def maize_plant(hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle, separate=False):
# def maize_plant(x, separate=False):

    # hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle = x

    #all_xyz, all_fac = [], []
    #lmax = 1.15
    #hNt = 2.5

    height = hs[-1]

    #Li = np.sqrt(LA / 0.089)
    Li = np.sqrt(LA/0.75/leafAspectRatio)
    Wi = Li*leafAspectRatio

    leaf_xyz, leaf_fac = [], []
    #Azi = 0
    #area = 0
    PlantHeight = 0
    for i in range(Nt):
        # if leaf_Mask[i]:
        #     continue
        xyz, fac = new_leaf(order = leaf_angle[i], length = Li[i], width = Wi[i]) #Li[i]*0.119
        xyz[:, -1] += hs[i]
        if i == Nt-1:
            PlantHeight = np.max(xyz[:, -1])
        xyz = rotate2d(xyz, Azi[i], dims=[0, 1])

        #xyz = rotate2d(xyz, 90, dims=[1, 2])
        #Azi = (Azi + 180) % 360

        #vis_mesh(xyz, fac)
        leaf_xyz.append(xyz)
        leaf_fac.append(fac)
        #area += vis_mesh(xyz, fac)

    #print(area)
    if Nt > 7:
        stem_xyz, stem_fac = new_stem(hs[6], Nt, height, dS)
    else:
        stem_xyz, stem_fac = new_stem(0, Nt, height, dS)
    #stem_xyz = rotate2d(stem_xyz, 90, dims=[1, 2])

    #all_xyz, all_fac = wirteObj(leaf_xyz + [stem_xyz], leaf_fac + [stem_fac])  #
    #all_xyz = wirteObj(leaf_xyz + [stem_xyz], leaf_fac + [stem_fac])

    #all_xyz, all_fac = agg(leaf_xyz + [stem_xyz], leaf_fac + [stem_fac])  #
    #visualize_mesh(all_xyz, all_fac)
    #return all_xyz, all_fac
    #return all_xyz
    if separate:
        return leaf_xyz, stem_xyz, leaf_fac, stem_fac, PlantHeight
    else:
        return agg(leaf_xyz + [stem_xyz], leaf_fac + [stem_fac])