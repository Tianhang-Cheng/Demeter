import numpy as np
from utils import *
import math

Points = [0, -2.45,
          1, -3.25,
          2, -3.825,
          3, -4.14,
          4, -4.27,
          5, -4.11,
          6, -3.74,
          7, -3.13,
          8, -2.295,
          9, -1.241,
          10, 0,
          0, 2.45,
          1, 3.25,
          2, 3.825,
          3, 4.14,
          4, 4.27,
          5, 4.11,
          6, 3.74,
          7, 3.13,
          8, 2.295,
          9, 1.241,]

Facets = [1, 2, 12,
          2, 12, 13,
          2, 13, 3,
          3, 13, 14,
          3, 14, 4,
          4, 14, 15,
          4, 15, 5,
          5, 15, 16,
          5, 16, 6,
          6, 16, 17,
          6, 17, 7,
          7, 17, 18,
          7, 18, 8,
          8, 18, 19,
          8, 19, 9,
          9, 19, 20,
          9, 20, 10,
          10, 20, 21,
          10, 21, 11,]

Points1 = np.array([0,1,2,3,4,5,6,7,8,9,10,0,1,2,3,4,5,6,7,8,9])

# leaf_length = 2
# single_leaf_xyz[:,0] = single_leaf_xyz[:,0] * leaf_length
# single_leaf_xyz[:,1] = single_leaf_xyz[:,1] * 0.119 * leaf_length
# mesh = vis_mesh(single_leaf_xyz, single_leaf_facets)

# newPoints = [0, 0,
#           0, 1,
#           1, 0,
#           1, 1]
#
# newFacets = [1,2,3,
#              2,3,4]
#
# single_leaf_facets = np.zeros((10, 3))
# for i in range(10):
#     single_leaf_facets[i, 0] = i
#     single_leaf_facets[i, 1] = i + 1
#     single_leaf_facets[i, 2] = i + 2
#
# def new_leaf1(order = 1, length = 1, width = 0.119, theta = 160, mesh = 5):
#     arca = -0.0646 * order ** 2 + 0.7276 * order - 3.5258
#     scale = 1.085 * np.exp(-0.0906 * order)
#     length_t = length / 11.0
#
#     for i in range(10):
#
#         single_leaf_xyz = np.zeros((12, 3))
#         single_leaf_xyz[::2, 0] = i * length_t * scale
#         single_leaf_xyz[1::2, 0] = (i+1) * length_t * scale
#         single_leaf_xyz[:, -1] += arca * (single_leaf_xyz[:, 0] ** 2) - arca * single_leaf_xyz[:, 0]
#
#         width_t1 = width / 2 / 3.575 * (-10 * ((i * 0.1) ** 2) + 8 * i * 0.1 + 2)
#         width_t2 = width / 2 / 3.575 * (-10 * (((i + 1) * 0.1) ** 2) + 8 * (i + 1) * 0.1 + 2)
#
#         r_t1 = width_t1 / 10 / np.sin(theta / mesh * np.pi / 180 / 2)
#         r_t2 = width_t2 / 10 / np.sin(theta / mesh * np.pi / 180 / 2)
#         for j in range(mesh+1):
#             theta_t = j * theta / mesh * np.pi / 180
#             k = -1
#
#             if theta_t > np.pi/2:
#                 theta_t = np.pi - theta_t
#                 k = 1
#
#             single_leaf_xyz[j * 2, 1] = r_t1 * np.sin(theta_t)
#             single_leaf_xyz[j * 2, -1] += r_t1 * (1 + k*np.cos(theta_t))
#
#             single_leaf_xyz[j * 2 + 1, 1] = r_t2 * np.sin(theta_t)
#             single_leaf_xyz[j * 2 + 1, -1] += r_t2 * (1 + k*np.cos(theta_t))
#
#         vis_mesh(single_leaf_xyz, single_leaf_facets)
#         x = 0
#
#
#     return single_leaf_xyz

# xyz = new_leaf1()

def create_transform(A1, A2, B1, B2):
    # Calculate scaling factor
    scale = np.linalg.norm(B2-B1) / np.linalg.norm(A2-A1)

    # Calculate rotation angle
    theta = np.arctan2(B2[1]-B1[1], B2[0]-B1[0]) - np.arctan2(A2[1]-A1[1], A2[0]-A1[0])

    # Create transformation matrix
    transform_matrix = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta), B1[0] - scale * (np.cos(theta)*A1[0] - np.sin(theta)*A1[1])],
        [scale * np.sin(theta),  scale * np.cos(theta), B1[1] - scale * (np.sin(theta)*A1[0] + np.cos(theta)*A1[1])],
        [0, 0, 1]
    ])

    return transform_matrix


def transform_point(point, transform_matrix):
    # Create a 3D vector with the third component as 1
    point = np.array([point[0], point[1], 1])

    # Apply the transformation matrix
    transformed_point = np.dot(transform_matrix, point)

    # We only return the first two elements as the result (x and y coordinates)
    return transformed_point[:2]

def transform_points(points, transform_matrix):
    # Apply the transformation matrix to each point
    transformed_points = [transform_point(point, transform_matrix) for point in points]
    return transformed_points

#wave-form
def new_leaf2(meshL = 20, meshW = 10, theta = 100, order = 10, length = 1, width = 0.119):

    #width = 0.119*length

    single_leaf_xyz = np.zeros(((meshL+1)*(meshW+1), 3))
    lengthT = length/meshL
    #widthT = 1/meshW
    for i in range(21):
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1), 0] = i * lengthT
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1), 1] = (np.arange(0, 1.1, 1/meshW) - 0.5) * width / 3.575 * (-10 * ((i * 1/meshL) ** 2) + 8 * i * 1/meshL + 2)

    single_leaf_facets = np.zeros((2*meshL*meshW, 3), dtype=np.uint64)
    for i in range(meshL):
        for j in range(meshW):
            single_leaf_facets[i * meshW*2 + j*2, 0] = i * (meshW+1) + j
            single_leaf_facets[i * meshW*2 + j*2, 1] = i * (meshW+1) + j + 1
            single_leaf_facets[i * meshW*2 + j*2, 2] = i * (meshW+1) + j + 11
            single_leaf_facets[i * meshW*2 + j * 2+1, 0] = i * (meshW+1) + j + 1
            single_leaf_facets[i * meshW*2 + j * 2+1, 1] = i * (meshW+1) + j + 11
            single_leaf_facets[i * meshW*2 + j * 2+1, 2] = i * (meshW+1) + j + 12
    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    #bending perpendicular to the midrib
    for i in range(meshL+1):
        widthT = width / 2 / 3.575 * (-10 * ((i * 1/meshL) ** 2) + 8 * i * 1/meshL + 2)
        half = int(meshW/2)

        thetaT = theta * np.exp(-2 * i / meshL) * np.pi / 180 * np.sin(i*30*np.pi/90 + np.pi/60)

        if thetaT != 0:
            r = widthT / meshW / np.sin(thetaT / meshW)
            for j in range(half + 1):
                theta_t = (meshW/2 - j) * thetaT * 2 / meshW
                k = -1
                if theta_t > np.pi/2:
                    theta_t = np.pi - theta_t
                    k = 1

                single_leaf_xyz[i*(meshW+1) + j, 1] = r * np.sin(theta_t)
                single_leaf_xyz[i*(meshW+1) + j, -1] = r * (1 + k*np.cos(theta_t))

            single_leaf_xyz[i*(meshW+1)+half+1:i*(meshW+1)+meshW+1, 1] = np.flip(-1*single_leaf_xyz[i*(meshW+1):i*(meshW+1)+half, 1])
            single_leaf_xyz[i*(meshW+1)+half+1:i*(meshW+1)+meshW+1, -1] = np.flip(single_leaf_xyz[i*(meshW+1):i*(meshW+1)+half, -1])

    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    #bending along the midrib
    arca = -0.0646 * order ** 2 + 0.7276 * order - 3.5258
    scale = 1.085 * np.exp(-0.0906 * order)
    x = np.arange(0,meshL+1) * 1/meshL * scale
    z = arca * (x ** 2) - arca * x

    xLen = x[1:] - x[0:-1]
    zLen = z[1:] - z[0:-1]
    slopeNorm = np.sqrt(xLen ** 2 + zLen ** 2)
    xLen = xLen / slopeNorm * length / meshL
    zLen = zLen / slopeNorm * length / meshL

    newX = np.array([np.sum(xLen[:i]) for i in range(meshL + 1)])
    newZ = np.array([np.sum(zLen[:i]) for i in range(meshL + 1)])

    for i in range(meshL):
        transform_matrix = create_transform(single_leaf_xyz[i*(meshW+1)+5,[0,-1]], single_leaf_xyz[(i+1)*(meshW+1)+5,[0,-1]], np.array([newX[i],newZ[i]]), np.array([newX[i+1],newZ[i+1]]))
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1),[0,-1]] = np.array(transform_points(single_leaf_xyz[i*(meshW+1):(i + 1) * (meshW + 1),[0,-1]], transform_matrix))

    single_leaf_xyz[meshL*(meshW+1):, 0] = newX[-1]
    single_leaf_xyz[meshL*(meshW + 1):, -1] = newZ[-1]

    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    return single_leaf_xyz, single_leaf_facets

#non wave-form
def new_leaf1(meshL = 20, meshW = 10, theta = 160, order = 10, length = 1, width = 0.119):

    #width = 0.119*length

    single_leaf_xyz = np.zeros(((meshL+1)*(meshW+1), 3))
    lengthT = length/meshL
    #widthT = 1/meshW
    for i in range(21):
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1), 0] = i * lengthT
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1), 1] = (np.arange(0, 1.1, 1/meshW) - 0.5) * width / 3.575 * (-10 * ((i * 1/meshL) ** 2) + 8 * i * 1/meshL + 2)

    single_leaf_facets = np.zeros((2*meshL*meshW, 3), dtype=np.uint64)
    for i in range(meshL):
        for j in range(meshW):
            single_leaf_facets[i * meshW*2 + j*2, 0] = i * (meshW+1) + j
            single_leaf_facets[i * meshW*2 + j*2, 1] = i * (meshW+1) + j + 1
            single_leaf_facets[i * meshW*2 + j*2, 2] = i * (meshW+1) + j + 11
            single_leaf_facets[i * meshW*2 + j * 2+1, 0] = i * (meshW+1) + j + 1
            single_leaf_facets[i * meshW*2 + j * 2+1, 1] = i * (meshW+1) + j + 11
            single_leaf_facets[i * meshW*2 + j * 2+1, 2] = i * (meshW+1) + j + 12
    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    #bending perpendicular to the midrib
    if theta > 0:
        for i in range(meshL+1):
            widthT = width / 2 / 3.575 * (-10 * ((i * 1/meshL) ** 2) + 8 * i * 1/meshL + 2)

            thetaT = theta * np.exp(-2*i/meshL) * np.pi / 180 # * np.sin(i * np.pi / 60)
            r = widthT / meshW / np.sin(thetaT / meshW)

            half = int(meshW/2)
            for j in range(half + 1):
                theta_t = (meshW/2 - j) * thetaT * 2 / meshW
                k = -1
                if theta_t > np.pi/2:
                    theta_t = np.pi - theta_t
                    k = 1

                single_leaf_xyz[i*(meshW+1) + j, 1] = r * np.sin(theta_t)
                single_leaf_xyz[i*(meshW+1) + j, -1] = r * (1 + k*np.cos(theta_t))

            single_leaf_xyz[i*(meshW+1)+half+1:i*(meshW+1)+meshW+1, 1] = np.flip(-1*single_leaf_xyz[i*(meshW+1):i*(meshW+1)+half, 1])
            single_leaf_xyz[i*(meshW+1)+half+1:i*(meshW+1)+meshW+1, -1] = np.flip(single_leaf_xyz[i*(meshW+1):i*(meshW+1)+half, -1])

    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    #bending along the midrib
    arca = -0.0646 * order ** 2 + 0.7276 * order - 3.5258
    scale = 1.085 * np.exp(-0.0906 * order)
    x = np.arange(0,meshL+1) * 1/meshL * scale
    z = arca * (x ** 2) - arca * x

    xLen = x[1:] - x[0:-1]
    zLen = z[1:] - z[0:-1]
    slopeNorm = np.sqrt(xLen ** 2 + zLen ** 2)
    xLen = xLen / slopeNorm * length / meshL
    zLen = zLen / slopeNorm * length / meshL

    newX = np.array([np.sum(xLen[:i]) for i in range(meshL + 1)])
    newZ = np.array([np.sum(zLen[:i]) for i in range(meshL + 1)])

    for i in range(meshL):
        transform_matrix = create_transform(single_leaf_xyz[i*(meshW+1)+5,[0,-1]], single_leaf_xyz[(i+1)*(meshW+1)+5,[0,-1]], np.array([newX[i],newZ[i]]), np.array([newX[i+1],newZ[i+1]]))
        single_leaf_xyz[i*(meshW+1):(i+1)*(meshW+1),[0,-1]] = np.array(transform_points(single_leaf_xyz[i*(meshW+1):(i + 1) * (meshW + 1),[0,-1]], transform_matrix))

    single_leaf_xyz[meshL*(meshW+1):, 0] = newX[-1]
    single_leaf_xyz[meshL*(meshW + 1):, -1] = newZ[-1]

    #vis_mesh(single_leaf_xyz, single_leaf_facets)

    return single_leaf_xyz, single_leaf_facets

#xyz = new_leaf2(order = 5)

#simple leaf
def new_leaf(order = 10, length = 1, width = 0.119):
    single_leaf_xyz = []
    # for i in range(0, len(Points), 2):
    #     single_leaf_xyz.append([Points[i] / 10, Points[i + 1] / 8.54, 0])
    lP = Points1/10
    wP = (-11.62*(lP**2) + 9.125*lP + 2.4786) / 8.4
    wP[11:] *= -1
    for i in range(0, len(Points1)):
        single_leaf_xyz.append([lP[i], wP[i], 0])
    single_leaf_facets = []
    for i in range(0, len(Facets), 3):
        single_leaf_facets.append([Facets[i] - 1, Facets[i + 1] - 1, Facets[i + 2] - 1])

    single_leaf_xyz = np.array(single_leaf_xyz)
    single_leaf_facets = np.array(single_leaf_facets, dtype=np.uint64)

    #single_leaf_xyz[:, 0] *= length
    #single_leaf_xyz[:, 1] *= width

    arca = -0.0646 * order**2 + 0.7276*order - 3.5258
    scale = 1.085*np.exp(-0.0906*order)
    single_leaf_xyz[:, 0] = single_leaf_xyz[:,0] * scale
    single_leaf_xyz[:,-1] = arca*(single_leaf_xyz[:,0]**2) - arca*single_leaf_xyz[:,0]

    #compute new corrdinates
    xLen = single_leaf_xyz[1:11, 0] - single_leaf_xyz[0:10, 0]
    zLen = single_leaf_xyz[1:11, -1] - single_leaf_xyz[0:10, -1]
    slopeNorm = np.sqrt(xLen**2 + zLen**2)
    xLen = xLen / slopeNorm * length / 10.0
    zLen = zLen / slopeNorm * length / 10.0

    single_leaf_xyz[:, 0] = 0
    single_leaf_xyz[:, -1] = 0
    newX = np.array([np.sum(xLen[:i]) for i in range(11)])
    newZ = np.array([np.sum(zLen[:i]) for i in range(11)])

    single_leaf_xyz[0:11, 0] = newX
    single_leaf_xyz[11:, 0] = newX[:-1]

    single_leaf_xyz[0:11, -1] = newZ
    single_leaf_xyz[11:, -1] = newZ[:-1]

    #width = length * 0.119
    single_leaf_xyz[:, 1] = single_leaf_xyz[:, 1] * width
    #vis_mesh(single_leaf_xyz, single_leaf_facets)
    return single_leaf_xyz, single_leaf_facets

# Azi = 0
# for i in range(19):
#     xyz, fac = new_leaf(1, i+1, (i+1)*0.119)
#     xyz = rotate2d(xyz, Azi, dims=[0, 1])
#     xyz = rotate2d(xyz, 90, dims=[1, 2])
#     mesh = vis_mesh(xyz, fac)
#     Azi = (Azi + 180) % 360

stemPoints = [
    -1/2, np.sqrt(3)/2,
    -1, 0,
    -1/2, -np.sqrt(3)/2,
    1/2, -np.sqrt(3)/2,
    1, 0,
    1/2, np.sqrt(3)/2,
]

stemFacets = [
    1, 2, 7,
    2, 7, 8,
    2, 3, 8,
    3, 8, 9,
    3, 4, 9,
    4, 9, 10,
    4, 5, 10,
    5, 10, 11,
    5, 6, 11,
    6, 11, 12,
    6, 12, 1,
    1, 12, 7
]

def new_stem(h7, Nt = 10, height = 2.5, diameter = 0.025):
    stem_xyz = []
    for i in range(0, len(stemPoints), 2):
        stem_xyz.append([stemPoints[i] / 2, stemPoints[i + 1] / 2, 0])
    for i in range(0, len(stemPoints), 2):
        stem_xyz.append([stemPoints[i] / 2, stemPoints[i + 1] / 2, 1])

    stem_facets = []
    for i in range(0, len(stemFacets), 3):
        stem_facets.append([stemFacets[i] - 1, stemFacets[i + 1] - 1, stemFacets[i + 2] - 1])

    if Nt > 7:
        stem_topPoints = np.array(stemPoints)*(1-0.17/6*Nt)
        stem_topFacets = np.array(stemFacets) + 6

        for i in range(0, len(stem_topPoints), 2):
            stem_xyz.append([stem_topPoints[i] / 2, stem_topPoints[i + 1] / 2, height])

        for i in range(0, len(stem_topFacets), 3):
            stem_facets.append([stem_topFacets[i] - 1, stem_topFacets[i + 1] - 1, stem_topFacets[i + 2] - 1])

        height = h7

    stem_xyz = np.array(stem_xyz)
    stem_xyz[:,0:2] = stem_xyz[:,0:2] * diameter
    stem_xyz[6:12, -1] = stem_xyz[6:12, -1] * height

    stem_facets = np.array(stem_facets, dtype=np.uint16)

    return stem_xyz, stem_facets

# xyz, fac = new_stem()
# xyz = rotate2d(xyz, 90, dims=[1, 2])
# mesh = vis_mesh(xyz, fac)