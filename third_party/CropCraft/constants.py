import numpy as np

Points = [0,0,
    1,0,
    2,0,
    3.1547,0,
    4.3094,0,
    5,0,   ## 5.4641
    0.2588,1,
    1.5,1,
    2.5774,1,
    3.7312,1.2, ##
    1.2,1.78,
    2.3,2.,
    0.2588,-1,
    1.5,-1,
    2.5774,-1,
    3.7312,-1.2, ## -1
    1.2,-1.78, ## 1.1056, -1.7889
    2.3,-2.]  ## 2, -2
single_leaf_xyz = []
for i in range(0, len(Points), 2):
    single_leaf_xyz.append([Points[i]/5, Points[i + 1]/4., 0])

# Points_New = [[ 0.        , 0.        , 0.        ],
#  [ 0.0473637 , 0.25      , 0.        ],
#  [ 0.18301275, 0.        , 0.        ],
#  [ 0.27451915, 0.25      , 0.        ],
#  [ 0.36602551, 0.        , 0.        ],
#  [ 0.47169709, 0.25      , 0.        ],
#  [ 0.57735032, 0.        , 0.        ],
#  [ 0.68285722, 0.30      , 0.        ],
#  [ 0.78867519, 0.        , 0.        ],
#  [ 1.        , 0.        , 0.        ],
#  [ 0.2023389 , 0.447225  , 0.        ],
#  [ 0.36602551, 0.5       , 0.        ],
#  [ 0.0473637 ,-0.25      , 0.        ],
#  [ 0.27451915,-0.25      , 0.        ],
#  [ 0.47169709,-0.25      , 0.        ],
#  [ 0.68285722,-0.30      , 0.        ],
#  [ 0.2023389 ,-0.447225  , 0.        ],
#  [ 0.36602551,-0.48562497, 0.        ]]
# single_leaf_xyz = Points_New


Facets = [1,7,2,
    2,7,8,
    2,8,3,
    3,8,9,
    3,9,4,
    4,9,10,
    4,10,5,
    5,10,6,
    7,11,8,
    8,11,12,
    8,12,9,
    9,12,10,
    13,1,2,
    13,2,14,
    14,2,3,
    14,3,15,
    15,3,4,
    15,4,16,
    16,4,5,
    16,5,6,
    17,13,14,
    17,14,18,
    18,14,15,
    18,15,16]
single_leaf_facets = []
for i in range(0, len(Facets), 3):
    single_leaf_facets.append([Facets[i] - 1, Facets[i + 1] - 1, Facets[i + 2] - 1])
    
def new_leaf():
    return np.array(single_leaf_xyz), np.array(single_leaf_facets, dtype=np.uint32)


petiolePoints = [
    0.25,   0,
    0.125,  0.2165,
    -0.125, 0.2165,
    -0.25,  0,
    -0.125, -0.2165,
    0.125,  -0.2165,
]
petiole_xyz = []
for i in range(0, len(petiolePoints), 2):
    petiole_xyz.append([petiolePoints[i] * 2, petiolePoints[i + 1] * 2, 0])
for i in range(0, len(petiolePoints), 2):
    petiole_xyz.append([petiolePoints[i] * 2, petiolePoints[i + 1] * 2, 1])

    
petioleFacets = [
    1, 2, 8,
    8 ,7, 1,
    2 ,3, 9,
    9 ,8, 2,
    3 ,4, 10,
    10 ,9, 3,
    4 ,5, 11,
    11 ,10, 4,
    5 ,6, 12,
    12 ,11, 5,
    6 ,1, 7,
    7 ,12, 6
]

petiole_facets = []
for i in range(0, len(petioleFacets), 3):
    petiole_facets.append([petioleFacets[i] - 1, petioleFacets[i + 1] - 1, petioleFacets[i + 2] - 1])
    
def new_petiole():
    return np.array(petiole_xyz), np.array(petiole_facets, dtype=np.uint16)