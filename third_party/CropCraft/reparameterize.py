import numpy as np

DUMMY_NODE = {'node_info': {'lat1L': 0, 'lat1W': 0, 'lat1ang': 90, 
                            'lat2L': 0, 'lat2W': 0, 'lat2ang': 90, 
                            'midL': 0, 'midW': 0, 'midang': 0,
                            'pet1len': 0, 'pet1ang': 0,
                            'pet2len': 0, 'pet2ang': 0}, 
            'node_rotation': np.random.normal(loc=0, scale=45), 
            'node_ang': 0, 
            'node_height': 0}


# consider adding per-plant seed
def soybean_info_parametric(params, with_dummies=False):
    """Returns soybean plant info with custom parameterization."""
    
    # This is where fixed ratios between parameters are hard-coded
    # from M_stru_mean.txt
    BASE_PET1_ANG = np.array([59.0, 51.0, 49.0, 43.0, 37.0, 35.0, 33.0, 30.0, 28.0, 26.0, 24.0, 22.0, 19.0, 17.0, 16.0, 15.0])
    BASE_PET1_LEN = np.array([100.0, 125.0, 150.0, 180.0, 200.0, 220.0, 230.0, 220.0, 200.0, 180.0, 150.0, 125.0, 100.0, 75.0, 50.0, 25.0]) / 10
    BASE_NODE_DIST = np.array([22.8, 24.05, 30.15, 42.65, 48.35, 52.4, 61.1, 61.85, 75.95, 71.4, 72.32, 64.05, 51.83, 42.17, 39.5, 22.75]) / 10
    BASE_LEAF_L = np.array([51.82, 70.91, 77.24, 94.83, 87.5, 87.47, 93.9, 101.21, 108.53, 108.34, 98.2, 90.11, 84.63, 69.86, 58.42, 58.42]) / 10
    # from M_branch_probability.txt
    BRANCH_NODE_COUNT_MEANS = [0.25, 1.3, 1.0, 0.55, 0.8, 0.5]
    
    LEAF_W_TO_L = params['LEAF_W_TO_L']
    # LEAF_L = np.random.normal(loc=params['LEAF_L'], scale=params['LEAF_L_STD'])
    # LEAF_W = LEAF_L * LEAF_W_TO_L
    LEAF_L_SCALE = params['LEAF_L_SCALE']
    LEAF_W_SCALE = params['LEAF_L_SCALE'] * LEAF_W_TO_L
    
    PET_ANG_SCALE = params['PET_ANG_SCALE']
    PET2_ANG = 5.
    
    BRANCH_NODE_ANG = 60. 
    BRANCH_ANG = 50. 
    
    NODE_DIST_SCALE = params['NODE_DIST_SCALE']
    
    PET_LEN_SCALE = params['PET_LEN_SCALE']
    PET2_LEN = 3.4
    
    NODE_FRAC = params['NODE_COUNT'] - int(params['NODE_COUNT'])
    NODE_COUNT = int(params['NODE_COUNT'])
    BRANCH_COUNT = min(max(0, params['NODE_COUNT'] - 7), 6)  # branches start growing at node 8
    BRANCH_PET1_LEN = 12.
    BRANCH_PET1_ANG = 35.
    BRANCH_NODE_DIST = 8.
    BRANCH_LEAF_L = 8.  # base
    
    BASE_PET1_ANG = BASE_PET1_ANG[len(BASE_PET1_ANG) - NODE_COUNT - 1:]
    BASE_PET1_ANG += np.random.normal(loc=0, scale=params['PET1_ANG_STD'], size=BASE_PET1_ANG.shape)
    BASE_PET1_LEN = BASE_PET1_LEN[len(BASE_PET1_LEN) - NODE_COUNT - 1:]
    BASE_NODE_DIST = BASE_NODE_DIST[len(BASE_NODE_DIST) - NODE_COUNT - 1:]
    BASE_LEAF_L = BASE_LEAF_L[len(BASE_LEAF_L) - NODE_COUNT - 1:]
    
    main_stem_info = [{'node_info': {'lat1L': BASE_LEAF_L[i] * LEAF_L_SCALE, 'lat1W': BASE_LEAF_L[i] * LEAF_W_SCALE, 'lat1ang': 90, 
                                     'lat2L': BASE_LEAF_L[i] * LEAF_L_SCALE, 'lat2W': BASE_LEAF_L[i] * LEAF_W_SCALE, 'lat2ang': 90, 
                                     'midL': BASE_LEAF_L[i] * LEAF_L_SCALE, 'midW': BASE_LEAF_L[i] * LEAF_W_SCALE, 'midang': 0,
                                     'pet1len': BASE_PET1_LEN[i] * PET_LEN_SCALE, 'pet1ang': BASE_PET1_ANG[i] * PET_ANG_SCALE,
                                     'pet2len': PET2_LEN * PET_LEN_SCALE, 'pet2ang': PET2_ANG * PET_ANG_SCALE}, 
                       'node_rotation': i * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                       'node_ang': 0, 
                       'node_height': BASE_NODE_DIST[i] * NODE_DIST_SCALE} 
                      for i in range(int(NODE_COUNT))]
    TOP_LEAF_L = BASE_LEAF_L[NODE_COUNT] * LEAF_L_SCALE * NODE_FRAC
    TOP_LEAF_W = BASE_LEAF_L[NODE_COUNT] * LEAF_W_SCALE * NODE_FRAC
    main_stem_info += [{'node_info': {'lat1L': TOP_LEAF_L, 'lat1W': TOP_LEAF_W, 'lat1ang': 90, 
                                      'lat2L': TOP_LEAF_L, 'lat2W': TOP_LEAF_W, 'lat2ang': 90, 
                                      'midL': TOP_LEAF_L, 'midW': TOP_LEAF_W, 'midang': 0,
                                      'pet1len': BASE_PET1_LEN[NODE_COUNT] * PET_LEN_SCALE * NODE_FRAC, 'pet1ang': BASE_PET1_ANG[NODE_COUNT] * PET_ANG_SCALE,
                                      'pet2len': PET2_LEN * PET_LEN_SCALE * NODE_FRAC, 'pet2ang': PET2_ANG * PET_ANG_SCALE}, 
                        'node_rotation': (NODE_COUNT) * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                        'node_ang': 0, 
                        'node_height': BASE_NODE_DIST[NODE_COUNT] * NODE_DIST_SCALE * NODE_FRAC}]
    
    if with_dummies:
        main_stem_info += [DUMMY_NODE for _ in range(14 - NODE_COUNT)]

    branches_info = []
    for j in range(int(BRANCH_COUNT)):
        low = int(BRANCH_NODE_COUNT_MEANS[j])
        if np.random.rand() < BRANCH_NODE_COUNT_MEANS[j] - low:
            BRANCH_NODE_COUNT = low + 1
        else:
            BRANCH_NODE_COUNT = low
        if BRANCH_NODE_COUNT > 0:
            default_node_info = {'lat1L': BRANCH_LEAF_L * LEAF_L_SCALE, 'lat1W': BRANCH_LEAF_L * LEAF_W_SCALE, 'lat1ang': 90, 
                                 'lat2L': BRANCH_LEAF_L * LEAF_L_SCALE, 'lat2W': BRANCH_LEAF_L * LEAF_W_SCALE, 'lat2ang': 90, 
                                 'midL': BRANCH_LEAF_L * LEAF_L_SCALE, 'midW': BRANCH_LEAF_L * LEAF_W_SCALE, 'midang': 0,
                                 'pet1len': BRANCH_PET1_LEN * PET_LEN_SCALE, 'pet1ang': BRANCH_PET1_ANG * PET_ANG_SCALE,
                                 'pet2len': PET2_LEN * PET_LEN_SCALE, 'pet2ang': PET2_ANG * PET_ANG_SCALE}
            default_branch_info = [{'node_info': default_node_info, 
                                    'node_rotation': 0, #np.random.randint(360), 
                                    'node_ang': BRANCH_NODE_ANG if i % 2 == 0 else -BRANCH_NODE_ANG,
                                    'node_height': BRANCH_NODE_DIST * NODE_DIST_SCALE} for i in range(int(BRANCH_NODE_COUNT) - 1)]
            default_branch_info += [{'node_info': default_node_info, 
                                    'node_rotation': 0, #np.random.randint(360), 
                                    'node_ang': 0,
                                    'node_height': BRANCH_NODE_DIST * NODE_DIST_SCALE}]
            branches_info.append({'branch_info': default_branch_info, 
                                  'branch_rotation': j * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                                  'branch_height': np.sum(BASE_NODE_DIST[:j+1]) * NODE_DIST_SCALE, 
                                  'branch_ang': BRANCH_ANG + np.random.normal(0, params['PET1_ANG_STD'])})
            
        elif with_dummies:
            branches_info.append({'branch_info': [DUMMY_NODE for _ in range(2)], 
                                  'branch_rotation': j * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                                  'branch_height': np.sum(BASE_NODE_DIST[:j+1]) * NODE_DIST_SCALE, 
                                  'branch_ang': 0})


    if BRANCH_COUNT > 0 and int(BRANCH_COUNT) < len(BRANCH_NODE_COUNT_MEANS):
        j = int(BRANCH_COUNT)
        low = int(BRANCH_NODE_COUNT_MEANS[j])
        if np.random.rand() < BRANCH_NODE_COUNT_MEANS[j] - low:
            BRANCH_NODE_COUNT = low + 1
        else:
            BRANCH_NODE_COUNT = low
        if BRANCH_NODE_COUNT > 0:
            BRANCH_TOP_LEAF_L = BRANCH_LEAF_L * LEAF_L_SCALE * NODE_FRAC
            BRANCH_TOP_LEAF_W = BRANCH_LEAF_L * LEAF_W_SCALE * NODE_FRAC
            default_node_info = {'lat1L': BRANCH_TOP_LEAF_L, 'lat1W': BRANCH_TOP_LEAF_W, 'lat1ang': 90, 
                                 'lat2L': BRANCH_TOP_LEAF_L, 'lat2W': BRANCH_TOP_LEAF_W, 'lat2ang': 90, 
                                 'midL': BRANCH_TOP_LEAF_L, 'midW': BRANCH_TOP_LEAF_W, 'midang': 0,
                                 'pet1len': BRANCH_PET1_LEN * PET_LEN_SCALE * NODE_FRAC, 'pet1ang': BRANCH_PET1_ANG * PET_ANG_SCALE,
                                 'pet2len': PET2_LEN * PET_LEN_SCALE * NODE_FRAC, 'pet2ang': PET2_ANG * PET_ANG_SCALE}
            default_branch_info = [{'node_info': default_node_info, 
                                    'node_rotation': 0, #np.random.randint(360), 
                                    'node_ang': BRANCH_NODE_ANG if i % 2 == 0 else -BRANCH_NODE_ANG,
                                    'node_height': BRANCH_NODE_DIST * NODE_DIST_SCALE * NODE_FRAC} for i in range(int(BRANCH_NODE_COUNT) - 1)]
            default_branch_info += [{'node_info': default_node_info, 
                                    'node_rotation': 0, #np.random.randint(360), 
                                    'node_ang': 0,
                                    'node_height': BRANCH_NODE_DIST * NODE_DIST_SCALE * NODE_FRAC}]
            branches_info.append({'branch_info': default_branch_info, 
                                  'branch_rotation': j * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                                  'branch_height': np.sum(BASE_NODE_DIST[:j+1]) * NODE_DIST_SCALE, 
                                  'branch_ang': BRANCH_ANG + np.random.normal(0, params['PET1_ANG_STD'])})
            
        elif with_dummies:
            branches_info.append({'branch_info': [DUMMY_NODE for _ in range(2)], 
                                  'branch_rotation': j * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                                  'branch_height': np.sum(BASE_NODE_DIST[:j+1]) * NODE_DIST_SCALE, 
                                  'branch_ang': 0})
            

    if with_dummies:
        total_node_height = np.sum(BASE_NODE_DIST[:int(NODE_COUNT)]) * NODE_DIST_SCALE + BASE_NODE_DIST[int(NODE_COUNT)] * NODE_DIST_SCALE * NODE_FRAC
        for b in branches_info:
            b['branch_info'] += [DUMMY_NODE for _ in range(2 - len(b['branch_info']))]
        for j in range(0 if BRANCH_COUNT == 0 else (int(BRANCH_COUNT) + 1), len(BRANCH_NODE_COUNT_MEANS)):
            _ = np.random.rand()
            branches_info.append({'branch_info': [DUMMY_NODE for _ in range(2)], 
                                  'branch_rotation': j * 180 + np.random.normal(loc=0, scale=params['AZIMUTH_STD']), 
                                  'branch_height': min(total_node_height, np.sum(BASE_NODE_DIST[:j+1]) * NODE_DIST_SCALE), 
                                  'branch_ang': 0})
        # print(len(main_stem_info), BRANCH_COUNT, len(branches_info), [len(b['branch_info']) for b in branches_info])
    return main_stem_info, branches_info


def maize_info_parametric(params):

    BASE_hs = np.array([0.03578, 0.04043, 0.11405, 0.20333, 0.31082, 0.42405, 0.55527, 
                        0.7035, 0.854, 1.00727, 1.1679, 1.327, 1.48, 1.64, 1.79, 1.93, 2.05, 2.16]) * 100
    BASE_LA = np.array([0.00012, 0.00034, 0.000272, 0.0255, 0.00416, 0.00658, 0.01374, 0.02501, 0.03643, 
                        0.04901, 0.05913, 0.05637, 0.05128, 0.04519, 0.03713, 0.0285, 0.01981, 0.01278]) * 10000
    BASE_leaf_angle = np.arange(1, 19)

    leafAspectRatio = [0.0888, 0.0935, 0.0982, 0.1029, 0.1076, 0.1123, 0.117, 0.1217, 0.1264, 0.1311, 0.1358, 0.1405, 0.1452, 0.1452, 0.1546, 0.1593, 0.164, 0.1687]
    leafAspectRatio = leafAspectRatio[:int(params['NODE_COUNT'])]
    dS = 0.0244 * 100
    Nt = int(params['NODE_COUNT'])

    hs = BASE_hs[:Nt] * params['NODE_DIST_SCALE']
    LA = BASE_LA[:Nt] * params['LEAF_L_SCALE'] ** 2
    leaf_angle = BASE_leaf_angle + params['LEAF_ORDER_SHIFT']

    # make Azi alternating 0 and 1 like [1, 0, 1, 0, ...]
    Azi = np.arange(Nt).astype(np.float32) % 2
    if np.random.rand() < 0.5:
        Azi = 1 - Azi
    Azi *= 180
    Azi += np.random.normal(loc=0, scale=params['AZIMUTH_STD'], size=Nt)

    return hs, dS, Nt, LA, Azi, leafAspectRatio, leaf_angle