"""Group predicted points into organ instances.

The network predicts semantics and a per-point boundary score, not instances, so
organs are recovered afterwards. Both consumers of that step live here so they
cannot drift apart: `script_auto_reconstruction/recon.py`, which fits a Demeter
graph to the organs, and `script_point_transformer/viz_predictions.py`, which
scores them. Before this module each had its own copy of the threshold, the
density estimate and the clustering, so an improvement measured by one did not
reach the other.

Measured on the 11 held-out soybean plants, best IoU per annotated organ scored
against the whole organ:

    spacing     coverage 0.466, 1.71x too many organs, 74% of points assigned
    fixed       coverage 0.453, 0.96x,                 73% assigned
    graph_cut   coverage 0.511, 0.95x,                 87% assigned

`spacing` is what the published pipeline does and stays the default here, so
reconstruction output does not change unless a caller asks for another method.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# Points scoring above this are treated as organ junctions. Coordinates are
# normalised to unit radius, so the value is comparable across scans.
BOUNDARY_KEEP = {"soybean": 0.15}
DEFAULT_BOUNDARY_KEEP = 0.4

MIN_CLUSTER = 8
SPACING_EPS_SCALE = 0.9
FIXED_GRID = 0.004
FIXED_EPS = 0.012
GRAPH_CUT_KEEP = 0.25
GRAPH_CUT_NEIGHBOURS = 16
GRAPH_CUT_LINK = 3.0


def boundary_threshold(species):
    return BOUNDARY_KEEP.get(species, DEFAULT_BOUNDARY_KEEP)


def density_threshold(coord, k=10, kdtree=None, batch=100_000):
    """T = 2 * mean over points of the k-th nearest-neighbour distance.

    The same quantity `prepare_data.py` uses to truncate the boundary target,
    measured here over the whole cloud rather than within one organ.
    """
    tree = kdtree if kdtree is not None else cKDTree(coord)
    spacings = []
    for start in range(0, len(coord), batch):
        distance, _ = tree.query(coord[start:start + batch], k=k, workers=-1)
        spacings.append(np.nan_to_num(distance, nan=0.0, posinf=0.0).max(axis=-1))
    return float(np.concatenate(spacings).mean() * 2)


def _voxelise(coord, grid):
    _, first, inverse = np.unique(np.floor(coord / grid).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    return first, inverse


def _compact(labels):
    ids = np.unique(labels[labels >= 0])
    lookup = {int(v): i for i, v in enumerate(ids)}
    return np.array([lookup.get(int(v), -1) for v in labels], dtype=np.int64)


def group_organs(coord, boundary_score, semantic=None, species="soybean",
                 method="spacing", threshold=None, kdtree=None):
    """Per-point organ label, -1 where the point is left unassigned.

    `spacing` deletes the junction points and clusters the rest at a radius taken
    from the cloud's own spacing. Since preprocessing already puts every plant at
    unit radius that ties the radius to point density rather than organ size --
    across the released plants point count spans 77x while plant size spans 1.7x,
    so dense scans split their organs.

    `fixed` removes the density dependence by voxelising first, which corrects the
    organ count but not the IoU.

    `graph_cut` deletes nothing, which is what caps the other two: it cuts kNN
    edges that straddle a junction or a class change, so every point keeps an
    organ. It needs `semantic`.
    """
    coord = np.asarray(coord, dtype=np.float64)
    score = np.asarray(boundary_score).reshape(-1)
    if threshold is None:
        threshold = boundary_threshold(species)
    labels = np.full(len(coord), -1, dtype=np.int64)

    if method == "graph_cut":
        if semantic is None:
            raise ValueError("graph_cut needs predicted semantics")
        return _graph_cut(coord, score, np.asarray(semantic).reshape(-1))

    keep = score < threshold
    if not keep.any():
        raise ValueError(f"No point scores below {threshold}; check the boundary head")
    if method == "spacing":
        eps = density_threshold(coord, kdtree=kdtree) * SPACING_EPS_SCALE
        labels[keep] = DBSCAN(eps=eps, min_samples=MIN_CLUSTER,
                              n_jobs=-1).fit_predict(coord[keep])
    elif method == "fixed":
        if semantic is None:
            raise ValueError("fixed needs predicted semantics")
        inside = coord[keep]
        first, inverse = _voxelise(inside, FIXED_GRID)
        centres = inside[first]
        classes = np.asarray(semantic).reshape(-1)[keep][first]
        voxel = np.full(len(centres), -1, dtype=np.int64)
        nxt = 0
        for cls in np.unique(classes):
            part = classes == cls
            found = DBSCAN(eps=FIXED_EPS, min_samples=MIN_CLUSTER,
                           n_jobs=-1).fit_predict(centres[part])
            voxel[part] = np.where(found >= 0, found + nxt, -1)
            if (found >= 0).any():
                nxt += found.max() + 1
        labels[keep] = voxel[inverse]
    else:
        raise ValueError(f"method must be spacing, fixed or graph_cut; got {method!r}")
    return _compact(labels)


def _graph_cut(coord, score, semantic, grid=FIXED_GRID):
    first, inverse = _voxelise(coord, grid)
    centres, centre_score, classes = coord[first], score[first], semantic[first]
    distance, index = cKDTree(centres).query(centres, k=GRAPH_CUT_NEIGHBOURS + 1,
                                             workers=-1)
    src = np.repeat(np.arange(len(centres)), GRAPH_CUT_NEIGHBOURS)
    dst = index[:, 1:].ravel()
    edge = ((distance[:, 1:].ravel() < GRAPH_CUT_LINK * grid)
            & (np.maximum(centre_score[src], centre_score[dst]) < GRAPH_CUT_KEEP)
            & (classes[src] == classes[dst]))
    graph = coo_matrix((np.ones(edge.sum()), (src[edge], dst[edge])),
                       shape=(len(centres),) * 2)
    _, found = connected_components(graph, directed=False)
    sizes = np.bincount(found)
    found = np.where(sizes[found] >= MIN_CLUSTER, found, -1)
    return _compact(found)[inverse]


def as_clusters(coord, labels, semantic=None):
    """Adapt per-point labels to the dict-of-arrays shape recon.py fits against."""
    ids = np.unique(labels[labels >= 0])
    points, indices, semantics = {}, {}, {}
    for new, old in enumerate(ids):
        mask = labels == old
        points[new] = coord[mask]
        indices[new] = np.full(int(mask.sum()), new, dtype=np.int64)
        if semantic is not None:
            semantics[new] = np.asarray(semantic).reshape(-1)[mask]
    return indices, len(ids), points, semantics
