"""The coordinate frame the point network is trained and used in.

Two entry points put a plant into that frame — `script_point_transformer/prepare_data.py`
for an annotated plant with a fitted graph, and `script_auto_reconstruction/normalize_data.py`
for a new scan with two clicks. They cannot share an implementation, because they
start from different information, but they have to agree on the answer. They did
not: `prepare_data.py` stopped at the graph's own frame, which puts the main stem
on +Z, while `normalize_data.py` targeted +X. The released checkpoint scores 0.97
on +X input and 0.90 on +Z, so a scan that went through the interactive path and a
plant that went through the batch path were 90 degrees apart.

Both now read the axis from here.
"""

import numpy as np


# decode.py and graph.pkl work in a frame whose main stem is +Z. The network is
# trained on +X: that is what the released checkpoint expects, and what config.py
# assumes when it augments with a free rotation about x and only jitters y and z.
CANONICAL_STEM_AXIS = "z"
TRAINING_STEM_AXIS = "x"

AXES = {"x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0])}


def training_axis_vector():
    """The direction a normalised main stem should point, as a unit vector."""
    return AXES[TRAINING_STEM_AXIS].copy()


def canonical_to_training():
    """Row-vector rotation carrying the canonical stem axis onto the training one.

    Applied as ``coord @ canonical_to_training()``. Any residual roll about the
    stem is irrelevant: the training augmentation rotates freely about that axis.
    """
    source, target = AXES[CANONICAL_STEM_AXIS], AXES[TRAINING_STEM_AXIS]
    axis = np.cross(source, target)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:                      # already aligned, or opposed
        return np.eye(3) if source @ target > 0 else -np.eye(3)
    axis = axis / norm
    angle = np.arctan2(norm, float(source @ target))
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    matrix = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return matrix.T                       # column-vector matrix -> row-vector form


def principal_axis(points):
    """Dominant direction of a point set, with how dominant it is."""
    centred = np.asarray(points, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    _, singular, basis = np.linalg.svd(centred, full_matrices=False)
    elongation = float(singular[0] / singular[1]) if singular[1] > 0 else np.inf
    return basis[0], elongation


def check_stem_axis(label, coord, stem_mask, axis=None, tolerance=0.7,
                    min_points=10, min_elongation=3.0):
    """Fail loudly if the main stem did not land on the axis we claim.

    The frame drifted once because alignment was only assumed, never checked, and
    nothing downstream noticed until the released checkpoint lost seven points of
    accuracy. A stem too short or too isotropic to have a direction is skipped.
    """
    axis = axis or TRAINING_STEM_AXIS
    stem = np.asarray(coord)[np.asarray(stem_mask, dtype=bool)]
    if len(stem) < min_points:
        return
    direction, elongation = principal_axis(stem)
    if elongation < min_elongation:
        return
    alignment = abs(direction["xyz".index(axis)])
    if alignment < tolerance:
        found = "xyz"[int(np.argmax(abs(direction)))]
        raise ValueError(f"{label}: main stem points along {found}, not {axis} "
                         f"(|cos| {alignment:.2f}); the alignment is wrong")


# --- the normalisation transform -------------------------------------------
#
# The same four numbers describe how a plant was put into the training frame,
# but they reach consumers two ways: prepare_data.py records them per sample in
# the dataset manifest, normalize_data.py pickles them beside a single scan. Both
# shapes are read here so run.py and recon.py do not each carry their own reader.

TRANSFORM_KEYS = ("rotation", "bbox_center", "radius", "normalize_divisor")


def load_transform(folder, name=None):
    """Read the normalisation transform for one plant, from either format.

    `folder` holds `transform.pkl`, or is a dataset split whose parent holds
    `manifest.json`; `name` selects the sample in the manifest.
    """
    import json
    import pickle
    from pathlib import Path

    folder = Path(folder)
    pickled = folder / "transform.pkl"
    if pickled.is_file():
        with pickled.open("rb") as handle:
            stored = pickle.load(handle)
    else:
        manifest = folder.parent / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Need {pickled} or {manifest}; neither describes how {name or folder} "
                "was normalised")
        samples = json.loads(manifest.read_text())["samples"]
        if name not in samples:
            raise KeyError(f"{name} is not in {manifest}")
        stored = samples[name]
    transform = {key: stored[key] for key in TRANSFORM_KEYS if key in stored}
    for key in ("rotation", "bbox_center"):
        transform[key] = np.asarray(transform[key], dtype=np.float64)
    if "radius" not in transform:
        raise KeyError(f"Transform for {name or folder} has no radius")
    return transform


def save_transform(folder, transform):
    import pickle
    from pathlib import Path

    path = Path(folder) / "transform.pkl"
    with path.open("wb") as handle:
        pickle.dump(transform, handle)
    return path


def scan_scale(transform):
    """Divisor that takes normalised coordinates back to the scan's own units."""
    return float(transform.get("normalize_divisor", transform["radius"]))


def to_scan_frame(coord, transform):
    """Undo the normalisation: training frame -> the original scan's coordinates."""
    return (np.asarray(coord) * scan_scale(transform)) @ transform["rotation"].T \
        + transform["bbox_center"]
