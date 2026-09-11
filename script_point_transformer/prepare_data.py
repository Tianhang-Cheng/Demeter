"""Build Pointcept samples from the released per-organ point clouds (CPU only)."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from utils.frames import (CANONICAL_STEM_AXIS, TRAINING_STEM_AXIS,
                          canonical_to_training, check_stem_axis)


REPO = Path(__file__).resolve().parents[1]
CLASS_MAP = {0: 0, 1: 1, 2: 3, 3: 4}
CLASS_NAMES = ["leaf", "other_stem", "main_stem", "flower", "fruit"]
MAIN_STEM_CLASS = CLASS_NAMES.index("main_stem")


def read_annotations(folder):
    classes = {}
    for line in (folder / "info/class.txt").read_text().splitlines():
        if line.strip():
            key, value = map(int, line.split())
            classes[key] = value
    parents = {}
    for line in (folder / "info/parent.txt").read_text().splitlines():
        if line.strip():
            key, value = map(int, line.split("->"))
            parents[key] = value
    roots = [key for key, value in parents.items() if value == -1]
    if len(roots) != 1 or classes.get(roots[0]) != 1:
        raise ValueError(f"{folder}: expected exactly one root annotated as stem (class 1)")
    return classes, roots[0]


def inverse_distances(points, retained, k=10):
    """Match Plant's cross-instance distances and density-adaptive threshold.

    points contains all original points; retained contains the non-black query
    points. Cross-instance reference points deliberately include black points,
    as in the archived preprocessing script. Self KNN includes the query itself.
    """
    if len(points) < 2 or len(retained) < 2:
        raise ValueError("At least two nonempty organ instances are required")
    distances, spacings = [], []
    for organ_id, query in retained.items():
        rest = np.concatenate([p for key, p in points.items() if key != organ_id])
        distances.append(cKDTree(rest).query(query, k=1)[0])
        self_dist = cKDTree(query).query(query, k=k)[0]
        # The original script replaces missing neighbors (inf) and NaNs by 0.
        self_dist = np.nan_to_num(self_dist, nan=0.0, posinf=0.0, neginf=0.0)
        spacings.append(self_dist.max(axis=1))
    threshold = float(2 * np.concatenate(spacings).mean())
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Zero/invalid density threshold: check duplicate or singleton organs")
    distance = np.concatenate(distances).astype(np.float32)
    target = (threshold / np.maximum(distance, threshold)).astype(np.float32)
    return target, threshold


def graph_rotation(folder, root, stem_axis=TRAINING_STEM_AXIS):
    """Return the raw -> training-frame row-vector rotation.

    ``stem_axis="z"`` stops at decode.py's canonical frame; the default carries on
    to the +X frame the network is trained and used in.
    """
    state = torch.load(folder / "graph.pkl", map_location="cpu", weights_only=True)
    q = np.asarray(state[f"M_quat_{root}"], dtype=np.float64).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError(f"{folder}: invalid main-stem quaternion")
    # Demeter stores w,x,y,z; scipy takes x,y,z,w. Its matrix is the
    # column-vector raw->canonical matrix used by decode.raw_to_canonical_transform.
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T
    if stem_axis == TRAINING_STEM_AXIS:
        return rotation @ canonical_to_training()
    if stem_axis != CANONICAL_STEM_AXIS:
        raise ValueError(f"stem_axis must be x or z, got {stem_axis!r}")
    return rotation


def prepare_sample(folder, alignment="graph", keep_black=False,
                   stem_axis=TRAINING_STEM_AXIS):
    import open3d as o3d

    classes, root = read_annotations(folder)
    paths = sorted((folder / "raw").glob("*.ply"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"{folder}/raw/*.ply is missing; restore sample_params.tar.gz")
    points, retained, colors = {}, {}, {}
    for path in paths:
        organ_id = int(path.stem)
        if organ_id in points:
            raise ValueError(f"Duplicate organ ID {organ_id} in {folder}")
        if organ_id not in classes or classes[organ_id] not in CLASS_MAP:
            raise ValueError(f"{path}: missing or unsupported class annotation")
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points).copy()
        if not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError(f"{path}: empty or non-finite coordinates")
        rgb = np.asarray(pcd.colors).copy() if pcd.has_colors() else np.full_like(xyz, 0.5)
        if not np.isfinite(rgb).all():
            raise ValueError(f"{path}: non-finite colors")
        points[organ_id] = xyz
        mask = np.ones(len(xyz), dtype=bool) if keep_black else rgb.sum(axis=1) > 0
        if mask.any():
            retained[organ_id] = xyz[mask]
            colors[organ_id] = rgb[mask]
    if root not in retained:
        raise ValueError(f"{folder}: main stem has no retained points")
    inv_dists, threshold = inverse_distances(points, retained)
    all_points = np.concatenate(list(points.values()))
    radii = np.sort(np.linalg.norm(all_points - all_points.mean(axis=0), axis=1))
    radius = float(radii[int(len(radii) * 0.95)])
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError(f"{folder}: invalid normalization radius")
    xyz = np.concatenate(list(retained.values()))
    center = xyz.mean(axis=0)
    rotation = (graph_rotation(folder, root, stem_axis) if alignment == "graph"
                else np.eye(3))
    coord = ((xyz - center) @ rotation / radius).astype(np.float32)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    instance = np.concatenate([np.full(len(p), key, dtype=np.int64) for key, p in retained.items()])
    semantic = np.concatenate([
        np.full(len(p), 2 if key == root else CLASS_MAP[classes[key]], dtype=np.int64)
        for key, p in retained.items()
    ])
    if alignment == "graph":
        check_stem_axis(folder, coord, semantic == MAIN_STEM_CLASS, stem_axis)
    # Keep the archived RGB convention (0..1) for checkpoint compatibility.
    arrays = dict(coord=coord, color=np.concatenate(list(colors.values())).astype(np.float32),
                  normal=np.asarray(cloud.normals).astype(np.float32),
                  semantic_gt5=semantic, instance_gt=instance, inv_dists=inv_dists)
    sample = {key: torch.from_numpy(value.copy()) for key, value in arrays.items()}
    sample["scene_id"] = folder.name
    metadata = dict(source=str(folder.resolve()), points=len(coord), organs=len(retained),
                    main_stem=root, threshold_raw=threshold, threshold_normalized=threshold / radius,
                    radius=radius, normalize_divisor=radius, bbox_center=center.tolist(),
                    rotation=rotation.tolist(), alignment=alignment,
                    semantic_counts=np.bincount(semantic, minlength=5).tolist())
    return sample, metadata


def make_splits(names, test_fraction, seed):
    if not 0 <= test_fraction < 1:
        raise ValueError("Test fraction must be nonnegative and less than 1")
    # Keep scans such as 10_i and 10_o together to avoid same-plant leakage.
    groups = sorted({name.rsplit("_", 1)[0] if name.endswith(("_i", "_o")) else name for name in names})
    np.random.default_rng(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * test_fraction)) if test_fraction else 0
    if n_test >= len(groups):
        raise ValueError("Too few plant groups for train/test; use --test-fraction 0 for a smoke test")
    group_split = {g: "test" if i < n_test else "train"
                   for i, g in enumerate(groups)}
    result = {key: [] for key in ("train", "test")}
    for name in sorted(names):
        group = name.rsplit("_", 1)[0] if name.endswith(("_i", "_o")) else name
        result[group_split[group]].append(name)
    return result


def select_splits(names, test_fraction, seed, split_file=None):
    preset_path = split_file or Path(__file__).with_name("soybean_split.json")
    preset = json.loads(preset_path.read_text()) if preset_path.is_file() else None
    if split_file is not None and preset is None:
        raise FileNotFoundError(preset_path)
    if split_file is not None or (test_fraction is None and preset is not None
                                 and set(sum(preset.values(), [])) == set(names)):
        if set(preset) != {"train", "test"}:
            raise ValueError("Split file must contain only train and test lists")
        flat = preset["train"] + preset["test"]
        if len(flat) != len(set(flat)) or set(flat) != set(names):
            raise ValueError("Split file must assign every selected sample exactly once")
        groups = {}
        for split, samples in preset.items():
            for name in samples:
                group = name.rsplit("_", 1)[0] if name.endswith(("_i", "_o")) else name
                if group in groups and groups[group] != split:
                    raise ValueError(f"Plant group {group} appears in both splits")
                groups[group] = split
        return preset
    return make_splits(names, 0.15 if test_fraction is None else test_fraction, seed)


def write_sample(source, output, split, name, alignment, keep_black, stem_axis):
    sample, info = prepare_sample(source / name, alignment, keep_black, stem_axis)
    path = output / split / f"{name}.pth"
    temporary = path.with_suffix(".pth.tmp")
    torch.save(sample, temporary)
    temporary.replace(path)
    return name, dict(split=split, **info)


def save_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "sample_params/soybean/instances")
    parser.add_argument("--output", type=Path, default=REPO / "data/point_transformer/soybean")
    parser.add_argument("--samples", nargs="+", help="Optional subset of sample directory names")
    split_options = parser.add_mutually_exclusive_group()
    split_options.add_argument("--test-fraction", type=float,
                               help="Create a new split with this test fraction (default: published split or 0.15)")
    split_options.add_argument("--split-file", type=Path, help="JSON containing explicit train/test sample lists")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--alignment", choices=("graph", "none"), default="graph",
                        help="Use the fitted main-stem rotation; none is only for already aligned scans")
    parser.add_argument("--keep-black", action="store_true", help="Keep black points (changes legacy targets)")
    parser.add_argument("--stem-axis", choices=("x", "z"), default="x",
                        help="Axis the main stem is aligned to; x matches the released "
                             "checkpoint, normalize_data.py and the augmentation in config.py")
    parser.add_argument("--jobs", type=int, default=1, help="Number of CPU preprocessing processes")
    parser.add_argument("--resume", action="store_true", help="Continue a matching incomplete manifest")
    args = parser.parse_args()
    try:
        if not args.source.is_dir():
            raise FileNotFoundError(f"Missing {args.source}; restore the processed sample_params archive")
        names = args.samples or sorted(p.name for p in args.source.iterdir() if p.is_dir())
        if not names or len(names) != len(set(names)):
            raise ValueError("Expected a nonempty list of unique samples")
        for name in names:
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid sample name: {name}")
            folder = args.source / name
            for rel in ("raw", "info/class.txt", "info/parent.txt"):
                if not (folder / rel).exists():
                    raise FileNotFoundError(f"Missing {folder / rel}; restore sample_params.tar.gz (see readme)")
            if args.alignment == "graph" and not (folder / "graph.pkl").is_file():
                raise FileNotFoundError(f"Missing {folder}/graph.pkl")
        splits = select_splits(names, args.test_fraction, args.seed, args.split_file)
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        settings = dict(source=str(args.source.resolve()), splits=splits, seed=args.seed,
                        alignment=args.alignment, keep_black=args.keep_black,
                        stem_axis=args.stem_axis)
        manifest = dict(version=2, status="in_progress", settings=settings, seed=args.seed,
                        splits=splits, samples={}, class_names=CLASS_NAMES, class_map=CLASS_MAP,
                        keep_black=args.keep_black, color_range=[0, 1],
                        stem_axis=args.stem_axis,
                        inv_dist_formula="T / max(d, T)",
                        threshold="2 * mean(max(within-instance KNN(k=10, including self)))")
        if args.resume:
            manifest = json.loads((args.output / "manifest.json").read_text())
            if manifest.get("settings") != settings:
                raise ValueError("Resume settings differ from the saved source/splits/preprocessing")
            for name, info in manifest["samples"].items():
                if not (args.output / info["split"] / f"{name}.pth").is_file():
                    raise FileNotFoundError(f"Completed sample is missing: {name}")
        elif args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"Output must be empty to prevent stale samples/split leakage: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        tasks = []
        for split, sample_names in splits.items():
            (args.output / split).mkdir(exist_ok=True)
            for name in sample_names:
                if name not in manifest["samples"]:
                    tasks.append((args.source, args.output, split, name, args.alignment,
                                  args.keep_black, args.stem_axis))
        save_manifest(args.output, manifest)

        def record(result):
            name, info = result
            manifest["samples"][name] = info
            save_manifest(args.output, manifest)
            print(f"[{len(manifest['samples'])}/{len(names)}] {info['split']}/{name}: "
                  f"{info['points']:,} points; T={info['threshold_raw']:.6g}", flush=True)

        if args.jobs == 1:
            for task in tasks:
                print(f"Preparing {task[2]}/{task[3]}", flush=True)
                record(write_sample(*task))
        else:
            # Avoid nested OpenMP pools in each Open3D/SciPy worker.
            import os
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            print(f"Preparing {len(tasks)} samples with {args.jobs} CPU processes", flush=True)
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(write_sample, *task): task[3] for task in tasks}
                for future in as_completed(futures):
                    try:
                        record(future.result())
                    except Exception as exc:
                        for pending in futures:
                            pending.cancel()
                        raise ValueError(f"{futures[future]}: {exc}; rerun with --resume after fixing the input") from exc
        manifest["status"] = "complete"
        save_manifest(args.output, manifest)
        print(f"Saved dataset to {args.output.resolve()}")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
