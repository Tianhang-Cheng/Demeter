"""Repository-root entry points for Pointcept training, testing and reconstruction."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
POINTCEPT = REPO / "third_party/PointTransformer_V3/Pointcept"
CONFIG = Path(__file__).with_name("config.py")


def pointcept_command(mode, options, gpus):
    return [sys.executable, str(POINTCEPT / "tools" / f"{mode}.py"),
            "--config-file", str(CONFIG), "--num-gpus", str(gpus), "--options",
            *[f"{key}={value}" for key, value in options.items()]]


def execute(command):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(POINTCEPT), str(REPO), env.get("PYTHONPATH", "")])
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def require_runtime():
    missing = [name for name in ("torch", "pointops", "torch_scatter", "torch_cluster",
                                 "torch_geometric", "SharedArray", "addict", "yapf", "tensorboard")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError("Missing Pointcept dependencies: " + ", ".join(missing)
                         + ". Follow script_auto_reconstruction/readme.md#environment in Linux/CUDA.")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Pointcept training/inference requires a CUDA GPU; preprocessing is CPU-only")


def require_empty(folder):
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Choose an empty output directory (prevents stale predictions): {folder}")


def require_split(root, split):
    manifest = root / "manifest.json"
    if manifest.is_file() and json.loads(manifest.read_text()).get("status") != "complete":
        raise ValueError(f"Dataset preparation is incomplete: {manifest}; resume preprocessing first")
    samples = list((root / split).glob("*.pth"))
    if not samples:
        raise ValueError(f"No .pth samples in {root / split}; run prepare_data.py first")
    return samples


def stage_inference(sample_path, output):
    """Stage coordinates in exactly the order used by the network and reconstruction."""
    import numpy as np
    import open3d as o3d
    import torch

    # Local inputs include legacy NumPy-based .pth files from the released repo.
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    arrays = {}
    for key in ("coord", "color", "normal"):
        value = sample[key]
        arrays[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if arrays[key].ndim != 2 or arrays[key].shape[1] != 3 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key} in {sample_path}")
    coord = arrays["coord"]
    if len(coord) == 0 or any(len(v) != len(coord) for v in arrays.values()):
        raise ValueError("Input arrays must have the same nonzero point count")
    manifest_path = sample_path.parent.parent / "manifest.json"
    if manifest_path.is_file():
        info = json.loads(manifest_path.read_text())["samples"][sample_path.stem]
        transform = {k: info[k] for k in ("rotation", "bbox_center", "radius", "normalize_divisor")}
        transform["rotation"] = np.asarray(transform["rotation"])
        transform["bbox_center"] = np.asarray(transform["bbox_center"])
    elif (sample_path.parent / "transform.pkl").is_file():
        with (sample_path.parent / "transform.pkl").open("rb") as f:
            transform = pickle.load(f)
    else:
        raise FileNotFoundError("Need the generated manifest.json or a transform.pkl beside the normalized sample")
    output.mkdir(parents=True, exist_ok=True)
    staged = {k: torch.as_tensor(v.copy()).float() for k, v in arrays.items()}
    staged["scene_id"] = sample_path.stem
    # Ground-truth labels are deliberately omitted from the reconstruction input.
    torch.save(staged, output / "normalized_pcd.pth")
    with (output / "transform.pkl").open("wb") as f:
        pickle.dump(transform, f)
    click_path = sample_path.parent / "rotation_click.txt"
    if click_path.is_file():
        shutil.copy2(click_path, output / click_path.name)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord))
    cloud.colors = o3d.utility.Vector3dVector(arrays["color"])
    o3d.io.write_point_cloud(str(output / "pcd_unit_radius.ply"), cloud)
    divisor = transform.get("normalize_divisor", transform["radius"])
    cloud.points = o3d.utility.Vector3dVector(
        (coord * divisor) @ np.asarray(transform["rotation"]).T + transform["bbox_center"])
    o3d.io.write_point_cloud(str(output / "pcd.ply"), cloud)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "test", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--weight", type=Path, required=name != "train")
        p.add_argument("--gpus", type=int, default=1)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--dry-run", action="store_true", help="Validate paths and print commands without writing/running")
        if name != "infer":
            p.add_argument("--data-root", type=Path, default=REPO / "data/point_transformer/soybean")
        if name == "train":
            p.add_argument("--epochs", type=int, default=150)
            p.add_argument("--batch-size", type=int, default=4)
            p.add_argument("--resume", action="store_true", help="Resume --weight in an existing run directory")
        elif name == "test":
            p.add_argument("--split", choices=("test",), default="test")
        else:
            p.add_argument("--sample", type=Path, required=True, help="Generated split/name.pth or normalized_pcd.pth")
            p.add_argument("--reconstruct", action="store_true", help="Also fit the Demeter graph and export its mesh")
            p.add_argument("--species", default="soybean")
    args = parser.parse_args()
    try:
        args.output = args.output.resolve()
        if args.gpus < 1 or args.workers < 0:
            raise ValueError("gpus must be positive and workers nonnegative")
        if args.weight:
            args.weight = args.weight.resolve()
            if not args.weight.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {args.weight}. See the checkpoint download instructions.")
        options = dict(save_path=str(args.output), num_worker=args.workers)
        if args.weight:
            options["weight"] = str(args.weight)
        if args.command == "train":
            if args.epochs < 1 or args.batch_size < 1 or args.batch_size % args.gpus:
                raise ValueError("epochs must be positive; batch size must be positive and divisible by gpus")
            if args.resume and not args.weight:
                raise ValueError("--resume requires --weight")
            if not args.resume:
                require_empty(args.output)
            root = args.data_root.resolve()
            if len(require_split(root, "train")) < args.batch_size:
                raise ValueError("Training set is smaller than batch size; reduce --batch-size")
            require_split(root, "test")
            options.update({f"data.{split}.data_root": str(root) for split in ("train", "test")})
            options.update(epoch=args.epochs, eval_epoch=args.epochs, batch_size=args.batch_size, resume=args.resume)
        elif args.command == "test":
            require_empty(args.output)
            root = args.data_root.resolve()
            require_split(root, args.split)
            options.update({"data.test.data_root": str(root), "data.test.split": args.split,
                            "batch_size": args.gpus})
        else:
            require_empty(args.output)
            args.sample = args.sample.resolve()
            if not args.sample.is_file():
                raise FileNotFoundError(args.sample)
            options.update({"save_path": str(args.output / "pointcept"),
                            "data.test.data_root": str(args.output.parent),
                            "data.test.split": args.output.name, "batch_size": args.gpus})
        command = pointcept_command("train" if args.command == "train" else "test", options, args.gpus)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            if args.command == "infer":
                print(f"Stage {args.sample} in {args.output}; copy both predictions back after inference.")
                if args.reconstruct:
                    print("Then fit params/plant_graph.pth and export predict.ply.")
            return
        require_runtime()
        if args.command == "infer":
            stage_inference(args.sample, args.output)
        execute(command)
        if args.command == "infer":
            for name in ("normalized_pcd_pred.npy", "normalized_pcd_pred_dist.npy"):
                shutil.copy2(args.output / "pointcept/result" / name, args.output / name)
            if args.reconstruct:
                execute([sys.executable, str(REPO / "script_auto_reconstruction/recon.py"),
                         "--data_folder", str(args.output), "--species", args.species, "--no-viz"])
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
