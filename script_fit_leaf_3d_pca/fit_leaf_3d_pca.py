"""Fit the 3D leaf-shape PCA consumed by Demeter's decoder.

For each species this loads the fitted leaf-deformation array
``leaf_deformed_mesh/{species}.npy`` of shape ``(n_leaf, 43, 45, 3)``, fits the
repo's :class:`~utils.pca.NodePCA`, and writes
``sample_params/{species}/3d_leaf_pca.pth`` -- the exact basis read back by
``decode.py``. This reproduces the shipped ``3d_leaf_pca.pth`` files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.pca import NodePCA  # noqa: E402

H, W = 43, 45
DEFAULT_SPECIES = ["soybean", "rose", "ribes", "maize", "pepper", "tobacco"]


def fit_species(
    species: str,
    data_dir: Path,
    out_dir: Path,
    n_components: float = 0.995,
) -> NodePCA:
    """Fit and save the 3D leaf PCA for a single species.

    Args:
        species: Species name, e.g. ``"soybean"``.
        data_dir: Folder holding ``{species}.npy`` arrays of shape
            ``(n, H, W, 3)``.
        out_dir: ``sample_params`` root; the basis is written to
            ``{out_dir}/{species}/3d_leaf_pca.pth``.
        n_components: Forwarded to :class:`NodePCA`; a float keeps that fraction
            of variance (``0.995`` matches the shipped bases).

    Returns:
        The trained :class:`NodePCA`.
    """
    data_path = data_dir / f"{species}.npy"
    data = np.load(data_path)  # (n, H, W, 3)
    n = data.shape[0]
    print(f"[{species}] loaded {n} leaves, grid {tuple(data.shape[1:])}")

    pca = NodePCA(n_components=n_components)
    pca.train(data.reshape(n, -1))  # (n, H*W*3) -> (n, k)

    species_dir = out_dir / species
    species_dir.mkdir(parents=True, exist_ok=True)
    pca.save(str(species_dir / "3d_leaf_pca.pth"))
    return pca


def main() -> None:
    """CLI entry point: fit the 3D leaf PCA for one or all species."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="all",
        help="Species to fit, or 'all' for every array found in --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "leaf_deformed_mesh",
        help="Folder of {species}.npy leaf-deformation arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "sample_params",
        help="sample_params root to write {species}/3d_leaf_pca.pth into.",
    )
    parser.add_argument(
        "--n-components",
        type=float,
        default=0.995,
        help="Variance fraction (float) or component count (int) to keep.",
    )
    args = parser.parse_args()

    if args.species == "all":
        species_list = [p.stem for p in sorted(args.data_dir.glob("*.npy"))]
        if not species_list:
            raise FileNotFoundError(f"no *.npy arrays found in {args.data_dir}")
    else:
        species_list = [args.species]

    for species in species_list:
        fit_species(species, args.data_dir, args.out_dir, args.n_components)


if __name__ == "__main__":
    main()
