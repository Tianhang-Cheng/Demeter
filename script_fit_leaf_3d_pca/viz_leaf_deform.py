"""Visualize the fitted 3D leaf-deformation dataset.

Each ``leaf_deformed_mesh/{species}.npy`` is an array of shape
``(n_leaf, 43, 45, 3)``: ``n_leaf`` fitted leaf surfaces, each a fixed
``43 x 45`` grid of 3D points in a canonical frame (blade base at the origin,
main axis along ``+y``). This renders two figures:

* ``leaf_deform_gallery.png`` -- one row per species, random sample leaves.
* ``leaf_deform_means.png``   -- the per-species mean leaf shape.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_by_species(data_dir: Path) -> dict[str, np.ndarray]:
    """Load every ``{species}.npy`` array grouped by species.

    Args:
        data_dir: Folder of ``{species}.npy`` arrays of shape ``(n, H, W, 3)``.

    Returns:
        Mapping species -> array of shape ``(n, H, W, 3)``.
    """
    grids: dict[str, np.ndarray] = {}
    for path in sorted(data_dir.glob("*.npy")):
        grids[path.stem] = np.load(path)
    if not grids:
        raise FileNotFoundError(f"no *.npy arrays found in {data_dir}")
    return grids


def set_equal_aspect(ax, grid: np.ndarray) -> None:
    """Give a 3D axis a data-proportional box so the leaf is not distorted.

    Args:
        ax: A matplotlib 3D axis.
        grid: The ``(H, W, 3)`` points drawn on it.
    """
    pts = grid.reshape(-1, 3)
    ranges = pts.max(0) - pts.min(0)
    ax.set_box_aspect(tuple(ranges + 1e-6))


def draw_leaf(ax, grid: np.ndarray, title: str = "") -> None:
    """Render one leaf grid as a height-colored 3D surface + wireframe.

    Args:
        ax: A matplotlib 3D axis.
        grid: ``(H, W, 3)`` surface points.
        title: Optional subplot title.
    """
    x, y, z = grid[..., 0], grid[..., 1], grid[..., 2]
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.15, alpha=0.4)
    set_equal_aspect(ax, grid)
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)


def viz_gallery(
    grids: dict[str, np.ndarray],
    out_path: Path,
    n_cols: int = 5,
    seed: int = 0,
) -> None:
    """Render a gallery: one row per species, ``n_cols`` random samples each.

    Args:
        grids: Species -> array of shape ``(n, H, W, 3)``.
        out_path: Destination PNG.
        n_cols: Number of sample leaves per species.
        seed: RNG seed for reproducible sampling.
    """
    rng = random.Random(seed)
    species_order = sorted(grids, key=lambda s: -len(grids[s]))
    n_rows = len(species_order)

    fig = plt.figure(figsize=(2.4 * n_cols, 2.4 * n_rows))
    for r, species in enumerate(species_order):
        samples = grids[species]
        idx = rng.sample(range(len(samples)), min(n_cols, len(samples)))
        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            if c < len(idx):
                title = f"{species} (n={len(samples)})" if c == 0 else ""
                draw_leaf(ax, samples[idx[c]], title)
            else:
                ax.set_axis_off()
    fig.suptitle("Leaf deformation templates — random samples per species", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def viz_means(grids: dict[str, np.ndarray], out_path: Path) -> None:
    """Render the mean leaf shape of each species in a single row.

    Args:
        grids: Species -> array of shape ``(n, H, W, 3)``.
        out_path: Destination PNG.
    """
    species_order = sorted(grids, key=lambda s: -len(grids[s]))
    n = len(species_order)
    fig = plt.figure(figsize=(2.6 * n, 3.0))
    for i, species in enumerate(species_order):
        mean_grid = grids[species].mean(0)
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        draw_leaf(ax, mean_grid, f"{species} mean")
    fig.suptitle("Per-species mean leaf shape", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    """CLI entry point: render the gallery and per-species mean figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "leaf_deformed_mesh",
        help="Folder of {species}.npy leaf-deformation arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Folder to write the PNG figures into.",
    )
    args = parser.parse_args()

    grids = load_by_species(args.data_dir)
    total = sum(len(v) for v in grids.values())
    print(
        f"loaded {total} leaves across {len(grids)} species: "
        + ", ".join(
            f"{k}={len(v)}" for k, v in sorted(grids.items(), key=lambda x: -len(x[1]))
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    viz_gallery(grids, args.out_dir / "leaf_deform_gallery.png")
    viz_means(grids, args.out_dir / "leaf_deform_means.png")


if __name__ == "__main__":
    main()
