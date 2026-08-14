"""Fit a 3D leaf-shape PCA for one species and visualize its leading modes.

Loads ``leaf_deformed_mesh/{species}.npy`` (shape ``(n_leaf, 43, 45, 3)``),
fits the repo's :class:`~utils.pca.NodePCA` (same basis used by ``decode.py``),
then renders a grid where each row is a principal component swept from
``-2 sigma`` to ``+2 sigma`` with the other components held at zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.pca import NodePCA  # noqa: E402

H, W = 43, 45


def draw_leaf(ax, grid: np.ndarray, lims: tuple, title: str = "") -> None:
    """Render one leaf grid as a height-colored 3D surface with fixed limits.

    Args:
        ax: A matplotlib 3D axis.
        grid: ``(H, W, 3)`` surface points.
        lims: ``((xmin,xmax),(ymin,ymax),(zmin,zmax))`` shared across all cells.
        title: Optional subplot title.
    """
    x, y, z = grid[..., 0], grid[..., 1], grid[..., 2]
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.15, alpha=0.35)
    (xl, yl, zl) = lims
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_zlim(zl)
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def viz_modes(
    pca: NodePCA,
    out_path: Path,
    species: str,
    n_modes: int = 4,
    sigmas: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> None:
    """Render leading PCA modes swept over +/- sigma.

    Args:
        pca: A trained :class:`NodePCA` (``coeff_std``/``components`` populated).
        out_path: Destination PNG.
        species: Species name, used only for the figure title.
        n_modes: Number of leading components to show (one per row).
        sigmas: Sweep positions in units of each component's std.
    """
    n_modes = min(n_modes, pca.components.shape[0])
    coeff_std = pca.coeff_std.cpu().numpy()
    n_comp = pca.components.shape[0]

    # Pre-decode every frame so we can pick shared axis limits.
    frames = np.empty((n_modes, len(sigmas), H, W, 3), dtype=np.float32)
    for r in range(n_modes):
        for c, s in enumerate(sigmas):
            coeff = np.zeros(n_comp, dtype=np.float32)
            coeff[r] = s * coeff_std[r]
            frames[r, c] = pca.decode(coeff).reshape(H, W, 3)

    pts = frames.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    lims = ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]))

    evr = pca.explained_variance_ratio.cpu().numpy()
    n_rows, n_cols = n_modes, len(sigmas)
    fig = plt.figure(figsize=(2.5 * n_cols, 2.6 * n_rows))
    for r in range(n_rows):
        for c, s in enumerate(sigmas):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            title = f"{s:+.0f}$\\sigma$"
            if c == 0:
                title = f"PC{r + 1} ({evr[r] * 100:.1f}% var)\n" + title
            draw_leaf(ax, frames[r, c], lims, title)
    fig.suptitle(f"{species} leaf PCA — leading modes swept -2σ … +2σ", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    """CLI entry point: fit the PCA for one species and render its modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        default="soybean",
        help="Species whose {species}.npy is loaded and fit.",
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
        default=Path(__file__).resolve().parent / "assets",
        help="Folder to write the PNG figure into.",
    )
    parser.add_argument(
        "--n-modes", type=int, default=4, help="Number of leading modes to show."
    )
    args = parser.parse_args()

    data = np.load(args.data_dir / f"{args.species}.npy")  # (n, H, W, 3)
    n = data.shape[0]
    print(f"loaded {n} {args.species} leaves, grid {tuple(data.shape[1:])}")

    pca = NodePCA(n_components=0.995)
    pca.train(data.reshape(n, -1))  # (n, H*W*3) -> (n, k)

    evr = pca.explained_variance_ratio.cpu().numpy()
    print(f"kept {pca.components.shape[0]} components for 99.5% variance")
    print(
        "top-6 explained-variance ratios: "
        + ", ".join(f"{v * 100:.1f}%" for v in evr[:6])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.species}_leaf_pca_modes.png"
    viz_modes(pca, out_path, args.species, n_modes=args.n_modes)


if __name__ == "__main__":
    main()
