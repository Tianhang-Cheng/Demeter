"""Render a point-cloud prediction gallery for a PointTransformer run.

Reads ``<run>/result/<plant>_pred.npy`` and ``<plant>_pred_dist.npy`` produced by
``run.py train``/``test`` (or an ``infer`` output directory), pairs them with the
ground truth in the prepared dataset, and writes offscreen renders plus an
``index.html`` gallery and ``metrics.json``.

    python script_point_transformer/viz_predictions.py \
        --run outputs/point_transformer/soybean_e30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


REPO = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["leaf", "other_stem", "main_stem", "flower", "fruit"]
CLASS_COLORS = np.array([
    [0.24, 0.70, 0.29],   # leaf
    [0.96, 0.62, 0.12],   # other stem
    [0.78, 0.24, 0.20],   # main stem
    [0.86, 0.32, 0.76],   # flower
    [0.22, 0.47, 0.92],   # fruit
])


def heat(values):
    """Map [0, 1] scores to an RGB ramp without pulling in matplotlib."""
    v = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    stops = np.array([[0.19, 0.07, 0.23], [0.13, 0.42, 0.56],
                      [0.12, 0.67, 0.47], [0.71, 0.87, 0.17], [0.99, 0.91, 0.15]])
    pos = v * (len(stops) - 1)
    low = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - low)[:, None]
    return stops[low] * (1 - frac) + stops[low + 1] * frac


FOV = 45.0


def render(coord, colors, width, height, point_size, azimuths, elevation):
    """Render one coloured cloud from several azimuths; returns uint8 images."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)  # keep label colours exact
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = point_size
    renderer.scene.add_geometry("cloud", cloud, material)

    low, high = coord.min(axis=0), coord.max(axis=0)
    center = (low + high) / 2
    corners = np.array(np.meshgrid(*zip(low - center, high - center))).reshape(3, -1).T
    up = np.array([0.0, 0.0, 1.0])
    tan_v = np.tan(np.radians(FOV) / 2)
    tan_h = tan_v * width / height
    images = []
    for azimuth in azimuths:
        a, e = np.radians(azimuth), np.radians(elevation)
        # Preprocessing leaves the main stem along +Z, so orbit around Z.
        direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        # Fit the box tightly in this view: distance covers both screen axes plus depth.
        right = np.cross(-direction, up)
        right /= np.linalg.norm(right)
        screen_up = np.cross(right, -direction)
        distance = 1.06 * max(np.abs(corners @ screen_up).max() / tan_v,
                              np.abs(corners @ right).max() / tan_h) + (corners @ direction).max()
        renderer.setup_camera(FOV, center, center + direction * distance, up)
        images.append(np.asarray(renderer.render_to_image()))
    del renderer
    return images


# Instance clustering, matching script_auto_reconstruction/recon.py for soybean:
# drop the points the boundary head calls a junction, then cluster the rest with
# DBSCAN at a radius derived from the cloud's own point spacing.
BOUNDARY_KEEP = 0.15
DBSCAN_EPS_SCALE = 0.9
DBSCAN_MIN_SAMPLES = 8


def instance_labels(coord, pred_dist, pred_semantic, mode="spacing",
                    grid=0.004, eps=0.012):
    """Group points into organ instances; -1 marks points left unassigned.

    ``spacing`` reproduces recon.py: delete the points the boundary head scores
    above the threshold, then DBSCAN the rest at a radius taken from the cloud's own
    point spacing. Preprocessing already normalises every plant to unit radius, so
    that ties the radius to point density rather than organ size -- point count
    spans 77x across the 78 released plants while plant size spans 1.7x, and dense
    scans get a small radius and split their organs.

    ``fixed`` removes the density dependence: voxelise to a fixed grid, then cluster
    each predicted semantic class at a fixed radius. On the held-out plants this
    corrects the organ count (1.7x -> 1.0x) without improving IoU.

    ``graph-cut`` drops the deletion step, which is what caps the others: it builds a
    kNN graph over voxels and cuts edges that straddle a predicted junction or a
    class change, so every point keeps an organ. It assigns 87% of points rather
    than 74% and scores the best of the three, but a third of annotated organs still
    come out below 0.25 IoU, so the remaining error is not the clustering's to fix.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.cluster import DBSCAN

    if mode == "graph-cut":
        return graph_cut_labels(coord, pred_dist, pred_semantic, grid)
    keep = pred_dist < BOUNDARY_KEEP
    labels = np.full(len(coord), -1, dtype=np.int64)
    if not keep.any():
        return labels, keep, 0.0
    if mode == "spacing":
        # T = 2 * mean over points of the 10th-nearest-neighbour distance.
        neighbours, _ = cKDTree(coord).query(coord, k=10, workers=-1)
        radius = float(np.nan_to_num(neighbours, posinf=0.0).max(axis=-1).mean() * 2
                       * DBSCAN_EPS_SCALE)
        labels[keep] = DBSCAN(eps=radius, min_samples=DBSCAN_MIN_SAMPLES,
                              n_jobs=-1).fit_predict(coord[keep])
        return labels, keep, radius

    inside = coord[keep]
    _, first, inverse = np.unique(np.floor(inside / grid).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    centres, classes = inside[first], pred_semantic[keep][first]
    voxel_labels = np.full(len(centres), -1, dtype=np.int64)
    nxt = 0
    for cls in np.unique(classes):
        part = classes == cls
        found = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES,
                       n_jobs=-1).fit_predict(centres[part])
        voxel_labels[part] = np.where(found >= 0, found + nxt, -1)
        if (found >= 0).any():
            nxt += found.max() + 1
    labels[keep] = voxel_labels[inverse]
    return labels, keep, eps


GRAPH_CUT_THRESHOLD = 0.25
GRAPH_CUT_NEIGHBOURS = 16
GRAPH_CUT_LINK = 3.0


def graph_cut_labels(coord, pred_dist, pred_semantic, grid=0.004):
    """Connected components of a kNN graph with junction and class edges removed."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    _, first, inverse = np.unique(np.floor(coord / grid).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    centres, score, classes = coord[first], pred_dist[first], pred_semantic[first]
    distance, index = cKDTree(centres).query(centres, k=GRAPH_CUT_NEIGHBOURS + 1, workers=-1)
    src = np.repeat(np.arange(len(centres)), GRAPH_CUT_NEIGHBOURS)
    dst = index[:, 1:].ravel()
    edge = ((distance[:, 1:].ravel() < GRAPH_CUT_LINK * grid)
            & (np.maximum(score[src], score[dst]) < GRAPH_CUT_THRESHOLD)
            & (classes[src] == classes[dst]))
    graph = coo_matrix((np.ones(edge.sum()), (src[edge], dst[edge])),
                       shape=(len(centres),) * 2)
    _, found = connected_components(graph, directed=False)
    sizes = np.bincount(found)
    found = np.where(sizes[found] >= DBSCAN_MIN_SAMPLES, found, -1)
    ids = np.unique(found[found >= 0])
    lookup = np.full(found.max() + 2, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids))
    labels = lookup[found][inverse]
    return labels, labels >= 0, float(GRAPH_CUT_THRESHOLD)


def instance_metrics(gt_instance, pred_instance, keep):
    """Counts, plus the best IoU per annotated organ, scored two ways.

    ``coverage`` is measured only on the points the boundary filter kept, so it
    rates the clustering alone; ``coverage_all`` scores against the whole organ, so
    the points the filter discarded count against it. Reporting only the first
    rewards a filter that throws more away, which is why both are here.
    """
    assigned = pred_instance >= 0
    ids, sizes = np.unique(pred_instance[assigned], return_counts=True)
    best_kept, best_all = [], []
    for organ in np.unique(gt_instance):
        mask = gt_instance == organ
        hit_ids, hit_counts = np.unique(pred_instance[mask & assigned], return_counts=True)
        if not len(hit_ids):
            best_kept.append(0.0)
            best_all.append(0.0)
            continue
        hit_sizes = sizes[np.searchsorted(ids, hit_ids)]
        best_kept.append(float((hit_counts / ((mask & assigned).sum() + hit_sizes - hit_counts)).max()))
        best_all.append(float((hit_counts / (mask.sum() + hit_sizes - hit_counts)).max()))
    return dict(
        gt_instances=int(len(np.unique(gt_instance))),
        pred_instances=int(len(ids)),
        kept=float(keep.mean()),
        coverage=float(np.mean(best_kept)),
        matched=float(np.mean([b >= 0.5 for b in best_kept])),
        coverage_all=float(np.mean(best_all)),
        matched_all=float(np.mean([b >= 0.5 for b in best_all])),
    )


def instance_colors(labels):
    """Distinct hues per instance via a golden-angle rotation; grey for dropped points."""
    ids = np.unique(labels[labels >= 0])
    lookup = {int(i): n for n, i in enumerate(ids)}
    hue = np.array([(lookup.get(int(v), 0) * 0.61803398875) % 1.0 for v in labels])
    sat = np.where(labels >= 0, 0.62, 0.0)
    val = np.where(labels >= 0, 0.92, 0.80)
    i = np.floor(hue * 6).astype(int) % 6
    f = hue * 6 - np.floor(hue * 6)
    p, q, t = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
    return np.select([(i == n)[:, None] for n in range(6)],
                     [np.stack([val, t, p], -1), np.stack([q, val, p], -1),
                      np.stack([p, val, t], -1), np.stack([p, q, val], -1),
                      np.stack([t, p, val], -1), np.stack([val, p, q], -1)])


def confusion_counts(gt_semantic, pred_semantic):
    """Per-class intersection, union and ground-truth support, for pooling."""
    counts = np.zeros((3, len(CLASS_NAMES)), dtype=np.int64)
    for index in range(len(CLASS_NAMES)):
        gt_mask, pred_mask = gt_semantic == index, pred_semantic == index
        counts[:, index] = ((gt_mask & pred_mask).sum(), (gt_mask | pred_mask).sum(),
                            gt_mask.sum())
    return counts


def metrics(gt_semantic, pred_semantic, gt_dist, pred_dist, counts):
    intersection, union, support = counts
    per_class = {
        name: dict(
            support=int(support[index]),
            iou=float(intersection[index] / union[index]) if union[index] else None,
            recall=float(intersection[index] / support[index]) if support[index] else None,
        )
        for index, name in enumerate(CLASS_NAMES)}
    ious = [c["iou"] for c in per_class.values() if c["iou"] is not None and c["support"]]
    return dict(
        points=int(len(gt_semantic)),
        accuracy=float((gt_semantic == pred_semantic).mean()),
        miou=float(np.mean(ious)) if ious else None,
        dist_mae=float(np.abs(gt_dist - pred_dist).mean()),
        dist_mse=float(((gt_dist - pred_dist) ** 2).mean()),
        # Reconstruction thresholds the boundary score, so its ranking matters
        # more than its absolute calibration.
        dist_corr=correlation(gt_dist, pred_dist),
        dist_mean=float(pred_dist.mean()),
        dist_max=float(pred_dist.max()),
        per_class=per_class,
    )


def correlation(gt, pred):
    if gt.std() == 0 or pred.std() == 0:
        return None
    return float(np.corrcoef(gt, pred)[0, 1])


def pooled(counts, summary):
    """Dataset-level metrics: counts summed over plants, not per-plant means.

    This is the pooled mIoU / mAcc that Pointcept's evaluator reports, so classes
    absent from the split still count as zero in the 5-class means.
    """
    intersection, union, support = counts
    per_class = {
        name: dict(
            support=int(support[index]),
            iou=float(intersection[index] / union[index]) if union[index] else 0.0,
            recall=float(intersection[index] / support[index]) if support[index] else 0.0,
            plants=sum(1 for s in summary if s["per_class"][name]["support"]),
        )
        for index, name in enumerate(CLASS_NAMES)}
    points = sum(s["points"] for s in summary)
    return dict(
        plants=len(summary),
        points=points,
        accuracy=float(sum(s["accuracy"] * s["points"] for s in summary) / points),
        miou=float(np.mean([c["iou"] for c in per_class.values()])),
        macc=float(np.mean([c["recall"] for c in per_class.values()])),
        dist_mae=float(sum(s["dist_mae"] * s["points"] for s in summary) / points),
        dist_corr=(None if any(s["dist_corr"] is None for s in summary) else
                   float(np.mean([s["dist_corr"] for s in summary]))),
        gt_instances=sum(s["gt_instances"] for s in summary),
        pred_instances=sum(s["pred_instances"] for s in summary),
        kept=float(sum(s["kept"] * s["points"] for s in summary) / points),
        coverage=float(np.mean([s["coverage"] for s in summary])),
        matched=float(np.mean([s["matched"] for s in summary])),
        coverage_all=float(np.mean([s["coverage_all"] for s in summary])),
        matched_all=float(np.mean([s["matched_all"] for s in summary])),
        per_class=per_class,
    )


def find_ground_truth(name, data_root):
    for split in ("test", "train"):
        path = data_root / split / f"{name}.pth"
        if path.is_file():
            return path, split
    raise FileNotFoundError(f"No prepared ground truth for {name} under {data_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory holding result/")
    parser.add_argument("--data-root", type=Path, default=REPO / "data/point_transformer/soybean")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <run>/viz")
    parser.add_argument("--samples", nargs="*", default=None, help="Plant names; default all found")
    parser.add_argument("--max-points", type=int, default=400_000, help="Render subsample cap")
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--azimuths", type=float, nargs="+", default=[0.0, 90.0])
    parser.add_argument("--elevation", type=float, default=8.0)
    parser.add_argument("--instance-eps", choices=("graph-cut", "spacing", "fixed"),
                        default="graph-cut",
                        help="graph-cut assigns every point and scores best on the held-out "
                             "plants; spacing reproduces recon.py; fixed makes the DBSCAN "
                             "radius independent of point density")
    parser.add_argument("--instance-grid", type=float, default=0.004,
                        help="voxel size for --instance-eps fixed")
    parser.add_argument("--instance-radius", type=float, default=0.012,
                        help="DBSCAN radius for --instance-eps fixed")
    args = parser.parse_args()

    run = args.run.resolve()
    result = run / "result"
    output = (args.output or run / "viz").resolve()
    names = args.samples or sorted(p.name[: -len("_pred.npy")] for p in result.glob("*_pred.npy"))
    if not names:
        parser.exit(1, f"Error: no *_pred.npy in {result}\n")
    output.mkdir(parents=True, exist_ok=True)

    summary, totals = [], np.zeros((3, len(CLASS_NAMES)), dtype=np.int64)
    for name in names:
        gt_path, split = find_ground_truth(name, args.data_root.resolve())
        sample = torch.load(gt_path, map_location="cpu", weights_only=True)
        coord = sample["coord"].numpy()
        pred_semantic = np.load(result / f"{name}_pred.npy").reshape(-1)
        pred_dist = np.load(result / f"{name}_pred_dist.npy").reshape(-1)
        gt_semantic = sample["semantic_gt5"].numpy().reshape(-1)
        gt_dist = sample["inv_dists"].numpy().reshape(-1)
        if not len(pred_semantic) == len(pred_dist) == len(coord):
            raise ValueError(f"{name}: predictions do not match the {len(coord)} prepared points")

        counts = confusion_counts(gt_semantic, pred_semantic)
        totals += counts
        record = metrics(gt_semantic, pred_semantic, gt_dist, pred_dist, counts)
        gt_instance = sample["instance_gt"].numpy().reshape(-1)
        pred_instance, boundary_keep, threshold = instance_labels(
            coord, pred_dist, pred_semantic, args.instance_eps,
            args.instance_grid, args.instance_radius)
        record.update(name=name, split=split, dbscan_threshold=threshold,
                      **instance_metrics(gt_instance, pred_instance, boundary_keep))

        keep = np.arange(len(coord))
        if len(coord) > args.max_points:
            keep = np.random.default_rng(2025).choice(len(coord), args.max_points, replace=False)
        panels = {
            "rgb": sample["color"].numpy()[keep],
            "gt_semantic": CLASS_COLORS[gt_semantic[keep]],
            "pred_semantic": CLASS_COLORS[pred_semantic[keep]],
            "error": np.where((gt_semantic == pred_semantic)[keep, None],
                              np.array([0.82, 0.82, 0.84]), np.array([0.85, 0.12, 0.12])),
            "gt_dist": heat(gt_dist[keep]),
            "pred_dist": heat(pred_dist[keep]),
            "gt_instance": instance_colors(gt_instance[keep]),
            "pred_instance": instance_colors(pred_instance[keep]),
        }
        record["views"] = {}
        for panel, colors in panels.items():
            images = render(coord[keep], colors, args.width, args.height,
                            args.point_size, args.azimuths, args.elevation)
            files = []
            for index, image in enumerate(images):
                relative = f"{name}_{panel}_v{index}.png"
                o3d.io.write_image(str(output / relative), o3d.geometry.Image(image))
                files.append(relative)
            record["views"][panel] = files
        summary.append(record)
        print(f"{name} ({split}): acc {record['accuracy']:.3f} mIoU {record['miou']:.3f} "
              f"| instances {record['pred_instances']}/{record['gt_instances']} "
              f"coverage {record['coverage']:.3f}", flush=True)

    dataset = pooled(totals, summary)
    (output / "metrics.json").write_text(json.dumps(
        dict(run=str(run), data_root=str(args.data_root), dataset=dataset,
             samples=summary), indent=2))
    (output / "index.html").write_text(build_page(run, summary, dataset))
    print(f"\npooled over {dataset['plants']} plants: accuracy {dataset['accuracy']:.3f} "
          f"mIoU {dataset['miou']:.3f} mAcc {dataset['macc']:.3f} "
          f"boundary corr {dataset['dist_corr']:.3f}")
    print(f"instances ({args.instance_eps} eps): {dataset['pred_instances']} predicted vs "
          f"{dataset['gt_instances']} annotated "
          f"({dataset['pred_instances'] / dataset['gt_instances']:.2f}x), "
          f"kept {dataset['kept']:.3f}")
    print(f"  coverage {dataset['coverage']:.3f} on kept points, "
          f"{dataset['coverage_all']:.3f} end to end; "
          f"matched@0.5 {dataset['matched']:.3f} / {dataset['matched_all']:.3f}")
    print(f"Wrote {output / 'index.html'}")


PANEL_TITLES = {
    "rgb": "Input RGB",
    "gt_semantic": "GT semantics",
    "pred_semantic": "Predicted semantics",
    "error": "Errors (red)",
    "gt_dist": "GT boundary score",
    "pred_dist": "Predicted boundary score",
    "gt_instance": "GT organ instances",
    "pred_instance": "Clustered instances",
}


def build_page(run, summary, dataset):
    def cell(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    legend = "".join(
        f'<span class="chip"><i style="background:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"></i>{n}</span>'
        for n, c in zip(CLASS_NAMES, CLASS_COLORS))
    rows = "".join(
        f"<tr><td><a href='#{s['name']}'>{s['name']}</a></td><td>{s['split']}</td>"
        f"<td>{s['points']:,}</td><td>{cell(s['accuracy'])}</td><td>{cell(s['miou'])}</td>"
        + "".join(f"<td>{cell(s['per_class'][n]['iou'])}</td>" for n in CLASS_NAMES)
        + f"<td>{cell(s['dist_mae'], 4)}</td></tr>"
        for s in summary)
    # Pooled over every point in the split, so absent classes count as zero.
    mean_row = (f"<tr class='mean'><td>pooled</td><td>{dataset['plants']} plants</td>"
                f"<td>{dataset['points']:,}</td><td>{cell(dataset['accuracy'])}</td>"
                f"<td>{cell(dataset['miou'])}</td>"
                + "".join(f"<td>{cell(dataset['per_class'][n]['iou'])}</td>" for n in CLASS_NAMES)
                + f"<td>{cell(dataset['dist_mae'], 4)}</td></tr>")

    blocks = []
    for s in summary:
        panels = "".join(
            f"<figure><figcaption>{PANEL_TITLES[p]}</figcaption>"
            + "".join(f"<img src='{f}' loading='lazy' alt='{p}'>" for f in files)
            + "</figure>"
            for p, files in s["views"].items())
        blocks.append(
            f"<section id='{s['name']}'><h2>{s['name']} <small>{s['split']} split &middot; "
            f"{s['points']:,} points &middot; acc {cell(s['accuracy'])} &middot; "
            f"mIoU {cell(s['miou'])} &middot; boundary MAE {cell(s['dist_mae'], 4)}</small></h2>"
            f"<div class='panels'>{panels}</div></section>")

    return f"""<!doctype html>
<meta charset="utf-8"><title>PointTransformer soybean predictions</title>
<style>
:root {{ color-scheme: light; }}
body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 24px; max-width: 1400px; color: #1b1b1f; }}
h1 {{ margin: 0 0 4px; font-size: 22px; }}
p.meta {{ color: #5c5c66; margin: 0 0 18px; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; margin-bottom: 28px; }}
th, td {{ border-bottom: 1px solid #e3e3e8; padding: 5px 10px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
tr.mean td {{ font-weight: 600; border-top: 2px solid #1b1b1f; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }}
.chip i {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
section {{ border-top: 1px solid #e3e3e8; padding-top: 14px; margin-top: 14px; }}
h2 small {{ font-weight: 400; color: #5c5c66; font-size: 13px; }}
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
figure {{ margin: 0; }}
figcaption {{ font-weight: 600; margin-bottom: 4px; }}
img {{ width: 100%; max-width: 100%; border: 1px solid #e3e3e8; border-radius: 6px; background: #fff; }}
</style>
<h1>PointTransformer soybean predictions</h1>
<p class="meta">{run}</p>
<p>{legend}</p>
<table><thead><tr><th>plant</th><th>split</th><th>points</th><th>acc</th><th>mIoU</th>
{''.join(f'<th>IoU {n}</th>' for n in CLASS_NAMES)}<th>boundary MAE</th></tr></thead>
<tbody>{rows}{mean_row}</tbody></table>
{''.join(blocks)}
"""


if __name__ == "__main__":
    main()
