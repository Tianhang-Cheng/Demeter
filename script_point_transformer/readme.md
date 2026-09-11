# PointTransformer: released point clouds to training and reconstruction

Run the commands below from the repository root. This directory owns the
Demeter-specific data preparation, configuration and launch commands. Pointcept
stays under `third_party/`; generated data and experiment outputs are ignored by Git.

```text
sample_params/soybean/instances/<plant>/   # downloaded source, preserved
  raw/<organ>.ply                         # per-organ points in the raw scan frame
  info/class.txt                         # organ ID -> semantic class
  info/parent.txt                        # topology; root has parent -1
  graph.pkl                              # fitted main-stem orientation
script_point_transformer/
  prepare_data.py                         # CPU preprocessing
  config.py                              # portable training recipe
  run.py                                 # train / test / infer (+ reconstruction)
  test_prepare_data.py                    # CPU regression tests
data/point_transformer/soybean/
  train/<plant>.pth
  test/<plant>.pth
  manifest.json                          # splits, transforms, thresholds, counts
outputs/point_transformer/<run>/          # config, logs, checkpoints, test predictions
outputs/reconstruction/<plant>/          # staged input, predictions, graph, mesh
```

## 1. Download prepared data (moving to another machine)

The prepared dataset contains **67 train / 11 test samples**, no validation
set, and 76,362,794 points. Training uses these files directly; you can skip
source-data restoration and preprocessing below.

Download the **2.24 GB archive** from
[Hugging Face](https://huggingface.co/datasets/TianhangCheng7/DemeterData/tree/main/point_transformer).
The published archive's size and SHA-256 have been verified against the local
release. All 79 archive members also match the original local files byte-for-byte.

From the repository root on the new machine:

```bash
hf download TianhangCheng7/DemeterData point_transformer/soybean.tar.gz point_transformer/soybean.tar.gz.sha256 --repo-type dataset --local-dir outputs/downloads
# Linux: check the downloaded archive before extracting.
(cd outputs/downloads/point_transformer && sha256sum -c soybean.tar.gz.sha256)
tar -xzf outputs/downloads/point_transformer/soybean.tar.gz
```

Install the download CLI with `pip install -U huggingface_hub` if needed.
Extract into a clean checkout, or first move any existing
`data/point_transformer/soybean/` elsewhere to avoid mixing old splits.
The archive restores `train/`, `test/` and `manifest.json` under that directory.
Allow about 4.28 GB for the extracted data, plus space for the archive.
Then continue with [training](#3-train-or-fine-tune).

For reproducible downloads, add
`--revision b5f4e16387a4d7b27ef85d2d87f2c0673448b6cd` to `hf download`.
The archive SHA-256 is
`f8c34f2464a65721bbe195c86077cf12f914af4f8a0ab9e35bd7b5f9ac990a59`.

### Optional: restore source point clouds to regenerate the data

Use **the processed per-organ point clouds**, not the 607 unsegmented raw meshes.
The small parameter examples in a checkout alone do not contain the required raw
point clouds. Download the approximately 1.54 GB archive described in the
[main README](../readme.md#full-soybean-parametric-meshes-and-corresponding-point-cloud-segmentations-78-plants):

```bash
hf download TianhangCheng7/DemeterData sample_params.tar.gz --repo-type dataset --local-dir .
tar -xzf sample_params.tar.gz
```

This restores `sample_params/`, including `raw/*.ply`, annotations and fitted
graphs. If you have edited that directory, extract to a separate location and
pass its `sample_params/soybean/instances` directory with `--source` instead.

## 2. Prepare train and test samples

Preprocessing needs Python, NumPy, SciPy, PyTorch and Open3D; it runs on CPU,
including Windows, without Pointcept's CUDA extensions.

```bash
python script_point_transformer/prepare_data.py
```

Use `--jobs 4` to process four plants concurrently if sufficient RAM is
available. Large scans can have over two million points. Progress is recorded
after each sample; resume an interrupted run with the same arguments plus
`--resume`. Do not modify source files between the original run and its resume.

The released soybean split contains **67 train / 11 test samples**, with no
validation set. The exact lists are versioned in
[`soybean_split.json`](soybean_split.json), and are used automatically when
preparing all 78 released samples.

For another collection, the default is approximately 85% train / 15% test,
using seed 2025. Names ending in `_i` and `_o` with the same prefix stay
together. Use `--test-fraction` to create a new split, or `--split-file` to
supply explicit train/test lists. The manifest records the actual split; this
does not claim to reproduce the original paper's training partition.

For a quick one-plant preprocessing check:

```bash
python script_point_transformer/prepare_data.py --samples 24_o --test-fraction 0 --output outputs/point_transformer/data_smoke
```

An output directory must be empty unless using `--resume`. Use a new output directory when changing the
source, split or preprocessing settings; this prevents stale files leaking into
another split. A failure reports the sample being processed. The manifest's
status becomes `complete` only after all samples succeed; training and testing
reject an incomplete dataset.

### Target definition and stored fields

For every retained point `p_i`, compute the nearest Euclidean distance `d_i` to
points from **other organ instances**, regardless of their semantic class.
Within its own instance, query `k=10` nearest neighbors, including itself.
Let `r_i` be the largest of those ten distances. Compute a single threshold per
plant and a scalar target per point:

```python
T = 2 * mean(r_i)
inv_dists[i] = T / max(d_i, T)
```

Points within `T` of another instance receive 1; points farther away receive
`T / d_i`. The threshold adapts to point spacing. There is no maximum-value
normalization and no exponential transform.

The implementation follows the archived Plant preprocessing details: black
points are removed from queries and saved data, but remain in other-instance
reference clouds. Missing self-neighbors for small organs contribute zero,
matching the original infinity handling. Missing RGB is filled with gray.
`--keep-black` explicitly changes the filtering behavior. Degenerate plants
with zero density threshold fail with an error.

The fitted root quaternion aligns the main stem to **+Z**, which is what all 78
released plants come out as (the main stem's principal axis is within 17 degrees
of +Z on every one). Coordinates are centered at the retained points' mean and
divided by the sorted 95th-percentile radius computed from all original points
around their mean. Normals are estimated in normalized coordinates (radius 0.1,
at most 30 neighbors). Distances and `T` are both computed before scaling, so
their ratio is scale invariant. Use `--alignment none` only for input already
aligned to +Z. Note that `config.py`'s augmentation rotates about x as though the
stem were along that axis; [`config_fixed.py`](config_fixed.py) corrects it.

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `coord` | N x 3, float32 | Centered, aligned, normalized coordinates |
| `color` | N x 3, float32 | RGB in [0, 1], matching the archived data |
| `normal` | N x 3, float32 | Estimated normals |
| `semantic_gt5` | N, int64 | 0 leaf, 1 other stem, 2 main stem, 3 flower, 4 fruit |
| `instance_gt` | N, int64 | Original numeric organ ID from the PLY filename |
| `inv_dists` | N, float32 | Truncated inverse-distance target |
| `scene_id` | string | Plant directory name |

Source class IDs 0/1/2/3 map to 0/1/3/4; the unique root identified by
`parent.txt` is assigned class 2. The saved tensors can be read with
`torch.load(path, map_location="cpu", weights_only=True)`.

Compatibility note: the archived pipeline stores RGB in [0, 1] even though the
inherited Pointcept `NormalizeColor` divides by 127.5. This recipe preserves that
behavior, including the existing color augmentations, to match the released
checkpoint. Changing color scaling requires a coordinated training/inference
recipe change and retraining.

## 3. Train or fine-tune

Use the Linux/CUDA `demeter` environment from the [main README](../readme.md#2-requirements)
and the extra dependencies in [the reconstruction environment instructions](../script_auto_reconstruction/readme.md#environment).
Preprocessing does not need a GPU; network training/inference needs the bundled
`pointops` CUDA extension and the PyG dependencies.

Despite the `PointTransformer_V3` directory name, the released Demeter recipe
uses **`PT-v2m2-custom`**, with a semantic head and a scalar distance head. The
new config inherits this architecture and its training transforms. The loss is
semantic cross-entropy plus `15 * MSE(predicted_inv_dist, inv_dists)`.

```bash
python script_point_transformer/run.py train --output outputs/point_transformer/soybean
```

Defaults: 150 epochs, total batch size 4, one GPU and four workers. There is
no validation loader or best-checkpoint selection. Training saves
`model/model_last.pth`, then evaluates the **final checkpoint** on the test set
once. Test results are not used to choose an epoch. The run directory also
contains `config.py`, `train.log`, TensorBoard events and `result/` predictions.

```bash
# Initialize from an existing checkpoint in a new run directory.
python script_point_transformer/run.py train --weight third_party/PointTransformer_V3/Pointcept/exp/soybean3d/plant3/model/model_last.pth --output outputs/point_transformer/finetune

# Resume an interrupted run; use the same epoch/batch settings as that run.
python script_point_transformer/run.py train --resume --weight outputs/point_transformer/soybean/model/model_last.pth --output outputs/point_transformer/soybean

# Explicit held-out evaluation, with a fresh result directory.
python script_point_transformer/run.py test --weight outputs/point_transformer/soybean/model/model_last.pth --output outputs/point_transformer/soybean_test
```

Use `--data-root` for a different generated dataset. `--epochs`, `--batch-size`,
`--gpus` and `--workers` configure training; `--workers 0` is supported for
debugging. Training requires nonempty train/test splits and enough training
samples for a complete batch. `--dry-run` checks paths and prints the command
without launching Pointcept or writing outputs.

### Recipes

`--recipe released` (the default) reproduces the published config exactly and is
what the released checkpoint was trained with. `--recipe fixed` selects
[`config_fixed.py`](config_fixed.py), which corrects three defects measured in
that config and inherits everything else unchanged:

| | released | fixed |
| --- | --- | --- |
| colour units | `[0, 1]` data through `[0, 255]` transforms | `color_scale=255.0` |
| azimuth rotation | ±180° about **x**, while plants align to +Z | ±180° about **z** |
| boundary loss | MSE on the logit clipped to `[0, 1]` | `binary_cross_entropy_with_logits` |

The colour defect is the largest: `ChromaticJitter` added noise with std 9.39 on
top of a signal with std 0.185, and at test time `NormalizeColor` left a
near-constant −0.997 (std 0.001), so three of the nine input channels were noise
while training and a constant while testing. Colour is also the cue that
separates petiole from leaf — the weakest class — and it does so consistently
within a plant, in all 25 plants sampled.

Swapping MSE for cross-entropy needs two accompanying changes, both recorded in
`config_fixed.py`: `dist_loss_weight` drops from 15 to 1 (cross-entropy starts at
ln 2 whatever the target, so weight 15 starts 17x above the released boundary term
and flattens the semantic head), and `dist_prior` seeds the head's final bias at
logit of the target mean (AdamW moves a parameter by about the learning rate per
step, so a 100-epoch run cannot travel the −2.07 the target's mean requires;
raising the loss weight does not help, because Adam is invariant to gradient
scale).

`config_fixed.py` is not compatible with the released checkpoint, because the
colour fix changes the input distribution. Compare recipes at equal epochs — the
default epoch counts differ enough to confound the comparison otherwise.

### Throughput

Training is **CPU-bound on the data pipeline**, not on the GPU. Each plant holds
0.8-2 million points, and the per-sample transform chain (`ElasticDistortion`,
`GridSample`) dominates; the network itself needs roughly 0.13 s per iteration.
With the default `--workers 4`, one epoch of the 67-plant split measured about
20 s on one RTX PRO 6000, of which only ~2 s was GPU compute. Raising
`--workers` to 24 on a many-core host cut the same epoch to about 7 s. Set
`--workers` to what the machine's cores and RAM allow before tuning anything else.

The larger cost is the final test pass: the released recipe averages **13
test-time augmentations** per plant, roughly 6 min per plant, so evaluating 11
test plants takes about 65 min regardless of how long training took. `--tta`
trades that accuracy averaging for time:

| `--tta` | views | held-out evaluation |
| --- | --- | --- |
| `full` (default) | 13 | released recipe, ~6 min per plant |
| `rot4` | 4 x-axis rotations | ~3x faster |
| `single` | identity only | ~13x faster, ~30 s per plant |

`--tta` applies to `train` (its automatic final test), `test` and `infer`. Any
value other than `full` writes the derived config next to the run's other
outputs, so the exact augmentation list stays recorded. Report headline numbers
from `--tta full`; the smaller presets are for iteration and visualization.

### Train on another machine

Get the repository code and follow [the prepared-data download instructions](#1-download-prepared-data-moving-to-another-machine)
on the training machine. The data loader only needs the `.pth` files; inference staging also
uses the manifest's transforms. Recorded source paths are provenance and do
not need to exist on the training machine.

After installing the Linux/CUDA dependencies, run a short trial and evaluate
its final checkpoint on test:

```bash
python script_point_transformer/run.py train --data-root data/point_transformer/soybean --epochs 5 --output outputs/point_transformer/soybean_trial
```

Test evaluation runs automatically after training. For a separate test run:

```bash
python script_point_transformer/run.py test --data-root data/point_transformer/soybean --weight outputs/point_transformer/soybean_trial/model/model_last.pth --output outputs/point_transformer/soybean_trial_test
```

Full network training has not been run locally. The data preparation and CPU
integration checks do not establish test-set accuracy.

## 4. Infer and reconstruct

Choose a sample from `manifest.json`'s test split. This example uses `24_o` from
the one-plant preprocessing check above; substitute a held-out path for evaluation:

```bash
python script_point_transformer/run.py infer --sample outputs/point_transformer/data_smoke/train/24_o.pth --weight outputs/point_transformer/soybean/model/model_last.pth --output outputs/reconstruction/24_o --reconstruct
```

The command stages the input under the new output directory, omits ground-truth
labels, restores the coordinate-transform metadata, runs Pointcept and copies
both `normalized_pcd_pred.npy` and `normalized_pcd_pred_dist.npy` into place.
`--reconstruct` then runs boundary filtering, DBSCAN, topology inference and
Demeter fitting without visualization windows. It writes:

```text
outputs/reconstruction/24_o/
  normalized_pcd.pth
  normalized_pcd_pred.npy
  normalized_pcd_pred_dist.npy
  transform.pkl
  pcd.ply, pcd_unit_radius.ply
  pointcept/                 # inference config/log/results
  graph.pkl                  # fitted Demeter state
  params/plant_graph.pth     # fitted state also saved by reconstruction
  params/info/               # class.txt, parent.txt
  fit/                      # individual organ fits
  predict.ply               # reconstructed triangle mesh
```

Without `--reconstruct`, the command stops after predictions. To fit later:

```bash
python script_auto_reconstruction/recon.py --data_folder outputs/reconstruction/24_o --species soybean --no-viz
```

For a **new, unannotated scan**, first use the existing interactive alignment
step, then pass its normalized file to the same inference command:

```bash
python script_auto_reconstruction/normalize_data.py --point_path sample_point_cloud/val/65_i/pcd.ply
python script_point_transformer/run.py infer --sample sample_point_cloud/val/65_i/normalized_pcd.pth --weight third_party/PointTransformer_V3/Pointcept/exp/soybean3d/plant3/model/model_last.pth --output outputs/reconstruction/65_i --reconstruct
```

The published checkpoint is available as described in the [reconstruction guide](../script_auto_reconstruction/readme.md#step-2-infer-semantics-and-boundary-scores).
The existing reconstruction method is approximate and still depends on scan and
prediction quality; creating the training data does not guarantee a successful
fit for every plant. Other species can use `--source` and `--species`, but the
published checkpoint and this default training recipe target soybean.

## 5. Inspect predictions

`viz_predictions.py` turns a run's `result/` directory into a browsable report.
It pairs every `<plant>_pred.npy` / `<plant>_pred_dist.npy` with the prepared
ground truth, renders both heads offscreen from two azimuths, and writes
`index.html` plus `metrics.json`:

```bash
python script_point_transformer/viz_predictions.py --run outputs/point_transformer/soybean
```

Each plant gets six panels: input RGB, ground-truth semantics, predicted
semantics, a red error mask, and the ground-truth and predicted boundary
scores. The table reports per-plant accuracy, mIoU, per-class IoU and boundary
mean absolute error, computed on the full point set rather than on voxels, so
these numbers are comparable across plants but are not Pointcept's own mIoU.

Rendering uses Open3D's EGL offscreen renderer, so no display is needed.
`--samples` limits which plants are drawn, `--max-points` caps the render
subsample (metrics always use every point), and `--width`, `--height`,
`--point-size`, `--azimuths` and `--elevation` control the renders. Pass
`--output` to write elsewhere than `<run>/viz`.

### Grouping points into organs

The network predicts semantics and a boundary score, not instances; organs are
grouped afterwards by [`utils/instances.py`](../utils/instances.py), which
`script_auto_reconstruction/recon.py` also calls — so the numbers below describe
exactly the organs the reconstruction fits its graph to, and a change to the
grouping reaches both. `--instance-eps` selects the method (`recon.py` takes the
same choice as `--instance-method`). Measured on the 11 held-out plants against
438 annotated organs:

| `--instance-eps` | organs | coverage | matched@0.5 | assigned |
| --- | --- | --- | --- | --- |
| `graph-cut` (default) | 417 (0.95x) | **0.511** | 0.491 | 0.875 |
| `spacing` (recon.py) | 750 (1.71x) | 0.466 | 0.503 | 0.740 |
| `fixed` | 419 (0.96x) | 0.453 | 0.477 | 0.726 |

Coverage is the best IoU per annotated organ against the **whole** organ, so points
left unassigned count against it; `coverage` in `metrics.json` scores only the
assigned points and is the more flattering of the two.

`spacing` reproduces `recon.py`: delete the points scored above 0.15, then DBSCAN at
a radius taken from the cloud's own spacing. Because preprocessing already puts every
plant at unit radius, that ties the radius to point density rather than organ size —
point count spans 77x across the released plants while plant size spans 1.7x, so dense
scans split their organs. `fixed` removes the density dependence and corrects the organ
count, but not the IoU. `graph-cut` drops the deletion step, which is what caps both:
it cuts kNN edges that straddle a junction or a class change, so every point keeps an
organ.

None of the three fixes the real limit. Per-organ IoU is bimodal — 34% of organs land
above 0.75 and 29% below 0.25, the latter being the thin petioles, which fragment.
Substituting a **perfect** boundary score raises coverage only from 0.504 to 0.537, and
seeding organs from the stem/petiole skeleton and growing geodesically scored *worse*
(0.477), because organs being merged is already rare. Grouping is not where the
remaining error is; a learned instance head is the change that would move it.

`recon.py` defaults to `spacing`, so reconstruction output is unchanged until you pass
`--instance-method graph_cut`; only this report defaults to the better one.

## Verification

```bash
python -m unittest discover -s script_point_transformer -p 'test_*.py' -v
```

These CPU tests check targets against brute-force Euclidean distances, small
and degenerate organs, plant-level split isolation, semantic/instance labels,
nonidentity coordinate transforms, saved tensor files and inference staging.
They do not execute the CUDA network or establish trained-model accuracy.

## Publishing the prepared dataset

Package the complete local dataset with the standard-library script below.
It validates the train/test file lists and plant isolation, preserves the
files byte-for-byte, and writes an archive and SHA-256 checksum. Use a new
`--output` directory when rebuilding an existing release.

```bash
python script_point_transformer/package_data.py
hf auth login
hf upload TianhangCheng7/DemeterData outputs/releases/point_transformer point_transformer --repo-type dataset
```

Publishing requires write access to the dataset repository. The manifest's
original source paths are retained as provenance; those paths do not need to
exist on a machine using the prepared data.
