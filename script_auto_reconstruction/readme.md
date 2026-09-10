# Reconstruct Demeter from a point cloud

For training from the released segmented point clouds, use the
[PointTransformer guide](../script_point_transformer/readme.md). This guide
covers a new scan with an existing checkpoint. Run commands from the repository
root. Reconstruction depends on scan completeness and prediction quality.

## Environment

Use the Linux/CUDA `demeter` environment from the [main README](../readme.md#2-requirements).

```bash
conda activate demeter
pip install torch_scatter torch_cluster torch_geometric==2.5.3 addict SharedArray yapf==0.30.0 tensorboard
pip install third_party/PointTransformer_V3/Pointcept/libs/pointops
```

The bundled recipe uses `PT-v2m2-custom`, despite the third-party directory's
name. It does not require `spconv`. The repository's tested PyG version is 2.5.3.
Keep `yapf` pinned: Pointcept's config dump calls `FormatCode(..., verify=True)`,
which yapf removed after 0.31.

`pointops` compiles CUDA kernels for the architectures PyTorch was built for. On
a GPU newer than the wheel's targets (for example Blackwell, `sm_120`), name the
architecture and the matching toolkit explicitly:

```bash
CUDA_HOME=/path/to/cuda-12.8 TORCH_CUDA_ARCH_LIST="12.0" \
  pip install --no-build-isolation third_party/PointTransformer_V3/Pointcept/libs/pointops
```

Newer PyTorch also needs one edit in the vendored Pointcept: its schedulers
forward `verbose=` to `torch.optim.lr_scheduler`, which dropped that argument in
PyTorch 2.7. Dropping the forwarded argument restores the default behaviour:

```bash
sed -i '/^ *verbose=verbose,$/d' third_party/PointTransformer_V3/Pointcept/pointcept/utils/scheduler.py
```

## Step 1: align and normalize the scan

```bash
python script_auto_reconstruction/normalize_data.py --point_path sample_point_cloud/val/65_i/pcd.ply
```

Click the main stem bottom first, then a point toward its top. The script aligns
this direction to +X, centers the scan and divides by its 95th-percentile radius.
It saves `normalized_pcd.pth` and `transform.pkl` beside the scan. Existing
rotation clicks can be reused.

<img src="../assets/before_annotate.png" alt="Main stem alignment" width="300">

## Step 2: infer semantics and boundary scores

Download the published checkpoint from
[DemeterPointSeg](https://huggingface.co/TianhangCheng7/DemeterPointSeg/tree/main):

```bash
hf download TianhangCheng7/DemeterPointSeg exp.zip --local-dir .
unzip -o exp.zip -d third_party/PointTransformer_V3/Pointcept/
```

Use the shared runner to stage the scan, predict both outputs, and copy them
into a fresh reconstruction directory:

```bash
python script_point_transformer/run.py infer --sample sample_point_cloud/val/65_i/normalized_pcd.pth --weight third_party/PointTransformer_V3/Pointcept/exp/soybean3d/plant3/model/model_last.pth --output outputs/reconstruction/65_i
```

It produces `normalized_pcd_pred.npy` (semantic IDs) and
`normalized_pcd_pred_dist.npy` (truncated inverse-distance scores). Larger
scores indicate proximity to another organ; these are not signed distances.
Add `--reconstruct` to also run Step 3 without visualization windows.

## Step 3: build the graph and mesh

```bash
python script_auto_reconstruction/recon.py --data_folder outputs/reconstruction/65_i --species soybean --no-viz
```

Omit `--no-viz` for interactive visualization. Fitting may take roughly a minute
per organ, depending on the point count and GPU. Outputs include `graph.pkl`,
`params/plant_graph.pth`, `params/info/{parent,class}.txt`, individual fits in
`fit/`, and the reconstructed triangle mesh `predict.ply`.

## Step 4: decode again

`predict.ply` is already a mesh. To regenerate it from the fitted parameters,
arrange the graph and annotations in a separate instance folder:

```bash
mkdir -p outputs/reconstruction/65_i/decoded/info
cp outputs/reconstruction/65_i/graph.pkl outputs/reconstruction/65_i/decoded/graph.pkl
cp outputs/reconstruction/65_i/params/info/*.txt outputs/reconstruction/65_i/decoded/info/
python decode.py --data_folder sample_params --species soybean --instance_folder outputs/reconstruction/65_i/decoded --output outputs/reconstruction/65_i/decoded.ply
```

The species PCA models come from `sample_params/`; the fitted plant comes from
`--instance_folder`. `--output` exports without opening a visualization window.
Generated results stay under `outputs/`, preserving the released source data.

## Pipeline visualization

Boundary scores identify points to remove before clustering:

<img src="../assets/distance.png" alt="Boundary scores" width="300">

Semantics distinguish the main stem, other stems and leaves:

<img src="../assets/semantics.png" alt="Semantic predictions" width="300">

DBSCAN separates the remaining points into organ instances:

<img src="../assets/init_segmentation.png" alt="Initial instances" width="300">

Removed points are assigned to the nearest cluster:

<img src="../assets/fixed_segmentation.png" alt="Completed instances" width="300">

The fitted reconstruction is compared with the input:

<img src="../assets/recon.png" alt="Reconstruction" width="300">
