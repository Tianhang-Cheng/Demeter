# Demeter: A Parametric Model of Crop Plant Morphology from the Real World (ICCV 2025)

[Project Page](https://tianhang-cheng.github.io/Demeter/) | [Mesh Dataset](https://huggingface.co/datasets/TianhangCheng7/DemeterData)

<img src="assets/teaser.png" alt="Demeter " width="600">

Demeter is a plant parametric models that is learned from 3D scans of real-world plants. It explicitly models the plant as a graph of stem and leaf.

## 1. Data

### Processed data

Small processed 3D parametric plant examples are included in this repository.

### Raw data

The raw soybean mesh data can be found on [DemeterData](https://huggingface.co/datasets/TianhangCheng7/DemeterData). It contains 607 unprocessed meshes for 3D generation and representation learning. These raw scans are neither aligned nor segmented.

<img src="assets/dataset_row.png" alt="Demeter " width="600">

### 3D leaf deformation data (for PCA training)

Per-species arrays of fitted 3D leaf surfaces `[n_leaf, 43, 45, 3]` (hosted on
[DemeterData](https://huggingface.co/datasets/TianhangCheng7/DemeterData)) are
the samples used to fit `sample_params/<species>/3d_leaf_pca.pth`. 

### Full soybean parametric meshes and corresponding point-cloud segmentations (78 plants)

The complete processed dataset is hosted on
[DemeterData](https://huggingface.co/datasets/TianhangCheng7/DemeterData) as a
single archive. It contains all 78 soybean instances, the sample instances for
the other species, and their fitted parametric graphs and point-cloud
segmentations. Within each `sample_params/<species>/instances/<sample>/`
directory, `raw/*.ply` stores the per-organ segments in the original scan
coordinate frame;

Install the Hugging Face CLI with `pip install -U huggingface_hub`, then run the
following commands from the repository root to restore the complete processed
`sample_params/` folder:

```bash
set -euo pipefail
REPO="TianhangCheng7/DemeterData"
ARCHIVE="sample_params.tar.gz"
hf download "$REPO" "$ARCHIVE" --repo-type dataset --local-dir .
tar -xzf "$ARCHIVE"
rm -f "$ARCHIVE"
echo "Done. Processed data restored under ./sample_params"
```

## 2. Requirements

### Environment (Tested)

+ Linux
+ Python 3.11
+ CUDA 12.1
+ Pytorch 2.5.0

### Dependencies
Install PyTorch and other dependencies.
```bash
conda create -n demeter python=3.11 -y
conda activate demeter
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# basic dependencies for decoding
pip install -r requirements.txt
```

Install in editable mode

```
pip install -e .
```

for reconstruction from 3d point cloud (`script_auto_reconstruction`), a few extra dependencies (Point Transformer / Pointcept) are installed into this same `demeter` environment — no separate environment is needed; see [script_auto_reconstruction/readme.md](script_auto_reconstruction/readme.md#environment). We still recommend using manual annotation to create demeter parameters for now.

## 3. Usage

### a) Visualize parametric plant & segmented point cloud

decode demeter parameter to 3d mesh of soybean

```python
python decode.py --data_folder sample_params --sample_name 24_o --species soybean

python decode.py --data_folder sample_params --sample_name 08 --species ribes 

python decode.py --data_folder sample_params --sample_name 10008da --species maize

python decode.py --data_folder sample_params --sample_name 1 --species tobacco

python decode.py --data_folder sample_params --sample_name 02 --species rose
```

Visualize both the parametric mesh and the original segmented point cloud.
Download and extract the processed dataset above before running these examples.
With `--show_segmentation_color`, each organ segment receives a distinct color;
otherwise, the original RGB colors are used.

```bash
python viz_segmentation.py --sample_name 24_o --species soybean --data_folder sample_params --show_segmentation_color

python viz_segmentation.py --sample_name 08 --species ribes  --data_folder sample_params

python viz_segmentation.py --sample_name 2_i --species soybean --data_folder sample_params 
```

Visualize every soybean in the `instances` folder:

```bash
python viz_segmentation.py --batch --species soybean --data_folder sample_params --instance_root instances
```

### b) Reconstruction parametric plant from point cloud

+ (Optional): Fitting 2D leaf shape from images if you need other species
[script_process_leaf_contour/readme.md](script_process_leaf_contour/readme.md)

+ (Optional): Fitting 3D leaf shape from aligned leaf mesh array. See
[script_fit_leaf_3d_pca/readme.md](script_fit_leaf_3d_pca/readme.md) for
downloading the data and (re)fitting / visualizing the 3D leaf PCA.

+ Step 1: Get point cloud
Monocular RGB video -> Raw 3D point clouds \
[third_party/2d-gaussian-splatting/readme.md](third_party/2d-gaussian-splatting/readme.md)

+ Step 2, Option A: manual annotate point clouds -> Demeter parameters \
[script_manual_annotation/readme.md](script_manual_annotation/readme.md)

+ Step 2, Option B: automatic multi-stage 3D point clouds -> Demeter parameters \
[script_auto_reconstruction/readme.md](script_auto_reconstruction/readme.md)
Note that this method is not accurate, so it's more recommanded to use manual segementation

+ Step 2, Option C: automatic feed-forward one-pass 3D point clouds -> Demeter parameters \
Working in progress.

+ Others: raw 3D point clouds -> baseline L-system parameteres \
[third_party/CropCraft/readme.md](third_party/CropCraft/readme.md)

### c) Simulation

Please refer to [Helios Tutorial](https://github.com/PlantSimulationLab/PyHelios/blob/master/docs/plugin_photosynthesis.md) for now.

## 4. Release Note

- [ ] editing tutorial (TBD)
- [x] full soybean 2d image dataset (2026-8-13)
- [x] learning leaf shape PCA from 2D leaf scanns (2026-5-26)
- [x] release 3D leaf deformation arrays for 3D leaf PCA training (2026-8-13)
- [x] building demeter representation from your own annotated 3d point cloud (2026-4-24)
- [x] full soybean 3d dataset (2025-12-17)
- [x] sample data of other species (2025-11-1)
- [x] sample data of soybean (2025-10-7)
- [x] decoding (2025-10-7)
- [x] reconstruction from 3d point cloud (2025-10-8)
- [x] L-system baseline (2025-10-13)

## 5. Acknowledgement
This project is supported by NSF Awards #1847334 #2331878, #2340254, #2312102, #2414227, and #2404385. We greatly appreciate the NCSA for providing computing resources.

## 6. License
This code is released under the **Academic Research License (Non-Commercial)**.
For commercial inquiries, please contact [shenlong@illinois.edu](mailto:shenlong@illinois.edu). 
For code issue and academic collaboration, please contact [tcheng12@illinois.edu](mailto:tcheng12@illinois.edu).
