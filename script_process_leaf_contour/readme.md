## Full 2D leaf dataset
We build the 2d leaf shape PCA on [FGLIR dataset](https://github.com/NUFE-AIAG/FGLIR) for soybean and on [Folio leaf dataset](https://archive.ics.uci.edu/dataset/338/folio) dataset for other species.

## Install SAM2 for auto segmentation
The sam2 code is already in the repo. Please follow the official [installation guidance](https://github.com/facebookresearch/sam2#installation).
```bash
cd script_process_leaf_contour/sam2
pip install -e .

# download sam2 ckpt
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## Annotation Pipeline
+ step 0: rename

+ step 1: segmentation using sam2

+ step 2: click keypoint

+ step 3: rotate according

+ step 4~6: find contour and fit 2d PCA