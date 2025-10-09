# Demeter: A Parametric Model of Crop Plant Morphology from the Real World (ICCV 2025)

[Project Page](https://tianhang-cheng.github.io/Demeter-project.github.io/) | [Dataset (Coming soon)]()

<img src="assets/teaser.png" alt="Demeter " width="600">

Demeter is a plant parametric models that is learned from 3D scans of real-world plants. It explicitly models the plant as a graph of stem and leaf.

## Requirements

### Environment (Tested)

+ Linux
+ Python 3.11
+ CUDA 12.1
+ Pytorch 2.5.0

### Dependencies
Install PyTorch (not included here) and other dependencies:
```bash
conda create -n demeter python=3.11 -y
conda activate demeter

# basic dependencies for decoding
pip install -r requirements.txt
```

for reconstruction from 3d point cloud, it is recommended to create a new envrionment following instruction in [Pointcept](https://github.com/Pointcept/Pointcept)

## Data

the sample data is already included in the code.

## Decode fitted parametric plant

decode demeter parameter to 3d mesh of soybean

```python
python decode.py
```

## Reconstruction from point cloud

For detailed instructions on reconstructing Demeter representations from raw 3D point clouds, see [script_reconstruction/readme.md](script_reconstruction/readme.md).


## Release Plan

- [x] sample data of soybean (2025-10-7)
- [x] decoding (2025-10-7)
- [ ] editing tutorial (TBD)
- [ ] sample data of other species (TBD)
- [x] reconstruction from 3d point cloud (2025-10-8)
- [ ] fitting demeter representation from annotated 3d point cloud (TBD)
- [ ] learning leaf shape PCA from 2D leaf scanns (TBD)
- [ ] L-system baseline (TBD)
- [ ] full soybean 3d dataset (TBD)

## Acknowledgement
This project is supported by NSF Awards #1847334 #2331878, #2340254, #2312102, #2414227, and #2404385. We greatly appreciate the NCSA for providing computing resources.

## License
This code is released under the **Academic Research License (Non-Commercial)**.  
For commercial inquiries, please contact [shenlong@illinois.edu](mailto:shenlong@illinois.edu).