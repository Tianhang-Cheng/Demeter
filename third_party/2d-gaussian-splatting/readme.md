# reconstruction 2dgs and extract mesh from video

## Install Environment

please follow [https://github.com/hbb1/2d-gaussian-splatting]([2DGS](https://github.com/hbb1/2d-gaussian-splatting))

## Run SfM

```bash

cd third_party/2d-gaussian-splatting

# extract frames (the output folder should have name 'input')
python extract_frames.py --video_path ../../sample_video/001.mp4 --output_dir ../../sample_video/001/input --target_fps 10

# Run sfm to get camera poses (may takes minutes depending on the resolution and video length)
python convert.py -s ../../sample_video/001
```

## Train 2DGS

```bash

# Train 2dgs (change to your path)
python train.py -s ../../sample_video/001 --iteration 15000

# Extract Mesh from weight
python render.py -s ../../sample_video/001 -m ../../sample_video/001 --skip_test --skip_train --mesh_res 1024 --depth_trunc 5 

```

the output mesh will be found in ```sample_video\001\train\ours_5000\fuse_decimated.ply```

<p align="center">
  <img src="../../assets/2dgs.jpg" alt="Description">
  The reconstruction result of 2DGS.
</p>

<p align="center">
  <img src="../../assets\structure.png" alt="Description">
  The folder structure.
</p>
