# PCA Component Images

This folder contains the interpolation images for each PCA component.

## Folder Structure

```
static/interpolation/
├── stacked/           # Original stacked interpolation images (50 frames)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── pca1/             # PCA Component 1 interpolation images
│   ├── 000000.jpg    # Frame 0 (start)
│   ├── 000001.jpg    # Frame 1
│   ├── ...
│   └── 000049.jpg    # Frame 49 (end)
├── pca2/             # PCA Component 2 interpolation images
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── pca3/             # PCA Component 3 interpolation images
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── pca4/             # PCA Component 4 interpolation images
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

## Image Naming Convention

- All images should be named with 6-digit zero-padded numbers: `000000.jpg`, `000001.jpg`, etc.
- Each PCA component folder should contain the same number of frames (currently 50)
- Frame 0 (000000.jpg) represents the start state
- Frame 49 (000049.jpg) represents the end state
- Intermediate frames show the interpolation between start and end states

## Adding Your PCA Component Images

1. For each PCA component, place your interpolation images in the corresponding folder:
   - `pca1/` for PCA Component 1
   - `pca2/` for PCA Component 2
   - `pca3/` for PCA Component 3
   - `pca4/` for PCA Component 4

2. Ensure all images follow the naming convention (000000.jpg, 000001.jpg, etc.)

3. Make sure all PCA folders have the same number of frames

4. The JavaScript will automatically load and display these images when you move the sliders

## Current Status

- Placeholder images have been created for testing
- Each PCA folder currently contains 3 sample images (frames 0, 25, and 49)
- Replace these with your actual PCA component interpolation sequences
