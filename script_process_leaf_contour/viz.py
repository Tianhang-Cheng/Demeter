import numpy as np
from PIL import Image

def combine_grid(images, grid_shape=(6, 6), gap=0, gap_value=255):
    """
    Combine small images into a single large image with optional gaps.
    
    Parameters
    ----------
    images : numpy.ndarray
        Array of shape [rows, cols, H, W] or [rows, cols, H, W, C].
    grid_shape : tuple of int, default (6, 6)
        Grid arrangement (rows, cols).
    gap : int, default 0
        Number of pixels between images.
    gap_value : int or tuple, default 255
        Fill value for the gap (255 = white for grayscale, (255,255,255) for RGB).
    
    Returns
    -------
    big_img : numpy.ndarray
        Combined image as a numpy array.
    """
    rows, cols = grid_shape
    H, W = images.shape[2], images.shape[3]
    C = 1 if images.ndim == 4 else images.shape[4]

    # Total canvas size including gaps
    out_h = rows * H + (rows - 1) * gap
    out_w = cols * W + (cols - 1) * gap

    if C == 1:
        canvas = np.full((out_h, out_w), gap_value, dtype=images.dtype)
    else:
        canvas = np.full((out_h, out_w, C), gap_value, dtype=images.dtype)

    # Paste each image into the canvas
    for r in range(rows):
        for c in range(cols):
            y = r * (H + gap)
            x = c * (W + gap)
            canvas[y:y+H, x:x+W, ...] = images[r, c]

    return canvas
# Example usage:
# Suppose your array is [6, 6, size, size] (grayscale)



# species = 'papaya'
# species = 'soybean'
# species = 'geranium'
# species = 'maize'
# species = 'ficus'
import numpy as np

import cv2

seq = ['ficus','geranium', 'maize', 'papaya','soybean']
h, w = 400, 400

images = []
for i in seq:
    for j in seq:
        img_path = f'/home/tianhang/Desktop/texture/{i}_{j}.png'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to h, w
        img = cv2.resize(img, (h, w))

        images.append(img)

imgs = np.stack(images, axis=0).reshape(5, 5, h, w, 3)

big = combine_grid(imgs, (5, 5), gap=15)
# Save with PIL
Image.fromarray(big).save(f"/home/tianhang/Desktop/texture/combined.png")

seq = ['ficus','geranium', 'maize', 'papaya','soybean']
h, w = 400, 400

images = []
for i in seq:
    img_path = f'/home/tianhang/Desktop/texture/{i}.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to h, w
    img = cv2.resize(img, (h, w))

    images.append(img)

imgs = np.stack(images, axis=0).reshape(5, 1, h, w, 3)

big = combine_grid(imgs, (5, 1), gap=15)
# Save with PIL
Image.fromarray(big).save(f"/home/tianhang/Desktop/texture/t.png")

_=1