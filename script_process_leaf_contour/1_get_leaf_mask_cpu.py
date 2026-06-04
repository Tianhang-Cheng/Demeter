import argparse
import os
import tqdm
import imageio.v3 as iio
import numpy as np
from PIL import Image
from rembg import remove, new_session


def parse_args():
    parser = argparse.ArgumentParser(description='Generate leaf masks using rembg (CPU only, no torch).')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    parser.add_argument('--model', default='u2net', help='rembg model name (default: u2net).')
    return parser.parse_args()


def main():
    args = parse_args()
    image_folder = os.path.abspath(args.data_dir)
    mask_folder = image_folder + '_mask'
    os.makedirs(mask_folder, exist_ok=True)

    session = new_session(args.model)

    image_list = sorted(f for f in os.listdir(image_folder) if f.lower().endswith('.jpg'))

    for image_name in tqdm.tqdm(image_list):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        # remove() returns RGBA; alpha channel is the foreground mask
        result = remove(image, session=session)
        alpha = np.array(result)[:, :, 3]  # uint8, 0–255

        # Binarize to match SAM2 output convention (0 or 255)
        mask = np.where(alpha >= 128, np.uint8(255), np.uint8(0))

        iio.imwrite(os.path.join(mask_folder, image_name), mask)


if __name__ == '__main__':
    main()
