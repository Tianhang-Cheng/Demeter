import argparse
import glob
import os

import cv2
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Interactively annotate leaf keypoints.')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    parser.add_argument(
        '--mask-dir',
        default=None,
        help='Mask folder. Defaults to {data-dir}_mask.',
    )
    parser.add_argument(
        '--keypoints-dir',
        default=None,
        help='Keypoint output folder. Defaults to {data-dir}_keypoints.',
    )
    return parser.parse_args()


MAX_WIDTH = 1920
MAX_HEIGHT = 1080


def resize_to_fit(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def annotate_image(image_path, image_folder, keypoint_folder):
    filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    index = os.path.basename(image_path).split('.')[0]

    if os.path.exists(os.path.join(keypoint_folder, filename)):
        print(f'Annotations already exist for {filename}')
        return

    print(f'Annotating {image_path}...')

    img = cv2.imread(image_path)[..., 0]
    rgb = cv2.imread(os.path.join(image_folder, index + '.jpg'))
    rgb[img == 0] = 0
    img_resized, scale = resize_to_fit(rgb, MAX_WIDTH, MAX_HEIGHT)
    keypoints = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scaled_x, scaled_y = int(x / scale), int(y / scale)
            keypoints.append((scaled_x, scaled_y))
            cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Annotate', img_resized)

    cv2.imshow('Annotate', img_resized)
    cv2.setMouseCallback('Annotate', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = os.path.join(keypoint_folder, filename)
    with open(output_path, 'w') as f:
        for x, y in keypoints:
            f.write(f'{x} {y}\n')
    print(f'Annotations saved to {output_path}')


def main():
    args = parse_args()
    image_folder = os.path.abspath(args.data_dir)
    mask_folder = args.mask_dir or image_folder + '_mask'
    keypoint_folder = args.keypoints_dir or image_folder + '_keypoints'
    os.makedirs(keypoint_folder, exist_ok=True)

    image_paths = glob.glob(os.path.join(mask_folder, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths, desc='Annotating images'):
        annotate_image(image_path, image_folder, keypoint_folder)


if __name__ == '__main__':
    main()
