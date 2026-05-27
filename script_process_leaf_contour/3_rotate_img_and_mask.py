
import sys
sys.path.append('../Plant')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import matplotlib.pyplot as plt
import tqdm
import imageio.v2 as iio
import numpy as np
import argparse
from scipy.interpolate import splprep, splev

def rotate_image(image, mask, angle):
    # Get image size
    h, w = image.shape[:2]

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    # Compute the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the transformation matrix for translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Apply the affine transformation with the new size
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0)

    return rotated_image, rotated_mask, M, new_w, new_h

def parse_args():
    parser = argparse.ArgumentParser(description='Rotate leaf images and masks to vertical.')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    parser.add_argument(
        '--mask-dir',
        default=None,
        help='Mask folder. Defaults to {data-dir}_mask.',
    )
    parser.add_argument(
        '--keypoints-dir',
        default=None,
        help='Keypoint folder. Defaults to {data-dir}_keypoints.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_foler = os.path.abspath(args.data_dir)
    mask_folder = args.mask_dir or image_foler + '_mask'
    keypoint_folder = args.keypoints_dir or image_foler + '_keypoints'
    rot_folder = image_foler + '_rotated'
    os.makedirs(rot_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    rot_mask_folder = image_foler + '_rotated_mask'
    os.makedirs(rot_mask_folder, exist_ok=True)

    new_keypoint_folder = os.path.join(rot_mask_folder, 'keypoints')
    os.makedirs(new_keypoint_folder, exist_ok=True)

    image_list = os.listdir(image_foler)
    image_list.sort()
    image_list = [image_name for image_name in image_list if 'jpg' in image_name]

    # temp = '/home/tianhang/data/folio/Folio Leaf Dataset/Folio/soybeans_rotated_mask/temp/filtered.txt'
    # with open(temp, 'r') as f:
    #     lines = f.readlines()
    #     lines = [line.strip() for line in lines]

    for index, image_name in enumerate(tqdm.tqdm(image_list)):

        name = image_name.split('.')[0]

        # read image and mask
        image_path = os.path.join(image_foler, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        if not os.path.exists(mask_path):
            # raise ValueError(f'Mask not found: {mask_path}, please use SAM2 to generate mask first')
            print(f'Mask not found: {mask_path}, skip this image')
            continue
        keypoint_path = os.path.join(keypoint_folder, image_name.replace('.jpg', '.txt'))
        extra_keypoints_path = os.path.join(keypoint_folder, image_name.replace('.jpg', 'extra.txt'))
        image = cv2.imread(image_path)
        image_copy = image.copy()
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        keypoints = None
        if os.path.exists(keypoint_path):
            keypoints = np.loadtxt(keypoint_path)
            if len(keypoints) > 0:
                keypoints = keypoints.astype(np.int32)
                if len(keypoints.shape) == 1:
                    keypoints = keypoints[None]
            else:
                keypoints = None
            # keypoints = np.flip(keypoints) 

        # plt.figure()
        # plt.scatter(keypoints[0, 0], keypoints[0, 1], c='r', s=100)
        # # if extra_keypoints is not None:
        # #     plt.scatter(extra_keypoints[:, 0], extra_keypoints[:, 1], c='g', s=100)
        # plt.imshow(image_copy)
        # plt.show()

        mask_indices = np.argwhere(mask > 0)
        mask_indices = mask_indices[np.argsort(mask_indices[:, 0])] # sort by the y coordinate

        # all inference
        if keypoints is None:
            # get the lowest and highest point of the leaf by y coordinate
            lowest_point = mask_indices[-1]
            highest_point = mask_indices[0]
            # lowest_point = np.flip(lowest_point) # (height, width) to (width, height)
            # highest_point = np.flip(highest_point) # (height, width) to (width, height)
            keypoints = np.stack([lowest_point, highest_point], axis=0)
        
        # inference 1
        elif keypoints.shape[0] == 1:
            lowest_point = keypoints.copy()[0]
            lowest_point = np.flip(lowest_point) # (height, width)
            highest_point = mask_indices[0]
        
        # all annotation
        else:
            lowest_point = keypoints.copy()[0] # (width, height)
            highest_point = keypoints.copy()[1]
            lowest_point = np.flip(lowest_point) # (height, width)
            highest_point = np.flip(highest_point)
        # del keypoints

        # rotate the image to make the leaf vertical
        angle = np.arctan((lowest_point[1] - highest_point[1]) / (lowest_point[0] - highest_point[0])) * 180 / np.pi 
        print('angle = {}'.format( angle))

        # center = (image.shape[1] // 2, image.shape[0] // 2)
        # M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        # rotate the image, making the background white

        # rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=0)
        # rotated_mask = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]), borderValue=0)
        rotated_image, rotated_mask, M, new_w, new_h = rotate_image(image, mask, -angle)

        temp = np.zeros_like(image)
        temp[int(lowest_point[0]), int(lowest_point[1])] = 255
        temp = cv2.warpAffine(temp, M, (new_w, new_h), borderValue=0)
        lowest_point = np.argwhere(temp > 0)
        lowest_point = np.mean(lowest_point, axis=0)[:-1]
        temp = np.zeros_like(image)
        temp[int(highest_point[0]), int(highest_point[1])] = 255
        temp = cv2.warpAffine(temp, M, (new_w, new_h), borderValue=0)
        highest_point = np.argwhere(temp > 0)
        highest_point = np.mean(highest_point, axis=0)[:-1]

        # plt.figure()
        # plt.imshow(rotated_mask)
        # # # plt.imshow(image_copy)
        # plt.scatter(lowest_point[1], lowest_point[0], c='r', s=100)
        # plt.scatter(highest_point[1], highest_point[0], c='b', s=100)
        # for point in extra_keypoints_rot:
        #     plt.scatter(point[1], point[0], c='g', s=100)
        # plt.show()

        # lowest_point (height, width)

        rotated_image_path = os.path.join(rot_folder, image_name)
        cv2.imwrite(rotated_image_path, rotated_image)
        print(f'Saved rotated image to {rotated_image_path}')

        rotated_mask_path = os.path.join(rot_mask_folder, image_name)
        cv2.imwrite(rotated_mask_path, rotated_mask)
        print(f'Saved rotated mask to {rotated_mask_path}')

        point_ = np.array([lowest_point, highest_point])
        np.savetxt(os.path.join(new_keypoint_folder, image_name.replace('.jpg', '.txt')), point_)

        print('\n')