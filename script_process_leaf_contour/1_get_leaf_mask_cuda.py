import argparse
import os
import sys
from pathlib import Path
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()




SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Generate leaf masks with SAM2.')
    parser.add_argument('--data-dir', required=True, help='Folder containing .jpg leaf images.')
    parser.add_argument(
        '--sam-checkpoint',
        default=str(SCRIPT_DIR / 'sam2/checkpoints/sam2.1_hiera_large.pt'),
        help='Path to SAM2 checkpoint.',
    )
    parser.add_argument(
        '--sam-config',
        default='configs/sam2.1/sam2.1_hiera_l.yaml',
        help='SAM2 model config.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_foler = os.path.abspath(args.data_dir)
    mask_folder = image_foler + '_mask'
    os.makedirs(mask_folder, exist_ok=True)

    # Avoid shadowing the installed sam2 package by the local sam2/ repo folder.
    script_dir = str(SCRIPT_DIR)
    for path_entry in ('', script_dir):
        while path_entry in sys.path:
            sys.path.remove(path_entry)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_list = os.listdir(image_foler)
    image_list.sort()
    image_list = [image_name for image_name in image_list if 'jpg' in image_name]

    import tqdm
    import imageio.v3 as iio
    import cv2
    for index, image_name in enumerate(tqdm.tqdm(image_list)):
        image_path = os.path.join(image_foler, image_name)
        image = Image.open(image_path).convert('RGB')
        predictor.set_image(image)

        h, w = image.size

        input_point = np.array([[h // 2, w // 2]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

        if masks[0][w // 2, h // 2] == 0:
            target_mask = 255 - masks[0].astype(np.uint8) * 255
        else:
            target_mask = masks[0].astype(np.uint8) * 255

        pad = 10
        target_mask[0:pad] = 0
        target_mask[-pad:] = 0
        target_mask[:, 0:pad] = 0
        target_mask[:, -pad:] = 0

        iio.imwrite(os.path.join(mask_folder, image_name), target_mask)


if __name__ == '__main__':
    main()