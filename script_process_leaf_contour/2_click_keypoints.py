import cv2
import os
import glob
import tqdm

# species = 'papaya'
# species = 'geranium'
# species = 'betel'
# species = 'thevetia'
# species = 'ficus' 
species = 'soybean'
# species = 'maize'

IMAGE_FOLDER = f"sample_leaf_data/{species}"  # Path to the folder containing images
MASK_FOLDER = f"sample_leaf_data/{species}_mask"  # Path to the folder containing masks
KEYPOINT_FOLDER = f"sample_leaf_data/{species}_keypoints"  # Path to save keypoint annotations
os.makedirs(KEYPOINT_FOLDER, exist_ok=True)  # Create the keypoint folder if it doesn't exist

# open3d setting
MAX_WIDTH = 1920  # Max width to fit within screen
MAX_HEIGHT = 1080  # Max height to fit within screen

# Get image list
# image_paths = glob.glob(os.path.join(OUTPUT_FOLDER, "*.png"))  # Adjust extension if needed
image_paths = glob.glob(os.path.join(MASK_FOLDER, "*.jpg"))  # Adjust extension if needed

def resize_to_fit(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # Ensure the image is not enlarged
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale

def annotate_image(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    
    index = os.path.basename(image_path).split('.')[0]

    if os.path.exists(os.path.join(KEYPOINT_FOLDER, filename)):
        print(f"Annotations already exist for {filename}")
        return
    else:
        print(f"Annotating {image_path}...")

    img = cv2.imread(image_path)[..., 0]
    rgb = cv2.imread(os.path.join(IMAGE_FOLDER, index + '.jpg'))
    rgb[img == 0] = 0
    img_resized, scale = resize_to_fit(rgb, MAX_WIDTH, MAX_HEIGHT)
    keypoints = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scaled_x, scaled_y = int(x / scale), int(y / scale)
            keypoints.append((scaled_x, scaled_y))
            cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Annotate", img_resized)

    cv2.imshow("Annotate", img_resized)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save keypoints to file
    output_path = os.path.join(KEYPOINT_FOLDER, filename)
    with open(output_path, "w") as f:
        for x, y in keypoints:
            f.write(f"{x} {y}\n")
    print(f"Annotations saved to {output_path}")

for image_path in tqdm.tqdm(image_paths, desc="Annotating images"):
    annotate_image(image_path)