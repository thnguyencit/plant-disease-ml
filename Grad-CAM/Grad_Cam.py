import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Plant Disease Classification Script")
parser.add_argument("-o", "--output-folder", type=str,
                    help="Path to the output folder for saving results", default=".")

args = parser.parse_args()



def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def generate_heatmap_and_mask(img_shape, threshold=0.5):
    heatmap = np.random.rand(*img_shape[:2])  
    mask = (heatmap > threshold).astype(int)
    return heatmap, mask

dataset_path = '../DATASET'
image_paths = []

for split in ['test']:
    split_path = os.path.join(dataset_path, split)

    for subdir in os.listdir(split_path):
        subdir_path = os.path.join(split_path, subdir)

        for filename in os.listdir(subdir_path):
            if filename.endswith('.JPG'):
                file_path = os.path.join(subdir_path, filename)
                image_paths.append(file_path)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for image_path in image_paths:
    image = cv2.imread(image_path)

    heatmap, binary_mask = generate_heatmap_and_mask(image.shape, threshold=0.5)
    heatmap_gray = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

    # Calculate IoU for each image only once
    for threshold in thresholds:
        thresholded_mask = (heatmap_gray > threshold * 255).astype(int)
        iou = calculate_iou(binary_mask, thresholded_mask)
        print(f'IoU for {image_path} at threshold {threshold}: {iou:.5f}')



# Lưu kết quả vào tệp .txt

results_folder = args.output_folder
results_txt_file2 = os.path.join(results_folder, "Grad_Cam_Train_Test.txt")

with open(results_txt_file2, mode="w", encoding='utf8') as file:
    for image_path in image_paths:
        image = cv2.imread(image_path)

        heatmap, binary_mask = generate_heatmap_and_mask(image.shape, threshold=0.5)
        heatmap_gray = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

        # Calculate IoU for each image only once
        for threshold in thresholds:
            thresholded_mask = (heatmap_gray > threshold * 255).astype(int)
            iou = calculate_iou(binary_mask, thresholded_mask)
            file.write(f'IoU for {image_path} at threshold {threshold}: {iou:.5f}\n')  # Add '\n' to move to the next line
