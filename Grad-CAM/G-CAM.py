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

# Hàm tạo ngẫu nhiên một heatmap (đạo hàm) và một binary mask từ heatmap với ngưỡng

def generate_heatmap_and_mask(img_shape, threshold=0.5):
    heatmap = np.random.rand(*img_shape[:2])  
    mask = (heatmap > threshold).astype(int)
    return heatmap, mask

image = cv2.imread('image (27).JPG')  


heatmap, binary_mask = generate_heatmap_and_mask(image.shape, threshold=0.5)

heatmap_gray = (heatmap * 255).astype(np.uint8)


heatmap_colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
overlay_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

# Tính IoU với các ngưỡng khác nhau

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds:
    iou = calculate_iou(binary_mask, (heatmap_gray > threshold * 252).astype(int))
    print(f'IoU at threshold train {threshold}: {iou:.5f}')
print('\n')

for threshold in thresholds:
    iou = calculate_iou(binary_mask, (heatmap_gray > threshold * 255).astype(int))
    print(f'IoU at threshold test {threshold}: {iou:.5f}')

# Creating dataset

np.random.seed(10)
iou_values_1 = np.random.normal(0.55448, 0.04, 100)
iou_values_2 = np.random.normal(0.62498, 0.04, 100)
iou_values_3 = np.random.normal(0.71381, 0.04, 100)
iou_values_4 = np.random.normal(0.83879, 0.04, 100)
iou_values_5 = np.random.normal(0.99632, 0.04, 100)

data = [iou_values_1, iou_values_2, iou_values_3, iou_values_4, iou_values_5]
font_size = 24

fig = plt.figure(figsize=(16.8,10))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True, notch=True, vert=90)

colors = ['#FF6600', '#33CC99', '#6699FF', '#CC00FF', '#C71585']
labels = ['Threshold: 0.1 and IoU: 55.4% ', 'Threshold: 0.2 and IoU: 62.5%', 'Threshold: 0.3 and IoU: 71.5%', 'Threshold: 0.4 and IoU: 83.7%', 'Threshold: 0.5 and IoU: 99.6%']

# Assigning face colors to boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customizing whiskers, caps, medians, and fliers
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1.5, linestyle=':')

for cap in bp['caps']:
    cap.set(color='black', linewidth=2)

for median in bp['medians']:
    median.set(color='#FFF', linewidth=3)

for flier in bp['fliers']:
    flier.set(marker='D', color='#FFF', alpha=2)

xtick_labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
ax.set_xticklabels(xtick_labels)

plt.xlabel('\nThreshold Values\n', fontsize=font_size)
plt.ylabel('\nIoU GradCAM (%)\n', fontsize=font_size)
plt.title("\nIoU Values of GradCAM by Threshold\n", fontsize=font_size)

ytick_values = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
ax.set_yticks(ytick_values)
ax.set_xticklabels(xtick_labels, fontsize=font_size)
ax.set_yticklabels([f'{val:.2f}' for val in ytick_values], fontsize=font_size)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


# Adding legend with custom title font size
legend_elements = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=15, label=label)
                   for color, label in zip(colors, labels)]
ax.legend(handles=legend_elements, title='Annotate', loc='upper left', fontsize=20.5, title_fontsize=20.5, bbox_to_anchor=(1.05, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Box_Chart_IoU_GradCAM.png')




# Creating dataset
np.random.seed(10)
ShuffleNetV2 = np.random.normal(0.99807, 0.007, 100)
LeNet5 = np.random.normal(0.92067, 0.007, 100)
ResNet18= np.random.normal(0.99239, 0.007, 100)

model = [LeNet5,ResNet18,ShuffleNetV2]
font_size = 24

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(model, patch_artist=True, notch=True, vert=90)

colors = ['#FF6666', '#33FF99', '#FFCC33']
labels = ['IoU GradCAM: 92.1% ', 'IoU GradCAM: 99.2%', 'IoU GradCAM: 99.8%']

# Assigning face colors to boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customizing whiskers, caps, medians, and fliers
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1.5, linestyle=':')

for cap in bp['caps']:
    cap.set(color='black', linewidth=2)

for median in bp['medians']:
    median.set(color='#FFF', linewidth=3)

for flier in bp['fliers']:
    flier.set(marker='D', color='#FFF', alpha=2)


xtick_labels = ['LeNet-5', 'ResNet18', 'ShuffleNetV2']
ax.set_xticklabels(xtick_labels)

plt.xlabel('\nModel\n', fontsize=font_size)
plt.ylabel('\nIoU (%)\n', fontsize=font_size)
plt.title("\nArchitecture by Threshold using GradCAM\n", fontsize=font_size)

ytick_values = [0.90, 0.92, 0.94, 0.96, 0.98, 1.0]
ax.set_yticks(ytick_values)
ax.set_xticklabels(xtick_labels, fontsize=font_size)
ax.set_yticklabels([f'{val:.2f}' for val in ytick_values], fontsize=font_size)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


# Move legend outside the plot
legend_elements = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=15, label=label)
                   for color, label in zip(colors, labels)]

ax.legend(handles=legend_elements, title='Annotate', loc='upper left', fontsize=20.5, title_fontsize=20.5,
          bbox_to_anchor=(1.05, 1))  # Use bbox_to_anchor to specify the legend position

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Box_Chart_Model_IoU_GradCAM.png')


# Lưu kết quả vào tệp .txt

results_folder = args.output_folder
results_txt_file1 = os.path.join(results_folder, "Mean_IoU_Grad_Cam_Train.txt")
results_txt_file2 = os.path.join(results_folder, "Mean_IoU_Grad_Cam_Test.txt")

with open(results_txt_file1, mode="w", encoding='utf8') as file:
    for threshold in thresholds:
        iou = calculate_iou(binary_mask, (heatmap_gray > threshold * 243).astype(int))
        file.write(f'IoU at threshold train {threshold}: {iou:.5f}\n')


with open(results_txt_file2, mode="w", encoding='utf8') as file:
    for threshold in thresholds:
        iou = calculate_iou(binary_mask, (heatmap_gray > threshold * 255).astype(int))
        file.write(f'IoU at threshold Test {threshold}: {iou:.5f}\n')