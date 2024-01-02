# Import Dependencies

from keras.preprocessing import image
from torchsummary import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import StepLR
import argparse
from io import StringIO
import sys
from colorama import Fore, Back, Style, init
from PIL import Image
import torchvision.transforms.functional as TF
import re
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D
import cv2
from ultralytics import YOLO
from skimage import filters, segmentation

# Import Dataset
# <b > Dataset Link(Plant Vliiage Dataset): < /b > <br >
# <a href = 'https://data.mendeley.com/datasets/tywbtsjrjv/1'> https: // data.mendeley.com/datasets/tywbtsjrjv/1 </a>
# Data augmentation and normalization


parser = argparse.ArgumentParser(
    description="Plant Disease Classification Script")
parser.add_argument("-t", "--train-folder",
                    help="Path to the train folder", required=True)
parser.add_argument("-r", "--test-folder",
                    help="Path to the test folder", required=True)
parser.add_argument("-e", "--epochs", type=int,
                    help="Number of epochs", default=10)
parser.add_argument("-k", "--num-classes", type=int,
                    help="Number of classes", default=39)
parser.add_argument("-b", "--batch-size", type=int,
                    help="Batch size for data loaders", default=32)
parser.add_argument("-g", "--use-gpu", action="store_true",
                    help="Use GPU if available")
parser.add_argument("-o", "--output-folder", type=str,
                    help="Path to the output folder for saving results", default=".")
parser.add_argument("-s", "--show", type=int, default=0,
                    help="If >0, show results by a chart")
args = parser.parse_args()



# Tách lá bằng YOLOv8x
# Load a YOLOv8x model

YOLOv8x = YOLO("yolov8x-seg.pt")

image_files = ["YOLOv8x_TEST/1.png", "YOLOv8x_TEST/2.png", "YOLOv8x_TEST/3.png"]

# Loop through each object in the results

def YOLO():
    for file in image_files:
        image = cv2.imread(file)

        for obj in YOLOv8x.predict(show=True, source=image, show_labels=False, save=True, project="YOLOv8x_IMAGES/IMAGES01", name=".")[0]:
            if len(obj) >= 4:
                x, y, w, h = obj[:4]
                class_index = int(obj[5])
                class_name = YOLOv8x.names[class_index]

                # Convert the coordinates to integers

                x, y, w, h = int(x), int(y), int(w), int(h)

                # Draw a rectangle around the object (each leaf)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw the class label for each leaf
                
                cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


YOLO()
cv2.waitKey(1000)
cv2.destroyAllWindows()


# Hàm xử lý ngưỡng mềm với hồi quy Lasso + Phân đoạn (Thresholding + Segmentation).
# Threshold value

threshold = 0.5
font_size = 18

def Denoising_Images(image):

    # Apply denoising methods here, for example, GaussianBlur

    Denoising_Images = cv2.GaussianBlur(image, (5, 5), 0)
    return Denoising_Images


def Thresholding_Segmentation_Leaf(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  
            
            # Gọi hàm YOLO để tách lá 
            
            YOLO()
            
            # Áp dụng tính năng khử nhiễu cho hình ảnh trước khi chuyển nó sang thang độ xám

            Denoising_Images = Denoising_Images(image)
    
            # Chuyển đổi hình ảnh thành ảnh grayscale

            gray_image = cv2.cvtColor(Denoising_Images, cv2.COLOR_RGB2GRAY)
            
            # Thực hiện phân đoạn để làm sạch ảnh grayscale

            kernel = np.ones((5, 5), np.uint8)
            leaf_mask = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            
            # Áp dụng ngưỡng cho maske sau phân đoạn

            _, binary_leaf_mask = cv2.threshold(leaf_mask, threshold, 255, cv2.THRESH_BINARY)
            
            # Đảo ngược maske

            binary_leaf_mask = cv2.bitwise_not(binary_leaf_mask)


            return binary_leaf_mask
    else:
            return image



def soft_threshold(alpha, beta):
    if beta > alpha:
        return beta - alpha
    elif beta < -alpha:
        return beta + alpha
    else:
        return 0
    
class GradientDescentUnivariateLassoRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=0.5, threshold=threshold):
        self.learning_rate, self.iterations, self.alpha = learning_rate, iterations, alpha
        self.threshold = threshold  # Add threshold as a parameter

    def generate_synthetic_data(self):

        # Generate synthetic data for Lasso regression

        n_samples = 16200  # Adjust as needed
        n_features = 100   # Adjust as needed

        # Generate random synthetic data for demonstration

        X = np.random.rand(n_samples, n_features)

        # Ensure y is a 2D array with shape (n_samples, 1)

        y = np.random.rand(n_samples, 1)

        return X, y

    def fit(self):

        # Generate synthetic data for Lasso regression

        X, y = self.generate_synthetic_data()

        def soft_threshold(alpha, beta):
            if beta > alpha:
                return beta - alpha
            elif beta < -alpha:
                return beta + alpha
            else:
                return 0

        def gradient(X, y, alpha, beta):
            n = len(X)
            ols_term = -2 * np.sum(X * (y - (beta * X))) / n
            soft_term = soft_threshold(alpha, beta) / n
            return ols_term + soft_term

        beta = 0.5
        for _ in range(self.iterations):
            grad = gradient(X, y, self.alpha, beta)
            beta = beta - self.learning_rate * grad
             # Apply threshold
            beta = max(min(beta, self.threshold), -self.threshold)

        self.beta = beta

    def predict(self, X):
        return X * self.beta

    def plot_soft_threshold_and_segmentation(self, beta_values, segmented_images):
    
        variables = [f'{i}' for i in range(1, len(beta_values) + 1)]

         # Tạo một danh sách các biểu đồ đường tương ứng với các hình ảnh ngưỡng và segmentation
        fig, ax = plt.subplots(figsize=(15, 8))

        # Vẽ biểu đồ ngưỡng
        ax.plot(variables, beta_values, label='Original Betas', marker='o', linestyle='-', color='b')
        
        # Vẽ biểu đồ Segmentation
        for i, segmented_image in enumerate(segmented_images):
            ax.plot(variables, segmented_image, label=f'Segmentation {i}', marker='x', linestyle='--')

        ax.axhline(y=self.alpha, color='r', linestyle='--', label=f'Alpha Threshold = {self.alpha}')
        ax.set_xlabel('\nVariables', fontsize=font_size)
        ax.set_ylabel('Values', fontsize=font_size)
        ax.yaxis.set_tick_params(labelsize=font_size)
        ax.xaxis.set_tick_params(labelsize=font_size)
        ax.set_title('Soft-Thresholding And Segmentation With Lasso Regression\n', fontsize=font_size)
        plt.legend(fontsize=font_size, loc='lower left')
        plt.tight_layout()



# Custom transformation to apply the threshold

class ThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, image_tensor):
        return torch.threshold(image_tensor, self.threshold, 0.0)
    
    def __str__(self):
        return f"ThresholdTransform(Threshold_Value={self.threshold})"


# Tiền xử lý

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ThresholdTransform(threshold),  # Apply threshold using the custom transform
    transforms.Lambda(lambda x: Thresholding_Segmentation_Leaf(x)),  # Áp dụng Thresholding_Segmentation_Leaf
   
])

# Update with your dataset path

data_path = "..\..\..\DATASET"


# Sử dụng các tham số args.train_folder, args.test_folder

train_dataset = datasets.ImageFolder(os.path.join(
    data_path, args.train_folder), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(
    data_path, args.test_folder), transform=transform)



# In thông tin của từng dataset con trong ConcatDataset

dataset = ConcatDataset([train_dataset, test_dataset])




# Format CMD

print("\n        ------------------------------------------  LENET ARCHITECTURE  ---------------------------------------  ")

# Khởi tạo colorama

init(autoreset=True)

# Độ dài của dấu gạch ngang

line_length = 120

# Chuỗi dấu gạch ngang màu trắng

white_line = Back.WHITE + " " * line_length + Style.RESET_ALL
print(f"{white_line}\n")


# Thông tin dữ liệu

print("Dataset Information:\n")
for i, ds in enumerate(dataset.datasets):
    print(f"Dataset {i + 1}:")
    print("Number of datapoints:", len(ds))
    print("Root location:", ds.root)
    print("Transform:", ds.transform)
    print()


# Chia dữ liệu

indices = list(range(len(dataset)))
train_size = int(np.floor(0.80 * len(dataset)))  # train_size


print("\nDataset and Samples:")
print("---------------------------------------------------------------------------")
print(f"This is a constant value : {0}")
print(f"Dataset length : {len(dataset)}")
print(f"Length of train size : {train_size}")
print(f"Length of test size : {len(dataset)-train_size}")
np.random.shuffle(indices)


# Split into Train and Test
split_size = int(np.floor(0.80 * train_size))  # split_size
train_indices, test_indices = (
    indices[:split_size],
    indices[train_size:],
)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
class_to_idx = {}


# Lặp qua các đối tượng trong dataset.datasets

for ds in dataset.datasets:
    class_to_idx.update(ds.class_to_idx)
print("\n\nClass to index mapping:")
print("---------------------------------------------------------------------------")
for class_name, index in class_to_idx.items():
    print(f"{class_name.ljust(40)}{index}")

targets_size = len(class_to_idx)
print("\nTotal number of datapoints:", targets_size)


# Model
# <b>Convolution Aithmetic Equation : </b>(W - F + 2P) / S + 1 <br>
# W = Input Size<br>
# F = Filter Size<br>
# P = Padding Size<br>
# S = Stride <br>


# Transfer Learning LeNet
# Original Modeling


# Create an instance of the Lasso regression model

lasso_model = GradientDescentUnivariateLassoRegression(
    iterations=1000, alpha=0.5, threshold=threshold)

class LeNet(nn.Module):
    def __init__(self, num_classes=39):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Adjusted input size
        self.fc1 = nn.Linear(in_features=16*53*53, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

         # Adjust the input size (64 is just an example)

        n_features = 100
        self.lasso_input = nn.Linear(n_features, 64)  # Adjust 64 according to your needs


    def forward(self, x, lasso_predictions_tensor=None):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        if lasso_predictions_tensor is not None:

        # Pass Lasso predictions through the lasso_input layer

         lasso_input_result = self.lasso_input(lasso_predictions_tensor)

        # Concatenate Lasso input with the output of the previous layers

         x = torch.cat((x, lasso_input_result), dim=1)

        return x


# CPU cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = LeNet(targets_size)
model.to(device)


print("\n\nModel Summary:")
summary(model, (3, 224, 224))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


# Batch Gradient Descent

def batch_gd(model, criterion, train_loader, test_loader, optimizer, scheduler, epochs):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    train_accuracies = np.zeros(epochs)
    test_accuracies = np.zeros(epochs)
    

    for epoch in range(epochs):
        start_time = datetime.now()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        model.train()

    # Train the model on the test dataset

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = output.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader)
        train_accuracy =  correct_train / total_train

    # Test the model on the test dataset

        test_loss = 0.0
        correct_test = 0
        total_test = 0

        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)

                test_loss += loss.item()

                _, predicted = output.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader)
        test_accuracy =  correct_test / total_test

        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss
        train_accuracies[epoch] = train_accuracy
        test_accuracies[epoch] = test_accuracy

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.5f}, "
              f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.5f}, "
              f"Duration: {duration}")

    return train_losses, test_losses, train_accuracies, test_accuracies, duration


# Batch Size

batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler
)



# Train the model

print("\n\nTraining Results:")
print("----------------------------------------------------------------")
train_losses, test_losses, train_accuracies, test_accuracies, duration = batch_gd(
    model, criterion, train_loader, test_loader, optimizer, scheduler, args.epochs
)


# Save the Model

results_folder = args.output_folder

model_filename = os.path.join(results_folder, 'LeNet_File_4.pt')

# Lưu trạng thái của mô hình vào tệp
torch.save(model.state_dict(), model_filename)

# Load Model

model.load_state_dict(torch.load(model_filename))

# Đặt mô hình vào chế độ đánh giá
model.eval()


# Calculate Accuracy

def accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


train_accuracy = accuracy(train_loader)
test_accuracy = accuracy(test_loader)


print("\n\nAccuracy Results of LeNet:")
print("----------------------------------------------------------------")
print(f"Train Accuracy: {train_accuracy: .5f}")
print(f"Test Accuracy: {test_accuracy: .5f}")




# Single Image Prediction

transform_index_to_disease = train_dataset.class_to_idx


# reverse the index

transform_index_to_disease = dict(
    [(value, key) for key, value in transform_index_to_disease.items()]
)

data = pd.read_csv("disease_info.csv", encoding="cp1252")


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    print("Original : ", image_path[12:-4])
    pred_csv = data["Disease Name"][index]
    print(pred_csv)


# Wrong Prediction

print("\n\nWrong Prediction:")
print("----------------------------------------------------------------")
prediction("../Test_Image/Apple.JPG")
prediction("../Test_Image/Cherry.JPG")
prediction("../Test_Image/Corn.JPG")
prediction("../Test_Image/Grape.JPG")
prediction("../Test_Image/Tomato_Heathy.JPG")
prediction("../Test_Image/Peach.JPG")
prediction("../Test_Image/Potato.JPG")
prediction("../Test_Image/Squash.JPG")
prediction("../Test_Image/Strawberry.JPG")
prediction("../Test_Image/Tomato_Virus.JPG")
prediction("../Test_Image/Pepper.JPG")



# Hiển thị hình ảnh

results_images_size = os.path.join(results_folder, "Img_Size.png")
results_images_four_thresold = os.path.join(results_folder, "Result_Thres.png")
Threshold_And_Segmentation_Analysis = os.path.join(results_folder, "Threshold_And_Segmentation_Analysis.png")


plt.figure(figsize=(9, 7))
img_size = image.load_img('../../../Dataset/train/Cherry_Healthy/image (2).JPG')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.imshow(img_size)

if args.show > 0:
    plt.tight_layout()
    plt.savefig(f"{results_images_size}")

# Danh sách các tên tệp hình ảnh

image_files = ['../../../Dataset/train/Strawberry_Leaf_Scorch/image (12).JPG', '../../../Dataset/train/Apple_Healthy/image (150).JPG',
               '../../../Dataset/train/Grape_Healthy/image (41).JPG', '../../../Dataset/train/Tomato_Tomato_Yellow_Leaf_Curl_Virus/image (255).JPG']

# Tạo lưới 4x3 để hiển thị hình ảnh (4 hàng và 3 cột)

fig, axes = plt.subplots(4, 3, figsize=(20, 8))
axes[0, 0].set_title(
    '\nImage before applying \n threshold and segmentation\n\n', fontsize=font_size)
axes[0, 1].set_title('\nImage grayscale\n\n', fontsize=font_size)
axes[0, 2].set_title(
    '\nImage after applying denoising\n threshold and segmentation \n (Threshold Value = 0.5)\n', fontsize=font_size)

for i, file_name in enumerate(image_files):

    # Đọc hình ảnh

    image_before = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # Chuyển đổi hình ảnh sang ảnh grayscale (để áp dụng ngưỡng)

    gray_image = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)

    # Áp dụng ngưỡng (thresholding) cho hình ảnh grayscale

    thresholded_image = transform(Image.fromarray(gray_image))

    # Áp dụng Thresholding_Segmentation_Leaf cho hình ảnh sau khi áp dụng ngưỡng

    segmented_image = Thresholding_Segmentation_Leaf(thresholded_image[0])

    sobel = filters.sobel(segmented_image)
    blurred = filters.gaussian(sobel, sigma=2.0)
    
    ym = blurred.shape[0] // 2
    xm = blurred.shape[1] // 2
    
    markers = np.zeros(blurred.shape)
    
    # using corners of the image as background

    markers[0, 0:2 * xm] = 1
    markers[2 * ym - 1, 0:2 * xm] = 1
    markers[0:2 * ym, 0] = 1
    markers[0:2 * ym, 2 * xm - 1] = 1
    
    # using middle part of the image as foreground

    markers[ym - 50:ym + 50, xm - 20:xm + 20] = 2

    mask = segmentation.watershed(blurred, markers)
    

    # Hiển thị hình ảnh trước khi áp dụng ngưỡng + segmentation

    axes[i, 0].imshow(cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB))
    axes[i, 0].axis('off')

    # Hiển thị ảnh grayscale

    axes[i, 1].imshow(gray_image, cmap='gray')
    axes[i, 1].axis('off')

    # Hiển thị hình ảnh sau khi áp dụng ngưỡng + segmentation

    axes[i, 2].imshow(mask, cmap='gray')
    axes[i, 2].axis('off')

plt.tight_layout()


if args.show > 0:
    plt.tight_layout()
    plt.savefig(f"{results_images_four_thresold}")


beta_values = np.array([0.2, -0.1, 0.5, 0.3, -0.4])
segmented_images = [np.array([-0.1, 0.1, 0.3, 0.2, -0.3]), np.array([0.2, 0.0, 0.4, 0.1, -0.2])]  

# Thay bằng dữ liệu thực tế của bạn
# Gọi hàm plot_soft_threshold để vẽ biểu đồ đường

# Tạo một đối tượng của lớp GradientDescentUnivariateLassoRegression

lasso_Grad = GradientDescentUnivariateLassoRegression()

# lasso_Grad.plot_soft_threshold(beta_values)  
lasso_Grad.plot_soft_threshold_and_segmentation(beta_values, segmented_images)

if args.show > 0:
    plt.tight_layout()
    plt.savefig(f"{Threshold_And_Segmentation_Analysis}")



# Lưu kết quả vào tệp .txt


results_txt_file2 = os.path.join(results_folder, "LeNet_File_2.txt")

with open(results_txt_file2, mode="w", encoding='utf8') as file:

    file.write("\n  ---------------------------------------------------------  LENET ARCHITECTURE  ------------------------------------------------------  \n")

    file.write("\nDataset Information:\n")
    file.write("---------------------------------------------------------------------------\n")
    for i, ds in enumerate(dataset.datasets):
        file.write(f"Dataset {i + 1}:\n")
        file.write(f"Number of datapoints: {len(ds)}\n")
        file.write(f"Root location: {ds.root}\n")
        file.write(f"Transform: {ds.transform}\n")
    file.write("---------------------------------------------------------------------------\n\n")

    file.write("\nDataset and Samples:\n")
    file.write("---------------------------------------------------------------------------\n")
    file.write(f"This is a constant value : {0}\n")
    file.write(f"Dataset length : {len(dataset)}\n")
    file.write(f"Length of train size : {train_size}\n")
    file.write(f"Length of test size: {len(dataset)-train_size}\n\n")


    file.write("\nClass to index mapping:\n")
    file.write("---------------------------------------------------------------------------\n")
    for class_name, index in class_to_idx.items():
        file.write(f"{class_name.ljust(40)}{index}\n")

    file.write(f"\nTotal number of datapoints: {targets_size}\n")
    file.write("---------------------------------------------------------------------------\n")

    info_model = f"\n\nModel Architecture (on {device}):\n{model.to(device)}"

    file.write(f"{info_model}\n")

    file.write("\n\nModel Summary:\n")
    original_stdout = sys.stdout
    sys.stdout = file

    # Gọi hàm summary và lưu kết quả vào file

    summary(model, (3, 224, 224))

    # Đặt lại luồng output

    sys.stdout = original_stdout

    file.write("\n\nTraining Results:\n")
    file.write("---------------------------------------------------------------------------\n")
    for epoch in range(args.epochs):
        file.write(f"Epoch [{epoch+1}/{args.epochs}] - ")
        file.write(
            f"Train Loss: {train_losses[epoch]:.5f}, Train Accuracy: {train_accuracies[epoch]:.5f}, ")
        file.write(
            f"Test Loss: {test_losses[epoch]:.5f}, Test Accuracy: {test_accuracies[epoch]:.5f}, ")
        file.write(f"Duration: {duration}\n")

    file.write("\n\nAccuracy Results of LeNet:\n")
    file.write("---------------------------------------------------------------------------\n")
    file.write(f"Train Accuracy: {train_accuracy: .5f}\n")
    file.write(f"Test Accuracy: {test_accuracy: .5f}\n")
    

# Trực quan hóa các dạng biểu đồ qua từng epchos...

with open(results_txt_file2, "r") as file:
    text = file.read()


# Extract training and test losses and accuracies using regular expressions

train_losses = re.findall(r'Train Loss: ([\d.]+)', text)
test_losses = re.findall(r'Test Loss: ([\d.]+)', text)
train_accuracies = re.findall(r'Train Accuracy: ([\d.]+)', text)
test_accuracies = re.findall(r'Test Accuracy: ([\d.]+)', text)




train_losses = [float(loss)  for loss in train_losses]
test_losses = [float(loss) for loss in test_losses]
train_accuracies = [float(acc) for acc in train_accuracies]
test_accuracies = [float(acc) for acc in test_accuracies]

# After the training loop

plt.figure(figsize=(18, 8))


# Plot train and test losses

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.xlabel('\nNo of Epochs\n', fontsize=font_size)
plt.ylabel('\nLoss\n', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size)
plt.title('\nLoss in training and testing\n', fontsize=font_size)


# Plot train and test accuracies

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.xlabel('\nNo of Epochs\n',fontsize=font_size)
plt.ylabel('\nAccuracy\n',fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size)
plt.title('\nAccuracy in training and testing\n',fontsize=font_size)


# Save file image

results_txt_file3 = os.path.join(results_folder, "LeNet_File_3.txt")
results_img_plot = os.path.join(results_folder, "Line_Plot.png")
results_Precise_details = os.path.join(results_folder, "Precise_Details.png")
results_confusion_matrix = os.path.join(results_folder, "Confusion_Matrix.png")
results_dataset_barchart = os.path.join(results_folder, "Dataset_Barchart.png")
results_normalized_confusion_matrix =os.path.join(results_folder, "Normalized_Confusion_Matrix.png")

# Save the plots as images if the argument is greater than 0

if args.show > 0:
    plt.tight_layout()
    plt.savefig(f"{results_img_plot}")


# Classification_report

def get_predictions(loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_predictions, all_labels


# Get predictions and labels

test_predictions, test_labels = get_predictions(test_loader)
class_names = [transform_index_to_disease[i] for i in range(len(class_to_idx))]

report_classification = classification_report(
    test_labels, test_predictions, target_names=class_names,digits=5)


# Precise_Details 

report = classification_report(
    test_labels, test_predictions, target_names=class_names, output_dict=True)


# Create a DataFrame from the classification report

class_accuracy_data = []
for class_name in class_names:
    data = {
        "Tên bệnh": class_name,
        "Số ảnh nhận dạng đúng": report[class_name]['recall'] * report[class_name]['support'],
        "Số ảnh nhận dạng sai": report[class_name]['support'] - report[class_name]['recall'] * report[class_name]['support'],
        "Độ chính xác (%)": report[class_name]['recall'] 
    }
    class_accuracy_data.append(data)

df_accuracy = pd.DataFrame(class_accuracy_data)


# Add the "Trung bình" row

average_accuracy = np.mean(df_accuracy["Độ chính xác (%)"])
# average_accuracy = f"{np.mean(df_accuracy['Độ chính xác (%)']):.5f}"

df_accuracy = df_accuracy.append({"Tên bệnh": "Trung bình",
                                  "Số ảnh nhận dạng đúng": np.sum(df_accuracy["Số ảnh nhận dạng đúng"]),
                                  "Số ảnh nhận dạng sai": np.sum(df_accuracy["Số ảnh nhận dạng sai"]),
                                  "Độ chính xác (%)": average_accuracy},
                                 ignore_index=True)

# Plot the DataFrame

plt.figure(figsize=(18, 8))
# df_accuracy["Độ chính xác (%)"] = df_accuracy["Độ chính xác (%)"].astype(float)
plt.bar(df_accuracy["Tên bệnh"], df_accuracy["Độ chính xác (%)"])
plt.xticks(rotation=45, ha="right",fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel("Disease name", fontsize=font_size)
plt.ylabel("\nAccuracy (%)\n", fontsize=font_size)
plt.title("\nPrecision details on each layer\n", fontsize=font_size)
plt.tight_layout()


# Save the plot as an image if the argument is greater than 0

if args.show > 0:
    plt.savefig(results_Precise_details)



# Confusion Matrix and Heatmap

cm = confusion_matrix(test_labels, test_predictions)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(12, 10))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", square=True)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('\nConfusion Matrix\n', fontsize=14)

# Save the confusion matrix to the output file
# Save the plot as an image if the argument is greater than 0

if args.show > 0:
    plt.savefig(results_confusion_matrix, bbox_inches='tight')
    plt.show()


# Calculate Normalized Confusion Matrix
# Plot Normalized Confusion Matrix as Heatmap
normalized_cm = confusion_matrix(test_labels, test_predictions, normalize='true')
df_normalized_cm = pd.DataFrame(normalized_cm, index=class_names, columns=class_names)

plt.figure(figsize=(15, 12))
sns.heatmap(df_normalized_cm, annot=True, cmap="Blues", square=True, fmt=".2f")

# Set spacing between cells
plt.subplots_adjust(wspace=10.5, hspace=10.5)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('\nNormalized Confusion Matrix\n', fontsize=14)



if args.show > 0:
    plt.savefig(results_normalized_confusion_matrix, bbox_inches='tight')
    plt.show()


# Calculate image counts for each disease class from the train_dataset

class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset:
    class_counts[label] += 1

for _, label in test_dataset:
    class_counts[label] += 1



# Get class names and image counts

disease_names = [transform_index_to_disease[i]
                 for i in range(len(class_counts))]
image_counts = class_counts


# Màu cho từng cột

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
          'cyan', 'magenta', 'indigo', 'lime', 'teal', 'maroon', 'olive', 'navy',
          'aquamarine', 'gold', 'sienna', 'lightcoral', 'darkorchid', 'peru', 'mediumvioletred', 'darkslategrey',
          'lightblue', 'darkorange', 'forestgreen', 'firebrick', 'mediumorchid', 'saddlebrown', 'hotpink', 'dimgray',
          'mediumturquoise', 'khaki', 'tomato', 'steelblue', 'darkgoldenrod', 'indianred', 'slategray']



# Kích thước của biểu đồ

plt.figure(figsize=(12, 10))


# Vẽ biểu đồ cột

bars = plt.bar(disease_names, image_counts, color=colors, edgecolor='white')

plt.xlabel("Type of disease", fontsize=14)
plt.ylabel("\nNumber of photos\n", fontsize=14)
plt.title("\nNumber of photos for each disease\n", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()


# Hiển thị số lượng ảnh bên trên mỗi cột

for i, (bar, count) in enumerate(zip(bars, image_counts)):
    plt.text(bar.get_x() + bar.get_width() / 2, count,
             str(count), ha='center', va='bottom', fontsize=10)


# Tạo dấu chấm màu tượng trưng ở bên trái

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=disease)
                   for color, disease in zip(colors, disease_names)]


# Tạo vùng ký hiệu bên ngoài biểu đồ

legend = plt.legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(1, 0.5), title="Type of disease")


# Save the plot as an image if the argument is greater than 0

if args.show > 0:
    plt.savefig(results_dataset_barchart, bbox_inches='tight')
    plt.show()



# Convert the image files into a .txt format

with open(results_txt_file3, "w", encoding="utf-8") as file:
    file.write("\n  ---------------------------------------------------------  CHART VISUALIZATION  ------------------------------------------------------  \n")
    file.write("\nLine Chart Plot Results:")
    file.write("\n---------------------------------------------------------------------------")
    file.write("\nTraining Losses:\n")
    formatted_train_losses = [f"{loss:.5f}" for loss in train_losses]
    file.write(", ".join(formatted_train_losses) + "\n")

    # Ghi test losses
    file.write("\nTest Losses:\n")
    formatted_test_losses = [f"{loss:.5f}" for loss in test_losses]
    file.write(", ".join(formatted_test_losses) + "\n")

    # Ghi dấu ngăn cách
    file.write("\n---------------------------------------------------------------------------\n")

    # Ghi training accuracies
    file.write("\nTraining Accuracies:\n")
    formatted_train_accuracies = [f"{accuracy:.5f}" for accuracy in train_accuracies]
    file.write(", ".join(formatted_train_accuracies) + "\n")

    # Ghi test accuracies
    file.write("\nTest Accuracies:\n")
    formatted_test_accuracies = [f"{accuracy:.5f}" for accuracy in test_accuracies]
    file.write(", ".join(formatted_test_accuracies) + "\n")

    file.write("\n  --------------------------------------------------------  CLASSIFICATION REPORT  -----------------------------------------------------  \n")
    file.write("\nClassification Report:\n\n")
    file.write(report_classification)

    file.write("\n\n  --------------------------------------------------------  ACCURACY OF DETAILS  -----------------------------------------------------  \n")
    file.write("\nPrecise details of each layer:\n\n")
    file.write(df_accuracy.to_string(index=False) + "\n\n")

    file.write("\n\n  --------------------------------------------------------  CONFUSION MATRIX  --------------------------------------------------------  \n")
    file.write("\nConfusion Matrix:\n")
    file.write(df_cm.to_string() + "\n\n")

    file.write("\n\n  --------------------------------------------------------  NORMALIZED CONFUSION MATRIX  --------------------------------------------------------  \n")
    file.write("\nConfusion Matrix:\n")
    float_format = "{:.2f}".format
    file.write(df_normalized_cm.to_string(float_format=float_format) + "\n\n")

    file.write("\n\n  --------------------------------------------------------  IMAGE COUNTS FOR EACH DISEASE CLASS  -------------------------------------  \n")
    file.write("\nClass Name".ljust(40) + "Image Count\n")
    file.write("---------------------------------------------------------------------------\n")

    for disease, count in zip(disease_names, image_counts):
       file.write(f"{disease.ljust(40)}{count}\n")

print(f"\nResults saved to folder")

