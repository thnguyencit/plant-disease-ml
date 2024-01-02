import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import ShuffleNetV2
import numpy as np
import torch
import pandas as pd
import time  # Import the time module
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from skimage import filters, segmentation
from skimage import io


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model =  ShuffleNetV2.ShuffleNetV2(39)    
model.load_state_dict(torch.load("ShuffleNetV2_File_4.pt"))
model.eval()


def prediction(image_path):
    start_time = time.time()  # Record the start time

    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
    input_data = Variable(input_data, requires_grad=True)

    output = model(input_data)

    index = torch.argmax(output).item()
    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][index] = 1
    one_hot_output = Variable(one_hot_output, requires_grad=True)

    output.backward(one_hot_output)

    gradients = input_data.grad.data.numpy()
    pooled_gradients = np.mean(gradients, axis=(0, 2, 3))
    activations = input_data.data.numpy()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = np.mean(activations, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    eps = 1e-8
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap /= np.max(heatmap)

    heatmap_resized = cv2.resize(heatmap[0], (image.width, image.height))
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay_img = cv2.addWeighted(np.array(image), 0.5, heatmap_colored, 0.5, 0)


    heatmap_path = os.path.join('static/uploads', 'heatmap.jpg')
    cv2.imwrite(heatmap_path, heatmap_colored)

    overlay_path = os.path.join('static', 'uploads', 'overlay.jpg')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    output = output.detach().numpy()
    index = np.argmax(output)
    accuracy = round(torch.softmax(torch.from_numpy(output), dim=1)[0][index].item() * 100, 2)

    end_time = time.time()  # Record the end time
    duration = round(end_time - start_time, 4)  # Calculate the duration

    return index, os.path.basename(image_path), accuracy, duration, overlay_path


app = Flask(__name__)

@app.route('/')
def ai_engine_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred, filename, accuracy, duration, overlay_path = prediction(file_path)

        title = disease_info['Disease Name'][pred]
        description = disease_info['Description'][pred]
        prevent = disease_info['Possible Steps'][pred]

        for file_name in [file_path]:

            image_before = io.imread(file_name)

            gray_image = cv2.cvtColor(image_before, cv2.COLOR_BGR2GRAY)

            thresholded_image = gray_image

            sobel = filters.sobel(thresholded_image)
            blurred = filters.gaussian(sobel, sigma=2.0)

            ym = blurred.shape[0] // 2
            xm = blurred.shape[1] // 2

            markers = np.zeros(blurred.shape)
            

            markers[0, 0:2 * xm] = 1
            markers[2 * ym - 1, 0:2 * xm] = 1
            markers[0:2 * ym, 0] = 1
            markers[0:2 * ym, 2 * xm - 1] = 1
            

            markers[ym - 50:ym + 50, xm - 20:xm + 20] = 2

            mask = segmentation.watershed(blurred, markers)

            mask_path = os.path.join('static', 'uploads', 'mask.jpg')
            plt.imsave(mask_path, mask, cmap='gray')

        image_url = os.path.join('static/uploads', filename)

        supplement_name = supplement_info['Supplement Name'][pred]
        supplement_image_url = supplement_info['Supplement Image'][pred]
        supplement_buy_link = supplement_info['Buy Link'][pred]
        drug_description = supplement_info['Drug Description'][pred]


        timestamp = int(time.time())

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=f"{image_url}?{timestamp}", pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link,
                               drug=drug_description, accuracy=accuracy, duration=duration,
                               original_image_url=file_path, overlay_path=f"{overlay_path}?{timestamp}", mask_path=f"{mask_path}?{timestamp}")


if __name__ == '__main__':
    app.run(debug=True)
