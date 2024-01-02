import torch
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import numpy as np


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = Variable(img.unsqueeze(0))
    return img


def get_gradient_and_feature_map(img, model, target_layer):
    model.eval()

    def forward_hook(module, input, output):
        hooks['features'] = output

    def backward_hook(module, grad_input, grad_output):
        hooks['gradient'] = grad_output[0]

    hooks = {}
    hook_handles = []
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    output = model(img)
    model.zero_grad()
    output[0, output.argmax()].backward()
    feature_map = hooks['features']
    gradient = hooks['gradient']
    return feature_map, gradient



def generate_gradcam(img, model, target_layer):
    feature_map, gradient = get_gradient_and_feature_map(img, model, target_layer)
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def apply_gradcam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.detach().squeeze().numpy()), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


    img_np = np.array(img.detach().squeeze().permute(1, 2, 0))

    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    superimposed_img = heatmap_resized * 0.4 + img_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img


if __name__ == '__main__':
    img_path = 'image (41).JPG'
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1] 
    img = preprocess_image(img_path)
    mask = generate_gradcam(img, model, target_layer)
    result = apply_gradcam_on_image(img, mask)

    # Hiển thị ảnh gốc
    img_original = Image.open(img_path)
    img_original.show(title='Original Image')

    # Hiển thị và lưu ảnh sau khi áp dụng GradCAM
    result_image = Image.fromarray(result)
    result_image.show(title='GradCAM Result')

    # Lưu ảnh
    result_image.save('gradcam_mask.jpg')


