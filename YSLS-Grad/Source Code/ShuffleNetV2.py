import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Xây dựng lớp ShuffleNetV2

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=39, model_size='1x'):
        super(ShuffleNetV2, self).__init__()
        assert model_size in ['0.5x', '1x', '1.5x', '2x'], "Model size must be '0.5x', '1x', '1.5x', or '2x'."

        if model_size == '0.5x':
            self.shufflenet = models.shufflenet_v2_x0_5(pretrained=True)
        elif model_size == '1x':
            self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        elif model_size == '1.5x':
            self.shufflenet = models.shufflenet_v2_x1_5(pretrained=True)
        else:
            self.shufflenet = models.shufflenet_v2_x2_0(pretrained=True)

        # Change the fully connected layer to match the number of output classes
        self.shufflenet.fc = nn.Linear(self.shufflenet.fc.in_features, num_classes)
        
        # Assuming n_features is the number of input features for your Lasso model
        n_features = 100
        self.lasso_input = nn.Linear(n_features, 64)  # Adjust 64 according to your needs


    def forward(self, x, lasso_predictions_tensor=None):
        if lasso_predictions_tensor is not None:
            # Pass Lasso predictions through the lasso_input layer
            lasso_input_result = self.lasso_input(lasso_predictions_tensor)

            # Concatenate Lasso input with the output of the previous layers
            x = torch.cat((x, lasso_input_result), dim=1)

        return self.shufflenet(x)



idx_to_classes = {0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  4: 'Background_without_leaves',
                  5: 'Blueberry___healthy',
                  6: 'Cherry___Powdery_mildew',
                  7: 'Cherry___healthy',
                  8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                  9: 'Corn___Common_rust',
                  10: 'Corn___Northern_Leaf_Blight',
                  11: 'Corn___healthy',
                  12: 'Grape___Black_rot',
                  13: 'Grape___Esca_(Black_Measles)',
                  14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  15: 'Grape___healthy',
                  16: 'Orange___Haunglongbing_(Citrus_greening)',
                  17: 'Peach___Bacterial_spot',
                  18: 'Peach___healthy',
                  19: 'Pepper,_bell___Bacterial_spot',
                  20: 'Pepper,_bell___healthy',
                  21: 'Potato___Early_blight',
                  22: 'Potato___Late_blight',
                  23: 'Potato___healthy',
                  24: 'Raspberry___healthy',
                  25: 'Soybean___healthy',
                  26: 'Squash___Powdery_mildew',
                  27: 'Strawberry___Leaf_scorch',
                  28: 'Strawberry___healthy',
                  29: 'Tomato___Bacterial_spot',
                  30: 'Tomato___Early_blight',
                  31: 'Tomato___Late_blight',
                  32: 'Tomato___Leaf_Mold',
                  33: 'Tomato___Septoria_leaf_spot',
                  34: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  35: 'Tomato___Target_Spot',
                  36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  37: 'Tomato___Tomato_mosaic_virus',
                  38: 'Tomato___healthy'
    }






















