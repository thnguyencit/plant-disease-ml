# YSLS-Grad: A machine learning and Grad-Cam-based approach for identifying plant disease from leaves

[![Trạng thái xây dựng](https://img.shields.io/travis/username/repo.svg)](https://travis-ci.org/username/repo)
[![Giấy phép](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Phiên bản](https://img.shields.io/badge/version-v1.0.0-brightgreen.svg)](https://github.com/username/repo/releases)


# Summary

In the 4.0 technology revolution, artificial intelligence is increasingly developed and widely applied in all fields, especially smart agriculture. Development in the agricultural sector, especially crops, is one of the critical areas in economic development. The difficulties in growing fruit trees are the appearance of diseases such as brown spots, leaf cover fire, leaf fungus, powdery mildew, etc., which have reduced the yield and the quality of fruit cultivation. Therefore, detecting common plant leaf diseases is an urgent issue in agricultural development. This study proposes a plant leaf disease detection approach, namely YSLS-Grad, based on machine learning algorithms combined with an Interpretable Machine Learning Model such as Gradient-Weighted Class Activation Map (Grad-CAM). The proposed approach is: First, the You Only Look Once (YOLO) v8 is applied to separate the leaf area from plant images. Then, the Soft threshold-based noise reduction technique, Lasso regression, is used to reduce image noise. Next, the data are fed into the ShuffleNetV2 model refined with hyperparameter values for classifying plant leaf diseases. Then, after obtaining a trained model, Grad-CAM is deployed to locate the disease area on the plant leaf images. Based on essential scores on disease areas computed by Grad-CAM, we compare threshold values to perform the segmentation tasks. 


## Method
- YOLOv8, ShuffleNetV2, ResNet18, LeNet-5
- GradCAM
- Noise Reduction

## 
Technologies Used
Here are some key technologies that our project utilizes:

- [Python](https://www.python.org/)


## Support

If you encounter any issues or have questions, please feel free to contact us via [email](mailto:nphat77777@gmail.com) or [on our](https://github.com/thnguyencit/plant-disease-ml/tree/main).

---
**Note:** Don't forget to update all links and information to reflect the specifics of your project.
