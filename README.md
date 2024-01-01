# YSLS-Grad: A machine learning and Grad-Cam-based approach for identifying plant disease from leaves

[![Trạng thái xây dựng](https://img.shields.io/travis/username/repo.svg)](https://travis-ci.org/username/repo)
[![Giấy phép](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Phiên bản](https://img.shields.io/badge/version-v1.0.0-brightgreen.svg)](https://github.com/username/repo/releases)


# Mô tả ngắn về nghiên cứu

Trong cuộc cách mạng công nghệ 4.0, trí tuệ nhân tạo ngày càng được phát triển và ứng dụng rộng rãi trong mọi lĩnh vực. Tuy nhiên, có một lĩnh vực được phát triển trong nghiên cứu và rất được quan tâm gần đây đó là nông nghiệp thông minh. Phát triển trong ngành nông nghiệp, đặt biệt là cây trồng, là một trong những lĩnh vực quan trọng trong phát triển kinh tế. Những khó khăn trong quá trình trồng cây ăn trái là việc xuất hiện những loại bệnh như đốm nâu, cháy bìa lá, nấm lá, phấn trắng. Đã làm giảm cả sản lượng và chất lượng của việc trồng cây ăn trái. Vì vậy, việc phát hiện các bệnh trên cây phổ biến trên cây trồng nhằm giúp người dân nâng cao năng suất là vấn đề cấp thiết. Vì vậy trong nghiên cứu này chúng tôi đề xuất để tìm ra các phương pháp chẩn đoán bệnh trên cây dựa trên hình ảnh và được thực nghiệm trên tập dữ liệu chứa các hình ảnh của lá cây bị nhiễm bệnh và không bị nhiễm bệnh. Trong nghiên cứu này, thực hiện các công việc chính: sử dụng YOLOv8 để tách các thành lá cây và khử nhiễu hình ảnh, phân vùng hình ảnh sau đó áp dụng kỹ thuật lọc nhiễu dựa trên Ngưỡng mềm bằng hồi quy Lasso và tiếp tục thực hiện phương pháp giải thích kết quả Grad-CAM “Gradient-Weighted Class Activation Map” để xác định vị trí vùng bệnh trên ảnh của lá cây. Đồng thời, sử dụng mô hình được huấn luyện và đánh giá trên tập dữ liệu bao gồm 61.486 hình ảnh của 39 loại lá cây của 14 loài thực vật khác khau trong đó có 1 loại chứa hình ảnh nền không lá. Kết quả thực nghiệm cho thấy phương pháp phân đoạn và lọc nhiễu dựa trên Ngưỡng kết hợp với mô hình ShuffleNetV2 được tinh chỉnh các giá trị siêu tham số và đạt được kết quả tốt so với các mô hình LeNet-5, ResNet18 với mô hình ShuffleNetV2 đạt độ chính xác 99.9\% và thời gian chạy ngắn.


## Phương pháp nghiên cứu 
- Mô hình YOLOv8, ShuffleNetV2, ResNet18, LeNet-5
- GradCAM
- Giảm nhiễu

## Công Nghệ Sử Dụng

Dưới đây là một số công nghệ chính mà dự án của chúng tôi sử dụng:

- [Python](https://www.python.org/)


## Hỗ Trợ

Nếu bạn gặp bất kỳ vấn đề hoặc có câu hỏi, vui lòng liên hệ chúng tôi qua [email](mailto:nphat77777@gmail.com) hoặc [diễn đàn](https://github.com/NGUYENMINHPHAT).

---
**Chú ý:** Đừng quên cập nhật tất cả các liên kết và thông tin để phản ánh dự án cụ thể của bạn.
