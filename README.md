# Obesity Level Classification - Dự đoán và phân loại mức độ béo phì của một người
---
## Mục lục
- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Cài đặt](#cài-đặt)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

## Tổng quan
Béo phì được biết đến như một tình trạng xấu của sức khỏe, bị ảnh hưởng bởi nhiều yếu tố như tình trạng di truyền, thói quen ăn kiêng, hoạt động thể chất và tình trạng kinh tế xã hội. Mục đích của dự án này là sử dụng nhiều yếu tố để dự đoán khả năng béo phì của một cá nhân, một trong những yếu tố góp phần đáng kể vào nguy cơ mắc bệnh tim mạch. Bằng cách đào tạo một mô hình có thể dự đoán chính xác nguy cơ béo phì, chúng tôi hy vọng có thể phân loại những cá nhân có nguy cơ cao mắc bệnh béo phì. Dựa trên những hiểu biết này, chúng ta có thể phát triển các biện pháp can thiệp để ngăn ngừa béo phì và các vấn đề sức khỏe liên quan trong tương lai gần.

## Cấu trúc dự án
```
checkpoint/                 # triển khai model .joblib
data/                       # dữ liệu thô
.gitignore
app.py                      # app python với framework Gradio
ml.ipynb                    # file jupyter notebooks để train và phân tích dữ liệu
README.md
README-EN.md 
requirements.txt            # thư viện yêu cầu
```

## Tập dữ liệu
Đường link đến tập dữ liệu, lấy ở [Kaggle](https://www.kaggle.com): 
- [link1](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster)
- [link2](https://www.kaggle.com/competitions/playground-series-s4e2)

Nhóm chúng tôi lấy tập dữ liệu đó, sau đó tải về và bỏ vào project, và ghép 2 tập dữ liệu để dùng cho dự án
## Cài đặt

Để cài đặt và chạy dự án, hãy làm theo các bước sau:

1. Clone repository về máy tính của bạn:
    ```bash
    git clone https://github.com/GiaHuyJQK1706/Obesity_Level_Classification.git
    ```

2. Điều hướng đến thư mục dự án:
    ```bash
    cd Obesity_Level_Classification
    ```

3. Cài đặt các thư viện và công cụ cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

4. Phân tích dữ liệu và huấn luyện mô hình **(Lưu ý: khi thực hiện bước này có thể mất rất nhiều thời gian)** bằng cách chạy file ml.ipynb

5. Chạy ứng dụng:
    ```bash
    python app.py
    ```

## Công nghệ sử dụng
- **numpy**: Thư viện hỗ trợ tính toán nhanh chóng và hiệu quả với mảng đa chiều và các phép toán đại số tuyến tính.
- **pandas**: Cung cấp các cấu trúc dữ liệu mạnh mẽ như DataFrame để xử lý và phân tích dữ liệu.
- **tabulate**: Tạo bảng dữ liệu định dạng văn bản (text) từ dữ liệu thô (ví dụ: danh sách, DataFrame).
- **matplotlib**: Thư viện vẽ đồ thị 2D mạnh mẽ cho biểu đồ, đồ thị, trực quan hóa dữ liệu.
- **seaborn**: Thư viện xây dựng trên `matplotlib` để tạo các đồ thị thống kê trực quan, dễ hiểu.
- **scikit-learn**: Bộ công cụ học máy cung cấp thuật toán cho phân loại, hồi quy, phân cụm, và xử lý dữ liệu.
- **xgboost==2.0.3**: Thuật toán tăng cường gradient mạnh mẽ dành cho các bài toán dự đoán và học máy, tối ưu về tốc độ và hiệu quả.
- **flask**: Web framework đơn giản và nhẹ cho ứng dụng web Python, sử dụng để xây dựng API và trang web cơ bản.
- **gradio**: Thư viện tạo giao diện web đơn giản để thử nghiệm và tương tác với mô hình máy học dễ dàng.

## Tác giả
Đỗ Gia Huy - 20215060
