import gradio as gr
import pandas as pd
import numpy as np
import joblib
from scipy.stats import boxcox

numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
target_label = ['NObeyesdad']

def load_model(model):
    if model == "Logistic regression - Hồi quy logic":
        pred_model = joblib.load("checkpoint/logistic_regression.joblib")
    elif model == "K nearest neighbors - kNN":
        pred_model = joblib.load("checkpoint/knn.joblib")
    elif model == "Decision tree - Cây quyết định":
        pred_model = joblib.load("checkpoint/decision_tree.joblib")
    elif model == "Random forest - Rừng ngẫu nhiên":
        pred_model = joblib.load("checkpoint/random_forest.joblib")
    elif model == "XGBoost - Thuật toán dựa trên Gradient":
        pred_model = joblib.load("checkpoint/xg_boost.joblib")
    elif model == "Voting classifier - Phân loại kiểu biểu quyết":
        pred_model = joblib.load("checkpoint/votingClassifier.joblib")
    return pred_model

preprocessing = joblib.load("checkpoint/preprocessing.joblib")

label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

def predict_obesity_level(model_name, age, height, weight, fcvc, ncp, ch2o, faf, tue, 
                          gender, family_history_with_overweight, favc, caec, smoke, 
                          scc, calc, mtrans):
    model = load_model(model_name)
    
    x = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    
    if age > 0:
        try:
            x['Age'], _ = boxcox(x['Age'])
        except ValueError:
            x['Age'] = np.log1p(x['Age'])
    else:
        x['Age'] = np.log1p(x['Age'])
    
    # Anh xa cac truong thuoc tinh bang gia tri so
    x['FCVC'] = pd.cut(x['FCVC'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['NCP'] = pd.cut(x['NCP'], bins=[0.5,1.5,2.5,3.5,4.5], labels=[1,2,3,4]).astype('float64')
    x['CH2O'] = pd.cut(x['CH2O'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['FAF'] = pd.cut(x['FAF'], bins=[-0.5,0.5,1.5,2.5,3.5], labels=[0,1,2,3]).astype('float64')
    x['TUE'] = pd.cut(x['TUE'], bins=[-0.5,0.5,1.5,2.5], labels=[0,1,2]).astype('float64')
    
    int64_columns = x.select_dtypes(include='int64').columns
    x[int64_columns] = x[int64_columns].astype('float64')
    
    x = preprocessing.transform(x)
    x = pd.DataFrame(x, columns=preprocessing.get_feature_names_out())
    y = model.predict(x)
    return label_mapping[y[0]]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as app:
    gr.HTML("""
    <style>
    * {
        font-family: 'Segoe UI', sans-serif !important; /* Áp dụng font Segoe UI */
    }
    </style>
    """)
    gr.Markdown("# Obesity Level Classification - Dự đoán mức độ béo phì")
    gr.Markdown("Dự đoán mức độ béo phì dựa trên các yếu tố sức khỏe và lối sống khác nhau.")
    with gr.Group():
        gr.Markdown("## Những điều cần lưu ý khi sử dụng")
    gr.Markdown("### Chú ý 1: ")
    gr.Markdown("Giá trị tiêu thụ rau trong ngày **(FCVC)** có phạm vi từ 1 đến 3 (x100 gram).")
    gr.Markdown("Giá trị Số bữa ăn chính **(NCP)** có phạm vi từ 1 đến 4.")
    gr.Markdown("Giá trị tiêu thụ nước trong ngày **(CH2O)** có phạm vi từ 1 đến 3 (x1 lit).")
    gr.Markdown("Giá trị tần suất hoạt động thể chất **(FAF)** có phạm vi từ 0 đến 3.")
    gr.Markdown("Giá trị của Thời gian sử dụng thiết bị công nghệ trong ngày **(TUE)** có phạm vi từ 0 đến 2 (x2 giờ).")
    gr.Markdown("### Chú ý 2: ")
    gr.Markdown("Câu hỏi yes/no: **yes** được hiểu là có, luôn luôn hoặc thường xuyên; **no** được hiểu là không bao giờ hoặc rất hiếm khi")
    gr.Markdown("Câu hỏi tần suất: **no** được hiểu là không bao giờ hoặc hiếm khi; **Sometimes** là thỉnh thoảng có; **Frequently** là nhiều khi; **Always** là luôn luôn hoặc rất thường xuyên")
    gr.Markdown("### Chú ý 3: ")
    gr.Markdown("Có thể nhập số thập phân vào trường số với cú pháp `<phần nguyên>.<phần thập phân>` (**KHÔNG** dùng dấu phẩy ngăn cách)")
    gr.Markdown("Để tránh bị lỗi, phải bấm và nhập đầy đủ các trường, dù trường mặc định có giá trị như mong muốn")
    
    with gr.Group():
        gr.Markdown("## Thông tin cá nhân")
        with gr.Row():
            Age = gr.Number(label="Tuổi")
            Height = gr.Number(label="Chiều cao (m)")
            Weight = gr.Number(label="Cân nặng (kg)")
        with gr.Row():
            Gender = gr.Dropdown(label="Giới tính", choices=["Male", "Female"])
            Family_history = gr.Dropdown(label="Tiền sử gia đình có người thừa cân?", choices=["yes", "no"])

    with gr.Group():
        gr.Markdown("## Hoạt động lịch trình")
        with gr.Row():
            FAF = gr.Number(label="Tần suất hoạt động thể chất (FAF)", minimum=0, maximum=3, step=1)
            TUE = gr.Number(label="Thời gian sử dụng thiết bị công nghệ trong ngày (TUE)", minimum=0, maximum=2, step=1)
        with gr.Row():
            MTRANS = gr.Dropdown(label="Phương thức di chuyển (MTRANS)", choices=["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])
    
    with gr.Group():
        gr.Markdown("## Thói quen ăn uống")
        with gr.Row():
            FCVC = gr.Number(label="Tiêu thụ rau hằng ngày (x100 gram) (FCVC)", minimum=1, maximum=3, step=1)
            NCP = gr.Number(label="Số bữa ăn chính (NCP)", minimum=1, maximum=4, step=1)
        with gr.Row():
            FAVC = gr.Dropdown(label="Bạn có hay tiêu thụ thực phẩm có hàm lượng calo cao (Mỡ, đò ngọt, xào rán,...) không? (FAVC)", choices=["yes", "no"])
            CAEC = gr.Dropdown(label="Bạn có hay ăn vặt không (CAEC)", choices=["no", "Sometimes", "Frequently", "Always"])
            CH2O = gr.Number(label="Tiêu thụ nước hằng ngày (x1 lít) (CH2O)", minimum=1, maximum=3, step=1)

    with gr.Group():
        gr.Markdown("## Các yếu tố liên quan đến sức khỏe")
        with gr.Row():
            SMOKE = gr.Dropdown(label="Bạn có hút thuốc không (SMOKE)", choices=["yes", "no"])
            SCC = gr.Dropdown(label="Bạn có theo dõi lượng calo tiêu thụ không (SCC)", choices=["yes", "no"])
        with gr.Row():
            CALC = gr.Dropdown(label="Bạn có hay dùng đồ uống có cồn không (CALC)", choices=["no", "Sometimes", "Frequently", "Always"])
        
    Model = gr.Dropdown(
        label="Lựa chọn mô hình thuật toán dự đoán",
        choices=[
            "Logistic regression - Hồi quy logic",
            "K nearest neighbors - kNN",
            "Decision tree - Cây quyết định",
            "Random forest - Rừng ngẫu nhiên",
            "XGBoost - Thuật toán dựa trên Gradient",
            "Voting classifier - Phân loại kiểu biểu quyết"
        ]
    )

    Prediction = gr.Textbox(label="Obesity Level Classification (Mức độ béo phì của bạn): ")

    with gr.Row():
        submit_button = gr.Button("Bắt đầu dự đoán")
        submit_button.click(fn=predict_obesity_level,
                            outputs=Prediction,
                            inputs=[Model, Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE,
                                    Gender, Family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS
                                    ],
                            queue=True)
        clear_button = gr.ClearButton(components=[Prediction], value="Clear")
    
    with gr.Group():
        gr.Markdown("## Giải thích kết quả")
    gr.Markdown("**Insufficient_Weight (Thiếu cân):** Bạn có thể bị suy dinh dưỡng hoặc thiếu chất dinh dưỡng căn bản")
    gr.Markdown("**Normal_Weight (Bình thường):** Bạn có cơ thể cân đối hoặc hơi thon, tiếp tục phát huy.")
    gr.Markdown("**Overweight_Level_I (Thừa cân mức độ 1):** Bạn cũng có cơ thể cân đối dù cân nặng hơi thừa so với mức bình thường, khi bạn có kết quả này bạn cũng không phải quá lo lắng.")
    gr.Markdown("**Overweight_Level_II (Thừa cân mức độ 2):** Bạn đang trong giai đoạn thừa cân nhưng chưa đến mức có bệnh béo phì, bạn phải cẩn thận trong thói quen ăn uống để tránh bị béo phì.")
    gr.Markdown("**Obesity_Type_I (Béo phì mức độ 1):** Béo phì nhẹ đến trung bình, có thể gây vấn đề sức khỏe nếu không thay đổi chế độ ăn uống và vận động.")
    gr.Markdown("**Obesity_Type_II (Béo phì mức độ 2):** Béo phì trung bình đến nghiêm trọng, nguy cơ cao mắc các bệnh như tim mạch, tiểu đường.")
    gr.Markdown("**Obesity_Type_III (Béo phì mức độ 3):** Béo phì cực độ, nguy cơ mắc bệnh nguy hiểm, ảnh hưởng đến sức khỏe và chất lượng cuộc sống.")
        
    app.launch()