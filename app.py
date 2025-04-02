from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Tải mô hình đã lưu
model = joblib.load("diabetes_model.pkl")

# Khởi tạo ứng dụng Flask
app = Flask(__name__)


@app.route("/")
def home():
    return "Diabetes Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Nhận dữ liệu từ request
        data = request.get_json()

        # Chuyển dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Dự đoán kết quả
        prediction = model.predict(input_data)[0]

        # Trả về kết quả
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)



#test
