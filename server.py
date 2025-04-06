import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load mô hình
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["Nguy cơ mắc bệnh thấp", "Nguy cơ mắc bệnh cao"]

# Khởi tạo FastAPI app
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (origins)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các header
)


@app.get("/")
def read_root():
    return {"Message": "ML Model Deployments"}


@app.post("/predict")
def predict(inputs: dict):
    if isinstance(inputs.get('Notes'), list):
        inputs['Notes'] = inputs['Notes'][0]
    if isinstance(inputs.get('BMI_Category'), list):
        inputs['BMI_Category'] = inputs['BMI_Category'][0]

    # Chuyển inputs thành DataFrame
    inputs_df = pd.DataFrame([inputs])

    # Dự đoán
    predictions = model.predict(inputs_df)[0]
    predicted_class = class_names[predictions]

    return {"message": predicted_class}

