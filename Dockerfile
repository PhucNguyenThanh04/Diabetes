# Sử dụng image Python làm base image
FROM python:3.12
LABEL authors="Phuc"

WORKDIR /src

COPY model.pkl /src/model.pkl
COPY server.py /src/server.py

COPY requirements.txt /src/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]





















## Cài đặt thư viện cần thiết
#RUN pip install --upgrade pip
#RUN pip install fastapi uvicorn scikit-learn
#
## Copy mã nguồn vào container
#COPY . /app
#WORKDIR /app
#
## Cài đặt mô hình và các dependencies (ví dụ pickle)
#RUN pip install -r requirements.txt
#
## Expose port 8000 cho FastAPI
#EXPOSE 8000
#
## Chạy ứng dụng FastAPI
#CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
