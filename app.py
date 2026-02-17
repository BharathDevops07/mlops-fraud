from fastapi import FastAPI
from typing import List
import joblib
import numpy as np
import os
import boto3
import time

app = FastAPI()

# ===============================
# ENV CONFIG
# ===============================
MODEL_S3_PATH = os.getenv("MODEL_S3_PATH")

if not MODEL_S3_PATH:
    raise Exception("MODEL_S3_PATH environment variable is required")

LOCAL_MODEL_PATH = "/tmp/model.pkl"

# ===============================
# S3 DOWNLOAD FUNCTION
# ===============================
def download_model():
    print(f"Downloading model from {MODEL_S3_PATH}")

    path = MODEL_S3_PATH.replace("s3://", "")
    bucket = path.split("/")[0]
    key = "/".join(path.split("/")[1:])

    s3 = boto3.client("s3")

    for attempt in range(3):
        try:
            s3.download_file(bucket, key, LOCAL_MODEL_PATH)
            print("✅ Model downloaded")
            return
        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(3)

    raise Exception("❌ Failed to download model from S3")

# ===============================
# STARTUP MODEL LOAD
# ===============================
download_model()

print("Loading model into memory...")
model = joblib.load(LOCAL_MODEL_PATH)
print("✅ Model loaded")

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health():
    return {"status": "Fraud Detection API Running"}

# ===============================
# PREDICT ENDPOINT (JSON INPUT)
# ===============================
@app.post("/predict")
def predict(features: List[float]):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]

    return {"fraud_prediction": int(prediction)}
