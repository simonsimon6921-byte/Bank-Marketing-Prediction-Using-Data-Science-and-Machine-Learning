from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Bank Marketing Prediction API")

class Customer(BaseModel):
    age: int
    balance: int
    day: int
    duration: int
    campaign: int
    pdays: int
    previous: int

@app.get("/")
def home():
    return {"message": "Bank Marketing Prediction API is running"}

@app.post("/predict")
def predict(data: Customer):
    x = np.array(data.features).reshape(1, -1)
    prediction = model.predict(x)[0]

    return {
        "prediction": int(prediction)
    }
