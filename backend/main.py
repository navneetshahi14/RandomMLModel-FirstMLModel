from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Placement Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("model.pkl")


class StudentData(BaseModel):
    cgpa: float
    iq: float


@app.get("/")
def home():
    return {"message": "Placement Prediction API running 🚀"}


@app.post("/predict")
def predict(data: StudentData):
    try:
        features = np.array([[data.cgpa, data.iq]])
        prediction = model.predict(features)[0]

        result = "Placement Hoga ✅" if prediction == 1 else "Placement Nahi Hoga ❌"

        return {
            "cgpa": data.cgpa,
            "iq": data.iq,
            "prediction": int(prediction),
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}
