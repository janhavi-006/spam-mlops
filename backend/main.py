from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "spam_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Spam Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(msg: Message):
    spam_probability = float(model.predict_proba([msg.text])[0][1])
    return {
        "label": "spam" if spam_probability >= 0.5 else "ham",
        "spam_probability": round(spam_probability, 4),
    }