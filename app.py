from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class PenguinInput(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_length: float
    body_mass: float

# Load the model
model = joblib.load("penguin_model.joblib")
species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

app = FastAPI()

# === THIS WAS MISSING ===
@app.get("/")
def home():
    return {"message": "Penguin Prediction API is functioning!"}
# ========================

@app.post("/predict")
def predict(data: PenguinInput):
    features = np.array([[
        data.bill_length, 
        data.bill_depth, 
        data.flipper_length, 
        data.body_mass
    ]])
    
    prediction_id = model.predict(features)[0]
    return {"species": species_map[prediction_id]}
