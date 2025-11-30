from fastapi import FastAPI
# --- THESE TWO LINES WERE MISSING/BROKEN ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# -------------------------------------------
from pydantic import BaseModel
import joblib
import numpy as np

# Load Model
model = joblib.load("penguin_model.joblib")
species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

app = FastAPI()

# 1. Mount the "static" folder so the app can see the HTML file
# Note: We use directory="static" to be precise
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve the UI on the Home Page
@app.get("/")
def home():
    return FileResponse("static/index.html")

# 3. The Prediction Logic
class PenguinInput(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_length: float
    body_mass: float

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
