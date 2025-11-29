from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Define the Input Rules (Data Validation)
# If a user sends text instead of a number, the API will reject it automatically.
class PenguinInput(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_length: float
    body_mass: float

# 2. Load the Model (The "Brain") 
model = joblib.load("penguin_model.joblib")
species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

app = FastAPI()

# 3. Define the Endpoint
@app.post("/predict")
def predict(data: PenguinInput):
    # Convert data to the format the model expects (2D array)
    features = np.array([[
        data.bill_length, 
        data.bill_depth, 
        data.flipper_length, 
        data.body_mass
    ]])
    
    # Predict
    prediction_id = model.predict(features)[0]
    return {"species": species_map[prediction_id]}
