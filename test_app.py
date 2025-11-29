from fastapi.testclient import TestClient
from app import app

# 1. Create a "Test User" (Client)
client = TestClient(app)

# Test 1: Check if the API is alive
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Penguin Prediction API is functioning!"}

# Test 2: Check if the model logic is correct
def test_predict_penguin():
    # 1. Send fake data (Standard Gentoo penguin dimensions)
    payload = {
        "bill_length": 50.0,
        "bill_depth": 15.0,
        "flipper_length": 220.0,
        "body_mass": 5000.0
    }
    
    # 2. Hit the API
    response = client.post("/predict", json=payload)
    
    # 3. Check the Status (Did it crash?)
    assert response.status_code == 200
    
    # 4. Check the Logic (Is the answer correct?)
    result = response.json()
    assert result["species"] == "Gentoo"
