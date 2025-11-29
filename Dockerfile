# 1. The Foundation: Start with a lightweight Linux + Python
FROM python:3.9-slim

# 2. The Setup: Create a folder inside the container
WORKDIR /app

# 3. The Dependencies: Copy and install requirements FIRST (for caching speed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. The Code: Copy everything else (app.py, model.joblib)
COPY . .

# 5. The Command: run the API when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
