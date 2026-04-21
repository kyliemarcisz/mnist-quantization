# app.py
# This is a web server. It receives your drawing and returns predictions.

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load all 4 model versions when the server starts
models = {
    "full":    joblib.load("models/model_full.pkl"),
    "float32": joblib.load("models/model_float32.pkl"),
    "float16": joblib.load("models/model_float16.pkl"),
}

# Route 1: serve the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route 2: receive a drawing and return predictions
# This is an API endpoint — the URL your JavaScript will call
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["pixels"]   # receive the drawing as pixel values
    pixels = np.array(data).reshape(1, -1)  # reshape for the model

    results = {}
    for name, model in models.items():
        prediction = model.predict(pixels)[0]        # what digit is it?
        confidence = model.predict_proba(pixels).max() * 100  # how sure?
        results[name] = {
            "prediction": prediction,
            "confidence": round(confidence, 1)
        }

    return jsonify(results)  # send results back as JSON

if __name__ == "__main__":
    app.run(debug=True)  # debug=True auto-reloads when you save