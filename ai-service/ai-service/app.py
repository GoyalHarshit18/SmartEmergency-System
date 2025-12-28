from flask import Flask, request, jsonify
import joblib
import os

print("🔥 Running ML Flask API from:", os.path.abspath(__file__))

# Load saved ML model + vectorizer
model = joblib.load("severity_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

def predict_severity(text: str):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

@app.route("/", methods=["GET"])
def home():
    return "🔥 ML Severity Prediction Service is Running (PORT 8080)"

@app.route("/severity", methods=["POST"])
def severity():
    data = request.json
    text = data.get("text", "")
    severity = predict_severity(text)
    return jsonify({"severity": severity})

if __name__ == "__main__":
    # use same port as before
    app.run(host="0.0.0.0", port=8080)
