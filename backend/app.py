from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)  # Allow all frontend requests

# Default route to confirm the backend is running
@app.route('/')
def index():
    return jsonify({'message': 'Flask API for Diabetes Prediction is running!'})

model = joblib.load('models/diabetes_rf_model.joblib')
scaler = joblib.load('models/diabetes_scaler.joblib')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json()  # Expecting JSON data
    # Expect features as a list in this order:
    # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    features = data.get('features')
    if features is None:
        return jsonify({'error': 'No features provided'}), 400
    
    # Convert to NumPy array, reshape, and scale the features
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Get prediction and probability (1 for diabetic, 0 for non-diabetic)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability)
    })

MODEL_PATH = 'lung_cancer_cnn_model.keras'
Model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dims
    return img

@app.route("/predict/cancer", methods=["POST"])
def predict_cancer():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files["file"]
    processed_image = preprocess_image(image)
    prediction = Model.predict(processed_image)[0][0]

    result = "Malignant" if prediction > 0.5 else "Benign"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

    return jsonify({"result": result, "confidence": confidence})

parkinsons_model = load_model('parkinsons_model.keras')
parkinsons_scaler = joblib.load('models/parkinsons_scaler.joblib')

@app.route('/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    data = request.get_json()
    features = data.get('features')

    if features is None:
        return jsonify({'error': 'No features provided'}), 400

    # Convert features to NumPy array and reshape
    features_array = np.array(features).reshape(1, -1)
    features_scaled = parkinsons_scaler.transform(features_array)
    features_reshaped = features_scaled.reshape(1, 1, -1)

    # Predict using the LSTM model
    prediction = parkinsons_model.predict(features_reshaped)[0][0]
    print(f"Raw Prediction: {prediction}")

    # Lowered threshold to 0.4 for better sensitivity
    result = "Parkinson's Detected" if prediction > 0.4 else "Healthy"
    confidence = float(prediction) if prediction > 0.4 else 1 - float(prediction)

    return jsonify({"result": result, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)