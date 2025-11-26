from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import joblib
from nn_model import HeartNN
from models import User, PredictionRecord
from pymongo import MongoClient
from bson import ObjectId
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/heart_prediction')

jwt = JWTManager(app)

# MongoDB connection
mongo_uri = app.config['MONGO_URI']
client = MongoClient(mongo_uri)
# Extract database name from URI or use default
db_name = mongo_uri.split('/')[-1] if '/' in mongo_uri else 'heart_prediction'
db = client[db_name] if db_name else client.get_database('heart_prediction')

# Load ML models (KEEPING EXISTING CODE UNCHANGED)
scaler = joblib.load("models/scaler.pkl")

model = HeartNN(input_dim=11)
model.load_state_dict(torch.load("models/nn_model.pth", map_location="cpu"))
model.eval()

# Encoding maps (KEEPING EXISTING CODE UNCHANGED)
sex_map = {"M": 1, "F": 0}
cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
angina_map = {"No": 0, "Yes": 1}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}

# Helper function for prediction (KEEPING EXISTING LOGIC)
def make_prediction(form_data):
    Age = float(form_data["Age"])
    Sex = sex_map[form_data["Sex"]]
    ChestPainType = cp_map[form_data["ChestPainType"]]
    RestingBP = float(form_data["RestingBP"])
    Cholesterol = float(form_data["Cholesterol"])
    FastingBS = 1 if form_data["FastingBS"] == "Yes" else 0
    RestingECG = ecg_map[form_data["RestingECG"]]
    MaxHR = float(form_data["MaxHR"])
    ExerciseAngina = angina_map[form_data["ExerciseAngina"]]
    Oldpeak = float(form_data["Oldpeak"])
    ST_Slope = slope_map[form_data["ST_Slope"]]

    data = np.array([[
        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    ]])

    scaled = scaler.transform(data)

    with torch.no_grad():
        logit = model(torch.tensor(scaled, dtype=torch.float32)).item()
        # Apply sigmoid to convert logit to probability
        prob = 1 / (1 + np.exp(-logit))

    pred = "High Risk" if prob >= 0.5 else "Low Risk"
    probability = round(prob * 100, 2)
    
    return pred, probability, {
        "Age": Age, "Sex": form_data["Sex"], "ChestPainType": form_data["ChestPainType"],
        "RestingBP": RestingBP, "Cholesterol": Cholesterol, "FastingBS": form_data["FastingBS"],
        "RestingECG": form_data["RestingECG"], "MaxHR": MaxHR, "ExerciseAngina": form_data["ExerciseAngina"],
        "Oldpeak": Oldpeak, "ST_Slope": form_data["ST_Slope"]
    }

# Legacy routes (keeping for backward compatibility)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pred, probability, inputs = make_prediction(request.form)
    return render_template("result.html",
                           prediction=pred,
                           probability=probability)

# New API routes
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    
    if User.find_by_email(data.get("email")):
        return jsonify({"error": "Email already exists"}), 400
    
    user = User.create(
        name=data.get("name"),
        email=data.get("email"),
        password=data.get("password")
    )
    
    access_token = create_access_token(identity=str(user["_id"]))
    return jsonify({
        "token": access_token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "createdAt": user["createdAt"].isoformat()
        }
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.find_by_email(data.get("email"))
    
    if not user or not check_password_hash(user["password"], data.get("password")):
        return jsonify({"error": "Invalid credentials"}), 401
    
    access_token = create_access_token(identity=str(user["_id"]))
    return jsonify({
        "token": access_token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "createdAt": user["createdAt"].isoformat()
        }
    }), 200

@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.find_by_id(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    record_count = PredictionRecord.count_by_user(user_id)
    
    return jsonify({
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "createdAt": user["createdAt"].isoformat(),
        "predictionCount": record_count
    }), 200

@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict_api():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    pred, probability, inputs = make_prediction(data)
    
    # Save prediction record
    record = PredictionRecord.create(
        user_id=user_id,
        inputs=inputs,
        prediction=pred,
        probability=probability
    )
    
    return jsonify({
        "prediction": pred,
        "probability": probability,
        "recordId": str(record["_id"]),
        "timestamp": record["timestamp"].isoformat()
    }), 200

@app.route("/api/records", methods=["GET"])
@jwt_required()
def get_records():
    user_id = get_jwt_identity()
    records = PredictionRecord.find_by_user(user_id)
    
    return jsonify([{
        "id": str(r["_id"]),
        "inputs": r["inputs"],
        "prediction": r["prediction"],
        "probability": r["probability"],
        "timestamp": r["timestamp"].isoformat()
    } for r in records]), 200

@app.route("/api/records/<record_id>", methods=["DELETE"])
@jwt_required()
def delete_record(record_id):
    user_id = get_jwt_identity()
    
    if PredictionRecord.delete(record_id, user_id):
        return jsonify({"message": "Record deleted"}), 200
    return jsonify({"error": "Record not found"}), 404

@app.route("/api/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    user = User.update(user_id, data)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "createdAt": user["createdAt"].isoformat()
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
