

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np

model = joblib.load("models/disease_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

def predict_disease(symptom_list):
    if len(symptom_list) != 5:
        raise ValueError("Exactly 5 symptoms are required")
    encoded_input = []
    for i, symptom in enumerate(symptom_list):
        col_name = f"Symptom_{i+1}"
        encoder = encoders[col_name]
        encoded_value = encoder.transform([symptom])[0]
        encoded_input.append(encoded_value)   
    encoded_input = np.array(encoded_input).reshape(1, -1)
    prediction_encoded = model.predict(encoded_input)[0]
    disease = encoders["Disease"].inverse_transform([prediction_encoded])[0]
    return disease
