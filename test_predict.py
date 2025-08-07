from src.predict import predict_disease

symptoms = ["fever", "headache", "nausea", "fatigue", "vomiting"]  # You can try your own

prediction = predict_disease(symptoms)
print(f"🩺 Predicted Disease: {prediction}")