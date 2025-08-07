import pandas as pd
import random
import os

# Define list of common symptoms and diseases
symptoms = [
    "fever", "headache", "chills", "nausea", "fatigue", "cough", "sore_throat", "shortness_breath",
    "vomiting", "abdominal_pain", "dehydration", "vision_blur", "dizziness", "weakness",
    "chest_pain", "breathlessness", "diarrhea", "rash", "joint_pain", "muscle_pain", "irritability"
]

diseases = ["Dengue", "COVID-19", "Typhoid", "Migraine", "Pneumonia", "Malaria", "Flu", "Jaundice", "Asthma"]

# Generate 500 fake patient records
data = []
for _ in range(500):
    selected_symptoms = random.sample(symptoms, 5)
    record = selected_symptoms + [random.choice(diseases)]
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data, columns=["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5", "Disease"])

# Define custom path (relative or absolute)
csv_path = "data/symptom_disease_dataset_large.csv"

# Ensure 'data/' folder exists
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Save CSV
df.to_csv(csv_path, index=False)

print(f"Dataset saved at: {csv_path}")
