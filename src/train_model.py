import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

from src.preprocess import load_and_encode_data

def train_and_save_model(csv_path, model_path):
    print(" Step 1: Loading and encoding data...")
    df, encoders = load_and_encode_data(csv_path)

    print("Step 2: Splitting data into train/test sets...")
    X = df.drop("Disease", axis=1)
    y = df["Disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 3: Training kNN model...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("Step 4: Evaluating model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.2f}")

    print("Step 5: Saving model and encoders...")
    joblib.dump(knn, model_path)
    joblib.dump(encoders, "models/label_encoders.pkl")

    print(f"Model saved to: {model_path}")
    print(" Label encoders saved to: models/label_encoders.pkl")

if __name__ == "__main__":
    print("🚀 Script started...")
    csv_path = "data/symptom_disease_dataset_large.csv"
    model_path = "models/disease_model.pkl"
    train_and_save_model(csv_path, model_path)
