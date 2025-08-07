from src.preprocess import load_and_encode_data

csv_path = "C:/Users/Nighthawk/Documents/Projects/Disease Prediction System/data/symptom_disease_dataset_large.csv"


df, encoders = load_and_encode_data(csv_path)

print("Data loaded successfully!")
print(df.head())
print(f"\nEncoders available for: {list(encoders.keys())}")
