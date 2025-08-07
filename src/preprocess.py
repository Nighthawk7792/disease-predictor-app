import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_encode_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()

    if 'Disease' not in df.columns:
        raise ValueError("Expected a 'Disease' column in the dataset.")
    label_encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    return df, label_encoders
