import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.dropna()
    return df

def preprocess_features(df):
    X = df.drop('label', axis=1)
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
