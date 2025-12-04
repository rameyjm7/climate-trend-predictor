import pandas as pd
import numpy as np
import pickle
import s3fs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def prepare_sequences():
    """
    Load cleaned weather data from S3, scale features, build sliding-window
    sequences using TimeSeriesSplit (the same pattern used in class),
    and write X_train / X_test / y_train / y_test back to S3.
    """

    s3 = s3fs.S3FileSystem()

    # Adjust based on your project S3 path
    CLEAN_DIR = "s3://ece5984-s3-rameyjm7/Project/transformed"
    SEQ_DIR   = "s3://ece5984-s3-rameyjm7/Project/sequences"

    # Load cleaned weather dataframe (JSON)
    with s3.open(f"{CLEAN_DIR}/weather_clean.json", "rb") as f:
        df = pd.read_json(f)

    # Sort by date (important for time-series)
    df = df.sort_values(by="date")

    # Set target (temperature_forecast)
    y = df["temperature"].astype(float).values

    # Feature columns for LSTM input
    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

    X_raw = df[feature_cols].astype(float)

    # Apply MinMaxScaler (same as class requirement)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    # Save scaler for inference
    with s3.open(f"{SEQ_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # TimeSeriesSplit — exactly like class
    tscv = TimeSeriesSplit(n_splits=10)

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train = X_scaled.iloc[train_idx].values
        X_test  = X_scaled.iloc[test_idx].values
        y_train = y[train_idx]
        y_test  = y[test_idx]

    # Save sequences to S3
    with s3.open(f"{SEQ_DIR}/X_train_weather.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with s3.open(f"{SEQ_DIR}/X_test_weather.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with s3.open(f"{SEQ_DIR}/y_train_weather.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with s3.open(f"{SEQ_DIR}/y_test_weather.pkl", "wb") as f:
        pickle.dump(y_test, f)

    print("Weather sequences prepared and stored to S3.")
