#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import load_model

from src.config import (
    USE_AWS,
    LOCAL_SEQUENCES,
    LOCAL_MODELS,
    LOCAL_LIVE,
    LOCAL_TRANSFORMED,
    S3_SEQUENCES,
    S3_MODELS,
    S3_LIVE,
    RDS_HOST,
    RDS_USER,
    RDS_PASS,
    RDS_DB
)

if USE_AWS:
    import s3fs
    from sqlalchemy import create_engine


# ----------------------------------------------------------------------
# Load most recent transformed clean data from local storage
# ----------------------------------------------------------------------
def load_local_clean(city, window):
    clean_path = os.path.join(LOCAL_TRANSFORMED, "combined_clean.json")
    df = pd.read_json(clean_path)
    df = df[df["city"] == city]
    df = df.sort_values("datetime")
    return df.tail(window)


# ----------------------------------------------------------------------
# Load normalizer weights and construct normalization layer
# ----------------------------------------------------------------------
def load_normalizer():
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "rb") as f:
            normalizer_weights = pickle.load(f)

        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "wb") as f:
            pickle.dump(normalizer_weights, f)

    else:
        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "rb") as f:
            normalizer_weights = pickle.load(f)

    norm = Normalization()
    norm.build(input_shape=(None, 4))
    norm.set_weights(normalizer_weights)
    return norm


# ----------------------------------------------------------------------
# Load trained LSTM model locally (copying from S3 if needed)
# ----------------------------------------------------------------------
def load_weather_model():
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        with s3.open(f"{S3_MODELS}/weather_lstm.keras", "rb") as f_in:
            with open(os.path.join(LOCAL_MODELS, "weather_lstm.keras"), "wb") as f_out:
                f_out.write(f_in.read())

    return load_model(os.path.join(LOCAL_MODELS, "weather_lstm.keras"))


# ----------------------------------------------------------------------
# Load latest cleaned window from RDS or local clean store
# ----------------------------------------------------------------------
def load_window(city, window):
    if USE_AWS:
        engine = create_engine(
            f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}"
        )
        df = pd.read_sql(
            f"SELECT * FROM weather_clean WHERE city='{city}' "
            f"ORDER BY datetime DESC LIMIT {window}",
            engine
        ).sort_values("datetime")
    else:
        df = load_local_clean(city, window)

    return df


# ----------------------------------------------------------------------
# Load live dataset (if exists)
# ----------------------------------------------------------------------
def load_live(city):
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        files = s3.glob(f"{S3_LIVE}/weather_live_all_cities_*.pkl")
        if not files:
            return None

        newest = sorted(files)[-1]
        with s3.open(newest, "rb") as f:
            live_df = pickle.load(f)

    else:
        files = [
            os.path.join(LOCAL_LIVE, f)
            for f in os.listdir(LOCAL_LIVE)
            if f.startswith("weather_live_all_cities_")
        ]
        if not files:
            return None

        newest = sorted(files)[-1]
        with open(newest, "rb") as f:
            live_df = pickle.load(f)

    row = live_df[live_df["city"] == city]
    if len(row) == 0:
        return None

    return float(row["temperature"].iloc[0])


# ----------------------------------------------------------------------
# Single step (+1 hour) forecast
# ----------------------------------------------------------------------
def predict_next_hour(city="Seattle", window=40):

    print("\nRunning +1 hour LSTM forecast...\n")

    normalizer = load_normalizer()
    model = load_weather_model()
    df = load_window(city, window)

    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

    window_np = df[feature_cols].astype(float).values
    window_norm = normalizer(window_np)
    X_input = np.array(window_norm).reshape(1, window, len(feature_cols))

    pred_c = float(model.predict(X_input)[0][0])
    pred_f = pred_c * 9.0 / 5.0 + 32.0

    last_c = float(df["temperature"].iloc[-1])
    last_f = last_c * 9.0 / 5.0 + 32.0

    live_c = load_live(city)
    live_f = live_c * 9.0 / 5.0 + 32.0 if live_c is not None else None

    print("==========================================================")
    print(f" {city} Weather – Forecast vs Live vs Model")
    print("==========================================================\n")
    print(f" Forecast Last Observed : {last_c:6.2f} C / {last_f:6.2f} F")
    if live_c is not None:
        print(f" Live Last Observed     : {live_c:6.2f} C / {live_f:6.2f} F")
    else:
        print(" Live Last Observed     : unavailable")
    print(f" Model Prediction (+1h) : {pred_c:6.2f} C / {pred_f:6.2f} F")
    print("==========================================================\n")

    return {
        "city": city,
        "forecast_last_temp_c": last_c,
        "forecast_last_temp_f": last_f,
        "live_last_temp_c": live_c,
        "live_last_temp_f": live_f,
        "pred_c": pred_c,
        "pred_f": pred_f
    }


# ----------------------------------------------------------------------
# Multi-step autoregressive forecast (+N hours)
# ----------------------------------------------------------------------
def predict_next_n_hours(city="Seattle", window=40, steps=8):

    print(f"\nRunning autoregressive LSTM forecast for next {steps} hours...\n")

    normalizer = load_normalizer()
    model = load_weather_model()
    df = load_window(city, window)

    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

    # Extract last timestamp
    last_ts = pd.to_datetime(df["datetime"].iloc[-1])

    # Use list so we can append predictions autoregressively
    window_data = df[feature_cols].astype(float).values.tolist()

    preds_c = []
    preds_f = []
    timestamps = []

    for step in range(steps):
        arr = np.array(window_data[-window:], dtype=float)
        arr_norm = np.array(normalizer(arr))
        arr_input = arr_norm.reshape(1, window, len(feature_cols))

        # Predict temperature
        pred_c = float(model.predict(arr_input)[0][0])
        pred_f = pred_c * 9.0 / 5.0 + 32.0

        preds_c.append(pred_c)
        preds_f.append(pred_f)

        # Compute timestamp for this step
        future_ts = last_ts + pd.Timedelta(hours=step + 1)
        timestamps.append(future_ts)

        # Append predicted step back into window
        window_data.append([
            pred_c,
            window_data[-1][1],
            window_data[-1][2],
            window_data[-1][3]
        ])

    print("==========================================================")
    print(f" {city} Weather – {steps}-Hour Forecast")
    print("==========================================================\n")

    for i, (ts, c, f) in enumerate(zip(timestamps, preds_c, preds_f), start=1):
        print(f" +{i:02d}h ({ts.strftime('%Y-%m-%d %H:%M')}) : {c:6.2f} C / {f:6.2f} F")

    print("==========================================================\n")

    return timestamps, preds_c, preds_f



# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Single-step forecast
    predict_next_hour()

    # Multi-step (+8h) forecast
    predict_next_n_hours(steps=8)
