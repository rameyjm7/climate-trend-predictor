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


def load_local_clean(city, window):
    clean_path = os.path.join(LOCAL_TRANSFORMED, "combined_clean.json")
    df = pd.read_json(clean_path)
    df = df[df["city"] == city]
    df = df.sort_values("datetime")
    return df.tail(window)


def predict_next_hour(city="Seattle", window=40):

    print("\nRunning 40-hour LSTM forecast...\n")

    # Load normalizer
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "rb") as f:
            normalizer_weights = pickle.load(f)
        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "wb") as f:
            pickle.dump(normalizer_weights, f)
    else:
        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "rb") as f:
            normalizer_weights = pickle.load(f)

    normalizer = Normalization()
    normalizer.build(input_shape=(None, 4))
    normalizer.set_weights(normalizer_weights)

    # Load model
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        with s3.open(f"{S3_MODELS}/weather_lstm.keras", "rb") as f_in:
            with open(os.path.join(LOCAL_MODELS, "weather_lstm.keras"), "wb") as f_out:
                f_out.write(f_in.read())

    model = load_model(os.path.join(LOCAL_MODELS, "weather_lstm.keras"))

    # Load forecast window
    if USE_AWS:
        engine = create_engine(f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}")
        df = pd.read_sql(
            f"SELECT * FROM weather_clean WHERE city='{city}' ORDER BY datetime DESC LIMIT {window}",
            engine
        ).sort_values("datetime")
    else:
        df = load_local_clean(city, window)

    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]
    X_window = df[feature_cols].astype(float).values

    X_norm = normalizer(X_window)
    X_input = np.array(X_norm).reshape(1, window, len(feature_cols))

    pred_c = float(model.predict(X_input)[0][0])
    pred_f = pred_c * 9/5 + 32

    forecast_last_c = float(df["temperature"].iloc[-1])
    forecast_last_f = forecast_last_c * 9/5 + 32

    # LIVE data
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        files = s3.glob(f"{S3_LIVE}/weather_live_all_cities_*.pkl")
        if files:
            newest = sorted(files)[-1]
            with s3.open(newest, "rb") as f:
                live_df = pickle.load(f)
        else:
            live_df = None
    else:
        files = [
            os.path.join(LOCAL_LIVE, f)
            for f in os.listdir(LOCAL_LIVE)
            if f.startswith("weather_live_all_cities_")
        ]
        if files:
            newest = sorted(files)[-1]
            with open(newest, "rb") as f:
                live_df = pickle.load(f)
        else:
            live_df = None

    if live_df is not None:
        row = live_df[live_df["city"] == city]
        live_c = float(row["temperature"].iloc[0]) if len(row) > 0 else None
    else:
        live_c = None

    live_f = live_c * 9/5 + 32 if live_c is not None else None

    print("==========================================================")
    print(f" {city} Weather – Forecast vs Live vs Model")
    print("==========================================================\n")
    print(f" Forecast Last Observed : {forecast_last_c:6.2f} C / {forecast_last_f:6.2f} F")
    if live_c is not None:
        print(f" Live Last Observed     : {live_c:6.2f} C / {live_f:6.2f} F")
    else:
        print(" Live Last Observed     : unavailable")
    print(f" Model Prediction (+1h) : {pred_c:6.2f} C / {pred_f:6.2f} F")
    print("==========================================================\n")

    return {
        "city": city,
        "forecast_last_temp_c": forecast_last_c,
        "forecast_last_temp_f": forecast_last_f,
        "live_last_temp_c": live_c,
        "live_last_temp_f": live_f,
        "pred_c": pred_c,
        "pred_f": pred_f
    }


if __name__ == "__main__":
    predict_next_hour()
