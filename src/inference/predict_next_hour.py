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
    LOCAL_CLEANED,
    S3_SEQUENCES,
    S3_MODELS,
    S3_LIVE,
)

if USE_AWS:
    import s3fs
    from sqlalchemy import create_engine
    from src.config import RDS_HOST, RDS_USER, RDS_PASS, RDS_DB


def load_latest_local_forecast(city, window):
    """
    Load last `window` rows from the locally cleaned dataset.
    Enables full offline inference.
    """
    cleaned_csv = os.path.join(LOCAL_CLEANED, "weather_clean.csv")

    if not os.path.exists(cleaned_csv):
        raise RuntimeError(
            f"Local cleaned dataset not found: {cleaned_csv}\n"
            "Run ingestion + sequence prep at least once with USE_AWS=True."
        )

    df = pd.read_csv(cleaned_csv)
    df_city = df[df["city"] == city].sort_values("datetime")

    if len(df_city) < window:
        raise RuntimeError(
            f"Not enough local rows for {city}. Need {window}, have {len(df_city)}."
        )

    return df_city.iloc[-window:]


def predict_next_hour(city="Seattle", window=40):

    print("\nRunning 40-hour LSTM forecast...\n")

    # ---------------------------------------------------------
    # Load X normalizer
    # ---------------------------------------------------------
    x_norm_path = os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl")
    if USE_AWS:
        s3 = s3fs.S3FileSystem()
        with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "rb") as f:
            weights = pickle.load(f)
        with open(x_norm_path, "wb") as f:
            pickle.dump(weights, f)
    else:
        with open(x_norm_path, "rb") as f:
            weights = pickle.load(f)

    x_norm = Normalization()
    x_norm.build(input_shape=(None, 4))
    x_norm.set_weights(weights)

    # ---------------------------------------------------------
    # Load Y normalizer
    # ---------------------------------------------------------
    y_norm_path = os.path.join(LOCAL_SEQUENCES, "y_normalizer_weights.pkl")
    if USE_AWS:
        with s3.open(f"{S3_SEQUENCES}/y_normalizer_weights.pkl", "rb") as f:
            yw = pickle.load(f)
        with open(y_norm_path, "wb") as f:
            pickle.dump(yw, f)
    else:
        with open(y_norm_path, "rb") as f:
            yw = pickle.load(f)

    y_norm = Normalization(axis=None)
    y_norm.build(input_shape=(None, 1))
    y_norm.set_weights(yw)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model_path = os.path.join(LOCAL_MODELS, "weather_lstm.keras")

    if USE_AWS:
        with s3.open(f"{S3_MODELS}/weather_lstm.keras", "rb") as f_in:
            with open(model_path, "wb") as f_out:
                f_out.write(f_in.read())

    model = load_model(model_path)

    # ---------------------------------------------------------
    # Load historical window (AWS or local CSV)
    # ---------------------------------------------------------
    if USE_AWS:
        engine = create_engine(
            f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}"
        )
        df = pd.read_sql(
            f"SELECT * FROM weather_clean WHERE city='{city}' "
            f"ORDER BY datetime DESC LIMIT {window}", engine
        )
        df = df.sort_values("datetime")

    else:
        df = load_latest_local_forecast(city, window)

    # ---------------------------------------------------------
    # Prepare model input
    # ---------------------------------------------------------
    feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]
    X_window = df[feature_cols].astype(float).values

    X_normed = x_norm(X_window)
    X_input = np.array(X_normed).reshape(1, window, len(feature_cols))

    # ---------------------------------------------------------
    # Predict normalized Y
    # ---------------------------------------------------------
    pred_y_norm = model.predict(X_input)[0][0]

    # ---------------------------------------------------------
    # Denormalize
    # ---------------------------------------------------------
    mu = y_norm.mean.numpy()
    sigma = np.sqrt(y_norm.variance.numpy())
    pred_c = float(mu + pred_y_norm * sigma)
    pred_f = pred_c * 9/5 + 32

    forecast_last_temp_c = float(df["temperature"].iloc[-1])
    forecast_last_temp_f = forecast_last_temp_c * 9/5 + 32

    # ---------------------------------------------------------
    # Load LIVE (AWS or local)
    # ---------------------------------------------------------
    live_last_c = None

    if USE_AWS:
        files = s3.glob(f"{S3_LIVE}/weather_live_all_cities_*.pkl")
        live_df = None
        if files:
            newest = sorted(files)[-1]
            with s3.open(newest, "rb") as f:
                live_df = pickle.load(f)
    else:
        files = [
            os.path.join(LOCAL_LIVE, f)
            for f in os.listdir(LOCAL_LIVE)
            if f.startswith("weather_live_all_cities_")
        ]
        live_df = None
        if files:
            newest = sorted(files)[-1]
            with open(newest, "rb") as f:
                live_df = pickle.load(f)

    if live_df is not None:
        row = live_df[live_df["city"] == city]
        if len(row) > 0:
            live_last_c = float(row["temperature"].iloc[0])

    live_last_f = live_last_c * 9/5 + 32 if live_last_c is not None else None

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    print("==========================================================")
    print(f" {city} Weather – Forecast vs Live vs Model")
    print("==========================================================\n")

    print(f" Forecast Last Observed : {forecast_last_temp_c:6.2f} C / {forecast_last_temp_f:6.2f} F")

    if live_last_c is not None:
        print(f" Live Last Observed     : {live_last_c:6.2f} C / {live_last_f:6.2f} F")
    else:
        print(" Live Last Observed     : unavailable")

    print(f" Model Prediction (+1h) : {pred_c:6.2f} C / {pred_f:6.2f} F")
    print("==========================================================\n")

    return {
        "city": city,
        "forecast_last_temp_c": forecast_last_temp_c,
        "forecast_last_temp_f": forecast_last_temp_f,
        "live_last_temp_c": live_last_c,
        "live_last_temp_f": live_last_f,
        "pred_c": pred_c,
        "pred_f": pred_f,
    }


if __name__ == "__main__":
    predict_next_hour()
