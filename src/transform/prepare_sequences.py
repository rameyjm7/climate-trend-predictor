#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
import s3fs
from tensorflow.keras.layers import Normalization
from utils.provenance_utils import ProvenanceTimer

from src.config import (
    USE_AWS,
    LOCAL_SEQUENCES,
    LOCAL_TRANSFORMED,
    LOCAL_HISTORICAL,
    S3_TRANSFORMED,
    S3_SEQUENCES,
)

def prepare_sequences():
    WINDOW = 40

    with ProvenanceTimer(
        stage="feature_extraction_weather",
        input_source="clean + historical",
        output_target="sequence generation"
    ) as p:

        # Load cleaned dataset
        if USE_AWS:
            s3 = s3fs.S3FileSystem()
            with s3.open(f"{S3_TRANSFORMED}/combined_clean.json", "rb") as f:
                clean_df = pd.read_json(f)

            local_clean_path = os.path.join(LOCAL_TRANSFORMED, "combined_clean.json")
            clean_df.to_json(local_clean_path)
        else:
            clean_df = pd.read_json(os.path.join(LOCAL_TRANSFORMED, "combined_clean.json"))

        # Load historical dataset
        hist_local_path = os.path.join(LOCAL_HISTORICAL, "historical_weather_all_cities.csv")
        if not os.path.exists(hist_local_path):
            raise RuntimeError("Historical dataset missing — run ingest_weather_historical first.")

        hist_df = pd.read_csv(hist_local_path)
        hist_df["datetime"] = pd.to_datetime(hist_df["datetime"])

        df = pd.concat([clean_df, hist_df], ignore_index=True)
        df = df.sort_values("datetime").reset_index(drop=True)

        feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

        X_raw = df[feature_cols].astype(float).values
        y_raw = df["temperature"].astype(float).values

        X_seq, y_seq = [], []
        for i in range(len(X_raw) - WINDOW):
            X_seq.append(X_raw[i:i+WINDOW])
            y_seq.append(y_raw[i+WINDOW])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        normalizer = Normalization()
        normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))
        normalizer_weights = normalizer.get_weights()

        # Save locally
        np.save(os.path.join(LOCAL_SEQUENCES, "X_train_weather.npy"), X_train)
        np.save(os.path.join(LOCAL_SEQUENCES, "X_test_weather.npy"), X_test)
        np.save(os.path.join(LOCAL_SEQUENCES, "y_train_weather.npy"), y_train)
        np.save(os.path.join(LOCAL_SEQUENCES, "y_test_weather.npy"), y_test)

        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "wb") as f:
            pickle.dump(normalizer_weights, f)

        # Save to S3
        if USE_AWS:
            with s3.open(f"{S3_SEQUENCES}/X_train_weather.npy", "wb") as f:
                np.save(f, X_train)
            with s3.open(f"{S3_SEQUENCES}/X_test_weather.npy", "wb") as f:
                np.save(f, X_test)
            with s3.open(f"{S3_SEQUENCES}/y_train_weather.npy", "wb") as f:
                np.save(f, y_train)
            with s3.open(f"{S3_SEQUENCES}/y_test_weather.npy", "wb") as f:
                np.save(f, y_test)
            with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "wb") as f:
                pickle.dump(normalizer_weights, f)

        p.commit(
            status="SUCCESS",
            records_in=len(df),
            records_out=len(X_seq),
            extra={"window_hours": WINDOW}
        )

if __name__ == "__main__":
    prepare_sequences()
