#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
import s3fs
import tensorflow as tf
from tensorflow.keras.layers import Normalization
from utils.provenance_utils import ProvenanceTimer

from src.config import (
    USE_AWS,
    S3_TRANSFORMED,
    S3_SEQUENCES,
    LOCAL_TRANSFORMED,
    LOCAL_SEQUENCES,
    LOCAL_CLEAN,
)


def prepare_sequences():

    WINDOW = 40

    with ProvenanceTimer(
        stage="feature_extraction_weather",
        input_source="clean data",
        output_target="sequence generation"
    ) as p:

        # ---------------------------------------------------------
        # Load cleaned dataset
        # ---------------------------------------------------------
        if USE_AWS:
            s3 = s3fs.S3FileSystem()
            remote_json = f"{S3_TRANSFORMED}/combined_clean.json"

            # Load dataframe from S3
            with s3.open(remote_json, "rb") as f:
                df = pd.read_json(f)

            # Save local copies for offline mode
            local_json = os.path.join(LOCAL_TRANSFORMED, "combined_clean.json")
            local_csv  = os.path.join(LOCAL_CLEAN, "weather_clean.csv")

            df.to_json(local_json, orient="records")
            df.to_csv(local_csv, index=False)

            print(f"Saved local cleaned JSON: {local_json}")
            print(f"Saved local cleaned CSV : {local_csv}")

        else:
            # Local-only mode
            local_json = os.path.join(LOCAL_TRANSFORMED, "combined_clean.json")
            local_csv  = os.path.join(LOCAL_CLEAN, "weather_clean.csv")

            if os.path.exists(local_json):
                df = pd.read_json(local_json)
            elif os.path.exists(local_csv):
                df = pd.read_csv(local_csv)
            else:
                raise RuntimeError(
                    f"No local cleaned dataset found.\n"
                    f"Expected:\n"
                    f"  {local_json}\n"
                    f"  {local_csv}\n"
                    f"Run once with USE_AWS=True first."
                )

        if df.empty:
            raise RuntimeError("Clean dataframe is empty.")

        # Sort by datetime or date
        df = df.sort_values(
            "datetime" if "datetime" in df.columns else "date"
        ).reset_index(drop=True)

        # ---------------------------------------------------------
        # Extract features
        # ---------------------------------------------------------
        feature_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

        X_raw = df[feature_cols].astype(float).values
        y_raw = df["temperature"].astype(float).values

        # ---------------------------------------------------------
        # Build sliding windows
        # ---------------------------------------------------------
        X_seq, y_seq = [], []
        for i in range(len(X_raw) - WINDOW):
            X_seq.append(X_raw[i:i+WINDOW])
            y_seq.append(y_raw[i + WINDOW])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Train/test split
        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        # ---------------------------------------------------------
        # Build Normalization layer
        # ---------------------------------------------------------
        normalizer = Normalization()
        normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))

        normalizer_weights = normalizer.get_weights()

        # ---------------------------------------------------------
        # Save everything locally (always)
        # ---------------------------------------------------------
        np.save(os.path.join(LOCAL_SEQUENCES, "X_train_weather.npy"), X_train)
        np.save(os.path.join(LOCAL_SEQUENCES, "X_test_weather.npy"), X_test)
        np.save(os.path.join(LOCAL_SEQUENCES, "y_train_weather.npy"), y_train)
        np.save(os.path.join(LOCAL_SEQUENCES, "y_test_weather.npy"), y_test)

        with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "wb") as f:
            pickle.dump(normalizer_weights, f)

        print(f"Saved local sequences to {LOCAL_SEQUENCES}")

        # ---------------------------------------------------------
        # Save to S3 if AWS enabled
        # ---------------------------------------------------------
        if USE_AWS:
            s3 = s3fs.S3FileSystem()

            def s3_save(path, arr):
                with s3.open(path, "wb") as f:
                    np.save(f, arr)

            s3_save(f"{S3_SEQUENCES}/X_train_weather.npy", X_train)
            s3_save(f"{S3_SEQUENCES}/X_test_weather.npy", X_test)
            s3_save(f"{S3_SEQUENCES}/y_train_weather.npy", y_train)
            s3_save(f"{S3_SEQUENCES}/y_test_weather.npy", y_test)

            with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "wb") as f:
                pickle.dump(normalizer_weights, f)

            print("Uploaded sequences + normalizer to S3.")

        p.commit(
            status="SUCCESS",
            records_in=len(df),
            records_out=len(X_seq),
            extra={"window_hours": WINDOW},
        )


if __name__ == "__main__":
    prepare_sequences()
