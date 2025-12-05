#!/usr/bin/env python3

import os
import numpy as np
import pickle
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Normalization

from src.config import (
    USE_AWS,
    LOCAL_SEQUENCES,
    LOCAL_MODELS,
    S3_SEQUENCES,
    S3_MODELS,
)

if USE_AWS:
    import s3fs

from utils.provenance_utils import ProvenanceTimer


def train_lstm_weather():

    WINDOW = 40

    with ProvenanceTimer(
        stage="train_lstm_weather",
        input_source="prepared sequences",
        output_target="trained model"
    ) as p:

        # ---------------------------------------------------------
        # Load sequences
        # ---------------------------------------------------------
        if USE_AWS:
            s3 = s3fs.S3FileSystem()

            X_train = np.load(s3.open(f"{S3_SEQUENCES}/X_train_weather.npy", "rb"))
            X_test  = np.load(s3.open(f"{S3_SEQUENCES}/X_test_weather.npy",  "rb"))
            y_train = np.load(s3.open(f"{S3_SEQUENCES}/y_train_weather.npy", "rb"))
            y_test  = np.load(s3.open(f"{S3_SEQUENCES}/y_test_weather.npy",  "rb"))

            with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "rb") as f:
                normalizer_weights = pickle.load(f)

            # Write mirrored copies locally
            np.save(os.path.join(LOCAL_SEQUENCES, "X_train_weather.npy"), X_train)
            np.save(os.path.join(LOCAL_SEQUENCES, "X_test_weather.npy"), X_test)
            np.save(os.path.join(LOCAL_SEQUENCES, "y_train_weather.npy"), y_train)
            np.save(os.path.join(LOCAL_SEQUENCES, "y_test_weather.npy"), y_test)
            with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "wb") as f:
                pickle.dump(normalizer_weights, f)

        else:
            X_train = np.load(os.path.join(LOCAL_SEQUENCES, "X_train_weather.npy"))
            X_test  = np.load(os.path.join(LOCAL_SEQUENCES, "X_test_weather.npy"))
            y_train = np.load(os.path.join(LOCAL_SEQUENCES, "y_train_weather.npy"))
            y_test  = np.load(os.path.join(LOCAL_SEQUENCES, "y_test_weather.npy"))
            with open(os.path.join(LOCAL_SEQUENCES, "normalizer_weights.pkl"), "rb") as f:
                normalizer_weights = pickle.load(f)

        n_features = X_train.shape[2]

        # ---------------------------------------------------------
        # Build X normalizer
        # ---------------------------------------------------------
        x_norm = Normalization()
        x_norm.build(input_shape=(None, n_features))
        x_norm.set_weights(normalizer_weights)

        X_train_norm = np.array(x_norm(X_train))
        X_test_norm  = np.array(x_norm(X_test))

        # ---------------------------------------------------------
        # NEW: y normalization (CRITICAL)
        # ---------------------------------------------------------
        y_norm = Normalization(axis=None)
        y_norm.adapt(y_train.reshape(-1, 1))

        # Save y-normalizer weights locally
        y_norm_path_local = os.path.join(LOCAL_SEQUENCES, "y_normalizer_weights.pkl")
        with open(y_norm_path_local, "wb") as f:
            pickle.dump(y_norm.get_weights(), f)

        # Save to S3 if online
        if USE_AWS:
            y_norm_path_s3 = f"{S3_SEQUENCES}/y_normalizer_weights.pkl"
            with s3.open(y_norm_path_s3, "wb") as f:
                pickle.dump(y_norm.get_weights(), f)

        # Normalize target labels
        y_train_norm = y_norm(y_train.reshape(-1, 1)).numpy()
        y_test_norm  = y_norm(y_test.reshape(-1, 1)).numpy()

        # ---------------------------------------------------------
        # Deep LSTM Model
        # ---------------------------------------------------------
        model = Sequential([
            LSTM(128, return_sequences=True, activation="tanh",
                 input_shape=(WINDOW, n_features)),
            Dropout(0.2),

            LSTM(64, return_sequences=False, activation="tanh"),
            Dropout(0.2),

            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        # ---------------------------------------------------------
        # Train Model
        # ---------------------------------------------------------
        history = model.fit(
            X_train_norm, y_train_norm,
            validation_data=(X_test_norm, y_test_norm),
            epochs=120,
            batch_size=16,
            shuffle=False,
            verbose=1
        )

        # ---------------------------------------------------------
        # Save model + graphs
        # ---------------------------------------------------------
        model_path_local = os.path.join(LOCAL_MODELS, "weather_lstm.keras")
        plot_path_local  = os.path.join(LOCAL_MODELS, "training_curve.png")

        with tempfile.TemporaryDirectory():

            model.save(model_path_local)

            plt.figure(figsize=(8,5))
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Val Loss")
            plt.legend()
            plt.title("Training vs Validation Loss (y-normalized)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(plot_path_local, dpi=150, bbox_inches="tight")
            plt.close()

            if USE_AWS:
                s3.put(model_path_local, f"{S3_MODELS}/weather_lstm.keras")
                s3.put(plot_path_local, f"{S3_MODELS}/training_curve.png")

        # ---------------------------------------------------------
        # Provenance
        # ---------------------------------------------------------
        p.commit(
            status="SUCCESS",
            records_in=len(X_train),
            records_out=1,
        )


if __name__ == "__main__":
    train_lstm_weather()
