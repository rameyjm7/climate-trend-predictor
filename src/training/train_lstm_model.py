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

from utils.provenance_utils import ProvenanceTimer
from src.config import (
    USE_AWS,
    LOCAL_SEQUENCES,
    LOCAL_MODELS,
    S3_MODELS,
    S3_SEQUENCES,
)

if USE_AWS:
    import s3fs


def train_lstm_weather():
    WINDOW = 40

    with ProvenanceTimer(
        stage="train_lstm",
        input_source="sequences",
        output_target="model"
    ) as p:

        if USE_AWS:
            s3 = s3fs.S3FileSystem()
            X_train = np.load(s3.open(f"{S3_SEQUENCES}/X_train_weather.npy", "rb"))
            X_test  = np.load(s3.open(f"{S3_SEQUENCES}/X_test_weather.npy",  "rb"))
            y_train = np.load(s3.open(f"{S3_SEQUENCES}/y_train_weather.npy", "rb"))
            y_test  = np.load(s3.open(f"{S3_SEQUENCES}/y_test_weather.npy",  "rb"))

            with s3.open(f"{S3_SEQUENCES}/normalizer_weights.pkl", "rb") as f:
                normalizer_weights = pickle.load(f)

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

        normalizer = Normalization()
        normalizer.build(input_shape=(None, n_features))
        normalizer.set_weights(normalizer_weights)

        X_train_norm = np.array(normalizer(X_train))
        X_test_norm  = np.array(normalizer(X_test))

        model = Sequential([
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_test_norm, y_test),
            epochs=20,
            batch_size=64,
            shuffle=False,
            verbose=1
        )

        os.makedirs(LOCAL_MODELS, exist_ok=True)
        model_path = os.path.join(LOCAL_MODELS, "weather_lstm.keras")
        model.save(model_path)

        # ------------------------------------------------------------
        # Plot training vs validation MAE
        # ------------------------------------------------------------
        mae = history.history.get("mae", [])
        val_mae = history.history.get("val_mae", [])

        plt.figure(figsize=(10, 6))
        plt.plot(mae, label="Train MAE")
        plt.plot(val_mae, label="Validation MAE")
        plt.title("Training vs Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        mae_plot_path = os.path.join(LOCAL_MODELS, "training_mae_curve.png")
        plt.savefig(mae_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Upload to S3 if enabled
        if USE_AWS:
            s3.put(model_path, f"{S3_MODELS}/weather_lstm.keras")
            s3.put(loss_plot_path, f"{S3_MODELS}/training_loss_curve.png")
            s3.put(mae_plot_path,  f"{S3_MODELS}/training_mae_curve.png")


if __name__ == "__main__":
    train_lstm_weather()
