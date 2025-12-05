#!/usr/bin/env python3

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import load_model

from src.config import LOCAL_SEQUENCES, LOCAL_MODELS

def evaluate_model():

    # Load model
    model_path = os.path.join(LOCAL_MODELS, "weather_lstm.keras")
    model = load_model(model_path)

    # Load sequences
    X_test = np.load(os.path.join(LOCAL_SEQUENCES, "X_test_weather.npy"))
    y_test = np.load(os.path.join(LOCAL_SEQUENCES, "y_test_weather.npy"))

    # Predictions
    preds = model.predict(X_test).reshape(-1)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("Evaluation Metrics:")
    print(f" MAE  : {mae:.4f}")
    print(f" RMSE : {rmse:.4f}")
    print(f" R2   : {r2:.4f}")

    out_dir = os.path.join(LOCAL_MODELS, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:200], label="Actual", linewidth=2)
    plt.plot(preds[:200], label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted (first 200 samples)")
    plt.xlabel("Sample")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"))
    plt.close()

    # Residuals histogram
    residuals = y_test - preds
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.title("Residual Distribution")
    plt.xlabel("Error (C)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(out_dir, "residual_histogram.png"))
    plt.close()


if __name__ == "__main__":
    evaluate_model()
