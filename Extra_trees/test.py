import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_PATH = "C:\\Users\\Jha_Sweety\\Desktop\\Internship\\Hoefding_Tree\\test\\"
MODEL_PATH = "Model_ExtraTrees.pkl"     
SCALER_PATH = "Scaler.pkl"

PLOT_DIR = "Plots"
METRIC_DIR = "Metrics"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

TARGET_COLUMN = "vinn"
FEATURE_COLUMNS = ["pd", "vinp", "vdd", "process", "temperature"]



with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


def load_files(folder):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            yield file, os.path.join(folder, file)


for file_name, file_path in load_files(TEST_PATH):

    print(f"Processing {file_name}")

    df = pd.read_csv(file_path)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    if df.empty:
        print("Skipped — empty after dropna")
        continue

    X = df[FEATURE_COLUMNS].values
    y_true = df[TARGET_COLUMN].values
  
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf

    metrics_df = pd.DataFrame([{
        "MAE": round(mae, 6),
        "MSE": round(mse, 6),
        "R2": round(r2, 6),
        "SNR": round(snr, 2)
    }])

    metrics_path = os.path.join(
        METRIC_DIR,
        file_name.replace(".csv", "_metrics.csv")
    )
    metrics_df.to_csv(metrics_path, index=False)

    plt.figure(figsize=(12, 6))

    plt.plot(
        y_true,
        color="blue",
        linewidth=2,
        label="Actual vinn"
    )

    plt.plot(
        y_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Predicted vinn"
    )

    plt.xlabel("Simulation Time")
    plt.ylabel("vinn")
    plt.title(f"Actual vs Predicted vinn - {file_name}")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(
        PLOT_DIR,
        file_name.replace(".csv", "_plot.png")
    )

    plt.savefig(plot_path, dpi=150)
    plt.close()


print("\nAll test files processed successfully.")
