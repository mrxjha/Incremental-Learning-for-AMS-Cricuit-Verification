import os
import time
import pickle
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "C:\\Users\\Jha_Sweety\\Desktop\\Internship\\Hoefding_Tree\\train\\"
MODEL_PATH = "Model_SGDRegressor.pkl"
SCALER_PATH = "Scaler.pkl"

TARGET_COLUMN = "vinn"
FEATURE_COLUMNS = ["pd", "vinp", "vdd", "process", "temperature"]


def load_files_from_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    yield file, df
            except Exception as e:
                print(f"Error loading {file}: {e}")


model = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=1e-5,          
    learning_rate="invscaling",
    eta0=0.03,           
    power_t=0.25,
    random_state=42
)


scaler = StandardScaler()

print("Incremental training started...")
start_time = time.time()
file_count = 0
scaler_fitted = False


for fname, df in load_files_from_folder(TRAIN_PATH):

    # Drop bad rows
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        continue

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # ✅ Fit scaler incrementally
    if not scaler_fitted:
        scaler.partial_fit(X)
        scaler_fitted = True
    else:
        scaler.partial_fit(X)

    X_scaled = scaler.transform(X)

    # ✅ Incremental model training
    model.partial_fit(X_scaled, y)

    file_count += 1

    if file_count % 5 == 0:
        print(f"Processed {file_count} files in {time.time() - start_time:.2f}s")


# ✅ Save model + scaler
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("\nTraining completed successfully.")
print("Model saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
