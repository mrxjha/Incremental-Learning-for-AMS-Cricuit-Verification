import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "C:\\Users\\Jha_Sweety\\Desktop\\Internship\\Hoefding_Tree\\train\\"
MODEL_PATH = "Model_ExtraTrees.pkl"
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


model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

scaler = StandardScaler()

print("Chunk-incremental training started (Extra Trees)...")
start_time = time.time()

X_all = []
y_all = []
file_count = 0


for fname, df in load_files_from_folder(TRAIN_PATH):

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        continue

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_all.append(X)
    y_all.append(y)

    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)

    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    model.fit(X_scaled, y_train)

    file_count += 1

    if file_count % 5 == 0:
        print(f"Processed {file_count} files in {time.time()-start_time:.2f}s")


with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("\nTraining completed.")
print("Model saved:", MODEL_PATH)
