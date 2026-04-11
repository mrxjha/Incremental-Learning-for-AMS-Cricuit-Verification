import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

TRAIN45_PATH = "C:\\Users\\Jha_Sweety\\Desktop\\Internship\\Increment_Ext_trees\\Train_45\\"

MODEL_PATH = "Model_ExtraTrees_100.pkl"
SCALER_PATH = "Scaler_100.pkl"

DATA_X_PATH = "TrainData_X.npy"
DATA_y_PATH = "TrainData_y.npy"

UPDATED_MODEL_PATH = "Model_ExtraTrees_updated.pkl"
UPDATED_SCALER_PATH = "Scaler_updated.pkl"

TARGET_COLUMN = "vinn"
FEATURE_COLUMNS = ["pd", "vinp", "vdd", "process", "temperature"]


def load_files(folder):
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            yield f, os.path.join(folder, f)

X_all = list(np.load(DATA_X_PATH, allow_pickle=True))
y_all = list(np.load(DATA_y_PATH, allow_pickle=True))

print(f"Loaded previous training chunks: {len(X_all)}")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

file_count = 0
start = time.time()

for fname, path in load_files(TRAIN45_PATH):

    print("Adding:", fname)

    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        continue

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_all.append(X)
    y_all.append(y)

    file_count += 1

print(f"Added {file_count} new files")

X_train = np.vstack(X_all)
y_train = np.concatenate(y_all)

print("Total samples after update:", len(y_train))

scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)

model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled, y_train)

pickle.dump(model, open(UPDATED_MODEL_PATH, "wb"))
pickle.dump(scaler, open(UPDATED_SCALER_PATH, "wb"))

np.save(DATA_X_PATH, np.array(X_all, dtype=object))
np.save(DATA_y_PATH, np.array(y_all, dtype=object))

print("\n✅ Model updated with 45 files")
print("Saved:", UPDATED_MODEL_PATH)
print("Time:", time.time() - start)
