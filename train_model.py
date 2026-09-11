# =============================================================
# train_model.py
# -------------------------------------------------------------
# Trains a Random Forest using ONLY wavelength columns
# (columns whose header is a number, e.g. "350", "400.5").
# Every other column is ignored.
#
# Saves: model_rf.pkl, saved_scaler.pkl,
#        model_info.json, class_labels.json
# =============================================================

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

def wavelength_columns(df):
    """Return column names whose header parses as a number."""
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except (ValueError, TypeError):
            pass
    # sort numerically
    return sorted(cols, key=lambda x: float(x))

# -------------------------------------------------------------
# 1. LOAD SPLITS
# -------------------------------------------------------------
train = pd.read_csv("train.csv")
val   = pd.read_csv("val.csv")
test  = pd.read_csv("test.csv")

y_train = train["label"].values
y_val   = val["label"].values
y_test  = test["label"].values

# Keep ONLY wavelength columns
wavelengths = wavelength_columns(train)

X_train = train[wavelengths]
X_val   = val[wavelengths]
X_test  = test[wavelengths]

print(f"Wavelengths used: {len(wavelengths)} "
      f"({float(wavelengths[0])} to {float(wavelengths[-1])} nm)")
print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# -------------------------------------------------------------
# 2. SCALE
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# -------------------------------------------------------------
# 3. TRAIN
# -------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train_s, y_train)

val_pred  = model.predict(X_val_s)
test_pred = model.predict(X_test_s)

val_acc  = accuracy_score(y_val,  val_pred)
val_f1   = f1_score(y_val,  val_pred,  average="weighted")
test_acc = accuracy_score(y_test, test_pred)

print(f"\nVal accuracy : {val_acc:.4f}")
print(f"Val F1       : {val_f1:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# -------------------------------------------------------------
# 4. SAVE
# -------------------------------------------------------------
joblib.dump(model,  "model_rf.pkl")
joblib.dump(scaler, "saved_scaler.pkl")

info = {
    "winner": "Random Forest",
    "validation_accuracy": float(val_acc),
    "validation_f1":       float(val_f1),
    "test_accuracy":       float(test_acc),
    "class_labels":        list(model.classes_),
    "n_wavelengths":       len(wavelengths),
    "wavelength_range_nm": [float(wavelengths[0]), float(wavelengths[-1])],
}
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)

with open("class_labels.json", "w") as f:
    json.dump(list(model.classes_), f, indent=2)

print("\nSaved: model_rf.pkl, saved_scaler.pkl, "
      "model_info.json, class_labels.json")