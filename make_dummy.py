# =============================================================
# make_dummy.py
# -------------------------------------------------------------
# Creates dummy_test.csv with the same wavelength columns as
# train.csv. Only numeric column headers are used.
# =============================================================

import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# Get exact columns from train.csv
train_cols = pd.read_csv("train.csv", nrows=0).columns.tolist()

# Keep ONLY numeric column headers (the wavelengths)
wavelengths = []
for c in train_cols:
    try:
        float(c)
        wavelengths.append(c)
    except ValueError:
        pass

wl = np.array([float(c) for c in wavelengths])
print(f"Using {len(wavelengths)} wavelength columns "
      f"({wl.min()} to {wl.max()} nm)")

# Base shape: rough leaf spectrum
def base_spectrum():
    s = 10 + 45 * ((wl > 700) & (wl < 1350))
    s += 30 * ((wl >= 1450) & (wl < 1900))
    s += 15 * ((wl >= 1950) & (wl < 2400))
    return s

rows = []

# 3 healthy rows
for i in range(3):
    spec = base_spectrum() + rng.normal(0, 1.5, len(wl)) + 3
    row = {c: v for c, v in zip(wavelengths, spec)}
    row.update({"Entry": f"H{i+1}", "Date": "2024-03-01", "Rep": "A"})
    rows.append(row)

# 3 diseased rows
for i in range(3):
    spec = base_spectrum() + rng.normal(0, 1.5, len(wl)) - 3
    row = {c: v for c, v in zip(wavelengths, spec)}
    row.update({"Entry": f"D{i+1}", "Date": "2024-03-01", "Rep": "A"})
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("dummy_test.csv", index=False)
print(f"Saved dummy_test.csv with {len(df)} rows, {len(df.columns)} columns")