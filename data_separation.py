# =============================================================
# data_separation.py
# -------------------------------------------------------------
# Creates a 2-class label (Healthy / Diseased) using one of
# two strategies controlled by LABEL_MODE:
#
#   "pathogen"   -> Diseased = inoculated with a pathogen
#                   Healthy  = control plants
#                   (Independent of visual symptoms -> good for
#                    pre-symptomatic detection.)
#
#   "timeseries" -> For each plant (Entry), find the first date
#                   FinalScore crosses ONSET_THRESHOLD. Every
#                   row for that plant is labelled Diseased,
#                   including rows BEFORE symptoms appear.
#                   Plants that never cross the threshold are
#                   Healthy.
#
# In both modes, the model is trained ONLY on wavelength
# columns, so it must learn from reflectance alone.
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# -------------------------------------------------------------
# 0. SETTINGS
# -------------------------------------------------------------
CSV_PATH          = "Au_Fus_Strawberry_data_cleaned (1).csv"
RANDOM_STATE      = 42
N_SPLITS          = 5

LABEL_MODE        = "timeseries"   # "timeseries" or "pathogen"
ONSET_THRESHOLD   = 3.0            # FinalScore >= this = visible disease
HEALTHY_CONTROL   = "Control"      # only used if LABEL_MODE == "pathogen"

# -------------------------------------------------------------
# 1. LOAD
# -------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df)}   Columns: {len(df.columns)}")

# -------------------------------------------------------------
# 2. BUILD THE 2-CLASS LABEL
# -------------------------------------------------------------
if LABEL_MODE == "pathogen":
    if "Pathogen" not in df.columns:
        raise ValueError("LABEL_MODE='pathogen' but no 'Pathogen' column found.")

    df["label"] = np.where(
        df["Pathogen"].astype(str).str.strip().str.lower() == HEALTHY_CONTROL.lower(),
        "Healthy",
        "Diseased",
    )

elif LABEL_MODE == "timeseries":
    # First date each plant shows visible disease
    onset = (
        df.loc[df["FinalScore"] >= ONSET_THRESHOLD]
          .groupby("Entry")["Date"]
          .min()
    )
    df["onset"] = df["Entry"].map(onset)

    # If a plant ever gets diseased -> all its rows are Diseased
    df["label"] = np.where(df["onset"].isna(), "Healthy", "Diseased")
    df = df.drop(columns=["onset"])

else:
    raise ValueError(f"Unknown LABEL_MODE: {LABEL_MODE}")

print("\nClass counts in the FULL dataset:")
print(df["label"].value_counts().to_string())

# -------------------------------------------------------------
# 3. BUILD THE GROUP COLUMN
# -------------------------------------------------------------
df["group"] = df["Entry"].astype(str) + "_" + df["Date"].astype(str)
print(f"\nUnique groups (plant+date): {df['group'].nunique()}")

# -------------------------------------------------------------
# 4. SPLIT 60 / 20 / 20 (grouped + stratified)
# -------------------------------------------------------------
sgkf = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_id = np.full(len(df), -1, dtype=int)
for fold, (_, test_idx) in enumerate(sgkf.split(df, df["label"], df["group"])):
    fold_id[test_idx] = fold

train_idx = np.where(np.isin(fold_id, [0, 1, 2]))[0]
val_idx   = np.where(fold_id == 3)[0]
test_idx  = np.where(fold_id == 4)[0]

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# -------------------------------------------------------------
# 5. PRINT A SPLIT REPORT
# -------------------------------------------------------------
def summarize(name, d):
    total = len(d)
    counts = d["label"].value_counts()
    h = counts.get("Healthy", 0)
    ds = counts.get("Diseased", 0)
    print(f"{name:<10} {total:>5} rows   Healthy={h:<5} Diseased={ds:<5}")

print("\n--- Split summary ---")
summarize("Train", train_df)
summarize("Val",   val_df)
summarize("Test",  test_df)

# Check no group leaks across piles
tg = set(train_df["group"])
vg = set(val_df["group"])
sg = set(test_df["group"])
assert tg.isdisjoint(vg), "Group leak: train <-> val"
assert tg.isdisjoint(sg), "Group leak: train <-> test"
assert vg.isdisjoint(sg), "Group leak: val   <-> test"
print("\n[OK] No group appears in more than one pile.")

for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    missing = {"Healthy", "Diseased"} - set(d["label"].unique())
    if missing:
        print(f"[WARN] {name} is missing classes: {missing}")

# -------------------------------------------------------------
# 6. SAVE
# -------------------------------------------------------------
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv",     index=False)
test_df.to_csv("test.csv",   index=False)

print("\nSaved: train.csv, val.csv, test.csv")
print("Done.")