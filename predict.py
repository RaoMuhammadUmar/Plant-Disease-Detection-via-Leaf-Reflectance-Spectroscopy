# =============================================================
# predict.py
# -------------------------------------------------------------
# Loads model + scaler and predicts Healthy/Diseased for a
# new CSV. Uses ONLY wavelength columns.
#
# Usage:
#   python predict.py val.csv
#   python predict.py test.csv
# =============================================================

import sys
import warnings
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

def wavelength_columns(df):
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except (ValueError, TypeError):
            pass
    return sorted(cols, key=lambda x: float(x))

# -------------------------------------------------------------
# 1. LOAD MODEL + SCALER
# -------------------------------------------------------------
model  = joblib.load("model_rf.pkl")
scaler = joblib.load("saved_scaler.pkl")

train_cols = pd.read_csv("train.csv", nrows=0).columns.tolist()
feature_cols = wavelength_columns(pd.DataFrame(columns=train_cols))

# -------------------------------------------------------------
# 2. LOAD NEW DATA
# -------------------------------------------------------------
csv_path = sys.argv[1] if len(sys.argv) > 1 else "new_data.csv"
df = pd.read_csv(csv_path)

X = df.reindex(columns=feature_cols)
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)
classes = list(model.classes_)

# -------------------------------------------------------------
# 3. BUILD RESULTS TABLE
# -------------------------------------------------------------
results = pd.DataFrame({
    "Entry": df["Entry"] if "Entry" in df.columns else "",
    "Date":  df["Date"]  if "Date"  in df.columns else "",
    "Rep":   df["Rep"]   if "Rep"   in df.columns else "",
    "Verdict": preds,
    "Confidence": (probs.max(axis=1) * 100).round(1),
})

# -------------------------------------------------------------
# 4. PRINT ROW-BY-ROW PREDICTIONS
# -------------------------------------------------------------
print()
print("=" * 78)
print(f"  PREDICTIONS  —  {csv_path}  ({len(df)} samples)")
print("=" * 78)
print(f"  {'#':<4} {'Entry':<15} {'Date':<12} {'Rep':<5} {'Verdict':<10} {'Conf':>6}")
print("-" * 78)
for i, row in results.iterrows():
    print(f"  {i+1:<4} {str(row['Entry']):<15} {str(row['Date']):<12} "
          f"{str(row['Rep']):<5} {row['Verdict']:<10} {row['Confidence']:>5.1f}%")
print("=" * 78)

# -------------------------------------------------------------
# 5. PER-PLANT SUMMARY (across dates)
# -------------------------------------------------------------
if "Entry" in df.columns:
    print()
    print("=" * 78)
    print("  PER-PLANT TRACKING  (how each plant evolves over time)")
    print("=" * 78)
    for entry, group in results.groupby("Entry", sort=True):
        counts = group["Verdict"].value_counts()
        healthy_n  = counts.get("Healthy", 0)
        diseased_n = counts.get("Diseased", 0)
        total = len(group)

        # Overall verdict for this plant
        if diseased_n > healthy_n:
            plant_verdict = "DISEASED"
        elif healthy_n > diseased_n:
            plant_verdict = "HEALTHY"
        else:
            plant_verdict = "UNCERTAIN"

        avg_conf = group["Confidence"].mean()
        print(f"  {entry:<15}  ->  {plant_verdict:<10} "
              f"({diseased_n}D / {healthy_n}H of {total}, avg conf {avg_conf:.1f}%)")
    print("=" * 78)

# -------------------------------------------------------------
# 6. FINAL VERDICT (overall)
# -------------------------------------------------------------
total_n    = len(results)
healthy_n  = (results["Verdict"] == "Healthy").sum()
diseased_n = (results["Verdict"] == "Diseased").sum()

# Average confidence for each class
healthy_conf  = results.loc[results["Verdict"] == "Healthy", "Confidence"].mean()
diseased_conf = results.loc[results["Verdict"] == "Diseased", "Confidence"].mean()

print()
print("=" * 78)
print("  FINAL VERDICT")
print("=" * 78)
print(f"  Total samples       : {total_n}")
print(f"  Predicted Diseased  : {diseased_n}  (avg confidence {diseased_conf:.1f}%)")
print(f"  Predicted Healthy   : {healthy_n}  (avg confidence {healthy_conf:.1f}%)")
print()

high_conf = (results["Confidence"] >= 80).sum()
mid_conf  = ((results["Confidence"] >= 60) & (results["Confidence"] < 80)).sum()
low_conf  = (results["Confidence"] < 60).sum()
print(f"  High-confidence  (>=80%) : {high_conf}")
print(f"  Medium-confidence(60-80%): {mid_conf}")
print(f"  Low-confidence   (<60%)  : {low_conf}")
print()

if diseased_n > healthy_n:
    overall = "DISEASE PRESENT"
elif healthy_n > diseased_n:
    overall = "PLANTS HEALTHY"
else:
    overall = "INCONCLUSIVE"
print(f"  OVERALL : {overall}")
print("=" * 78)

# -------------------------------------------------------------
# 7. SAVE
# -------------------------------------------------------------
out_path = csv_path.replace(".csv", "_predictions.csv")
df_out = df.copy()
df_out["predicted_label"] = preds
df_out["confidence_pct"] = (probs.max(axis=1) * 100).round(1)
df_out.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")