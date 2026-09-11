# =============================================================
# generate_plots.py
# -------------------------------------------------------------
# Shows 4 plots one after another (close each to see next):
#   1. Mean reflectance spectra per class
#   2. Confusion matrix
#   3. Top-20 most important wavelengths
#   4. PCA 2D scatter
#
# Uses ONLY wavelength columns (numeric headers).
# =============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

COLORS = {
    "Diseased": "#d62728",
    "Healthy":  "#2ca02c",
}

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
# 1. LOAD
# -------------------------------------------------------------
print("Loading model + scaler + test data...")
model  = joblib.load("model_rf.pkl")
scaler = joblib.load("saved_scaler.pkl")

class_names = list(model.classes_)
print(f"  Model classes: {class_names}")

test = pd.read_csv("test.csv")
y_test = test["label"].values

wavelength_cols = wavelength_columns(test)
wavelengths = np.array([float(c) for c in wavelength_cols])

X_test = test[wavelength_cols]
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

acc = (y_pred == y_test).mean()
print(f"  Wavelength columns: {len(wavelength_cols)}")
print(f"  Range: {wavelengths.min()} nm to {wavelengths.max()} nm")
print(f"  Test rows    : {len(test)}")
print(f"  Test accuracy: {acc:.4f}")

# -------------------------------------------------------------
# PLOT 1 — Mean reflectance spectra per class
# -------------------------------------------------------------
print("\nPlot 1: reflectance spectra by class...")
plt.figure(figsize=(10, 5))
for cls in class_names:
    mask = y_test == cls
    Xc = test.loc[mask, wavelength_cols].values
    mean_spec = Xc.mean(axis=0)
    std_spec  = Xc.std(axis=0)
    plt.plot(wavelengths, mean_spec,
             label=cls, color=COLORS.get(cls, "gray"), linewidth=2)
    plt.fill_between(wavelengths,
                     mean_spec - std_spec,
                     mean_spec + std_spec,
                     alpha=0.2, color=COLORS.get(cls, "gray"))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (mean ± std)")
plt.title("Light Fingerprint: Mean Reflectance by Class")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# PLOT 2 — Confusion matrix
# -------------------------------------------------------------
print("\nPlot 2: confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=class_names)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — Random Forest\nTest accuracy = {acc:.2%}")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# PLOT 3 — Top-20 feature importance
# -------------------------------------------------------------
print("\nPlot 3: feature importance...")
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_names  = np.array(wavelength_cols)
    order       = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(9, 5))
    plt.barh(range(20), importances[order][::-1], color="#1f77b4")
    plt.yticks(range(20), feat_names[order][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Most Important Wavelengths (Random Forest)")
    plt.tight_layout()
    plt.show()
else:
    print("  (model has no feature_importances_ — skipping)")

# -------------------------------------------------------------
# PLOT 4 — PCA 2D scatter
# -------------------------------------------------------------
print("\nPlot 4: PCA scatter...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(7, 6))
for cls in class_names:
    mask = y_test == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=cls, alpha=0.7,
                color=COLORS.get(cls, "gray"),
                edgecolors="k", linewidths=0.4)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA of Test Samples (2D Projection)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nAll plots shown. Done.")