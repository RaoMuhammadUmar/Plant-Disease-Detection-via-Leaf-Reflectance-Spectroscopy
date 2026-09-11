# Strawberry Disease Detection from Leaf Reflectance

Detects whether a strawberry plant is diseased by analysing the reflectance
spectrum of its leaves — **before symptoms are visible to the human eye**.

Unlike visual inspection or RGB imaging, this project uses the full
reflectance spectrum (339–2515 nm, 2177 wavelength bands), including the
Near-Infrared (NIR) and Short-Wave Infrared (SWIR) regions that the human
eye cannot see. The model learns to recognise internal physiological changes
(water loss, cell breakdown, chlorophyll degradation) that precede visible
symptoms.

---

## Key Figures

### 1. Mean Reflectance by Class — "The Light Fingerprint"

The two curves diverge most strongly between 700–1500 nm, a region
invisible to the human eye. This confirms the model is using internal
physiological signals, not visible colour.

![Reflectance Spectra](docs/Reflectance%20spectra.png)

### 2. Confusion Matrix — Test Accuracy 82.5%

47 of 58 diseased leaves correctly flagged. 38 of 45 healthy leaves
correctly cleared. Most errors occur on borderline plants, not clear cases.

![Confusion Matrix](docs/Confusion%20matrix.png)

### 3. Top-20 Most Important Wavelengths

The most discriminative wavelengths sit in the red-edge (~700 nm) and NIR
(~750–800 nm) regions — classic indicators of plant stress.

![Feature Importance](docs/Feature%20importance.png)

### 4. PCA of Test Samples

The two classes overlap but show a clear trend — the model is picking up a
real signal, not noise.

![PCA Scatter](docs/PCA%20scatter.png)

---

## Data

- Reflectance spectra, 339–2515 nm, 2177 wavelength bands
- ~500 samples from multiple strawberry plants over a growing season
- Four replicates (A, B, C, D) per plant per date
- (Dataset not included in the repository for privacy reasons)

## Method

**Labels** — a plant is labelled:
- `Diseased` if its `FinalScore` ever reaches 3.0 or higher during the season
- `Healthy` otherwise

Because every row of that plant (including rows from **before** symptoms
appeared) is labelled `Diseased`, the model is forced to learn
pre-symptomatic spectral signatures, not visible symptoms.

**Features** — only reflectance wavelengths. No visual scores, no
physiological sensors (no NDVI, Fv/Fm, stomatal conductance, etc.). Every
non-wavelength column is dropped before training.

**Model** — Random Forest (300 trees) with StandardScaler.

**Splitting** — 60% train / 20% validation / 20% test using
`StratifiedGroupKFold`. All four replicates of the same plant on the same
day stay in the same split, so the model can never memorise a plant.

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | 0.71 |
| Validation F1 | 0.71 |
| Test accuracy | 0.83 |

## How to Run

```bash
pip install -r requirements.txt

python data_separation.py    # creates train.csv / val.csv / test.csv
python train_model.py        # trains RF, saves model_rf.pkl + saved_scaler.pkl
python generate_plots.py     # shows 4 diagnostic plots
python predict.py test.csv   # predicts on unseen leaves
