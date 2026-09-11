cat > README.md << 'EOF'
# Strawberry Disease Detection from Leaf Reflectance

Detects whether a strawberry plant is diseased by analysing leaf
reflectance spectra — before symptoms are visible to the human eye.

## Data
- Reflectance spectra, 339–2515 nm, 2177 wavelength bands
- ~500 samples from multiple plants over a growing season
- (Dataset not included for privacy reasons)

## Method
- Labels: a plant is "Diseased" if it ever reaches `FinalScore >= 3.0`
  during the season; otherwise "Healthy". Pre-symptomatic spectra are
  therefore labelled Diseased.
- Features: **only** reflectance wavelengths. No visual or
  physiological features are used.
- Model: Random Forest (300 trees) + StandardScaler

## Results
- Validation accuracy: 0.71
- Test accuracy: 0.83

## How to run
```bash
pip install -r requirements.txt
python data_separation.py    # creates train/val/test splits
python train_model.py        # trains model, saves pkl + json
python generate_plots.py     # shows 4 diagnostic plots
python predict.py test.csv   # predicts on unseen leaves