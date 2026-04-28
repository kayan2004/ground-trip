## ML Workflow

This backend now includes a complete travel-style classification workflow built from `data/travel_destinations_labeled.csv`.

### Labels

The six target classes are:

- `Adventure`
- `Relaxation`
- `Culture`
- `Budget`
- `Luxury`
- `Family`

The labeled CSV is treated as a curated assignment artifact. It preserves the original destination features and adds:

- `travel_style`
- `label_status`
- `label_notes`

### Features

Training excludes `destination`, `country`, and the label/audit columns. The model uses:

- Categorical: `region`, `budget_level`, `tourism_level`
- Binary numeric: `has_hiking`, `has_beach`
- Continuous numeric: `culture_score`, `luxury_score`, `family_friendly`, `nightlife_level`, `avg_temp_peak`

### Training

Run the notebook from `backend/notebook/ml.ipynb`. The notebook now contains the full flow:

- exploratory data analysis
- compares Logistic Regression, Random Forest, and SVC
- uses 5-fold stratified cross-validation
- tunes Random Forest with grid search
- selects the winner by macro F1
- saves the trained model with `joblib`

### Artifacts

Training outputs are written to `artifacts/ml/`:

- `results.csv`
- `classification_report.json`
- `model_reports.json`
- `model_metadata.json`
- `best_model.joblib`

### Inference

Use the self-contained `predict_travel_style()` helper inside the notebook with a single destination-shaped feature dictionary to get a predicted travel style and probabilities when supported by the model.
