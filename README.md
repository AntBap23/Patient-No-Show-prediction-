# Patient No-Show Prediction

A healthcare-oriented machine learning and Streamlit project for identifying likely appointment no-shows, prioritizing outreach, and helping clinic teams protect schedule utilization.

## What Changed

This version is rebuilt around a reproducible training pipeline instead of missing pickle files.

- `modeling.py` now cleans the raw appointment data, engineers operational features, trains a LightGBM classifier, evaluates performance, and saves a reusable artifact.
- `app.py` is now a clinic-style command center with:
  - operations KPIs
  - high-risk segment views
  - patient-level risk triage
  - intervention guidance
  - model quality reporting

## Modeling Approach

The upgraded model uses healthcare operations signals that are common in access and scheduling workflows:

- age and age band
- appointment lead time
- same-day booking indicator
- scheduling hour
- neighborhood
- reminder outreach (`SMS_received`)
- social support proxy (`Scholarship`)
- chronic condition burden from hypertension, diabetes, alcoholism, and disability flags
- prior appointment count and prior missed-visit history for returning patients
- appointment weekday and month

Cleaning rules:

- remove invalid negative ages
- convert date columns to timestamps
- clip negative lead times to zero so scheduling artifacts do not create impossible waiting times

Model choice:

- gradient boosting with LightGBM
- class balancing enabled for the minority no-show class
- train/test split with stratification
- evaluation includes ROC-AUC, average precision, Brier score, precision, and recall

## Run The App

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

On first launch, the app will train the model from `patients.csv` and save an artifact to `artifacts/no_show_model.joblib`.

## Project Structure

```text
.
├── app.py
├── modeling.py
├── patients.csv
├── requirements.txt
└── artifacts/
    └── no_show_model.joblib
```

## Product Framing

This project is intended for operational decision support, including:

- reminder prioritization
- call-list generation
- rescheduling outreach
- waitlist and backfill planning
- schedule risk monitoring

It should not be used to restrict care access or replace staff judgment.
