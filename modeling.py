from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = "target"

NUMERIC_FEATURES = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "lead_days",
    "same_day_booking",
    "schedule_hour",
    "chronic_burden",
    "prior_appointments",
    "prior_no_shows",
    "prior_no_show_rate",
    "is_returning_patient",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Neighbourhood",
    "appointment_weekday",
    "appointment_month",
    "age_band",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "Age": "Age",
    "Scholarship": "Financial assistance",
    "Hipertension": "Hypertension",
    "Diabetes": "Diabetes",
    "Alcoholism": "Substance use history",
    "Handcap": "Disability support needs",
    "SMS_received": "SMS reminder received",
    "lead_days": "Lead time before visit",
    "same_day_booking": "Same-day booking",
    "schedule_hour": "Scheduling hour",
    "chronic_burden": "Chronic condition burden",
    "prior_appointments": "Prior appointments",
    "prior_no_shows": "Prior no-shows",
    "prior_no_show_rate": "Prior no-show rate",
    "is_returning_patient": "Returning patient",
    "Gender": "Gender",
    "Neighbourhood": "Neighborhood",
    "appointment_weekday": "Appointment weekday",
    "appointment_month": "Appointment month",
    "age_band": "Age band",
}

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

AGE_BAND_ORDER = ["0-17", "18-34", "35-49", "50-64", "65+"]


@dataclass(frozen=True)
class RiskRecommendation:
    tier: str
    label: str
    guidance: str
    action: str


RISK_RECOMMENDATIONS = [
    RiskRecommendation(
        tier="low",
        label="Low risk",
        guidance="Routine attendance pattern. Standard reminder workflow should be sufficient.",
        action="Send the normal reminder cadence and keep the slot unchanged.",
    ),
    RiskRecommendation(
        tier="moderate",
        label="Moderate risk",
        guidance="Some attendance friction is present. A lightweight confirmation touchpoint can help.",
        action="Trigger SMS confirmation and offer easy self-service rescheduling.",
    ),
    RiskRecommendation(
        tier="high",
        label="High risk",
        guidance="Material no-show risk. This patient would benefit from proactive outreach.",
        action="Add a call-back task, confirm transportation or access barriers, and flag for backfill planning.",
    ),
    RiskRecommendation(
        tier="critical",
        label="Critical risk",
        guidance="Very elevated no-show risk with meaningful operational impact if unattended.",
        action="Escalate to navigator outreach, confirm care barriers, and consider overbooking or waitlist protection.",
    ),
]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["ScheduledDay"] = pd.to_datetime(prepared["ScheduledDay"], utc=True)
    prepared["AppointmentDay"] = pd.to_datetime(prepared["AppointmentDay"], utc=True)

    prepared = prepared.loc[prepared["Age"] >= 0].copy()
    prepared = prepared.sort_values(
        ["PatientId", "ScheduledDay", "AppointmentID"]
    ).reset_index(drop=True)
    prepared[TARGET_COLUMN] = (prepared["No-show"] == "Yes").astype(int)

    lead_days = (
        prepared["AppointmentDay"] - prepared["ScheduledDay"]
    ).dt.total_seconds() / 86400

    prepared["lead_days"] = lead_days.clip(lower=0)
    prepared["same_day_booking"] = (prepared["lead_days"] < 1).astype(int)
    prepared["schedule_hour"] = prepared["ScheduledDay"].dt.hour
    prepared["appointment_weekday"] = prepared["AppointmentDay"].dt.day_name()
    prepared["appointment_month"] = prepared["AppointmentDay"].dt.strftime("%b")
    prepared["age_band"] = pd.cut(
        prepared["Age"],
        bins=[-1, 17, 34, 49, 64, 120],
        labels=AGE_BAND_ORDER,
    ).astype(str)
    prepared["chronic_burden"] = prepared[
        ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]
    ].sum(axis=1)
    patient_group = prepared.groupby("PatientId", sort=False)
    prepared["prior_appointments"] = patient_group.cumcount()
    prepared["prior_no_shows"] = (
        patient_group[TARGET_COLUMN].cumsum().shift(fill_value=0)
    )
    prepared["prior_no_show_rate"] = np.where(
        prepared["prior_appointments"] > 0,
        prepared["prior_no_shows"] / prepared["prior_appointments"],
        np.nan,
    )
    prepared["is_returning_patient"] = (
        prepared["prior_appointments"] > 0
    ).astype(int)

    prepared["appointment_weekday"] = pd.Categorical(
        prepared["appointment_weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    prepared["age_band"] = pd.Categorical(
        prepared["age_band"],
        categories=AGE_BAND_ORDER,
        ordered=True,
    )
    return prepared


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def aggregate_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = model_pipeline.named_steps["preprocessor"]
    model: LGBMClassifier = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    records = []
    for name, importance in zip(feature_names, importances):
        source_name = name.split("__", 1)[1]
        source_feature = resolve_source_feature(source_name)
        records.append(
            {
                "feature": FEATURE_LABELS.get(source_feature, source_feature),
                "importance": float(importance),
            }
        )

    importance_df = pd.DataFrame(records)
    importance_df = (
        importance_df.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    total = importance_df["importance"].sum()
    importance_df["importance_pct"] = np.where(
        total > 0,
        importance_df["importance"] / total,
        0,
    )
    return importance_df


def resolve_source_feature(source_name: str) -> str:
    for feature in sorted(MODEL_FEATURES, key=len, reverse=True):
        if source_name == feature or source_name.startswith(f"{feature}_"):
            return feature
    return source_name


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision_at_50": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_50": float(recall_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def build_artifact(data_path: str | Path = "patients.csv") -> dict[str, Any]:
    raw_df = pd.read_csv(data_path)
    prepared_df = prepare_dataset(raw_df)

    X = prepared_df[MODEL_FEATURES]
    y = prepared_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_probabilities)
    importance_df = aggregate_feature_importance(pipeline)

    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importance": importance_df,
        "feature_columns": MODEL_FEATURES,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "dataset_rows": int(len(prepared_df)),
        "positive_rate": float(prepared_df[TARGET_COLUMN].mean()),
    }
    return artifact


def save_artifact(
    artifact: dict[str, Any],
    artifact_path: str | Path = "artifacts/no_show_model.joblib",
) -> Path:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_or_train_artifact(
    data_path: str | Path = "patients.csv",
    artifact_path: str | Path = "artifacts/no_show_model.joblib",
) -> dict[str, Any]:
    path = Path(artifact_path)
    if path.exists():
        return joblib.load(path)

    artifact = build_artifact(data_path=data_path)
    save_artifact(artifact, artifact_path=artifact_path)
    return artifact


def build_prediction_frame(form_values: dict[str, Any]) -> pd.DataFrame:
    appointment_date = pd.Timestamp(form_values["appointment_date"])

    row = {
        "Age": int(form_values["Age"]),
        "Scholarship": int(form_values["Scholarship"]),
        "Hipertension": int(form_values["Hipertension"]),
        "Diabetes": int(form_values["Diabetes"]),
        "Alcoholism": int(form_values["Alcoholism"]),
        "Handcap": int(form_values["Handcap"]),
        "SMS_received": int(form_values["SMS_received"]),
        "lead_days": float(form_values["lead_days"]),
        "same_day_booking": int(float(form_values["lead_days"]) < 1),
        "schedule_hour": int(form_values["schedule_hour"]),
        "chronic_burden": int(
            form_values["Hipertension"]
            + form_values["Diabetes"]
            + form_values["Alcoholism"]
            + form_values["Handcap"]
        ),
        "prior_appointments": int(form_values["prior_appointments"]),
        "prior_no_shows": int(form_values["prior_no_shows"]),
        "prior_no_show_rate": (
            float(form_values["prior_no_shows"]) / float(form_values["prior_appointments"])
            if int(form_values["prior_appointments"]) > 0
            else np.nan
        ),
        "is_returning_patient": int(int(form_values["prior_appointments"]) > 0),
        "Gender": form_values["Gender"],
        "Neighbourhood": form_values["Neighbourhood"],
        "appointment_weekday": appointment_date.day_name(),
        "appointment_month": appointment_date.strftime("%b"),
        "age_band": pd.cut(
            [form_values["Age"]],
            bins=[-1, 17, 34, 49, 64, 120],
            labels=AGE_BAND_ORDER,
        )[0],
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def risk_recommendation(probability: float) -> RiskRecommendation:
    if probability < 0.15:
        return RISK_RECOMMENDATIONS[0]
    if probability < 0.30:
        return RISK_RECOMMENDATIONS[1]
    if probability < 0.50:
        return RISK_RECOMMENDATIONS[2]
    return RISK_RECOMMENDATIONS[3]


def case_drivers(
    form_values: dict[str, Any],
    prepared_df: pd.DataFrame,
) -> list[str]:
    overall_rate = prepared_df[TARGET_COLUMN].mean()
    drivers: list[str] = []

    if float(form_values["lead_days"]) >= prepared_df["lead_days"].quantile(0.75):
        drivers.append("Long booking lead time increases the chance of appointment drop-off.")

    if int(form_values["SMS_received"]) == 0:
        drivers.append("No reminder outreach is logged, which removes a common attendance safeguard.")

    if int(form_values["prior_appointments"]) > 0:
        no_show_rate = int(form_values["prior_no_shows"]) / max(
            int(form_values["prior_appointments"]), 1
        )
        if no_show_rate >= 0.5:
            drivers.append("The patient has missed a large share of prior appointments, which is a strong recurrence signal.")
        elif int(form_values["prior_no_shows"]) > 0:
            drivers.append("Prior missed visits increase the chance of another no-show without outreach.")

    if int(form_values["Scholarship"]) == 1:
        drivers.append("Financial-assistance patients can face higher transportation and access friction.")

    chronic_burden = (
        int(form_values["Hipertension"])
        + int(form_values["Diabetes"])
        + int(form_values["Alcoholism"])
        + int(form_values["Handcap"])
    )
    if chronic_burden >= 2:
        drivers.append("Multiple comorbidities can increase scheduling complexity and follow-through barriers.")

    weekday = pd.Timestamp(form_values["appointment_date"]).day_name()
    weekday_rate = (
        prepared_df.groupby("appointment_weekday", observed=False)[TARGET_COLUMN]
        .mean()
        .get(weekday)
    )
    if pd.notna(weekday_rate) and weekday_rate > overall_rate + 0.02:
        drivers.append(f"{weekday} appointments run above the clinic-wide no-show average in this dataset.")

    neighborhood_rate = (
        prepared_df.groupby("Neighbourhood")[TARGET_COLUMN].agg(["mean", "count"])
    )
    neighborhood = form_values["Neighbourhood"]
    if neighborhood in neighborhood_rate.index:
        row = neighborhood_rate.loc[neighborhood]
        if row["count"] >= 200 and row["mean"] > overall_rate + 0.03:
            drivers.append("The selected neighborhood trends above average for missed appointments in the historical data.")

    age = int(form_values["Age"])
    if 18 <= age <= 34:
        drivers.append("Younger adult patients are a higher-risk segment in this appointment history.")

    if not drivers:
        drivers.append("This profile stays close to baseline clinic behavior with no major risk amplifiers.")

    return drivers[:4]
