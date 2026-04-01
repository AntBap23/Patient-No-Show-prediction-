from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modeling import (
    AGE_BAND_ORDER,
    MODEL_FEATURES,
    WEEKDAY_ORDER,
    build_prediction_frame,
    case_drivers,
    load_or_train_artifact,
    prepare_dataset,
    risk_recommendation,
)

st.set_page_config(
    page_title="Clinic No-Show Command Center",
    page_icon="🏥",
    layout="wide",
)

DATA_PATH = Path("patients.csv")
ARTIFACT_PATH = Path("artifacts/no_show_model.joblib")


@st.cache_data(show_spinner=False)
def load_prepared_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    return prepare_dataset(raw)


@st.cache_resource(show_spinner="Training clinic no-show model...")
def load_artifact() -> dict:
    return load_or_train_artifact(DATA_PATH, ARTIFACT_PATH)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def metric_card(title: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_overview_tab(prepared_df: pd.DataFrame, artifact: dict) -> None:
    overall_rate = prepared_df["target"].mean()
    same_day_rate = prepared_df["same_day_booking"].mean()
    sms_coverage = prepared_df["SMS_received"].mean()
    median_lead = prepared_df["lead_days"].median()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Appointments", f"{len(prepared_df):,}", "Historical visits used for model training.")
    with c2:
        metric_card("No-show rate", format_pct(overall_rate), "Baseline missed-visit rate across the clinic.")
    with c3:
        metric_card("SMS coverage", format_pct(sms_coverage), "Share of appointments with reminder outreach logged.")
    with c4:
        metric_card("Median lead time", f"{median_lead:.1f} days", "Typical time between scheduling and appointment date.")

    st.markdown("### Operational Signals")
    left, right = st.columns((1.15, 1))

    weekday_risk = (
        prepared_df.groupby("appointment_weekday", observed=False)
        .agg(appointments=("target", "size"), no_show_rate=("target", "mean"))
        .reset_index()
        .sort_values("appointment_weekday")
    )
    weekday_risk["appointment_weekday"] = pd.Categorical(
        weekday_risk["appointment_weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    weekday_risk = weekday_risk.sort_values("appointment_weekday")

    age_band_risk = (
        prepared_df.groupby("age_band", observed=False)
        .agg(appointments=("target", "size"), no_show_rate=("target", "mean"))
        .reset_index()
    )
    age_band_risk["age_band"] = pd.Categorical(
        age_band_risk["age_band"],
        categories=AGE_BAND_ORDER,
        ordered=True,
    )
    age_band_risk = age_band_risk.sort_values("age_band")

    with left:
        fig = px.bar(
            weekday_risk,
            x="appointment_weekday",
            y="no_show_rate",
            color="appointments",
            color_continuous_scale="Tealgrn",
            labels={
                "appointment_weekday": "Appointment weekday",
                "no_show_rate": "No-show rate",
                "appointments": "Volume",
            },
            title="No-Show Rate by Appointment Weekday",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            age_band_risk,
            x="age_band",
            y="no_show_rate",
            markers=True,
            labels={"age_band": "Age band", "no_show_rate": "No-show rate"},
            title="Risk by Age Band",
        )
        fig.update_traces(line_color="#0f766e", marker_color="#0f766e")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### High-Risk Segments")
    neighborhoods = (
        prepared_df.groupby("Neighbourhood")
        .agg(appointments=("target", "size"), no_show_rate=("target", "mean"))
        .query("appointments >= 500")
        .sort_values(["no_show_rate", "appointments"], ascending=[False, False])
        .head(10)
        .reset_index()
    )
    neighborhoods["no_show_rate"] = neighborhoods["no_show_rate"].map(lambda value: f"{value:.1%}")
    st.dataframe(
        neighborhoods.rename(
            columns={
                "Neighbourhood": "Neighborhood",
                "appointments": "Appointments",
                "no_show_rate": "No-show rate",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Clinical interpretation: the model is best used to prioritize outreach and scheduling interventions, not to deny access or make treatment decisions."
    )


def build_prediction_tab(prepared_df: pd.DataFrame, artifact: dict) -> None:
    pipeline = artifact["pipeline"]
    overall_rate = prepared_df["target"].mean()
    neighborhoods = sorted(prepared_df["Neighbourhood"].unique().tolist())

    st.markdown("### Patient Risk Triage")
    st.write(
        "Enter a patient profile to estimate no-show risk and suggest an operational response for access teams, front-desk staff, or outreach coordinators."
    )

    with st.form("risk_form"):
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 0, 100, 38)
            gender = st.selectbox("Gender", ["F", "M"], format_func=lambda g: "Female" if g == "F" else "Male")
            neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)
            lead_days = st.slider("Lead time before appointment (days)", 0, 180, 14)
            appointment_date = st.date_input(
                "Appointment date",
                value=date.today() + timedelta(days=lead_days or 1),
                min_value=date.today(),
            )
            schedule_hour = st.slider("Scheduling hour", 6, 20, 10)
            scholarship = st.checkbox("Patient uses scholarship / financial support")
            sms_received = st.checkbox("Reminder SMS planned or received", value=True)
        with right:
            hipertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            alcoholism = st.checkbox("Alcohol use concern")
            handcap = st.checkbox("Disability support need")
            prior_appointments = st.number_input(
                "Prior appointments on record",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )
            prior_no_shows = st.number_input(
                "Prior missed appointments",
                min_value=0,
                max_value=int(prior_appointments),
                value=0,
                step=1,
            )

            st.markdown("#### Care-team framing")
            st.caption("Historical attendance matters. For new patients, leave prior appointments and prior missed appointments at zero.")
            submit = st.form_submit_button("Score patient risk", use_container_width=True)

    if not submit:
        return

    form_values = {
        "Age": age,
        "Gender": gender,
        "Neighbourhood": neighborhood,
        "lead_days": lead_days,
        "appointment_date": appointment_date,
        "schedule_hour": schedule_hour,
        "Scholarship": int(scholarship),
        "SMS_received": int(sms_received),
        "Hipertension": int(hipertension),
        "Diabetes": int(diabetes),
        "Alcoholism": int(alcoholism),
        "Handcap": int(handcap),
        "prior_appointments": int(prior_appointments),
        "prior_no_shows": int(prior_no_shows),
    }

    prediction_frame = build_prediction_frame(form_values)
    probability = float(pipeline.predict_proba(prediction_frame[MODEL_FEATURES])[:, 1][0])
    recommendation = risk_recommendation(probability)
    drivers = case_drivers(form_values, prepared_df)

    st.markdown("### Triage Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Predicted no-show risk", format_pct(probability), "Estimated probability that the patient misses the appointment.")
    with c2:
        uplift = probability - overall_rate
        metric_card("Above baseline", f"{uplift:+.1%}", "Difference versus the historical clinic average.")
    with c3:
        metric_card("Risk tier", recommendation.label, "Operational band used to trigger outreach intensity.")

    st.markdown(
        f"""
        <div class="callout callout-{recommendation.tier}">
            <div class="callout-title">{recommendation.label}</div>
            <div>{recommendation.guidance}</div>
            <div class="callout-action"><strong>Recommended action:</strong> {recommendation.action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.2, 1))
    with left:
        st.markdown("#### Likely Drivers")
        for driver in drivers:
            st.write(f"- {driver}")

    with right:
        risk_bar = pd.DataFrame(
            {
                "Category": ["Clinic average", "Patient risk"],
                "Rate": [overall_rate, probability],
            }
        )
        fig = px.bar(
            risk_bar,
            x="Category",
            y="Rate",
            color="Category",
            color_discrete_sequence=["#94a3b8", "#0f766e"],
            title="Patient Risk vs Clinic Baseline",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)


def build_model_tab(prepared_df: pd.DataFrame, artifact: dict) -> None:
    metrics = artifact["metrics"]
    importance_df = artifact["feature_importance"].head(10).copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}", "Ability to rank higher-risk appointments above lower-risk ones.")
    with c2:
        metric_card("Average precision", f"{metrics['average_precision']:.3f}", "Precision-recall performance for the minority no-show class.")
    with c3:
        metric_card("Brier score", f"{metrics['brier_score']:.3f}", "Probability calibration error. Lower is better.")

    st.markdown("### What the Model Learns")
    left, right = st.columns((1.1, 1))

    with left:
        fig = px.bar(
            importance_df.sort_values("importance_pct"),
            x="importance_pct",
            y="feature",
            orientation="h",
            labels={"importance_pct": "Share of model importance", "feature": "Feature"},
            title="Top Drivers in the Gradient Boosting Model",
            color="importance_pct",
            color_continuous_scale="Tealgrn",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Validation snapshot")
        st.write(
            f"- Trained on `{artifact['dataset_rows']:,}` appointment records after cleaning invalid ages and clipping negative lead times."
        )
        st.write(
            f"- Historical no-show prevalence in training data: `{format_pct(artifact['positive_rate'])}`."
        )
        st.write(
            f"- Default threshold precision: `{metrics['precision_at_50']:.3f}` and recall: `{metrics['recall_at_50']:.3f}`."
        )
        st.write(
            "- Intended use: outreach prioritization, reminder workflows, backfill planning, and operational forecasting."
        )
        st.write(
            "- Guardrail: use predictions alongside staff judgment and local access-equity policies."
        )

    confusion = pd.DataFrame(
        {
            "Outcome": ["True negatives", "False positives", "False negatives", "True positives"],
            "Count": [
                metrics["true_negatives"],
                metrics["false_positives"],
                metrics["false_negatives"],
                metrics["true_positives"],
            ],
        }
    )
    st.dataframe(confusion, use_container_width=True, hide_index=True)


def main() -> None:
    prepared_df = load_prepared_data()
    artifact = load_artifact()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Healthcare Operations Analytics</div>
            <h1>Clinic No-Show Command Center</h1>
            <p>
                A healthcare-oriented Streamlit experience for understanding missed appointments,
                prioritizing patient outreach, and grounding scheduling decisions in a stronger gradient boosting model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, prediction_tab, model_tab = st.tabs(
        ["Operations Overview", "Patient Risk Triage", "Model Quality"]
    )

    with overview_tab:
        build_overview_tab(prepared_df, artifact)
    with prediction_tab:
        build_prediction_tab(prepared_df, artifact)
    with model_tab:
        build_model_tab(prepared_df, artifact)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 28%),
                linear-gradient(180deg, #f4fbfa 0%, #edf7f6 42%, #f8fafc 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .stApp,
        .stApp div,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp li,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #0f172a;
        }
        .hero {
            background: linear-gradient(135deg, #ccfbf1 0%, #bae6fd 100%);
            border-radius: 24px;
            padding: 2rem 2.25rem;
            color: #0f172a;
            margin-bottom: 1.5rem;
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(14, 116, 144, 0.14);
        }
        .hero h1 {
            margin: 0.3rem 0 0.6rem;
            font-size: 2.4rem;
            color: #0f172a;
        }
        .hero p {
            margin: 0;
            max-width: 760px;
            color: #334155;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            color: #0f766e;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            min-height: 135px;
        }
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0.35rem 0;
        }
        .metric-help {
            color: #64748b;
            font-size: 0.9rem;
        }
        .callout {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 1rem 0 1.25rem;
            border: 1px solid transparent;
        }
        .callout-title {
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .callout-action {
            margin-top: 0.5rem;
        }
        .callout-low {
            background: #ecfdf5;
            border-color: #86efac;
            color: #14532d;
        }
        .callout-moderate {
            background: #eff6ff;
            border-color: #93c5fd;
            color: #1e3a8a;
        }
        .callout-high {
            background: #fff7ed;
            border-color: #fdba74;
            color: #9a3412;
        }
        .callout-critical {
            background: #fef2f2;
            border-color: #fca5a5;
            color: #991b1b;
        }
        button[kind="primary"],
        button[kind="secondary"],
        .stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            color: #0f172a !important;
            background: #dbeafe !important;
            border: 1px solid rgba(59, 130, 246, 0.18) !important;
        }
        [data-testid="stTabs"] button {
            color: #0f172a !important;
            background: rgba(255, 255, 255, 0.72) !important;
            border-radius: 12px 12px 0 0 !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f172a !important;
            background: #ccfbf1 !important;
        }
        [data-testid="stTabs"] button p,
        [data-testid="stTabs"] button div,
        [data-testid="stTabs"] button span {
            color: #0f172a !important;
        }
        .stSelectbox label,
        .stSlider label,
        .stDateInput label,
        .stCheckbox label,
        .stRadio label,
        .stNumberInput label,
        .stTextInput label,
        .stMarkdown,
        .stCaption,
        .stInfo,
        .stSuccess,
        .stWarning,
        .stError {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.1rem 1rem 0.3rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

main()
