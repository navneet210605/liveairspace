# streamlit_app.py

import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
from datetime import datetime
import numpy as np

# ----------------------------------
# CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Airspace Defence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡 Real-Time Airspace Defence Dashboard")

BASE_DIR = os.path.dirname(__file__)
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "features_dataset.csv")
EXPORT_CSV = os.path.join(BASE_DIR, "data", "processed", "powerbi_export.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------------
# REFRESH BUTTON
# ----------------------------------
if st.sidebar.button("🔄 Run"):
    st.rerun()



# ----------------------------------
# LOAD EVALUATION METRICS
# ----------------------------------

metrics_path = os.path.join(MODELS_DIR, "metrics.json")

if os.path.exists(metrics_path):
    import json
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
else:
    metrics = None

# ----------------------------------
# LOAD DATA
# ----------------------------------
if not os.path.exists(PROCESSED_CSV):
    st.error("❌ Processed dataset not found. Run live_pipeline.py first.")
    st.stop()

df = pd.read_csv(PROCESSED_CSV)

# Ensure columns exist
expected_cols = [
    "icao24","callsign","origin_country",
    "longitude","latitude","baro_altitude",
    "velocity","heading","vertical_rate",
    "geo_altitude","speed_diff","alt_diff","movement"
]
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0

# ----------------------------------
# LOAD MODELS
# ----------------------------------
try:
    reg_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))
    poly = joblib.load(os.path.join(MODELS_DIR, "poly_transform.pkl"))
    clf_model = joblib.load(os.path.join(MODELS_DIR, "classification_model.pkl"))
    anomaly_model = joblib.load(os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "behaviour_kmeans.pkl"))
except Exception as e:
    st.error(f"❌ Failed to load ML models: {e}")
    st.stop()

# ----------------------------------
# APPLY ML PREDICTIONS
# ----------------------------------
reg_features = ["velocity","heading","vertical_rate","geo_altitude","speed_diff","alt_diff"]
clf_features = ["velocity","heading","vertical_rate","geo_altitude","speed_diff","alt_diff","movement"]
cluster_features = clf_features  # same set

# Clean NaN/inf
for cols in [reg_features, clf_features]:
    df[cols] = df[cols].replace([None, np.inf, -np.inf], 0).fillna(0)

# Regression
X_reg_poly = poly.transform(df[reg_features])
df["pred_altitude"] = reg_model.predict(X_reg_poly)

# Classification
df["threat_level"] = clf_model.predict(df[clf_features])

# Anomaly (IsolationForest returns -1 (anomaly) and 1 (normal))
anomaly_raw = anomaly_model.predict(df[clf_features])
df["anomaly"] = (anomaly_raw == -1).astype(int)  # 1 = anomaly, 0 = normal

# Clustering (behavioural patterns)
df["cluster"] = kmeans.predict(df[cluster_features])

# ----------------------------------
# RISK SCORE & LOITERING LOGIC
# ----------------------------------

# Normalize some features
vel_norm = df["velocity"].clip(0, 350) / 350.0
alt_jump_norm = df["alt_diff"].abs().clip(0, 500) / 500.0

# Map clusters to a base risk weight (you can tune)
cluster_risk_map = {
    0: 0.2,
    1: 0.4,
    2: 0.7,
    3: 0.9
}
df["cluster_risk"] = df["cluster"].map(cluster_risk_map).fillna(0.3)

# Risk score formula (0–100)
df["risk_score"] = (
    20 * vel_norm +
    25 * alt_jump_norm +
    df["threat_level"] * 20 +
    df["anomaly"] * 25 +
    10 * df["cluster_risk"]
).clip(0, 100)

# Simple loitering heuristic (slow speed, small altitude change)
df["loitering"] = (
    (df["velocity"] < 80) &
    (df["movement"] == 1) &
    (df["alt_diff"].abs() < 50)
).astype(int)

# Timestamp for Power BI / monitoring
df["timestamp"] = datetime.now()

# ----------------------------------
# SIDEBAR FILTERS
# ----------------------------------
st.sidebar.header("🔍 Filters")

country_filter = st.sidebar.multiselect(
    "Origin Country",
    options=sorted(df["origin_country"].dropna().unique()),
    default=sorted(df["origin_country"].dropna().unique())
)

threat_filter = st.sidebar.multiselect(
    "Threat Level",
    options=sorted(df["threat_level"].unique()),
    default=sorted(df["threat_level"].unique())
)

risk_min, risk_max = st.sidebar.slider(
    "Risk Score Range",
    min_value=0, max_value=100, value=(0, 100)
)

flight_search = st.sidebar.text_input("Search Flight (Callsign / ICAO24)", "")

score_mode = st.sidebar.radio("Map Color Mode", ["Threat Level", "Anomaly", "Risk Score"])

filtered_df = df[
    (df["origin_country"].isin(country_filter)) &
    (df["threat_level"].isin(threat_filter)) &
    (df["risk_score"].between(risk_min, risk_max))
].copy()

if flight_search:
    mask = (
        filtered_df["callsign"].fillna("").str.contains(flight_search, case=False) |
        filtered_df["icao24"].fillna("").str.contains(flight_search, case=False)
    )
    filtered_df = filtered_df[mask]



st.header("📊 Machine Learning Model Evaluation")

if metrics:

    # --- Regression ---
    st.subheader("📈 Regression Model Metrics (Altitude Prediction)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['regression']['MAE']:.2f}")
    col2.metric("MSE", f"{metrics['regression']['MSE']:.2f}")
    col3.metric("RMSE", f"{metrics['regression']['RMSE']:.2f}")
    col4.metric("R² Score", f"{metrics['regression']['R2']:.3f}")

    # --- Classification ---
    st.subheader("🛡 Classification Metrics (Threat Level Model)")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Accuracy", f"{metrics['classification']['accuracy']:.2f}")
    col6.metric("Precision", f"{metrics['classification']['precision']:.2f}")
    col7.metric("Recall", f"{metrics['classification']['recall']:.2f}")
    col8.metric("F1 Score", f"{metrics['classification']['f1']:.2f}")

    # Confusion matrix
    st.write("### Confusion Matrix")
    st.table(pd.DataFrame(
        metrics["classification"]["confusion_matrix"],
        columns=["Pred 0","Pred 1","Pred 2"],
        index=["True 0","True 1","True 2"]
    ))

    # --- Anomaly detection ---
    st.subheader("🔎 Anomaly Detection Metrics")
    col9, col10 = st.columns(2)
    col9.metric("Anomaly Rate", f"{metrics['anomaly']['anomaly_rate']:.3f}")
    col10.metric("Model Stability", f"{metrics['anomaly']['stability_score']:.3f}")

else:
    st.info("⚠ No evaluation metrics found. Run train_models.py again.")

# ----------------------------------
# MAP
# ----------------------------------
st.subheader("🌍 Live Airspace Map")

if not filtered_df.empty:
    if score_mode == "Threat Level":
        color_col = "threat_level"
        color_continuous_scale = px.colors.sequential.Viridis
    elif score_mode == "Anomaly":
        color_col = "anomaly"
        color_continuous_scale = px.colors.sequential.Reds
    else:  # Risk Score
        color_col = "risk_score"
        color_continuous_scale = px.colors.sequential.Plasma

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude", lon="longitude",
        color=color_col, size="movement",
        hover_data=[
            "icao24","callsign","origin_country",
            "velocity","heading","geo_altitude",
            "pred_altitude","threat_level","anomaly",
            "cluster","risk_score","loitering"
        ],
        zoom=3,
        mapbox_style="carto-positron",
        color_continuous_scale=color_continuous_scale
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No flights match current filters.")

# ----------------------------------
# THREAT & RISK DISTRIBUTION
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Threat Level Distribution")
    if not filtered_df.empty:
        fig_threat = px.histogram(
            filtered_df,
            x="threat_level",
            color="threat_level",
            title="Flights by Threat Level"
        )
        st.plotly_chart(fig_threat, use_container_width=True)
    else:
        st.info("No data for threat distribution.")

with col2:
    st.subheader("🔥 Risk Score Distribution")
    if not filtered_df.empty:
        fig_risk = px.histogram(
            filtered_df,
            x="risk_score",
            nbins=20,
            title="Risk Score Histogram"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No data for risk distribution.")

# ----------------------------------
# ANOMALY & LOITERING
# ----------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔎 Anomaly vs Normal Flights")
    if not filtered_df.empty:
        fig_anomaly = px.pie(
            filtered_df,
            names="anomaly",
            title="Anomaly (1) vs Normal (0)"
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    else:
        st.info("No data for anomaly chart.")

with col4:
    st.subheader("♻ Loitering Detection")
    if not filtered_df.empty:
        fig_loiter = px.pie(
            filtered_df,
            names="loitering",
            title="Loitering (1) vs Non-Loitering (0)"
        )
        st.plotly_chart(fig_loiter, use_container_width=True)
    else:
        st.info("No data for loitering chart.")

# ----------------------------------
# PREDICTED vs ACTUAL ALTITUDE
# ----------------------------------
st.subheader("📈 Predicted vs Actual Altitude")

if not filtered_df.empty:
    fig_alt = px.scatter(
        filtered_df,
        x="geo_altitude",
        y="pred_altitude",
        color="risk_score",
        labels={"geo_altitude": "Actual Altitude", "pred_altitude": "Predicted Altitude"},
        title="Predicted vs Actual Altitude (Colored by Risk Score)",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig_alt, use_container_width=True)
else:
    st.info("No data available for altitude comparison.")

# ----------------------------------
# TABLE
# ----------------------------------
st.subheader(f"📁 Flight Data ({len(filtered_df)} records)")
st.dataframe(
    filtered_df[
        [
            "icao24","callsign","origin_country",
            "velocity","heading","geo_altitude",
            "pred_altitude","threat_level","anomaly",
            "cluster","risk_score","loitering","timestamp"
        ]
    ]
)

# ----------------------------------
# EXPORT FOR POWER BI
# ----------------------------------
st.subheader("📤 Export for Power BI")
# Add evaluation metrics into export for Power BI
if metrics:
    for key, sub in metrics.items():
        for mkey, value in sub.items():
            if isinstance(value, (int, float, str)):
                df[f"{key}_{mkey}"] = value

export_df = filtered_df.copy()
export_df.to_csv(EXPORT_CSV, index=False)

export_csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered CSV for Power BI",
    data=export_csv,
    file_name="airspace_powerbi_export.csv",
    mime="text/csv"
)

# ----------------------------------
# ALERTS
# ----------------------------------
high_risk = filtered_df[filtered_df["risk_score"] >= 70]
if not high_risk.empty:
    st.error(f"🚨 HIGH RISK ALERT: {len(high_risk)} flights with risk_score ≥ 70")

high_threat = filtered_df[filtered_df["threat_level"] == 2]
if not high_threat.empty:
    st.warning(f"⚠ High-Threat Aircraft Detected: {len(high_threat)} flights")

loiter_susp = filtered_df[filtered_df["loitering"] == 1]
if not loiter_susp.empty:
    st.info(f"♻ Loitering/Suspicious Hold: {len(loiter_susp)} flights")

# ----------------------------------
# TIMESTAMP
# ----------------------------------
st.sidebar.markdown(
    f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)




import streamlit as st

st.set_page_config(layout="wide")

st.title("✈️ Airspace Monitoring Dashboard")

powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiYjVkOTYwNzgtNzJiMC00ZjFkLThjZTItN2Y2ZTIzYzhiNWM3IiwidCI6ImUxNGU3M2ViLTUyNTEtNDM4OC04ZDY3LThmOWYyZTJkNWE0NiIsImMiOjEwfQ%3D%3D"

st.components.v1.iframe(
    src=powerbi_url,
    width=1300,
    height=650,
    scrolling=True
)

