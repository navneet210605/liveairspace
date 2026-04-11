# train_models.py

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import joblib

BASE_DIR = os.path.dirname(__file__)
PROC_CSV = os.path.join(BASE_DIR, "data", "processed", "features_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
if not os.path.exists(PROC_CSV):
    raise FileNotFoundError(f"Processed CSV not found at {PROC_CSV}. Run live_pipeline.py first.")

df = pd.read_csv(PROC_CSV)
df = df.dropna().reset_index(drop=True)

# -------------------------
# REGRESSION MODEL
# Predict "future_altitude" (next step geo_altitude)
# -------------------------
df["future_altitude"] = df["geo_altitude"].shift(-1)
df_reg = df.dropna(subset=["future_altitude"]).copy()

reg_features = ["velocity", "heading", "vertical_rate", "geo_altitude", "speed_diff", "alt_diff"]
X_reg = df_reg[reg_features]
y_reg = df_reg["future_altitude"]

poly = PolynomialFeatures(degree=2)
X_reg_poly = poly.fit_transform(X_reg)

reg_model = LinearRegression()
reg_model.fit(X_reg_poly, y_reg)

joblib.dump(reg_model, os.path.join(MODELS_DIR, "regression_model.pkl"))
joblib.dump(poly, os.path.join(MODELS_DIR, "poly_transform.pkl"))
print("✅ Saved regression_model.pkl and poly_transform.pkl")

# -------------------------
# CLASSIFICATION MODEL
# Synthetic threat_level based on simple rules
# -------------------------
df_cls = df.copy()

# Base rules for threat level (you can tune thresholds)
df_cls["threat_level"] = 0
df_cls.loc[(df_cls["velocity"] > 250) | (df_cls["alt_diff"].abs() > 200), "threat_level"] = 1
df_cls.loc[(df_cls["velocity"] > 300) | (df_cls["alt_diff"].abs() > 400), "threat_level"] = 2

clf_features = ["velocity", "heading", "vertical_rate", "geo_altitude", "speed_diff", "alt_diff", "movement"]
X_clf = df_cls[clf_features]
y_clf = df_cls["threat_level"]

clf_model = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_model.fit(X_clf, y_clf)

joblib.dump(clf_model, os.path.join(MODELS_DIR, "classification_model.pkl"))
print("✅ Saved classification_model.pkl")

# -------------------------
# ANOMALY DETECTION MODEL
# Isolation Forest
# -------------------------
anomaly_model = IsolationForest(contamination=0.01, random_state=42)
anomaly_model.fit(X_clf)

joblib.dump(anomaly_model, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
print("✅ Saved anomaly_model.pkl")

# -------------------------
# BEHAVIOURAL CLUSTERING (KMeans)
# -------------------------
cluster_features = ["velocity", "heading", "vertical_rate", "geo_altitude", "speed_diff", "alt_diff", "movement"]
X_cluster = df_cls[cluster_features]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_cluster)

joblib.dump(kmeans, os.path.join(MODELS_DIR, "behaviour_kmeans.pkl"))
print("✅ Saved behaviour_kmeans.pkl (KMeans clustering)")

print("🎉 All models trained and saved in /models")



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

metrics = {}

# ---- Regression Metrics ----
y_pred_reg = reg_model.predict(X_reg_poly)

MAE = mean_absolute_error(y_reg, y_pred_reg)
MSE = mean_squared_error(y_reg, y_pred_reg)
RMSE = MSE ** 0.5   # manually compute RMSE
R2 = r2_score(y_reg, y_pred_reg)

metrics["regression"] = {
    "MAE": float(MAE),
    "MSE": float(MSE),
    "RMSE": float(RMSE),
    "R2": float(R2)
}

# ---- Classification Metrics ----
y_pred_clf = clf_model.predict(X_clf)

metrics["classification"] = {
    "accuracy": float(accuracy_score(y_clf, y_pred_clf)),
    "precision": float(precision_score(y_clf, y_pred_clf, average="macro", zero_division=0)),
    "recall": float(recall_score(y_clf, y_pred_clf, average="macro", zero_division=0)),
    "f1": float(f1_score(y_clf, y_pred_clf, average="macro", zero_division=0)),
    "confusion_matrix": confusion_matrix(y_clf, y_pred_clf).tolist()
}

# ---- Anomaly Metrics ----
anom_pred = anomaly_model.predict(X_clf)
anom_pred_binary = (anom_pred == -1).astype(int)
anom_true_rate = anom_pred_binary.mean()

metrics["anomaly"] = {
    "anomaly_rate": float(anom_true_rate),
    "stability_score": float(1 - abs(0.01 - anom_true_rate))   # ideal contamination = 0.01
}

# Save metrics as JSON
METRIC_PATH = os.path.join(MODELS_DIR, "metrics.json")
with open(METRIC_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 Saved evaluation metrics → models/metrics.json")
