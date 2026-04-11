# live_pipeline.py

import requests
import pandas as pd
import time
import os
from datetime import datetime
import json

# -------------------------
# DIRECTORIES
# -------------------------
BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

API_URL = "https://opensky-network.org/api/states/all"

# -------------------------
# FETCH OPENSKY LIVE DATA
# -------------------------
def fetch_raw():
    try:
        r = requests.get(API_URL, timeout=8)
        r.raise_for_status()
        data = r.json()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(RAW_DIR, f"opensky_{ts}.json")

        with open(raw_path, "w") as f:
            json.dump(data, f)

        print(f"[RAW SAVED] {raw_path}")
        return data
    except Exception as e:
        print("Fetch error:", e)
        return None

# -------------------------
# PREPROCESS → CLEAN → FEATURE ENGINEER
# -------------------------
def preprocess(data):
    if data is None or "states" not in data or data["states"] is None:
        print("No states in data.")
        return None

    df = pd.DataFrame(data["states"])

    if df.empty:
        print("Empty states dataframe.")
        return None

    # OpenSky columns (indexes):
    # 0: icao24, 1: callsign, 2: origin_country,
    # 5: longitude, 6: latitude, 7: baro_altitude,
    # 9: velocity, 10: true_track, 11: vertical_rate,
    # 13: geo_altitude
    keep = [0, 1, 2, 5, 6, 7, 9, 10, 11, 13]
    df = df[keep]

    df.columns = [
        "icao24", "callsign", "origin_country",
        "longitude", "latitude",
        "baro_altitude", "velocity", "heading", "vertical_rate",
        "geo_altitude"
    ]

    # Basic cleaning
    numeric_cols = ["longitude", "latitude", "baro_altitude",
                    "velocity", "heading", "vertical_rate", "geo_altitude"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Feature engineering
    df = df.sort_values(by=["icao24"]).reset_index(drop=True)
    df["speed_diff"] = df["velocity"].diff().fillna(0)
    df["alt_diff"] = df["geo_altitude"].diff().fillna(0)
    df["movement"] = (df["velocity"] > 10).astype(int)

    # Drop rows with missing coordinates
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    out_path = os.path.join(PROC_DIR, "features_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"[PROCESSED UPDATED] {out_path} | rows: {len(df)}")
    return df


if __name__ == "__main__":
    print("🚀 LIVE PIPELINE STARTED (fetch + preprocess every 10 sec)")
    while True:
        raw = fetch_raw()
        preprocess(raw)
        time.sleep(10)
