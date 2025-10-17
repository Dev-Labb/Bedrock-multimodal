# app.py — RF Pattern‑of‑Life (PoL) Dashboard
# ----------------------------------------------------------------------------
# Quick start
#   1) pip install streamlit pandas numpy requests plotly
#   2) streamlit run app.py
#   3) Open http://localhost:8501
#
# What this app does
#   • Load RF measurement data from a CSV upload, an API (optional), or use
#     generated sample data.
#   • Build a baseline “Pattern of Life” by hour‑of‑day × frequency band.
#   • Visualize activity heatmaps and trends.
#   • Detect changes (anomalies) relative to the baseline using z‑scores.
#   • Export an anomalies report.
#
# Expected columns (your table can include more):
#   timestamp (datetime), frequency (Hz), signal_strength (dBm), modulation,
#   bandwidth (kHz or Hz), location, device_type, antenna_type, interference_type
# ----------------------------------------------------------------------------

import io
import json
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="RF Pattern‑of‑Life Dashboard", page_icon="📡", layout="wide")
st.title("📡 RF Pattern‑of‑Life (PoL) Dashboard")
st.caption("Analyze normal RF behavior and highlight changes over time.")

# ---------------------------- Utility functions -----------------------------

@st.cache_data(show_spinner=False)
def _infer_datetime(s):
    # Robust timestamp parser
    return pd.to_datetime(s, errors="coerce")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure minimal columns exist; create if missing
    defaults = {
        "frequency": np.nan,
        "signal_strength": np.nan,
        "modulation": None,
        "bandwidth": np.nan,
        "location": None,
        "device_type": None,
        "antenna_type": None,
        "interference_type": None,
    }
    if "timestamp" not in df.columns:
        raise ValueError("'timestamp' column is required.")
    df["timestamp"] = _infer_datetime(df["timestamp"])  
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df


def _bucket_freq_hz_to_band(freq_hz: float) -> str:
    if pd.isna(freq_hz):
        return "Unknown"
    f = float(freq_hz)
    # Simple buckets — tweak as needed for your environment
    if 2.3e9 <= f <= 2.5e9:
        return "2.4 GHz (Wi‑Fi/BT)"
    if 5.15e9 <= f <= 5.85e9:
        return "5 GHz (Wi‑Fi)"
    if 868e6 <= f <= 928e6:
        return "900 MHz (ISM)"
    if 400e6 <= f <= 470e6:
        return "UHF 400–470 MHz"
    if 700e6 <= f <= 900e6:
        return "700–900 MHz (Cell)"
    if 1.8e9 <= f <= 2.2e9:
        return "1.8–2.2 GHz (Cell)"
    if 3.3e9 <= f <= 4.2e9:
        return "3.3–4.2 GHz"
    if 24e9 <= f <= 30e9:
        return "27 GHz (mmWave)"
    return "Other"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df.copy())
    df = df.dropna(subset=["timestamp"])  # remove rows with bad timestamps
    df = df.sort_values("timestamp")

    # Derive features
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # Monday=0
    df["date"] = df["timestamp"].dt.date
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["year"] = df["timestamp"].dt.year
    df["freq_band"] = df["frequency"].apply(_bucket_freq_hz_to_band)
    df["count_one"] = 1  # helper for counts

    return df


def _make_sample(n_days: int = 10, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime.now() - timedelta(days=n_days)
    rows = []
    # Simulate a facility with daytime Wi‑Fi and sparse night activity.
    for d in range(n_days):
        for h in range(24):
            num = rng.integers(40, 120) if 8 <= h <= 18 else rng.integers(0, 8)
            for _ in range(int(num)):
                t = base + timedelta(days=d, hours=h, minutes=int(rng.integers(0, 60)))
                band_choice = rng.choice([
                    (2.45e9, "OFDM", -68, "Site‑A", "WiFi‑Router"),
                    (915e6, "FSK", -82, "Site‑A", "SensorNode"),
                    (5.3e9, "OFDM", -71, "Site‑A", "WiFi‑AP"),
                ], p=[0.6, 0.2, 0.2])
                f, mod, strength, loc, dev = band_choice
                # Add small noise
                strength = strength + rng.normal(0, 3)
                rows.append({
                    "timestamp": t,
                    "frequency": f,
                    "signal_strength": strength,
                    "modulation": mod,
                    "bandwidth": 20_000_000,
                    "location": loc,
                    "device_type": dev,
                    "antenna_type": "Omni",
                    "interference_type": None,
                })
    # Inject an anomaly: strong late‑night Wi‑Fi on last two nights
    for h in [0, 1, 2]:
        for _ in range(400):
            t = base + timedelta(days=n_days-1, hours=h, minutes=int(rng.integers(0, 60)))
            rows.append({
                "timestamp": t,
                "frequency": 2.45e9,
                "signal_strength": -60 + rng.normal(0, 2),
                "modulation": "OFDM",
                "bandwidth": 20_000_000,
                "location": "Site‑A",
                "device_type": "WiFi‑AP",
                "antenna_type": "Omni",
                "interference_type": None,
            })
    df = pd.DataFrame(rows)
    return df


def _build_baseline(df: pd.DataFrame, baseline_range: tuple[date, date], metric: str = "count"):
    start_d, end_d = baseline_range
    base = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
    if base.empty:
        return base, None

    if metric == "count":
        agg = base.groupby(["hour", "freq_band"], as_index=False)["count_one"].sum()
        agg = agg.rename(columns={"count_one": "value"})
    else:  # strength
        agg = base.groupby(["hour", "freq_band"], as_index=False)["signal_strength"].mean()
        agg = agg.rename(columns={"signal_strength": "value"})

    # Mean and std per (hour, band)
    stats = agg.groupby(["hour", "freq_band"]).agg(mean=("value", "mean"), std=("value", "std")).reset_index()
    return base, stats


def _score_period(df: pd.DataFrame, period_range: tuple[date, date], stats: pd.DataFrame, metric: str = "count"):
    start_d, end_d = period_range
    cur = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
    if cur.empty or stats is None or stats.empty:
        return cur, pd.DataFrame()

    if metric == "count":
        cur_agg = cur.groupby(["hour", "freq_band"], as_index=False)["count_one"].sum()
        cur_agg = cur_agg.rename(columns={"count_one": "value"})
    else:
        cur_agg = cur.groupby(["hour", "freq_band"], as_index=False)["signal_strength"].mean()
        cur_agg = cur_agg.rename(columns={"signal_strength": "value"})

    merged = pd.merge(cur_agg, stats, on=["hour", "freq_band"], how="left")
    merged["z"] = (merged["value"] - merged["mean"]) / merged["std"].replace(0, np.nan)

    # Rank anomalies by absolute z
    merged["abs_z"] = merged["z"].abs()
    merged = merged.sort_values("abs_z", ascending=False)
    return cur, merged


# ------------------------------- Data ingest -------------------------------
st.sidebar.header("Data")
source = st.sidebar.radio("Choose data source", ["Upload CSV", "Fetch from API", "Use sample data"], index=2)

df: pd.DataFrame | None = None

if source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload rf_measurements CSV", type=["csv"]) 
    if up:
        df = pd.read_csv(up)
elif source == "Fetch from API":
    st.sidebar.write("Provide an HTTP endpoint returning JSON array or CSV.")
    api_url = st.sidebar.text_input("API URL (GET)", placeholder="https://.../rf-data?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=1000")
    as_csv = st.sidebar.checkbox("Response is CSV", value=False)
    headers_txt = st.sidebar.text_area("Optional headers JSON", value="{}")
    if st.sidebar.button("Fetch", use_container_width=True) and api_url:
        try:
            headers = json.loads(headers_txt or "{}")
        except Exception:
            headers = {}
        r = requests.get(api_url, headers=headers, timeout=30)
        r.raise_for_status()
        if as_csv:
            df = pd.read_csv(io.StringIO(r.text))
        else:
            data = r.json()
            df = pd.DataFrame(data)
else:
    df = _make_sample(n_days=14)

if df is None or df.empty:
    st.info("Load data to begin. Use sample data if you just want to explore.")
    st.stop()

# Prepare
try:
    df = _prep(df)
except Exception as e:
    st.error(f"Data prep failed: {e}")
    st.stop()

# Optional filters
with st.sidebar.expander("Filters", expanded=False):
    locs = ["(all)"] + sorted([x for x in df["location"].dropna().unique()])
    devs = ["(all)"] + sorted([x for x in df["device_type"].dropna().unique()])
    mods = ["(all)"] + sorted([x for x in df["modulation"].dropna().unique()])
    sel_loc = st.selectbox("Location", options=locs)
    sel_dev = st.selectbox("Device type", options=devs)
    sel_mod = st.selectbox("Modulation", options=mods)

mask = pd.Series(True, index=df.index)
if sel_loc != "(all)":
    mask &= (df["location"] == sel_loc)
if sel_dev != "(all)":
    mask &= (df["device_type"] == sel_dev)
if sel_mod != "(all)":
    mask &= (df["modulation"] == sel_mod)

df_f = df[mask].copy()

# ------------------------------- Date ranges -------------------------------
min_date, max_date = df_f["date"].min(), df_f["date"].max()
colA, colB = st.columns(2)
with colA:
    st.subheader("Baseline window")
    base_range = st.date_input("Pick baseline date range", value=(max(min_date, max_date - timedelta(days=7)), max_date - timedelta(days=1)), min_value=min_date, max_value=max_date)
with colB:
    st.subheader("Current window")
    cur_range = st.date_input("Pick current date range", value=(max_date - timedelta(days=1), max_date), min_value=min_date, max_value=max_date)

metric = st.segmented_control("Metric", options=["count", "signal_strength"], default="count", help="Counts = activity volume; signal_strength = average dBm.")
thresh = st.slider("Anomaly threshold |z|", 1.5, 5.0, 3.0, 0.5)

# ------------------------------ Build baseline -----------------------------
base_df, stats = _build_baseline(df_f, base_range, metric=metric)
if stats is None or stats.empty:
    st.warning("No baseline stats — adjust the baseline range or filters.")
    st.stop()

cur_df, scored = _score_period(df_f, cur_range, stats, metric=metric)

# ------------------------------- KPI overview ------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Records (baseline)", f"{len(base_df):,}")
with col2:
    st.metric("Records (current)", f"{len(cur_df):,}")
with col3:
    st.metric("Freq bands", f"{df_f['freq_band'].nunique():,}")
with col4:
    st.metric("Max |z|", f"{np.nanmax(scored['abs_z']) if not scored.empty else 0:.2f}")

# -------------------------- Heatmaps: pattern vs delta ----------------------

# Baseline mean heatmap
heat_base = stats.pivot(index="freq_band", columns="hour", values="mean").fillna(0)
fig_base = px.imshow(
    heat_base,
    aspect="auto",
    labels=dict(x="Hour of day", y="Frequency band", color=f"Baseline mean ({'count' if metric=='count' else 'dBm'})"),
    title="Baseline Pattern of Life (hour × band)"
)
st.plotly_chart(fig_base, use_container_width=True)

# Current period value heatmap
if not cur_df.empty:
    if metric == "count":
        cur_agg = cur_df.groupby(["hour", "freq_band"], as_index=False)["count_one"].sum()
        cur_agg = cur_agg.rename(columns={"count_one": "value"})
    else:
        cur_agg = cur_df.groupby(["hour", "freq_band"], as_index=False)["signal_strength"].mean()
        cur_agg = cur_agg.rename(columns={"signal_strength": "value"})
    heat_cur = cur_agg.pivot(index="freq_band", columns="hour", values="value").fillna(0)
    fig_cur = px.imshow(
        heat_cur,
        aspect="auto",
        labels=dict(x="Hour of day", y="Frequency band", color=f"Current ({'count' if metric=='count' else 'dBm'})"),
        title="Current Period (hour × band)"
    )
    st.plotly_chart(fig_cur, use_container_width=True)

# Z‑score heatmap (delta)
if not scored.empty:
    z_mat = scored.pivot(index="freq_band", columns="hour", values="z")
    fig_z = px.imshow(
        z_mat,
        aspect="auto",
        color_continuous_midpoint=0,
        labels=dict(x="Hour of day", y="Frequency band", color="z‑score (Δ vs baseline)"),
        title="Change vs Baseline (z‑score) — red/blue = anomaly"
    )
    st.plotly_chart(fig_z, use_container_width=True)

# ----------------------------- Anomalies table ------------------------------

if not scored.empty:
    flagged = scored[np.abs(scored["z"]) >= thresh].copy()
    flagged = flagged.sort_values("abs_z", ascending=False)
    st.subheader("Anomalies (cells where current deviates from baseline)")
    st.dataframe(flagged[["hour", "freq_band", "value", "mean", "std", "z"]], use_container_width=True, hide_index=True)

    # Export
    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download anomalies CSV", data=csv, file_name="rf_anomalies.csv", mime="text/csv")
else:
    st.info("No anomalies computed — adjust your ranges or threshold.")

# ----------------------------- Time series views ----------------------------

st.subheader("Trends over time")
left, right = st.columns(2)

# Daily counts
with left:
    daily = df_f.groupby("date", as_index=False)["count_one"].sum()
    fig = px.line(daily, x="date", y="count_one", markers=True, title="Daily activity (records)")
    st.plotly_chart(fig, use_container_width=True)

# Hourly strength average
with right:
    hourly = df_f.groupby("hour", as_index=False)["signal_strength"].mean()
    fig = px.line(hourly, x="hour", y="signal_strength", markers=True, title="Average signal strength by hour (dBm)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------- Explanatory callouts ---------------------------

st.markdown(
    """
**How this works**  
• *Pattern of Life (PoL)* is the typical RF behavior by time‑of‑day and band.  
• We build a **baseline** using your chosen date range.  
• The **current** window is compared to the baseline using **z‑scores**:  
  \( z = (current - mean) / std \). Larger |z| means more unusual.

**What to look for**  
• Bright bands in the baseline heatmap = typical routine.  
• Strong red/blue in the z‑score heatmap = change vs. routine (potential anomaly).  
• Use filters to focus on a location, device type, or modulation.

**Tips**  
• Use the **count** metric to find unexpected activity volume (e.g., extra packets at night).  
• Use **signal_strength** to catch proximity/power changes (e.g., a transmitter moved closer).  
• Tune the threshold |z| to your noise level (2–3 is common; 4–5 is stricter).
"""
)
