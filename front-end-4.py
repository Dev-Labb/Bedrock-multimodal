import os
import io
import json
import math
import numpy as np
import requests
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta, timezone
import boto3

st.set_page_config(page_title="🧠 Multi-Modal Bedrock Test", page_icon="🧠")
st.title("🧠 Multi-Modal Bedrock Test")

# --------------------------------------------------------------------------------------
# DATA WINDOW LIMITS (hard constraint from your API's DB)
# --------------------------------------------------------------------------------------
DATA_MIN_DATE = date(2024, 9, 15)
DATA_MAX_DATE = date(2024, 11, 25)

def _parse_date(dstr):
    try:
        return pd.to_datetime(dstr).date()
    except Exception:
        return None

def _clamp_dates(start_str, end_str):
    """
    Clamp incoming dates to the allowed data range; if missing, default to full window.
    Returns (start_iso, end_iso, notes).
    """
    notes = []
    s = _parse_date(start_str) or DATA_MIN_DATE
    e = _parse_date(end_str) or DATA_MAX_DATE
    if s < DATA_MIN_DATE:
        notes.append(f"start clamped from {s.isoformat()} to {DATA_MIN_DATE.isoformat()}")
        s = DATA_MIN_DATE
    if e > DATA_MAX_DATE:
        notes.append(f"end clamped from {e.isoformat()} to {DATA_MAX_DATE.isoformat()}")
        e = DATA_MAX_DATE
    if s > e:
        # swap or collapse to edges if inverted
        notes.append(f"inverted range corrected: start={s.isoformat()} end={e.isoformat()}")
        s, e = min(s, e), max(s, e)
    return s.isoformat(), e.isoformat(), notes

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL    = "https://tisa6rznoj.execute-api.us-gov-west-1.amazonaws.com/dev/measurements"

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    # "Meta LLaMA3 2-1B Instruct": "meta.llama3-2-1b-instruct-v1:0"
}

VISION_CAPABLE = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": True,
    "amazon.nova-micro-v1:0": True,
    "meta.llama3-2-1b-instruct-v1:0": False,
}

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
model_id = MODELS[model_choice]

# --------------------------------------------------------------------------------------
# AWS client (secrets/env precedence)
# --------------------------------------------------------------------------------------
region = (st.secrets.get("aws", {}).get("region")
          if "aws" in st.secrets else os.getenv("AWS_REGION", "us-east-1"))
access_key = st.secrets.get("aws", {}).get("access_key_id") if "aws" in st.secrets else os.getenv("AWS_ACCESS_KEY_ID")
secret_key = st.secrets.get("aws", {}).get("secret_access_key") if "aws" in st.secrets else os.getenv("AWS_SECRET_ACCESS_KEY")
# session_token = st.secrets.get("aws", {}).get("session_token") if "aws" in st.secrets else os.getenv("AWS_SESSION_TOKEN")

if access_key and secret_key:
    brt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # aws_session_token=session_token
    )
else:
    brt = boto3.client("bedrock-runtime", region_name=region)

# --------------------------------------------------------------------------------------
# Session state stores
# --------------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # Converse-format (excluding injected SYSTEM_MSG)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # UI log

if "rf_store" not in st.session_state:
    st.session_state.rf_store = {}       # run-XXX -> {params, df, created, label}
if "rf_store_order" not in st.session_state:
    st.session_state.rf_store_order = []

if "baseline_store" not in st.session_state:
    st.session_state.baseline_store = {} # baseline-XXX -> {meta, frame, created, label}
if "baseline_order" not in st.session_state:
    st.session_state.baseline_order = []

def _mk_store_key(prefix: str, n_exist: int) -> str:
    return f"{prefix}-{n_exist+1:03d}"

# --------------------------------------------------------------------------------------
# Parser for common API Gateway response shapes
# --------------------------------------------------------------------------------------
def _parse_api_gateway_payload(resp) -> list[dict]:
    try:
        payload = resp.json()
    except Exception:
        try:
            payload = json.loads(resp.text or "")
        except Exception:
            return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        if "body" in payload:
            body = payload["body"]
            try:
                inner = json.loads(body) if isinstance(body, str) else body
            except Exception:
                inner = body
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict) and "results" in inner and isinstance(inner["results"], list):
                return inner["results"]
    return []

# --------------------------------------------------------------------------------------
# RF API caller + store (with date clamping)
# --------------------------------------------------------------------------------------
def call_rf_api(params: dict) -> dict:
    # Clamp dates to allowed window
    start_clamped, end_clamped, clamp_notes = _clamp_dates(params.get("start"), params.get("end"))
    limit = int(params.get("limit", 10))
    r = requests.get(
        RF_API_URL,
        params={"start": start_clamped, "end": end_clamped, "limit": limit},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    rows = _parse_api_gateway_payload(r)
    df = pd.DataFrame(rows)
    created = datetime.utcnow().isoformat() + "Z"
    key = _mk_store_key("run", len(st.session_state.rf_store))
    label = f"{key}: {start_clamped} → {end_clamped} (limit={limit})"
    st.session_state.rf_store[key] = {
        "params": {"start": start_clamped, "end": end_clamped, "limit": limit, "clamp_notes": clamp_notes},
        "df": df,
        "created": created,
        "label": label
    }
    st.session_state.rf_store_order.append(key)

    sample_cols = [
        c for c in [
            "measurement_timestamp","scc","site","carrier_frequency","frequency_band",
            "carrier_snr","relative_carrier_power","relative_noise_floor","bandwidth",
            "polarity","frequency_shift","signal_detection_status","id"
        ] if c in df.columns
    ]
    sample = df[sample_cols].head(12).to_dict(orient="records") if not df.empty else []

    return {
        "store_key": key,
        "label": label,
        "count": int(len(df)),
        "clamp_notes": clamp_notes,
        "sample": sample,
        "available_runs": [
            {"key": k, "label": st.session_state.rf_store[k]["label"], "count": int(len(st.session_state.rf_store[k]["df"]))}
            for k in st.session_state.rf_store_order
        ]
    }

# --------------------------------------------------------------------------------------
# Baseline + scoring (robust & JSON-safe)
# --------------------------------------------------------------------------------------
NUM_METRICS = [
    "carrier_snr", "relative_carrier_power", "relative_noise_floor",
    "carrier_frequency", "frequency_shift", "bandwidth"
]
CAT_METRICS = ["frequency_band", "polarity", "signal_detection_status"]

def _to_dt_utc(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def _hour_of_week(ts_col) -> pd.Series:
    ts = _to_dt_utc(ts_col)
    return ts.dt.dayofweek * 24 + ts.dt.hour  # 0..167

def _mad(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    med = float(np.median(s))
    return 1.4826 * float(np.median(np.abs(s - med)))

def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def _js_divergence(p: dict, q: dict) -> float:
    cats = set(p.keys()) | set(q.keys())
    if not cats:
        return 0.0
    eps = 1e-9
    pv = np.array([p.get(c, 0.0) + eps for c in cats], dtype=float); pv = pv / pv.sum()
    qv = np.array([q.get(c, 0.0) + eps for c in cats], dtype=float); qv = qv / qv.sum()
    m = 0.5 * (pv + qv)
    def _kl(a, b): return float(np.sum(a * np.log(a / b)))
    return float(0.5 * _kl(pv, m) + 0.5 * _kl(qv, m))

def _topk_dist(df, keys, col, k=5):
    counts = (df.groupby(keys + [col]).size()
                .groupby(keys)
                .apply(lambda s: (s / s.sum()).nlargest(k).to_dict())
                .reset_index(name=f"{col}_dist"))
    return counts

def build_baseline_df(df: pd.DataFrame,
                      group_keys=("scc", "site"),
                      ts_col="measurement_timestamp",
                      min_support=20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # ensure keys exist
    for c in group_keys:
        if c not in df.columns:
            df[c] = "__unknown__"
    if ts_col not in df.columns:
        df[ts_col] = pd.NaT

    d = df.copy()
    d["how"] = _hour_of_week(d[ts_col])
    keys = list(group_keys) + ["how"]

    rows = []
    for name, grp in d.groupby(keys):
        row = dict(zip(keys, name))
        row["n_train"] = int(len(grp))
        for m in NUM_METRICS:
            if m in grp.columns:
                s = pd.to_numeric(grp[m], errors="coerce").dropna()
                row[f"{m}_med"] = _safe_float(np.median(s)) if len(s) else None
                row[f"{m}_mad"] = _safe_float(_mad(s)) if len(s) else None
            else:
                row[f"{m}_med"] = None
                row[f"{m}_mad"] = None
        # per-hour counts baseline (median of count= len(grp) → just len(grp))
        row["count_med"] = _safe_float(len(grp))
        rows.append(row)
    base = pd.DataFrame(rows)

    for cat in CAT_METRICS:
        if cat in d.columns:
            dist = _topk_dist(d, keys, cat, k=5)
            base = base.merge(dist, on=keys, how="left")
        else:
            base[f"{cat}_dist"] = None

    base = base.loc[base["n_train"] >= int(min_support)].reset_index(drop=True)
    return base

def store_baseline(baseline_frame: pd.DataFrame, meta: dict) -> dict:
    key = _mk_store_key("baseline", len(st.session_state.baseline_store))
    created = datetime.utcnow().isoformat() + "Z"
    label = f"{key}: {meta.get('start','?')} → {meta.get('end','?')} [{meta.get('group_by','scc,site')}]"
    st.session_state.baseline_store[key] = {
        "meta": meta,
        "frame": baseline_frame,
        "created": created,
        "label": label
    }
    st.session_state.baseline_order.append(key)
    return {
        "baseline_key": key,
        "label": label,
        "rows": int(len(baseline_frame)),
        "available_baselines": [
            {"key": k, "label": st.session_state.baseline_store[k]["label"],
             "rows": int(len(st.session_state.baseline_store[k]["frame"]))}
            for k in st.session_state.baseline_order
        ]
    }

def _extract_segment(df, scc, site, start, end, ts_col="measurement_timestamp"):
    if df.empty:
        return df
    d = df.copy()
    if ts_col in d.columns:
        d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce", utc=True)
        try:
            s = pd.to_datetime(start, utc=True)
            e = pd.to_datetime(end, utc=True)
            d = d[(d[ts_col] >= s) & (d[ts_col] <= e)]
        except Exception:
            pass
    if scc is not None and "scc" in d.columns:
        d = d[d["scc"] == scc]
    if site is not None and "site" in d.columns:
        d = d[d["site"] == site]
    return d

def score_window_against_baseline(window_df: pd.DataFrame,
                                  baseline_df: pd.DataFrame,
                                  scc=None, site=None,
                                  ts_col="measurement_timestamp",
                                  persistence_k=3, persistence_m=6) -> dict:
    if window_df.empty or baseline_df.empty:
        return {"error": "Insufficient data to score window."}

    dfw = window_df.copy()
    dfw["how"] = _hour_of_week(dfw[ts_col])

    signals = []
    strong = []
    for m in NUM_METRICS:
        if m not in dfw.columns:
            continue
        s = pd.to_numeric(dfw[m], errors="coerce").dropna()
        if s.empty:
            continue
        cur_med = _safe_float(np.median(s))
        b = baseline_df[baseline_df["how"].isin(dfw["how"].dropna().unique())]
        b_med = _safe_float(np.nanmedian([x for x in b.get(f"{m}_med", pd.Series()).dropna().tolist() if x is not None])) if f"{m}_med" in b.columns else None
        b_mad = _safe_float(np.nanmedian([x for x in b.get(f"{m}_mad", pd.Series()).dropna().tolist() if x is not None])) if f"{m}_mad" in b.columns else None
        if cur_med is None or b_med is None or not b_mad or b_mad == 0:
            continue
        z = (cur_med - b_med) / (b_mad + 1e-6)
        signals.append({"metric": m, "z": _safe_float(z), "baseline_med": b_med, "current_med": cur_med})
        if abs(z) >= 3.0:
            strong.append({"metric": m, "z": _safe_float(z), "direction": "up" if z > 0 else "down",
                           "baseline_med": b_med, "current_med": cur_med})

    # Volume shift via count per hour
    volume = None
    if "measurement_timestamp" in dfw.columns:
        counts = dfw.groupby("how").size()
        cur_per_hour_med = _safe_float(np.median(counts.values)) if len(counts) else None
        b = baseline_df[baseline_df["how"].isin(counts.index.tolist())]
        base_counts_med = _safe_float(np.nanmedian([x for x in b.get("count_med", pd.Series()).dropna().tolist() if x is not None])) if "count_med" in b.columns else None
        base_counts_mad = _safe_float(_mad(pd.Series([x for x in b.get("count_med", pd.Series()).dropna().tolist() if x is not None]))) if "count_med" in b.columns else None
        if cur_per_hour_med is not None and base_counts_med is not None and base_counts_mad not in (None, 0.0):
            vol_z = (cur_per_hour_med - base_counts_med) / (base_counts_mad + 1e-6)
            volume = {"z": _safe_float(vol_z), "baseline_per_hour_med": base_counts_med, "current_per_hour_med": cur_per_hour_med}

    # Categorical divergence (JS)
    categorical = {}
    for cat in CAT_METRICS:
        if cat not in dfw.columns or f"{cat}_dist" not in baseline_df.columns:
            continue
        cur = (dfw.groupby(["how", cat]).size()
                 .groupby("how")
                 .apply(lambda s: (s / s.sum()).to_dict())
                 .to_dict())
        js_vals = []
        for _, brow in baseline_df.iterrows():
            how = int(brow["how"])
            bdist = brow.get(f"{cat}_dist") or {}
            cdist = cur.get(how) or {}
            js_vals.append(_js_divergence(bdist, cdist) if (bdist or cdist) else 0.0)
        if js_vals:
            categorical[cat] = {"js": _safe_float(float(np.median(js_vals)))}

    # Simple persistence on carrier_snr if available (k of last m hours exceed |z|>=2)
    persistence = None
    if "carrier_snr_med" in baseline_df.columns and "carrier_snr_mad" in baseline_df.columns and "measurement_timestamp" in dfw.columns and "carrier_snr" in dfw.columns:
        dfx = dfw.copy()
        dfx["measurement_timestamp"] = _to_dt_utc(dfx["measurement_timestamp"])
        dfx["hour_bin"] = dfx["measurement_timestamp"].dt.floor("H")
        hourly = dfx.groupby("hour_bin")["carrier_snr"].median().dropna()
        if not hourly.empty:
            bmed = float(np.nanmedian([x for x in baseline_df["carrier_snr_med"].dropna().tolist() if x is not None])) if "carrier_snr_med" in baseline_df.columns else 0.0
            bmad = float(np.nanmedian([x for x in baseline_df["carrier_snr_mad"].dropna().tolist() if x is not None])) if "carrier_snr_mad" in baseline_df.columns else 1e-6
            hz = ((hourly - bmed) / (bmad + 1e-6)).tail(persistence_m)
            elevated = int((hz.abs() >= 2.0).sum())
            persistence = {"k": elevated, "m": int(len(hz))}

    comps = []
    for s in signals:
        if s.get("z") is not None:
            comps.append(abs(s["z"]))
    if volume and volume.get("z") is not None:
        comps.append(abs(volume["z"]))
    for cat, v in (categorical or {}).items():
        if v and v.get("js") is not None:
            comps.append(min(5.0, 10.0 * float(v["js"])))  # scale JS into ~z range
    composite = max(comps) if comps else 0.0

    return {
        "entity": {"scc": scc, "site": site},
        "n_rows": int(len(dfw)),
        "strong_signals": strong[:5],
        "volume": volume,
        "categorical": (categorical or None),
        "persistence": persistence,
        "composite": _safe_float(composite)
    }

# --------------------------------------------------------------------------------------
# Tools (query + build_baseline + detect_pattern_change) with clamped dates
# --------------------------------------------------------------------------------------
def tool_query_rf_measurements(tool_input: dict) -> str:
    try:
        return json.dumps(call_rf_api(tool_input))
    except Exception as e:
        return json.dumps({"error": str(e)})

def tool_build_baseline(tool_input: dict) -> str:
    # Clamp requested training window
    start_c, end_c, notes = _clamp_dates(tool_input.get("start"), tool_input.get("end"))
    group_by = tool_input.get("group_by", "scc,site")
    min_support = int(tool_input.get("min_support", 20))
    limit = int(tool_input.get("limit", 500))
    # Pull data
    api_payload = call_rf_api({"start": start_c, "end": end_c, "limit": limit})
    run_key = api_payload.get("store_key")
    df = st.session_state.rf_store.get(run_key, {}).get("df", pd.DataFrame())
    groups = tuple([g.strip() for g in group_by.split(",") if g.strip()])
    base_df = build_baseline_df(df, group_keys=groups, ts_col="measurement_timestamp", min_support=min_support)
    stored = store_baseline(base_df, meta={"start": start_c, "end": end_c, "group_by": group_by, "min_support": min_support, "clamp_notes": notes})
    stored["clamp_notes"] = notes
    return json.dumps(stored)

def tool_detect_pattern_change(tool_input: dict) -> str:
    bkey = tool_input.get("baseline_key")
    if not bkey or bkey not in st.session_state.baseline_store:
        return json.dumps({"error": "Invalid or missing baseline_key."})
    cur_start_c, cur_end_c, notes = _clamp_dates(tool_input.get("current_start"), tool_input.get("current_end"))
    scc = tool_input.get("scc")
    site = tool_input.get("site")
    limit = int(tool_input.get("limit", 500))
    api_payload = call_rf_api({"start": cur_start_c, "end": cur_end_c, "limit": limit})
    run_key = api_payload.get("store_key")
    df = st.session_state.rf_store.get(run_key, {}).get("df", pd.DataFrame())
    if df.empty:
        return json.dumps({"error": "No data returned for current window.", "clamp_notes": notes})
    seg = _extract_segment(df, scc, site, cur_start_c, cur_end_c)
    base_df = st.session_state.baseline_store[bkey]["frame"]
    scored = score_window_against_baseline(seg, base_df, scc=scc, site=site)
    scored["clamp_notes"] = notes
    scored["window"] = {"start": cur_start_c, "end": cur_end_c}
    return json.dumps(scored)

# --------------------------------------------------------------------------------------
# Tool specs for Converse
# --------------------------------------------------------------------------------------
TOOLS = [
    {
        "toolSpec": {
            "name": "query_rf_measurements",
            "description": (
                "Query RF measurement data via API Gateway and store the result for later use. "
                "Dates are clamped to 2024-09-15..2024-11-25 if out of range. "
                "Returns: store_key, label, count, clamp_notes, sample, available_runs."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 10}
                    },
                    "required": ["start", "end"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "build_baseline",
            "description": (
                "Build a robust baseline of normal behavior by (scc, site, hour_of_week) over a historical period. "
                "Uses med/MAD for numeric metrics and top-k distributions for categoricals. "
                "Dates are clamped to 2024-09-15..2024-11-25. "
                "Returns: baseline_key, label, rows, available_baselines, clamp_notes."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "group_by": {"type": "string", "description": "Comma-separated columns, default 'scc,site'"},
                        "min_support": {"type": "integer", "minimum": 5, "default": 20},
                        "limit": {"type": "integer", "minimum": 50, "maximum": 500, "default": 500}
                    },
                    "required": ["start", "end"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "detect_pattern_change",
            "description": (
                "Compare a current time window against a stored baseline to flag significant deviations. "
                "Computes robust z-shifts, volume deltas, and categorical divergence with a simple persistence rule. "
                "Dates are clamped to 2024-09-15..2024-11-25. "
                "Returns: compact anomaly payload with composite score and clamp_notes."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "baseline_key": {"type": "string"},
                        "current_start": {"type": "string"},
                        "current_end": {"type": "string"},
                        "scc": {"type": "string"},
                        "site": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 50, "maximum": 500, "default": 500}
                    },
                    "required": ["baseline_key", "current_start", "current_end"]
                }
            }
        }
    }
]

# --------------------------------------------------------------------------------------
# System message: teach the model to build baseline first, then detect changes
# (Note the explicit date window so the model doesn't ask for out-of-range periods.)
# --------------------------------------------------------------------------------------
SYSTEM_MSG = (
    "You help RF data analysts assess pattern-of-life. "
    "ALWAYS respect the available data window: 2024-09-15 through 2024-11-25. "
    "To understand 'normal', first call build_baseline over a trailing period within that window "
    "(e.g., a few weeks). Then, to assess changes, call detect_pattern_change on a recent sub-window "
    "within the same bounds. Prioritize: measurement_timestamp, site, scc, carrier_frequency, frequency_band, "
    "carrier_snr, relative_carrier_power, relative_noise_floor, bandwidth, polarity, frequency_shift, "
    "signal_detection_status. Keep answers concise and reference the tool outputs. "
    "If the user asks general RF questions, answer directly."
)

# --------------------------------------------------------------------------------------
# Vision attachments for models that support images
# --------------------------------------------------------------------------------------
IMAGE_MIME_TO_FORMAT = {
    "image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg",
    "image/webp": "webp", "image/gif": "gif",
}

def build_user_content(text: str, files: list, model_id: str):
    content = []
    if text:
        content.append({"text": text})
    if not files:
        return content
    if not VISION_CAPABLE.get(model_id, False):
        st.info("Selected model is text-only; attached files will be ignored by the model.")
        return content
    for f in files:
        mime = getattr(f, "type", None) or ""
        if mime in IMAGE_MIME_TO_FORMAT:
            img_bytes = f.read()
            f.seek(0)
            content.append({"image": {"format": IMAGE_MIME_TO_FORMAT[mime], "source": {"bytes": img_bytes}}})
    return content

# --------------------------------------------------------------------------------------
# Converse (non-stream) with tools (kept simple, supports one tool-use round trip)
# --------------------------------------------------------------------------------------
def converse_with_tools(user_text: str, files=None, history=None):
    if history is None:
        history = []
    files = files or []

    messages = [{"role": "user", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)
    user_content = build_user_content(user_text, files, model_id)
    messages.append({"role": "user", "content": user_content})

    resp = brt.converse(
        modelId=model_id,
        toolConfig={"tools": TOOLS},
        messages=messages,
        inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 5000},
    )
    out_msg = resp.get("output", {}).get("message", {}) or {}
    out_content = out_msg.get("content", []) or []
    tool_uses = [c for c in out_content if "toolUse" in c]

    if tool_uses:
        tu = tool_uses[0]["toolUse"]
        tool_name = tu.get("name")
        tool_input = tu.get("input", {}) or {}
        # Dispatch
        if tool_name == "query_rf_measurements":
            tool_result_text = tool_query_rf_measurements(tool_input)
        elif tool_name == "build_baseline":
            tool_result_text = tool_build_baseline(tool_input)
        elif tool_name == "detect_pattern_change":
            tool_result_text = tool_detect_pattern_change(tool_input)
        else:
            tool_result_text = json.dumps({"error": f"Unknown tool {tool_name}"})

        # Append assistant toolUse + user toolResult
        messages.append({"role": "assistant", "content": out_content})
        messages.append({"role": "user", "content": [{
            "toolResult": {
                "toolUseId": tu["toolUseId"],
                "content": [{"text": tool_result_text}],
            }
        }]})

        resp2 = brt.converse(
            modelId=model_id,
            toolConfig={"tools": TOOLS},
            messages=messages,
            inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 5000},
        )
        final = resp2.get("output", {}).get("message", {}).get("content", []) or []
        final_text = "".join(c.get("text", "") for c in final if "text" in c)
        return final_text or "_No response_", messages

    # No tool-use path
    final_text = "".join(c.get("text", "") for c in out_content if "text" in c)
    return final_text or "_No response_", messages

# --------------------------------------------------------------------------------------
# Sidebar: show stored runs & baselines
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Stored RF runs")
    if st.session_state.rf_store_order:
        for key in st.session_state.rf_store_order[-15:][::-1]:
            meta = st.session_state.rf_store[key]
            st.caption(f"• {meta['label']}  (rows={len(meta['df'])})")
    else:
        st.caption("No stored runs yet.")

    st.subheader("Baselines")
    if st.session_state.baseline_order:
        for key in st.session_state.baseline_order[-15:][::-1]:
            meta = st.session_state.baseline_store[key]
            st.caption(f"• {meta['label']}  (rows={len(meta['frame'])})")
    else:
        st.caption("No baselines yet.")

# --------------------------------------------------------------------------------------
# Render prior chat turns
# --------------------------------------------------------------------------------------
for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# --------------------------------------------------------------------------------------
# Chat input (text + optional files). Defaults kept in bounds via SYSTEM_MSG/tools.
# --------------------------------------------------------------------------------------
prompt = st.chat_input(placeholder="Ask about RF PoL. Dates are limited to 2024-09-15..2024-11-25.", accept_file=True)

if prompt:
    text = getattr(prompt, "text", "") if prompt else ""
    files = prompt.get("files", []) if isinstance(prompt, dict) else []

    with st.chat_message("user"):
        if text:
            st.markdown(text)
        if files:
            for f in files:
                mime = getattr(f, "type", None) or ""
                name = getattr(f, "name", "uploaded_file")
                if mime.startswith("image/"):
                    st.image(f, caption=name)
                else:
                    st.write(f"📎 {name} ({mime or 'unknown type'})")

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, new_history = converse_with_tools(text, files=files, history=st.session_state.history)
                st.markdown(answer)
                st.session_state.history = new_history
                st.session_state.chat_log.append({"role": "user", "content": (text or "(file(s) only)")})
                st.session_state.chat_log.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Tools run failed: {e}")
