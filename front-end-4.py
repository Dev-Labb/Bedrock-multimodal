import os
import io
import json
import requests
import streamlit as st
import pandas as pd
from datetime import date, datetime
import boto3
from botocore.exceptions import NoCredentialsError, NoRegionError, ClientError

st.set_page_config(page_title="🧠 Multi-Modal Bedrock Test", page_icon="🧠")
st.title("🧠 Multi-Modal Bedrock Test")

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway.Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these will these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://tisa6rznoj.execute-api.us-gov-west-1.amazonaws.com/dev/generate/v2"
RF_API_URL    = "https://tisa6rznoj.execute-api.us-gov-west-1.amazonaws.com/dev/measurements"

# --------------------------------------------------------------------------------------------------------------------------------
# WARNING!!!!: I AM NOW USING CONVERSE API AND WILL NEED TO CONVERT TO CONVERSE STREAM DOWN THE LINE! 
# Model options - These must match keys in AWS Lambda's "ALLOWED_MODELS" varible. For more context please refer
# to the coinciding lamba function. The llama3 model is currently broken because of the format I used to invoke
# the model currently. My plan currrently is to change to models/formats that use AWS Converse API More info can
# be found here: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html  
# ---------------------------------------------------------------------------------------------------------------------------------

MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    # "Meta LLaMA3 2-1B Instruct": "meta.llama3-2-1b-instruct-v1:0"
}

# Which of the above are multimodal meaning they can use proper API. Meta Llamaa doesn't support what we're going with for now.
# Commented Llama out, but keeping for potential down the line testing though...

VISION_CAPABLE = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": True,
    "amazon.nova-micro-v1:0": True,
    "meta.llama3-2-1b-instruct-v1:0": False,
}

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
model_id = MODELS[model_choice]

# This is for storing my secrets in order to use AWS resources securely. 
# No idea yet how they will do it on unclass and how PM will prefer in other location yet.
PROFILE = os.getenv("AWS_PROFILE")
REGION  = os.getenv("AWS_REGION")
try:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    brt = session.client("bedrock-runtime")
except (NoCredentialsError, NoRegionError):
    session = boto3.Session(profile_name=PROFILE, region_name=REGION or "us-east-1")
    brt = session.client("bedrock-runtime")

# ----------------------- NEW: in-memory store so the model can reference prior pulls --------------------------
if "rf_store" not in st.session_state:
    # key -> {"params": {...}, "df": DataFrame, "created": iso, "label": str}
    st.session_state.rf_store = {}
if "rf_store_order" not in st.session_state:
    st.session_state.rf_store_order = []  # preserve order of creation

def _mk_store_key() -> str:
    n = len(st.session_state.rf_store) + 1
    return f"run-{n:03d}"

def _summarize_df(df: pd.DataFrame) -> dict:
    """
    Tiny summary that is cheap in tokens but useful for the model when referencing past pulls.
    Prioritize metrics in SYSTEM_MSG: timestamps, site, carrier_frequency, frequency_band, carrier_snr,
    relative_carrier_power, relative_noise_floor, pointing angles, bandwidth, polarity, frequency_shift,
    signal_detection_status.
    """
    if df.empty:
        return {"count": 0}

    # Convert timestamps if present (some APIs return strings)
    for col in ["measurement_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    summary = {"count": len(df), "columns": list(df.columns)}
    # Numeric columns of interest (only those present)
    numeric_cols = [
        "carrier_frequency", "carrier_snr", "relative_carrier_power", "relative_noise_floor",
        "bandwidth", "frequency_shift"
    ]
    present_num = [c for c in numeric_cols if c in df.columns]
    if present_num:
        desc = df[present_num].describe().to_dict()
        # keep only mean/min/max (reduce tokens)
        compact = {}
        for c, stats in desc.items():
            compact[c] = {k: float(v) for k, v in stats.items() if k in ("mean", "min", "max")}
        summary["numeric_summary"] = compact

    # Categorical quick counts
    for cat in ["site", "frequency_band", "polarity", "signal_detection_status"]:
        if cat in df.columns:
            vc = df[cat].astype(str).value_counts().head(5).to_dict()
            summary[f"top_{cat}"] = vc

    # Time span
    if "measurement_timestamp" in df.columns:
        ts = df["measurement_timestamp"].dropna()
        if not ts.empty:
            summary["time_range"] = {
                "min": ts.min().isoformat(),
                "max": ts.max().isoformat()
            }

    return summary

def _store_results(params: dict, results: list[dict]) -> dict:
    """
    Store the dataset in session and return store_key + light summary + a small sample.
    """
    df = pd.DataFrame(results)
    created = datetime.utcnow().isoformat() + "Z"
    key = _mk_store_key()
    label = f"{key}: {params.get('start','?')} → {params.get('end','?')} (limit={params.get('limit','')})"
    st.session_state.rf_store[key] = {
        "params": {k: v for k, v in params.items()},
        "df": df,
        "created": created,
        "label": label
    }
    st.session_state.rf_store_order.append(key)

    # prepare a cheap sample for the model (avoid blasting tokens)
    sample_cols = [
        c for c in [
            "measurement_timestamp", "site", "carrier_frequency", "frequency_band",
            "carrier_snr", "relative_carrier_power", "relative_noise_floor", "bandwidth",
            "polarity", "frequency_shift", "signal_detection_status", "id"
        ] if c in df.columns
    ]
    sample = df[sample_cols].head(12).to_dict(orient="records") if sample_cols else df.head(10).to_dict(orient="records")

    payload = {
        "store_key": key,
        "label": label,
        "params": params,
        "count": len(df),
        "summary": _summarize_df(df),
        "sample": sample,
        "available_runs": [
            {"key": k, "label": st.session_state.rf_store[k]["label"], "count": len(st.session_state.rf_store[k]["df"])}
            for k in st.session_state.rf_store_order
        ]
    }
    return payload

def _compare_runs(current_key: str, reference_key: str) -> dict:
    """
    Compare two stored runs by key: numeric deltas and (if applicable) per-day count diffs.
    Keep it compact for LLM consumption.
    """
    store = st.session_state.rf_store
    if current_key not in store or reference_key not in store:
        return {"error": "One or both run keys not found in store."}

    d1 = store[current_key]["df"].copy()
    d2 = store[reference_key]["df"].copy()

    # Ensure timestamps parsed (if present)
    for df in (d1, d2):
        if "measurement_timestamp" in df.columns:
            df["measurement_timestamp"] = pd.to_datetime(df["measurement_timestamp"], errors="coerce")

    res = {
        "current": {"key": current_key, "label": store[current_key]["label"], "count": len(d1)},
        "reference": {"key": reference_key, "label": store[reference_key]["label"], "count": len(d2)},
        "deltas": {},
        "per_day_counts_delta": None
    }

    # Numeric columns to compare (present in both)
    numeric_cols = [
        "carrier_frequency", "carrier_snr", "relative_carrier_power", "relative_noise_floor",
        "bandwidth", "frequency_shift"
    ]
    shared = [c for c in numeric_cols if c in d1.columns and c in d2.columns]
    def _mm(df, col):
        s = df[col].dropna()
        if s.empty:
            return None
        return {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())}

    for c in shared:
        a = _mm(d1, c); b = _mm(d2, c)
        if a and b:
            res["deltas"][c] = {
                "current": a, "reference": b,
                "delta_mean": (a["mean"] - b["mean"])
            }

    # Per-day counts if timestamp present
    if "measurement_timestamp" in d1.columns and "measurement_timestamp" in d2.columns:
        d1["day"] = d1["measurement_timestamp"].dt.date
        d2["day"] = d2["measurement_timestamp"].dt.date
        c1 = d1.groupby("day").size().rename("current")
        c2 = d2.groupby("day").size().rename("reference")
        joined = pd.concat([c1, c2], axis=1).fillna(0).astype(int)
        joined["delta"] = joined["current"] - joined["reference"]
        # keep short
        res["per_day_counts_delta"] = joined.reset_index().rename(columns={"index": "day"}) \
                                           .sort_values("day").tail(14).to_dict(orient="records")

    return res

# --------------------------------- Tools specs setup-----------------------------------------------------------------------------------------
# This is for allowing model to be able to use tools like external/internal API's. One thing we will need to do is make sure
# we copy the OpenAPI Json schema and plug it in as a tool. This will enable model to use API when it needs to based on intstructions
# you give it in System_MSG aka instructions to model system. Make sure this is sound or else model will return tons of errors... trust me....
# --------------------------------------------------------------------------------------------------------------------------------------------

TOOLS = [
    {
        "toolSpec": {
            "name": "query_rf_measurements",
            "description": (
                "Query RF measurement data via API Gateway and STORE the result for later comparison. "
                "Parameters: start (YYYY-MM-DD), end (YYYY-MM-DD), limit (integer). "
                "Returns: store_key, label, count, params, summary, sample, and a list of available_runs you can reference later."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end": {"type": "string", "description": "End date YYYY-MM-DD"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 10}
                    },
                    "required": ["start", "end"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "compare_rf_runs",
            "description": (
                "Compare two previously stored RF runs by store_key. "
                "Input: current_key, reference_key. "
                "Returns numeric deltas (mean/min/max) for key metrics and optional per-day count deltas."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "current_key": {"type": "string", "description": "Key of the current run (e.g., run-003)"},
                        "reference_key": {"type": "string", "description": "Key of the reference/baseline run (e.g., run-001)"}
                    },
                    "required": ["current_key", "reference_key"]
                }
            }
        }
    }
]

SYSTEM_MSG = (    
    """You help RF data analysts. The analysts are primarily concerned with pattern of life changes for
     each scc in a database. To answer questions about pattern of life you should first call query_rf_measurements with start, end, and optional limit.
     You should prioritize these metrics for pattern of life determinations: measurement_timestamp
     site, carrier_frequency, frequency_band, carrier_snr, relative_carrier_power, relative_noise_floor, pointing_information_azimuth,
     pointing_information_elevation, bandwidth, polarity, frequency_shift, and signal_detection_status. 
     The scc will be found using query_rf_measurements tool. When the user asks about RF measurements, call query_rf_measurements with start, end, and optional limit. 
     Analysts tend to use several of the metrics from the query_rf_mesurements tool to determine pattern of life for the scc.
     They are also concerned with changes from the normal pattern of life. You will also use the results from query_rf_measurements to determine those changes, but
     will look into: frequency_shifts and signal_subcarrier_snr on top of the other metrics mentioned to help determine why a pattern of life change may have occurred.
     If the user asks to compare with a prior pull, or mentions “previous”, use compare_rf_runs with the returned store_key values from earlier runs.
     Keep answers concise; include a short table only when necessary. Assume today's date is October 24 2025.
     Otherwise, if the user asks questions that aren't RF related answer them to the best of your ability."""
)

# Calls my RF API, but can replace with whatever backend API. Important to note params as these are the minimal required to call API
def call_rf_api(params: dict) -> dict:
    r = requests.get(
        RF_API_URL,
        params={
            "start": params.get("start"),
            "end": params.get("end"),
            "limit": params.get("limit", 10),
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # This took some debugging. Lots I did with previous stages, but basically, results will come in a nested dictinary and so you need to
    # unpack the json response (list containing dictionary values) and get only the contents of the body so model can read it. 
    # Use print if you have errors with response or I may just add error (try/catch) correction later if API changes again from Converse/Converse stream 
    # to correct again. It is very useful to see raw responses for debugging purposes.
    # --------------------------------------------------------------------------------------------------------------------------------------------------
    
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    elif isinstance(payload, dict) and "body" in payload:
        try:
            inner = json.loads(payload["body"])
            if isinstance(inner, dict) and "results" in inner:
                results = inner["results"]
            elif isinstance(inner, list):
                results = inner
            else:
                results = []
        except Exception:
            results = []
    else:
        results = []

    # NEW: store and return a compact wrapper (store_key + summary + small sample)
    stored = _store_results(params, results)
    return stored

def call_compare_tool(params: dict) -> dict:
    """
    Execute the comparison against stored datasets. The model supplies keys.
    """
    cur_key = params.get("current_key")
    ref_key = params.get("reference_key")
    return _compare_runs(cur_key, ref_key)

# Helps model understand differnt image formats as i added option for images and whatnot. Not all models understand images btw. I listed ones that do at top under "VISION_CAPABLE."

IMAGE_MIME_TO_FORMAT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

def build_user_content(text: str, files: list, model_id: str):
    """
    (Streamlit doesn't like triple quotes unless inside function like this so that's why 
    I have to use other comment styles above btw i.e. #---#). Plus I think above looks cooler/cleaner.
    Build Converse 'content' array for a user turn (AWS terms not mine)
    - Always includes the text (if provided).
    - If the model is vision-capable (able to see images), include supported images inline.
    - Unsupported files are ignored for the model, but we can make it previewable in the UI.
    
    """
    content = []
    if text:
        content.append({"text": text})

    if not files:
        return content

    if not VISION_CAPABLE.get(model_id, False):
        # Text-only: skips attachments for the API call
        st.info("Selected model is text-only; attached files will be ignored by the model.")
        return content

    # Add supported images as inline image parts. This may need some debugging down the line, but not priority now.
    for f in files:
        mime = getattr(f, "type", None) or ""
        if mime in IMAGE_MIME_TO_FORMAT:
            img_bytes = f.read()
            f.seek(0)  # reset for any subsequent UI previews
            content.append({
                "image": {
                    "format": IMAGE_MIME_TO_FORMAT[mime],
                    "source": {"bytes": img_bytes}
                }
            })
        else:
            # Non-image or unsupported type: ignore for the API call
            pass

    return content

# This adds in session for keeping track of files, what was said, etc. Still neeeds testing so...
def converse_with_tools(user_text: str, files=None, history=None):
    if history is None:
        history = []
    files = files or []

    # You will need to refer to Converse API formatting to understand this. Basically you have to define what role
    # is taking place based on documentation. Will add docs later, but just google converse API docs & you;ll get it.
    messages = [{"role": "user", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)

    user_content = build_user_content(user_text, files, model_id)
    messages.append({"role": "user", "content": user_content})
   
    #------------------------Round 1 for tool calls---------------------------------------------------------------------
    # Sets to allow tool calling with converse API and all the cool doo dads the kids are using these days like tokens
    # temp, etc. This can be tweaked later if stakeholders want certain responses back in certain form/tone/etc.
    #-------------------------------------------------------------------------------------------------------------------
    
    resp = brt.converse(
        modelId=model_id,
        toolConfig={"tools": TOOLS},
        messages=messages,
        inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 5000},
    )
    # Sets up for adding all tools we willl need to add for different sensors 
    # Right now we only have one so it will just use query_measurements based on how we defined it in TOOLS variable (up top)
    out_msg = resp.get("output", {}).get("message", {}) or {}
    out_content = out_msg.get("content", []) or []
    tool_uses = [c for c in out_content if "toolUse" in c]

    if tool_uses:
        tu = tool_uses[0]["toolUse"]
        tool_name = tu["name"]
        tool_input = tu.get("input", {})

        if tool_name == "query_rf_measurements":
            try:
                tool_result = call_rf_api(tool_input)   # returns store_key + summary + sample
                tool_result_text = json.dumps(tool_result)
            except Exception as e:
                tool_result_text = json.dumps({"error": str(e)})

        elif tool_name == "compare_rf_runs":
            try:
                compare_result = call_compare_tool(tool_input)
                tool_result_text = json.dumps(compare_result)
            except Exception as e:
                tool_result_text = json.dumps({"error": str(e)})

        else:
            tool_result_text = json.dumps({"error": f"Unknown tool {tool_name}"})

        # Keep the assistant's toolUse turn
        messages.append({"role": "assistant", "content": out_content})
        # Provide toolResult (Converse expects role='user')
        messages.append({
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"text": tool_result_text}],
                }
            }],
        })

        # Round 2: final answer after using tools and logic/instructions we giave it
        resp2 = brt.converse(
            modelId=model_id,
            toolConfig={"tools": TOOLS},
            messages=messages,
            inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 5000},
        )
        final = resp2.get("output", {}).get("message", {}).get("content", []) or []
        final_text = "".join(c.get("text", "") for c in final if "text" in c)
        return final_text or "_No response_", messages

    # No tool call path
    final_text = "".join(c.get("text", "") for c in out_content if "text" in c)
    return final_text or "_No response_", messages

# -------------------Setting up Chat Ui/user sessions (Still in beta seems to work but..)---------------------------------
# As stated above this allows sessions so model can keep track of what was asked before etc. streamlit has it's docs on it
# ------------------------------------------------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # store prior Converse-format messages (excluding the injected SYSTEM_MSG)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # [{'role': 'user'/'assistant', 'content': str}]

# Optional: Sidebar list of stored runs so we can see what the model can reference
with st.sidebar:
    st.subheader("Stored RF runs")
    if st.session_state.rf_store_order:
        for key in st.session_state.rf_store_order[-15:][::-1]:
            meta = st.session_state.rf_store[key]
            st.caption(f"• {meta['label']}  (rows={len(meta['df'])})")
    else:
        st.caption("No stored runs yet.")

# Renders the  prior chat turns
for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# --------------------------------- Chat input ---------------------------------------
# New widget test - supports text + optional file(s) now, but file ingestion needs testing..
prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True)

# Handles the submitted chat input/prompt from end users for both files and text.
if prompt:
    text = getattr(prompt, "text", "") if prompt else ""
    files = prompt.get("files", []) if isinstance(prompt, dict) else []

    # Shows user turn in UI widget (text + previews)
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

    # Run the now "tools-enabled" Converse version with session history and file input etc. 
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, new_history = converse_with_tools(text, files=files, history=st.session_state.history)
                st.markdown(answer)
                # Persist chat history (for model context) and UI log
                st.session_state.history = new_history
                st.session_state.chat_log.append({"role": "user", "content": (text or "(file(s) only)")})
                st.session_state.chat_log.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Tools run failed: {e}")
