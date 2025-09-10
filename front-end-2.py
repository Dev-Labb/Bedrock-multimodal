import streamlit as st
import requests
import pandas as pd
import json
import os
import boto3
from datetime import date as _date

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway.Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these will these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Bedrock model to use with Converse
MODEL_REGION = st.secrets.get("aws", {}).get("region", "us-east-1")  # Prefer secrets; fallback to us-east-1

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"  # (unused here)
RF_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements"

# ---------------------------------------------------------------------------------------------------------------
# AWS credentials for Bedrock: prefer Streamlit Secrets; fall back to env vars if set. 
# For production: use STS temp creds or call a backend you control (so keys never sit in the app).
# ---------------------------------------------------------------------------------------------------------------
aws_access_key_id = st.secrets.get("aws", {}).get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = st.secrets.get("aws", {}).get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = st.secrets.get("aws", {}).get("session_token") or os.getenv("AWS_SESSION_TOKEN")

session_kwargs = {"region_name": MODEL_REGION}
if aws_access_key_id and aws_secret_access_key:
    session_kwargs.update({
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key
    })
    if aws_session_token:
        session_kwargs["aws_session_token"] = aws_session_token

aws_sess = boto3.Session(**session_kwargs)
brt = aws_sess.client("bedrock-runtime", region_name=MODEL_REGION)

# Optional sanity check (uncomment if you want to print the principal)
# sts = aws_sess.client("sts")
# whoami = sts.get_caller_identity()["Arn"]
# st.caption(f"Using AWS principal: {whoami}")

st.title("🧠 Multi-Modal Bedrock Test (Converse + Tools)")

# ---------------------------------------------------------------------------------------------------------------
# Model options — keeping your pattern. These keys aren’t used by Converse directly, but we keep the UI consistent.
# ---------------------------------------------------------------------------------------------------------------
MODELS = {
    "Claude 3.5 Sonnet (Bedrock)": MODEL_ID,
    "Amazon Nova Micro (placeholder)": "amazon.nova-micro-v1:0",
    "Meta LLaMA3 2-1B Instruct (placeholder)": "meta.llama3-2-1b-instruct-v1:0"
}

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
prompt = st.text_area("Enter your prompt:")

# ---------------------------------------------------------------------------------------------------------------
# RF Data Query inputs (UI) — we’ll let the *model* decide when to use the tool, but we also expose inputs so
# analysts can explicitly set a range; we’ll pass these as context to the model. 
# NOTE: st.date_input returns a date (YYYY-MM-DD). We convert to string for the API query.
# ---------------------------------------------------------------------------------------------------------------
st.header("📡 RF Data Query to database (tool-aware)")
col1, col2, col3 = st.columns(3)
with col1:
    # Streamlit accepts "YYYY-MM-DD" or datetime/date; we’ll default to a known range
    start_date = st.date_input("Start Date", value=_date(2023, 5, 5))
with col2:
    end_date = st.date_input("End Date", value=_date(2023, 5, 6))
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=500, value=10)

# ---------------------------------------------------------------------------------------------------------------
# Bedrock Converse tools: define a single tool "query_rf_measurements" that the model can choose to call.
# The input schema MUST be JSON schema for the tool payload; keep it small and clear.
# ---------------------------------------------------------------------------------------------------------------
tool_spec = {
    "toolSpec": {
        "name": "query_rf_measurements",
        "description": (
            "Query RF measurement data from the backend API. "
            "Requires start and end (YYYY-MM-DD), optional limit."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end":   {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "limit": {"type": "integer", "description": "Max rows to return (1-500)"}
                },
                "required": ["start", "end"]
            }
        }
    }
}

def call_backend_api(start: str, end: str, limit: int):
    """Calls your API Gateway endpoint with query params and returns parsed JSON."""
    params = {"start": start, "end": end, "limit": limit}
    r = requests.get(RF_API_URL, params=params, timeout=20)
    # API Gateway proxy returns {statusCode, body, ...} — your Lambda builds body as JSON string: parse twice.
    data = r.json()
    # If your Lambda returns the new shape {"results":[...],"count":N} directly, just return data.
    # In your current proxy Lambda, body is the JSON payload string — handle both cases robustly:
    body = data.get("body")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            # body might already be raw JSON; fall back to whole object
            return data
    return data

def converse_once(messages, tools=None):
    """Thin wrapper around Bedrock converse for a single turn."""
    args = {
        "modelId": MODELS[model_choice],
        "messages": messages,
        "inferenceConfig": {"temperature": 0, "topP": 1, "topK": 250, "maxTokens": 1024}
    }
    if tools:
        args["tools"] = tools
    resp = brt.converse(**args)
    return resp

def run_converse_with_tool(prompt_text: str, start: str, end: str, limit: int):
    """
    1) Send user + system context with a tool spec.
    2) If the model asks to use the tool (toolUse), call your API Gateway.
    3) Send a toolResult message back to the model to get the final answer.
    """
    system_text = (
        "You are an RF data analyst assistant. "
        "When a user asks about RF measurements, use the query_rf_measurements tool if helpful. "
        "If start/end are missing, ask for them. Keep answers concise."
    )

    messages = [
        {"role": "system", "content": [{"text": system_text}]},
        {"role": "user", "content": [{"text": prompt_text}]},
        # Provide *hint* context so the model knows the preferred defaults/range the user set in the UI:
        {"role": "user", "content": [{"text": f"UI context: start={start}, end={end}, limit={limit}"}]},
    ]

    # 1) First call: give tool spec so the model can choose to call it
    resp = converse_once(messages, tools=[tool_spec])

    # Extract the assistant message
    out = resp["output"]["message"]["content"]

    # Look for a toolUse in the assistant content
    tool_uses = [c for c in out if c.get("type") == "toolUse"]
    if not tool_uses:
        # Model chose not to call tool; just return its text
        text_parts = [c["text"] for c in out if c.get("type") == "text"]
        return {"assistant_text": "\n".join(text_parts).strip(), "rows": None}

    # We’ll handle the first toolUse (you can add multi-tool logic if needed)
    tu = tool_uses[0]
    tool_name = tu["name"]
    tool_input = tu.get("input", {}) or {}
    tool_use_id = tu["toolUseId"]

    # Make sure we have start/end/limit — fill from UI hint if model omitted
    start_in = tool_input.get("start") or start
    end_in = tool_input.get("end") or end
    limit_in = int(tool_input.get("limit") or limit)

    # 2) Actually call your API Gateway
    api_data = call_backend_api(start_in, end_in, limit_in)  # expects {"results":[...], "count":N}
    rows = api_data.get("results") or []
    count = api_data.get("count", len(rows))

    # 3) Send toolResult back to the model so it can reason/summarize
    tool_result_msg = {
        "role": "user",
        "content": [{
            "type": "toolResult",
            "toolUseId": tool_use_id,
            "content": [
                # IMPORTANT: Send JSON as a string or as a {json: ...} object — Converse accepts either.
                {"json": {"results": rows, "count": count, "start": start_in, "end": end_in, "limit": limit_in}}
            ]
        }]
    }

    # Extend the conversation with the assistant’s toolUse and our toolResult
    messages.extend([
        {"role": "assistant", "content": out},  # echo the assistant message containing toolUse
        tool_result_msg
    ])

    # Final call: get the assistant’s summarized answer
    resp2 = converse_once(messages, tools=[tool_spec])
    out2 = resp2["output"]["message"]["content"]
    text_parts = [c["text"] for c in out2 if c.get("type") == "text"]
    final_text = "\n".join(text_parts).strip()
    return {"assistant_text": final_text, "rows": rows}

# -----------------------------------------------------------------------------------------
# Submit to Model (free-form chat that can also leverage the tool via UI context)
# -----------------------------------------------------------------------------------------
if st.button("Submit to Model"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Calling Bedrock (Converse + tools)..."):
            try:
                result = run_converse_with_tool(
                    prompt_text=prompt.strip(),
                    start=str(start_date),
                    end=str(end_date),
                    limit=int(limit),
                )
                st.success("Assistant:")
                st.markdown(result["assistant_text"] or "_(no text)_")

                # If rows returned, show a table
                if result["rows"]:
                    st.info(f"Rows returned: {len(result['rows'])}")
                    st.dataframe(pd.DataFrame(result["rows"]))
            except Exception as e:
                st.error(f"Bedrock tools call failed: {e}")

# -----------------------------------------------------------------------------------------------------------
# You can also keep your direct “Grab RF Data” button for deterministic calls (model not involved).
# Helpful for debugging the API end-to-end separately from the LLM.
# -----------------------------------------------------------------------------------------------------------
if st.button("Grab RF Data (direct API, no LLM)"):
    with st.spinner("Grabbing RF measurements..."):
        try:
            params = {"start": str(start_date), "end": str(end_date), "limit": int(limit)}
            r = requests.get(RF_API_URL, params=params, timeout=20)
            raw_json = r.json()
            # Your proxy returns {statusCode, body: "<string>"}; unwrap body if present:
            payload = json.loads(raw_json["body"]) if "body" in raw_json and isinstance(raw_json["body"], str) else raw_json
            df = pd.DataFrame(payload.get("results", []))
            st.success("RF Data Results")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
