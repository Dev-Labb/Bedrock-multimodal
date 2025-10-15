import os
import io
import json
import requests
import streamlit as st
import pandas as pd
from datetime import date
import boto3

st.set_page_config(page_title="🧠 Multi-Modal Bedrock Test", page_icon="🧠")
st.title("🧠 Multi-Modal Bedrock Test")

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway.Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these will these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
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
region = (st.secrets.get("aws", {}).get("region")
          if "aws" in st.secrets else os.getenv("AWS_REGION", "us-gov-west-1"))

access_key = st.secrets.get("aws", {}).get("access_key_id") if "aws" in st.secrets else os.getenv("AWS_ACCESS_KEY_ID")
secret_key = st.secrets.get("aws", {}).get("secret_access_key") if "aws" in st.secrets else os.getenv("AWS_SECRET_ACCESS_KEY")
# session_token = st.secrets.get("aws", {}).get("session_token") if "aws" in st.secrets else os.getenv("AWS_SESSION_TOKEN") #Will uncomment if I go back to temp sts sessions.

# ---- Best-effort Bedrock client: GovCloud often doesn't have Bedrock; don't crash if unavailable ----
brt = None
if access_key and secret_key:
    try:
        brt = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
            # aws_session_token=session_token
        )
    except Exception:
        brt = None
else:
    try:
        brt = boto3.client("bedrock-runtime", region_name=region) #brt is for bedrock Run Tiime so it knows the proper runtime and region to invoke models etc.
    except Exception:
        brt = None

# --------------------------------- Tools specs setup-----------------------------------------------------------------------------------------
# This is for allowing model to be able to use tools like external/internal API's. One thing we will need to do is make sure
# we copy the OpenAPI Json schema and plug it in as a tool. This will enable model to use API when it needs to based on intstructions
# you give it in System_MSG aka instructions to model system. Make sure this is sound or else model will return tons of errors... trust me....
# --------------------------------------------------------------------------------------------------------------------------------------------

TOOLS = [{
    "toolSpec": {
        "name": "query_rf_measurements",
        "description": (
            "Query RF measurement data via API Gateway. "
            "Parameters: start (YYYY-MM-DD), end (YYYY-MM-DD), limit (integer). "
            "Backend returns columns such as site, uuid, channel, classification, collect_id, telemetry_id, "
            "carrier_frequency, frequency_band, collection_mode, bandwidth, polarity, carrier_snr, "
            "relative_carrier_power, relative_noise_floor, confidence, frequency_shift, peak, "
            "pointing_information_azimuth/elevation/polarization/antenna_name, signal_* fields, "
            "satellite_* fields, measurement_timestamp, insert_timestamp, id."
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
}]

SYSTEM_MSG = (    
    "You help RF data analysts. "
    "When the user asks about RF measurements, call query_rf_measurements with start, end, and optional limit. "
    "If the user did not provide dates, ask for them before calling the tool."
    "Otherwise, if the user asks questions that aren't rf related answer them to the best of your ability." #Can coomment this out if we want to filter to only rf related
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
        return payload

    if isinstance(payload, dict) and "body" in payload:
        try:
            inner = json.loads(payload["body"])
            if isinstance(inner, dict) and "results" in inner:
                return inner
            if isinstance(inner, list):
                return {"results": inner, "count": len(inner)}
            return {"raw": inner}
        except Exception:
            return {"raw": payload}

    return {"raw": payload}

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
    # temp, etc. This can be tweaked later if PM wants certain responses back in certain form/tone/etc.
    #-------------------------------------------------------------------------------------------------------------------
    if brt is None:
        # If Bedrock is not available in this region/account (GovCloud), return a friendly message
        return ("_Bedrock is not available/configured in this environment. "
                "You can still use the RF Data Query panel below to fetch data directly._", messages)

    resp = brt.converse(
        modelId=model_id,
        toolConfig={"tools": TOOLS},
        messages=messages,
        inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 1024},
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
                tool_result = call_rf_api(tool_input)
                tool_result_text = json.dumps(tool_result)
            except Exception as e:
                tool_result_text = json.dumps({"error": str(e)})

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
                inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 1024},
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

# Render prior chat turns
for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# --------------------------------- Chat input section---------------------------------------
# New widget test - supports text + optional file(s) now, but file ingestion needs testing..
prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True)

# ---------- Small helper to pretty-order columns for the new table ----------
def _order_columns_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = [
        "site", "uuid", "channel", "classification", "collect_id", "telemetry_id",
        "measurement_timestamp", "insert_timestamp",
        "carrier_frequency", "frequency_band", "collection_mode", "bandwidth", "polarity",
        "carrier_snr", "relative_carrier_power", "relative_noise_floor", "confidence",
        "frequency_shift", "peak",
        "pointing_information_azimuth", "pointing_information_elevation",
        "pointing_information_polarization", "pointing_information_antenna_name",
        "signal_is_measured_signal", "signal_is_spread_signal", "signal_uuid",
        "signal_baud_rate", "signal_subcarrier_frequency", "signal_subcarrier_snr",
        "signal_confidence", "signal_detection_status", "signal_id",
        "signal_measured_telemetry_id", "signal_chip_rate", "signal_code_taps",
        "signal_code_fill", "signal_code_length", "signal_pn_order",
        "satellite_scc", "satellite_classification", "satellite_name",
        "satellite_international_designator", "satellite_launch_date", "satellite_object_type",
        "satellite_owner_code", "satellite_owner_name", "satellite_uuid",
        "id"
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest] if cols else df

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

# -----------------------------------------------------------------------------------------------------------
# This handles the RF Data Queries to the DB. Allows dynamic queries with 
# start/end/limit parmeters to call the Postgres API via my lambda fucntion/api gateway
# combo and return results. Looked at stremalit docs to help me set up.

###Note: Need to change to use datetime ISO format. To do so I need to add in correct parmeter after each input.
# This is currently why you'll get the wrong data back from database calls.### <--- Will update after fix.
#--------------------------------------------------------------------------------------------------------------
st.header("📡 RF Data Query to database")

col1, col2, col3 = st.columns(3)
with col1:
    # Using date objects for clean defaults that Streamlit expects
    start_date = st.date_input("Start Date", value=date(2023, 5, 5), min_value=date(2025, 9, 5))
with col2:
    end_date = st.date_input("End Date", value=date(2023, 5, 5), max_value=date(2025, 5, 5))
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=500, value=10) #setting defaults for limits, but want to hard code it in lambda too.

# Handles button widget and makes sure query parameters are passed to API
if st.button("Grab RF Data"):
    with st.spinner("Grabbing RF measurements..."):
        try:
            # Streamlit date_input returns date objects; stringify as YYYY-MM-DD for the API
            start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
            end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)

            params = {"start": start_str, "end": end_str, "limit": int(limit)}
            response = requests.get(RF_API_URL, params=params)

            # 🔎 Debugging visibility
            st.write("🔎 Params sent:", params)
            # st.write("🔍 Raw RF API response:", response.text)

            if response.status_code == 200:
                try:
                    payload = response.json()

                    # Support BOTH shapes:
                    # New proxy response: {"results":[...], "count":N}
                    # Old non-proxy-wrapped: {"statusCode":200,"body":"{\"results\":...}"}
                    if isinstance(payload, dict) and "results" in payload:
                        results = payload.get("results", [])
                        count = payload.get("count", len(results))
                    elif isinstance(payload, dict) and "body" in payload:
                        body_data = json.loads(payload["body"])
                        if isinstance(body_data, dict) and "results" in body_data:
                            results = body_data.get("results", [])
                            count = body_data.get("count", len(results))
                        elif isinstance(body_data, list):
                            results = body_data
                            count = len(results)
                        else:
                            raise ValueError("Unexpected body structure in legacy response.")
                    else:
                        raise ValueError("Unexpected response JSON structure from API.")

                    # Convert the parsed data (list of dictionaries) into a DataFrame
                    df = pd.DataFrame(results)

                    # Convert new timestamp columns if present and sort by measurement time
                    for ts_col in ["measurement_timestamp", "insert_timestamp"]:
                        if ts_col in df.columns:
                            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

                    if "measurement_timestamp" in df.columns:
                        df = df.sort_values(by="measurement_timestamp", ascending=True)

                    # Pretty-order columns for readability with new schema
                    df = _order_columns_for_display(df)

                    st.success(f"RF Data Results (count={count})")
                    if not df.empty:
                        st.dataframe(df)
                        # Optional CSV download (handy for analysts)
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV", data=csv, file_name="rf_measurements.csv", mime="text/csv")
                    else:
                        st.info("No rows returned for the selected range.")

                except Exception as parse_err:
                    st.error("Failed to parse JSON from RF API.")
                    st.text(f"Error: {parse_err}")
                    st.code(response.text)
            else:
                st.error(f"Error {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error(f"Request failed: {str(e)}")


