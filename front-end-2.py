import streamlit as st
import requests
import pandas as pd
import json  # Required to parse the stringified JSON inside the "body" field
from datetime import date  # <-- used for clean defaults to date_input
import boto3  # <-- added for Bedrock Converse (tools)

# ---------------------------------------------------------------------------------------------------------------
# A lot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway. Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements" 

# ---------------------------------------------------------------------------------------------------------------
# Model options - These must match keys in AWS Lambda's "ALLOWED_MODELS" variable. For more context please refer
# to the coinciding lambda function. The llama3 model is currently broken because of the format I used to invoke
# the model currently. My plan currently is to change to models/formats that use AWS Converse API. More info can
# be found here: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html  
# ---------------------------------------------------------------------------------------------------------------
MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    "Meta LLaMA3 2-1B Instruct": "meta.llama3-2-1b-instruct-v1:0"  # Keep exact IDs for Bedrock Converse
}

st.title("🧠 Multi-Modal Bedrock Test")

#-----------------------------------------------------------------------------------------
# This allows you to select which models you want based on the above model options.
# Passes both model selection and your prompt to model in json body. 
# -----------------------------------------------------------------------------------------

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
#prompt = st.text_area("Enter your prompt:")
prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True) 

if prompt and prompt.text:
    st.markdown(prompt.text)
if prompt and prompt["files"]:
    st.image(prompt["files"][0])'''

# -----------------------------------------------------------------------------------------------------------
# Option B: DIY tool calling (Bedrock Converse + tools) wired into a new button
# -----------------------------------------------------------------------------------------------------------

# ---- Bedrock client (region pulled from your AWS config/role; override if needed) ----
BEDROCK_REGION = "us-east-1"

# Prefer secrets; fall back to env/profile if not present
region = (st.secrets.get("aws", {}).get("region")
          if "aws" in st.secrets else os.getenv("AWS_REGION", "us-east-1"))

access_key = st.secrets.get("aws", {}).get("access_key_id") if "aws" in st.secrets else os.getenv("AWS_ACCESS_KEY_ID")
secret_key = st.secrets.get("aws", {}).get("secret_access_key") if "aws" in st.secrets else os.getenv("AWS_SECRET_ACCESS_KEY")
session_token = st.secrets.get("aws", {}).get("session_token") if "aws" in st.secrets else os.getenv("AWS_SESSION_TOKEN")

if access_key and secret_key:
    brt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token  # safe to pass None
    )
else:
    # Falls back to default credential chain (e.g., AWS_PROFILE / instance role)
    brt = boto3.client("bedrock-runtime", region_name=region)

# ---- Define the tool the model can call ----
TOOLS = [ {
    "toolSpec": {
        "name": "query_rf_measurements",
        "description": "Query RF measurement data via API Gateway.",
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
} ]

SYSTEM_MSG = (
    "You help RF data analysts. "
    "When the user asks about RF measurements, call query_rf_measurements with start, end, and optional limit. "
    "If the user did not provide dates, ask for them before calling the tool."
)

def call_rf_api(params: dict) -> dict:
    """
    Executes the actual HTTP GET to your API Gateway. Supports both your new proxy response
    ({"results":[...], "count":N}) and the old wrapped response ({"body":"..."}).
    """
    r = requests.get(RF_API_URL, params={
        "start": params.get("start"),
        "end": params.get("end"),
        "limit": params.get("limit", 10)
    }, timeout=30)
    r.raise_for_status()
    payload = r.json()

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

def converse_with_tools(user_text: str, history=None):
    """
    Minimal tool-use loop:
      1) Ask model (allow tool calling)
      2) If tool requested, execute it
      3) Send tool result back for final answer
    """
    if history is None:
        history = []

    messages = [{"role": "assistant", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)
    messages.append({"role": "user", "content": [{"text": user_text}]})

    # Round 1: let the model decide if it wants to call the tool
    resp = brt.converse(
        modelId=MODELS[model_choice],
        toolConfig={"tools": TOOLS},  # ✅ Correct placement
        messages=messages,
        inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 1024}
    )

    out = resp.get("output", {}).get("message", {})
    out_content = out.get("content", []) or []
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

            # Add the assistant turn that contained the toolUse
            messages.append({"role": "assistant", "content": out_content})
            # Provide the toolResult (Converse expects from 'user' role)
            messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"text": tool_result_text}]
                    }
                }]
            })

            # Round 2: model produces the final answer using the tool result
            resp2 = brt.converse(
                modelId=MODELS[model_choice],
                toolConfig={"tools": TOOLS},
                messages=messages,
                inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 1024}
            )
            final = resp2.get("output", {}).get("message", {}).get("content", []) or []
            final_text = "".join(c.get("text", "") for c in final if "text" in c)
            return final_text or "_No response_", messages

    final_text = "".join(c.get("text", "") for c in out_content if "text" in c)
    return final_text or "_No response_", messages

# ---- New button to use tools-enabled flow ----
if st.button("Ask (tools enabled)"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Thinking with tools…"):
            try:
                answer, _ = converse_with_tools(prompt)
                st.success("Answer (tools enabled):")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Tools run failed: {e}")



