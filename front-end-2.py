import os
import io
import json
import requests
import streamlit as st
import pandas as pd
from datetime import date
import boto3

# ---------------------------------- App/setup ----------------------------------
st.set_page_config(page_title="🧠 Multi-Modal Bedrock Test", page_icon="🧠")
st.title("🧠 Multi-Modal Bedrock Test")

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL    = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements"

MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    "Meta LLaMA3 2-1B Instruct": "meta.llama3-2-1b-instruct-v1:0"
}

# Which of the above are multimodal?
VISION_CAPABLE = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": True,
    "amazon.nova-micro-v1:0": True,
    "meta.llama3-2-1b-instruct-v1:0": False,
}

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
model_id = MODELS[model_choice]

# --------------------------- Bedrock client (Converse) -------------------------
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
        aws_session_token=session_token
    )
else:
    brt = boto3.client("bedrock-runtime", region_name=region)

# --------------------------------- Tools spec ----------------------------------
TOOLS = [{
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
}]

SYSTEM_MSG = (
    "You help RF data analysts. "
    "When the user asks about RF measurements, call query_rf_measurements with start, end, and optional limit. "
    "If the user did not provide dates, ask for them before calling the tool."
)

# ----------------------------- Utility: RF API call ----------------------------
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

# ------------------------ Helpers: content & file handling ---------------------
IMAGE_MIME_TO_FORMAT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

def build_user_content(text: str, files: list, model_id: str):
    """
    Build Converse 'content' array for a user turn.
    - Always includes the text (if provided).
    - If the model is vision-capable, include supported images inline.
    - Unsupported files are ignored for the model, but we'll preview them in the UI.
    """
    content = []
    if text:
        content.append({"text": text})

    if not files:
        return content

    if not VISION_CAPABLE.get(model_id, False):
        # Text-only: skip attachments for the API call
        st.info("Selected model is text-only; attached files will be ignored by the model.")
        return content

    # Add supported images as inline image parts
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

# -------------------------- Converse (tools-enabled) ---------------------------
def converse_with_tools(user_text: str, files=None, history=None):
    if history is None:
        history = []
    files = files or []

    # Inject "system" guidance as an assistant message (Converse does not support role=system)
    messages = [{"role": "assistant", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)

    user_content = build_user_content(user_text, files, model_id)
    messages.append({"role": "user", "content": user_content})

    # Round 1: allow tool calling
    resp = brt.converse(
        modelId=model_id,
        toolConfig={"tools": TOOLS},
        messages=messages,
        inferenceConfig={"temperature": 0, "topP": 1, "maxTokens": 1024},
    )

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

            # Round 2: final answer after tool
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

# ------------------------------- Chat UI state --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # store prior Converse-format messages (excluding the injected SYSTEM_MSG)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # [{'role': 'user'/'assistant', 'content': str}]

# Render prior chat turns
for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# --------------------------------- Chat input ---------------------------------
# New widget: supports text + optional file(s)
prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True)

# Handle a submitted chat input
if prompt:
    text = getattr(prompt, "text", "") if prompt else ""
    files = prompt.get("files", []) if isinstance(prompt, dict) else []

    # Echo user turn in UI (text + previews)
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

    # Run tools-enabled Converse
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
