import os
import io
import json
import requests
import streamlit as st
import pandas as pd
from datetime import date
import boto3

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway.Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these will these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="🧠 Multi-Modal Bedrock Test", page_icon="🧠")
st.title("🧠 Multi-Modal Bedrock Test")

# --------------------------------------------------------------------------------------------------------------------------------
# WARNING!!!!: THIS IS NOW HERE FOR LEGACY CODE PURPOSES. I AM NOW USING CONVERSE API AND WILL NEED TO CONVERT TO CONVERSE STREAM
# Model options - These must match keys in AWS Lambda's "ALLOWED_MODELS" varible. For more context please refer
# to the coinciding lamba function. The llama3 model is currently broken because of the format I used to invoke
# the model currently. My plan currrently is to change to models/formats that use AWS Converse API More info can
# be found here: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html  
# ---------------------------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL    = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements"

MODELS = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    "Meta LLaMA3 2-1B Instruct": "meta.llama3-2-1b-instruct-v1:0"
}

# Which of the above are multimodal meaning they can use proper API. Other (llama for now) doesn't support what we're going with now. 
# Keeping for potential down the line testing though...

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
    brt = boto3.client("bedrock-runtime", region_name=region) #brt is for bedrock Run Tiime so it knows the proper runtime and region to invoke models etc.

# --------------------------------- Tools specs setup-----------------------------------------------------------------------------------------
# This is for allowing model to be able to use tools like external/internal API's. One thing we will need to do is make sure
# we copy the OpenAPI Json file and plug it in as a tool. This will enable model to use API when it needs to based on intstructions
# you give it in System_MSG aka instructions to model system. Make sure this is sound or else model will return tons of errors... trust me....
# --------------------------------------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------------------------
# This took some debugging. Lots of from original code, but basically, results will come in a nested dictinary and so you need to
# unpack the dictionary list and get only the contents of the body so model can read it. Use print if you have errors with response
# or I may just add error (try/catch) correction later if API changes again from Converse/Converse stream to correct again.
# ---------------------------------------------------------------------------------------------------------------------------------
    
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

# Helps model understand differnt image formats as i added option for images and whatnot. Not all models understand images btw. 

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
    I have to use other comment styles above btw i.e. #---#). Plus I think above looks cooler.
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
    messages = [{"role": "assistant", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)

    user_content = build_user_content(user_text, files, model_id)
    messages.append({"role": "user", "content": user_content})
   
    #------------------------Round 1 for tool calls---------------------------------------------------------------------
    # Sets to allow tool calling with converse API and all the cool doo dads the kids are using these days like tokens
    # temp, etc. This can be tweaked later if PM wants certain responses back in certain form. 
    #-------------------------------------------------------------------------------------------------------------------
    
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

            # Round 2: final answer after using tools and logic
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

# --------------------------------- Chat input section!!!------------------------------------
# New widget test: supports text + optional file(s)
prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True)

# Handles the submitted chat input/prompt
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

    # Run the "tools-enabled" Converse version with session history and file input etc. 
    with st.chat_message("assistant"):
        with st.spinner("I'm Thinking… Saba stop RUSHING ME DAMN!....:("):
            try:
                answer, new_history = converse_with_tools(text, files=files, history=st.session_state.history)
                st.markdown(answer)
                # Persist chat history (for model context) and UI log
                st.session_state.history = new_history
                st.session_state.chat_log.append({"role": "user", "content": (text or "(file(s) only)")})
                st.session_state.chat_log.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Tools run failed: {e}")



