import streamlit as st
import requests
import pandas as pd
import json  # Required to parse the stringified JSON inside the "body" field
from datetime import date  # <-- used for clean defaults to date_input
import boto3  # <-- added for Bedrock Converse (tools)

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# These are my API endpoints to API Gateway.Currently, all API url's are under the "testing" stage in API Gateway.
# I plan to change staging names to dev/test/prod which means these will these two variables will likely change 
# to reflect the updated staging names. First variable is for talking to model. Second is for SQL query backend. 
# ----------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements" 

# ---------------------------------------------------------------------------------------------------------------
# Model options - These must match keys in AWS Lambda's "ALLOWED_MODELS" varible. For more context please refer
# to the coinciding lamba function. The llama3 model is currently broken because of the format I used to invoke
# the model currently. My plan currrently is to change to models/formats that use AWS Converse API More info can
# be found here: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html  
# ---------------------------------------------------------------------------------------------------------------
MODELS = {
    "Claude 3.5 Sonnet": "claude-sonnet",
    "Amazon Nova Micro": "nova-micro",
    "Meta LLaMA3 2-1B Instruct": "llama3-2-1b"  #Very important that you keep track of "exact" id/version of the model you use.
}

st.title("🧠 Multi-Modal Bedrock Test")

#-----------------------------------------------------------------------------------------
# This allows you to select which models you want based on the above model options.
# Passes both model selection and your prompt to model in json body. 
# -----------------------------------------------------------------------------------------

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
prompt = st.text_area("Enter your prompt:")

#------------------------------------------------------------------------------------------------------------
# FEATURE FIX TO TRY: I can probably just replace line 36 with the following code block and it will change 
# replace the st.text_area widget with st.chat_input without issue but I haven't tried it here yet, but have 
# made this widget elsewhere as quick test and just copied that code here based on documetnation:
   
#prompt = st.chat_input(placeholder="Enter prompt or add a file:", accept_file=True) 

#if prompt and prompt.text:
#    st.markdown(prompt.text)
#if prompt and prompt["files"]:
#    st.image(prompt["files"][0])'''
#------------------------------------------------------------------------------------------------------------

# Handles submit widget and makes sure you inputted a prompt 
if st.button("Submit to Model"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Calling model..."):
            # Sends POST request to model and configures headers.
            try:
                response = requests.post(
                    MODEL_API_URL,
                    headers={"Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "model": MODELS[model_choice]
                    }
                )

                # This is specifically for Debugging: Shows the raw API output before parsing
                st.write("🔍 Raw response for debugging. Will remove once more production ready:", response.text)

                #------------------------------------------------------------------------------
                # Basically, this looks for successful runs and gets the correct output.I added 
                # for troubleshooting different models as some runs returned 200 status code, 
                # but didn't give info back from the model. This helps, but you may need to 
                # still look at logs in cloudwatch for full details on errors.
                #-------------------------------------------------------------------------------
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        output = (
                            json_response.get("response") or
                            json_response.get("message") or
                            json_response.get("error") or
                            "No valid response found."
                        )
                        st.success("Model response:")
                        st.markdown(f"```\n{output.strip()}\n```")
                    except Exception as parse_err:
                        st.error("Failed to parse JSON from the response.")
                        st.text(f"Error: {parse_err}")
                else:
                    st.error(f"Error {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error(f"Request failed: {str(e)}")


# ============================
# Option B: DIY tool calling (Bedrock Converse + tools) wired into a new button
# ============================

# ---- Bedrock client (region pulled from your AWS config/role; override if needed) ----
BEDROCK_REGION = "us-east-1"
brt = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# ---- Define the tool the model can call ----
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

    # New proxy shape
    if isinstance(payload, dict) and "results" in payload:
        return payload

    # Old proxy-wrapped shape
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

    # Build the message list per Converse schema
    messages = [{"role": "system", "content": [{"text": SYSTEM_MSG}]}]
    messages.extend(history)
    messages.append({"role": "user", "content": [{"text": user_text}]})

    # Round 1: let the model decide if it wants to call the tool
    resp = brt.converse(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        toolConfig={"tools": TOOLS},
        messages=messages,
        inferenceConfig={"temperature": 0}
    )

    out = resp.get("output", {}).get("message", {})
    out_content = out.get("content", []) or []
    tool_uses = [c for c in out_content if "toolUse" in c]

    if tool_uses:
        tu = tool_uses[0]["toolUse"]
        tool_name = tu["name"]
        tool_input = tu.get("input", {})  # dict with start/end/limit

        if tool_name == "query_rf_measurements":
            try:
                tool_result = call_rf_api(tool_input)
                tool_result_text = json.dumps(tool_result)
            except Exception as e:
                tool_result = {"error": str(e)}
                tool_result_text = json.dumps(tool_result)

            # Add the assistant turn that contained the toolUse…
            messages.append({"role": "assistant", "content": out_content})
            # …then provide the toolResult (Converse expects it from the 'user' role)
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
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                toolConfig={"tools": TOOLS},
                messages=messages,
                inferenceConfig={"temperature": 0}
            )
            final = resp2.get("output", {}).get("message", {}).get("content", []) or []
            final_text = "".join(c.get("text", "") for c in final if "text" in c)
            return final_text or "_No response_", messages

    # No tool used → just return plain text
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
    start_date = st.date_input("Start Date", value=date(2023, 5, 5), min_value=date(2023, 5, 5))
with col2:
    end_date = st.date_input("End Date", value=date(2023, 5, 6), max_value=date(2023, 6, 11))
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=100, value=10) #setting defaults for limits, but want to hard code it in lambda too.

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

                    st.success(f"RF Data Results (count={count})")
                    if not df.empty:
                        st.dataframe(df)
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
