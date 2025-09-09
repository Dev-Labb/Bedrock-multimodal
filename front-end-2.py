import streamlit as st
import requests
import pandas as pd
import json  # Required to parse the stringified JSON inside the "body" field
from datetime import date  # <-- for clean defaults to date_input

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
    # ---- FIX: use clean date values Streamlit expects; keep your label/comments the same
    start_date = st.date_input("Start Date", value=date(2023, 5, 5), min_value=date(2023, 5, 5))
with col2:
    end_date = st.date_input("End Date", value=date(2023, 5, 6), max_value=date(2023, 6, 11))
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=100, value=10) #setting defaults for limits, but want to hard code it in lambda too.

# Handles button widget and makes sure query parameters are passed to API
if st.button("Grab RF Data"):
    with st.spinner("Grabbing RF measurements..."):
        try:
            # ---- FIX: Streamlit date_input returns date objects; stringify as YYYY-MM-DD for the API
            start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
            end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)

            params = {"start": start_str, "end": end_str, "limit": int(limit)}
            response = requests.get(RF_API_URL, params=params)

            # For debugging visibility
            st.write("🔎 Params sent:", params)
            # st.write("🔍 Raw RF API response:", response.text)

            if response.status_code == 200:
                try:
                    payload = response.json()

                    # ---- FIX: Support BOTH shapes:
                    # New proxy response: {"results":[...], "count":N}
                    # Old non-proxy-wrapped: {"statusCode":200,"body":"{\"results\":...}"}
                    if isinstance(payload, dict) and "results" in payload:
                        results = payload.get("results", [])
                        count = payload.get("count", len(results))
                    elif isinstance(payload, dict) and "body" in payload:
                        # Old shape: body contains a JSON string
                        body_data = json.loads(payload["body"])
                        # body_data may be {"results":[...], "count":N} OR a raw list
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
