import streamlit as st
import requests
import pandas as pd
import json
import re

# ---------------------------------------------------------------------------------------------------------------
# Alot of documentation for streamlit library can be found here: https://docs.streamlit.io/develop/api-reference/
# This is my current API url under "testing" though live I'll probably go with the standard dev/test/prod 
# setup for api's later on. I added RF_API_URL as well to test db backend
# ----------------------------------------------------------------------------------------------------------------

MODEL_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
RF_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements" 

# Model options — These must match keys in Lambda's ALLOWED_MODELS (this call will ultimately use lambda as a proxy first so look at the lambda function for more details).
MODELS = {
    "Claude 3.5 Sonnet": "claude-sonnet",
    "Amazon Nova Micro": "nova-micro",
    "Meta LLaMA3 2-1B Instruct": "llama3-2-1b"  # Very important that you keep track of "exact" id/version of the model you use.
}

st.title("🧠 Multi-Modal Bedrock Test")

# -------------------------------------------------------------------------
# Function to clean IQ data (if in complex string format)
# -------------------------------------------------------------------------
def clean_iq_data(iq_data_str: str) -> list:
    """
    Cleans raw IQ data (string of complex numbers) into a list of complex numbers.
    """
    iq_data_str = iq_data_str.replace('j', 'j ').replace('),', ') ,')  # Formatting clean-up for regex to work
    complex_pattern = r'\(([-+]?\d*\.\d+|\d+)([-+]\d*\.\d+|\d+)(j)\)'  # Regex pattern to capture complex numbers in (real, imag) format
    matches = re.findall(complex_pattern, iq_data_str)  # Extract the real and imaginary parts
    return [complex(float(real), float(imag)) for real, imag, _ in matches]

# -------------------------------------------------------------------------
# This allows you to select which models you want based on the above model
# selection under MODELS & type in prompts in prompt area box. Will be passed
# along in body.
# -------------------------------------------------------------------------

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
prompt = st.text_area("Enter your prompt:")

#handles submit element and makes sure you inputted a prompt 
if st.button("Submit to Model"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Calling model..."):
            try:
                response = requests.post(
                    MODEL_API_URL,
                    headers={"Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "model": MODELS[model_choice]
                    }
                )

                # 🔍 Debugging: Shows the raw API output before parsing
                st.write("🔍 Raw response for debugging. Will remove once production ready:", response.text)

                # Basically, this looks for successful runs and gets the correct output. 
                # I added for troubleshooting different models as some runs returned 200 code,
                # but didn't give info back. This helps, but you may need to still look at logs in cloudwatch.
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


# -------------------------------------------------------------------------
# This handles the RF Data Query Stuff to the DB
# Allows dynamic query parameters (start/end/limit) to call the Postgres API via my lambda fucntion/api gateway
# and return results. Looked at Streamlit docs to help me set up.
# -------------------------------------------------------------------------
st.header("📡 RF Data Query") 

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=300, value=100)  # setting defaults for limits, but want to hard code it in lambda too.

# handles submit element and makes sure query parameters are passed to API
if st.button("Fetch RF Data"):
    with st.spinner("Fetching measurements..."):
        try:
            params = {"start": str(start_date), "end": str(end_date), "limit": limit}
            response = requests.get(RF_API_URL, params=params)

            # 🔍 Debugging: Shows the raw RF API response. I may change this to build a table with pandas instead and keep old code for debugging. 
            st.write("🔍 Raw RF API response:", response.text)

            if response.status_code == 200:
                try:
                    data = response.json()

                    # Directly try to create a DataFrame from the response JSON
                    # Here, we need to clean IQ data if it's present in the response.
                    if isinstance(data, list):  # Ensure the response is a list of records
                        for entry in data:
                            if 'iq_data' in entry:
                                entry['iq_data'] = clean_iq_data(entry['iq_data'])  # Clean IQ data

                    # Create DataFrame from the cleaned data
                    df = pd.DataFrame(data)

                    # Optionally, style the dataframe (if you want)
                    styled_df = df.style.set_table_styles(
                        [{'selector': 'thead th', 
                          'props': [('background-color', '#f5f5f5'), ('color', 'black')]}]
                    ).hide_index()

                    st.success("RF Data Results")
                    st.dataframe(styled_df)  # display the styled dataframe

                except ValueError as ve:
                    st.error("Data format is not suitable for creating a DataFrame.")
                    st.text(f"Error: {ve}")

            else:
                st.error(f"Error {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error(f"Request failed: {str(e)}")
