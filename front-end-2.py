import streamlit as st
import requests
import json

API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"
DB_API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate/measurements"

MODELS = {
    "Claude 3.5 Sonnet": "claude-sonnet",
    "Amazon Nova Micro": "nova-micro",
    "Meta LLaMA3 2-1B Instruct": "llama3-2-1b"
}

st.title("🧠 Multi-Modal Bedrock + RDS RAG (with Summarization)")

model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
prompt = st.text_area("Enter your prompt:")

# Optional params for RF data
with st.expander("Add RF Data Context"):
    start = st.text_input("Start timestamp (YYYY-MM-DD)")
    end = st.text_input("End timestamp (YYYY-MM-DD)")
    limit = st.number_input("Limit results", value=10, min_value=1, max_value=100)

if st.button("Submit"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Fetching RF data + summarizing..."):
            rf_summary = ""
            try:
                # Step 1: Fetch RF Data
                db_resp = requests.get(DB_API_URL, params={
                    "start": start or None,
                    "end": end or None,
                    "limit": limit
                })
                if db_resp.status_code == 200:
                    data = db_resp.json()
                    if data:
                        # Step 2: Summarize RF Data with small model (Nova Micro)
                        summarization_prompt = f"""
Here are RF measurement rows:

{json.dumps(data[:limit], indent=2)}

Summarize the key patterns, anomalies, and useful stats in plain English.
Only highlight information useful for answering questions.
"""

                        sum_resp = requests.post(
                            API_URL,
                            headers={"Content-Type": "application/json"},
                            json={
                                "prompt": summarization_prompt,
                                "model": "nova-micro"  # force summarization with cheap model
                            }
                        )

                        if sum_resp.status_code == 200:
                            rf_summary = sum_resp.json().get("response", "")
                        else:
                            st.warning("Summarization failed, passing raw data.")
                            rf_summary = json.dumps(data[:limit], indent=2)
                    else:
                        st.info("No RF data returned for given filters.")
                else:
                    st.warning(f"DB API error {db_resp.status_code}")
            except Exception as e:
                st.warning(f"Could not fetch/summarize RF data: {e}")

            # Step 3: Combine user query + summary
            full_prompt = f"""
User query: {prompt}

Relevant RF context (summarized):
{rf_summary}
"""

            # Step 4: Send to main chosen model
            with st.spinner("Calling main model..."):
                try:
                    response = requests.post(
                        API_URL,
                        headers={"Content-Type": "application/json"},
                        json={"prompt": full_prompt, "model": MODELS[model_choice]}
                    )

                    st.write("🔍 Raw response:", response.text)

                    if response.status_code == 200:
                        json_response = response.json()
                        output = json_response.get("response", "No valid response found.")
                        st.success("Model response:")
                        st.markdown(f"```\n{output.strip()}\n```")
                    else:
                        st.error(f"Error {response.status_code}")
                        st.code(response.text)

                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

