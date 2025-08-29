import streamlit as st
import requests
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Config
API_BASE = "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/testing"
BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

st.title("📡 RF Data + Bedrock Chatbot")

# Input box
user_prompt = st.text_input("Enter your prompt")

# Query parameters
start_date = st.text_input("Start date (YYYY-MM-DD)", "2025-08-01")
end_date = st.text_input("End date (YYYY-MM-DD)", "2025-08-02")
limit = st.number_input("Limit rows", min_value=1, max_value=50, value=5)

def fetch_rf_data(start, end, limit):
    try:
        url = f"{API_BASE}/rf-data?start={start}&end={end}&limit={limit}"
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            return None, f"Error {resp.status_code}: {resp.text}"

        data = resp.json()

        # ✅ handle list vs dict safely
        if isinstance(data, list):
            subset = data[:limit]
        else:
            subset = [data]

        return subset, None

    except Exception as e:
        return None, str(e)

if user_prompt:
    rf_data, error = fetch_rf_data(start_date, end_date, limit)

    if error:
        st.error(f"Could not fetch RF data: {error}")
    else:
        st.subheader("📊 Raw RF Data (first rows)")
        st.json(rf_data)  # ✅ Debug: show raw API response

        # Build summarization prompt
        summarization_prompt = f"""
        Here are RF measurement rows:

        {json.dumps(rf_data, indent=2)}

        Summarize the key patterns, anomalies, and useful stats in plain English.
        Only highlight information useful for answering questions.
        """

        try:
            # First, summarize the RF data with Nova Micro
            summary_resp = client.responses.create(
                model="gpt-4.1-mini",  # You can switch to "gpt-4.1" if you want more power
                input=summarization_prompt
            )
            rf_summary = summary_resp.output_text

            st.subheader("📝 RF Data Summary")
            st.write(rf_summary)

            # Now, answer the user’s question using both summary + prompt
            final_prompt = f"""
            The user asked: "{user_prompt}"

            Here is a summary of relevant RF measurements:
            {rf_summary}

            Use both the question and the RF data summary to give the most accurate, grounded answer.
            """

            final_resp = client.responses.create(
                model=BEDROCK_MODEL,
                input=final_prompt
            )

            st.subheader("🤖 Chatbot Answer")
            st.write(final_resp.output_text)

        except Exception as e:
            st.error(f"Error during summarization or final response: {e}")
