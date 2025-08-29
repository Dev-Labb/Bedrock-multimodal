import streamlit as st
import requests
import json

# Config
API_BASE = "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/testing"
BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
SUMMARIZER_MODEL = "gpt-4.1-mini"  # lighter summarizer

st.title("📡 RF Data + Bedrock Chatbot")

# Inputs
user_prompt = st.text_input("Enter your prompt")
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

def call_bedrock(model, prompt):
    """Call Bedrock via API Gateway Lambda proxy"""
    url = f"{API_BASE}/generate"
    payload = {"modelId": model, "inputText": prompt}
    resp = requests.post(url, json=payload, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"Bedrock error {resp.status_code}: {resp.text}")

    return resp.json().get("outputText", "").strip()

if user_prompt:
    rf_data, error = fetch_rf_data(start_date, end_date, limit)

    if error:
        st.error(f"Could not fetch RF data: {error}")
    else:
        st.subheader("📊 Raw RF Data (first rows)")
        st.json(rf_data)

        # Summarization step
        summarization_prompt = f"""
        Here are RF measurement rows:

        {json.dumps(rf_data, indent=2)}

        Summarize the key patterns, anomalies, and useful stats in plain English.
        Only highlight information useful for answering questions.
        """
        try:
            rf_summary = call_bedrock(SUMMARIZER_MODEL, summarization_prompt)

            st.subheader("📝 RF Data Summary")
            st.write(rf_summary)

            # Final step: answer with Claude Sonnet
            final_prompt = f"""
            The user asked: "{user_prompt}"

            Here is a summary of relevant RF measurements:
            {rf_summary}

            Use both the question and the RF data summary to give the most accurate, grounded answer.
            """

            final_answer = call_bedrock(BEDROCK_MODEL, final_prompt)

            st.subheader("🤖 Chatbot Answer")
            st.write(final_answer)

        except Exception as e:
            st.error(f"Error during summarization or final response: {e}")
