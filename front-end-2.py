import streamlit as st
import requests

# This in my current API url under "testing" though live I'll probably go with the standard dev/test/prod setup for api's later on.
API_URL = "https://nj03mfzl37.execute-api.us-east-1.amazonaws.com/testing/generate"

# Model options — These must match keys in Lambda's ALLOWED_MODELS (this call will ultimately use lambda as a proxy first so look at the lambda function for more details).
MODELS = {
    "Claude 3.5 Sonnet": "claude-sonnet",
    "Amazon Nova Micro": "nova-micro",
    "Meta LLaMA3 2-1B Instruct": "llama3-2-1b" #Very important that you keep track of "exact" id/version of the model you use.
}

st.title("🧠 This my test app for upcoming project. Next step adding RAG woot!")

#This allows you to select which models you want based on the above model selection under MODELS & type in prompts in prompt area box
model_choice = st.selectbox("Select a model:", list(MODELS.keys()))
prompt = st.text_area("Enter your prompt:")

if st.button("Submit"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Calling model..."):
            #Pay close attention here. This is your headers and has severe impact on whether you will get the response you want. Can also be modified to get different formats/schemas back based on different backend setups.
            try:
                response = requests.post(
                    API_URL,
                    headers={"Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "model": MODELS[model_choice]
                    }
                )

                
                st.write("🔍 Raw response for debugging/full transparency:", response.text)

#Basically, this looks for 200 code response
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        output = (
                            json_response.get("response") or
                            json_response.get("message") or
                            json_response.get("error") or
                            "No valid response found."
                        )
                        #Added success/error catching for debugging later as I was getting errors on certain models. This helps, but you may need to still look at logs in cloudwatch to get full picture.
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








