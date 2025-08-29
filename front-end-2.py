new

import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# This is the API URL for the Lambda function, which is set up to handle queries to the Postgres database.
# The query parameters include "start", "end", and "limit" to filter data based on timestamps.
# We will modify the frontend to make sure it passes the right parameters to Lambda.
# ---------------------------------------------------------------------------------------------------------------
API_URL = "https://your-lambda-api-url.com"  # Replace with your actual Lambda API URL

# ---------------------------------------------------------------------------------------------------------------
# Streamlit frontend interface setup, with titles and input fields.
# This allows the user to specify a date range (start and end) and limit for the number of results.
# ---------------------------------------------------------------------------------------------------------------
st.title("📡 RF Data Query Interface")

# Input fields for start and end dates, as well as a limit on the number of results.
col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")
with col3:
    limit = st.number_input("Limit", min_value=1, max_value=100, value=50)  # Default limit is 50

# ---------------------------------------------------------------------------------------------------------------
# Function to format the date input as 'YYYY-MM-DD HH:MM:SS' (add 00:00:00 for start and 23:59:59 for end).
# ---------------------------------------------------------------------------------------------------------------
def format_date(date, is_start=True):
    if is_start:
        return date.strftime("%Y-%m-%d") + " 00:00:00"  # Start time at 00:00:00
    else:
        return date.strftime("%Y-%m-%d") + " 23:59:59"  # End time at 23:59:59

# ---------------------------------------------------------------------------------------------------------------
# Button to trigger the API request when the user submits the query.
# This will send the start, end, and limit parameters to the Lambda API.
# ---------------------------------------------------------------------------------------------------------------
if st.button("Submit Query to Lambda"):
    if not start_date or not end_date:
        st.warning("Please select both start and end dates.")
    else:
        with st.spinner("Querying RF Data..."):
            # Format the start and end dates as required by the Lambda function.
            formatted_start = format_date(start_date, is_start=True)
            formatted_end = format_date(end_date, is_start=False)

            # API query parameters: Pass the start, end, and limit to the Lambda function
            params = {
                "start": formatted_start,
                "end": formatted_end,
                "limit": limit
            }

            try:
                # Send GET request to the Lambda API
                response = requests.get(API_URL, params=params)

                # Check if the response is successful (HTTP status 200)
                if response.status_code == 200:
                    data = response.json()

                    # If the data returned is not empty, convert it to a Pandas DataFrame for display
                    if data:
                        df = pd.DataFrame(data)
                        st.success("Data fetched successfully!")
                        st.dataframe(df)  # Display the data as a table
                    else:
                        st.warning("No data found for the selected date range.")

                else:
                    # If the API request failed, display the error code and message
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                # In case of a network or API error, show the exception message
                st.error(f"Request failed: {str(e)}")

