import json
import psycopg2 #this is a libray for connecting to postgresql via pypthon
import os 
from datetime import datetime

# event: carries the incoming API request/query (with query parameters, headers, etc.)
# context: This is for when you run runtimes within the function. Not used here, but things like boto3 use them. 

def lambda_handler(event, context):
    try:

# You can use default "event" parameter to get the query string (query parameters) from users. Part of the query I'm setting up to make sure 
#That calls have a start, end, for timestamps and you can put limits. Similar to where I seen elsewhere.
        params = event.get("queryStringParameters") or {}
        start = params.get("start")   # should be in format like: "2025-08-01", but may need to modify to have time as well.
        end = params.get("end")       # should be same as above but with with end time
        limit = params.get("limit", 50)  # limits the amount of measurements and only gets last 50 by default

        # Connect to RDS, but in production will want to use AWS Secrets manager. For now storing in env variables.
        #Remember to use/add a user that has access to the table/tables, but has limited access outside of what it needs to do. 
        conn = psycopg2.connect(
            host=os.environ['DB_HOST'],
            database=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD'],
            port=5432
        )
        cur = conn.cursor()

        '''This will build the queries to be able to make call dynamically (from different date ranges etc.)
        #This was extremely tricky to figure out how to write, but the WHERE 1=1 adds the capability to add & to queries for API calls.
        This explains it better than I can in a comment: https://stackoverflow.com/questions/1264681/what-is-the-purpose-of-using-where-1-1-in-sql-statements'''

        query = "SELECT * FROM rf_measurements WHERE 1=1" # where 1=1 is important to easily add "AND" statements to queries/API calls
        values = [] 
        
        #Logic to query based on timestamp given in API Call. %s is a placeholder to help prevent SQL injection. 
        #I'm still learning some of the placeholders wildcards for like ?, %s, $1, but this should work for now. Double checking.
        #The %s helps ensure that values are treated as a literal value to search/query instead of an executable so that is helps prevent malicious commands like "DROP TABLE"
        if start:
            query += " AND timestamp >= %s" #If start is given in API call (will make requirement), it adds AND timestamp >= start time given to query the 'timestamp column for'
            values.append(start)

        if end:
            query += " AND timestamp <= %s" #for reference, one of the columns is "timestamp" so it can check that column and add values <= the query strings timestamp end.
            values.append(end)

        query += " ORDER BY timestamp DESC LIMIT %s" #I want the last {LIMIT} amount of entries. Note to self: need to find a way to have max limit at some point.
        values.append(limit)

        # Execute the modified/dynamic queries. 
        cur.execute(query, tuple(values)) #This is taking the query and adding values to the query, but with placeholder (%s) and set to tuple to make sure it doesn't change
        rows = cur.fetchall()

        # Get column names from my db table
        colnames = [desc[0] for desc in cur.description]

        '''Convert rows to JSON-friendly format by creating a zipping values and converting to a dictionary since json wants a the key value
        pairs. so it will look like this: [
        {"id": 1, "timestamp": "2025-08-01 12:00:00", "signal_strength": -70, ...},
        {"id": 2, "timestamp": "2025-08-01 12:01:00", "signal_strength": -68, ...}
        ]'''
        results = [dict(zip(colnames, row)) for row in rows]

        cur.close()
        conn.close() #These close database connections since we have results now. 

        return {
            "statusCode": 200,
            "body": json.dumps(results, default=str)  # default=str handles datetime objects
        }

    except Exception as e: # I may end up adding more error handling depending on my time, but these 2 will at least help me konw if it's response goes throuhg and/or if it's my backend server side.
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
