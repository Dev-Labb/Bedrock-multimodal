import json
import psycopg2
import os
import datetime
#Note to self: consider importing boto3.clientErrors or whatever the library is called again so I can catch errors
#Same for adding logging options for cloudwatch events etc. 

# DB settings. Will need to switch to something like AWS Secrets later instead of env
DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASSWORD']

#Defines all the parameters that can be passed to the API, and their types. I need to write rest of these out based on options in DB
PARAMS = {
    "start": {"type": "string", "required": True},
    "end": {"type": "string", "required": True},
    "limit": {"type": "integer", "required": False},
    "location": {"type": "string", "required": False},
    "device_type": {"type": "string", "required": False},
    "antenna_type": {"type": "string", "required": False},
    "interference_type": {"type": "string", "required": False}
}

#Defines the schema for the Agent/tool schema


def lambda_handler(event, context):
    try:
        # Extract query parameters (provided by Agent/tool schema)
        params = event.get("queryStringParameters", {}) or {}
        start = params.get("start")
        end = params.get("end")
        limit = int(params.get("limit", 50)) #-->should probably make this line up with other max limits.
        location = params.get("location")
        device_type = params.get("device_type")
        antenna_type = params.get("antenna_type")
        interference_type = params.get("interference_type")

        # Cap maximum rows
        limit = min(limit, 500) #Will keep it at 5oo for now, BUT need to look into upping runtimes limits for bigger queries.

        #---------------------------------------------------------------------------------------------------
        # Baseline query with all the little fun rf stuff you can query in db. NOTE: I'm thinking of addding
        # some additional columns like "Site" so will need to update this if I do.
        #---------------------------------------------------------------------------------------------------
        query = """
            SELECT id, timestamp, frequency, signal_strength, modulation,
                   bandwidth, location, device_type, antenna_type,
                   interference_type, latitude, longitude, altitude
            FROM rf_measurements
            WHERE timestamp BETWEEN %s AND %s
        """
        values = [start, end]

        # Adds in optional filters for "AND" statements. Needed this to be dynamic instead of my old static calls.
        if location:
            query += " AND location = %s"
            values.append(location)
        if device_type:
            query += " AND device_type = %s"
            values.append(device_type)
        if antenna_type:
            query += " AND antenna_type = %s"
            values.append(antenna_type)
        if interference_type:
            query += " AND interference_type = %s"
            values.append(interference_type)

        query += " ORDER BY timestamp ASC LIMIT %s"
        values.append(limit)

        # Connect to DB with predefined variables
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()

        #----------------------------------------------------------------------------------------
        # Execute query to DB and sorts by descending order. Taking note of the "cur.description"
        # portion. In psycop2 library you can pull metadata on each column you're pulling and get 
        # the name of each column that way which in this case is being used to help define columns.
        #-----------------------------------------------------------------------------------------
        cur.execute(query, tuple(values))
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        #Closes the connection once done
        cur.close()
        conn.close()

        # Convert rows to list of dicts. Needed to make it easy to match each column to a row value. Key/value pairs.
        # You also zip first so that they're tupled (did I make this word up maybe maybe I did) first. Thank god for zip existing.
        results = [dict(zip(colnames, row)) for row in rows]

        return {
            "statusCode": 200,
            "body": json.dumps({
                "results": results,
                "count": len(results)
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
