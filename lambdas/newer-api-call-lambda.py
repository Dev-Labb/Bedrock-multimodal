import json
import psycopg2
import os
from datetime import datetime

def lambda_handler(event, context):
    try:
        # Extract query params
        params = event.get("queryStringParameters") or {}
        start = params.get("start")   # e.g. 2025-08-01
        end = params.get("end")       # e.g. 2025-08-20
        limit = params.get("limit", 100)  # default 100

        # Connect to RDS
        conn = psycopg2.connect(
            host=os.environ['DB_HOST'],
            database=os.environ['DB_NAME'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD'],
            port=5432
        )
        cur = conn.cursor()

        # Build query dynamically
        query = "SELECT * FROM rf_measurements WHERE 1=1"
        values = []

        if start:
            query += " AND timestamp >= %s"
            values.append(start)

        if end:
            query += " AND timestamp <= %s"
            values.append(end)

        query += " ORDER BY timestamp DESC LIMIT %s"
        values.append(limit)

        # Execute
        cur.execute(query, tuple(values))
        rows = cur.fetchall()

        # Get column names
        colnames = [desc[0] for desc in cur.description]

        # Convert to list of dicts
        results = [dict(zip(colnames, row)) for row in rows]

        cur.close()
        conn.close()

        return {
            "statusCode": 200,
            "body": json.dumps(results, default=str)  # default=str handles datetime objects
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
