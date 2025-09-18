import json
import psycopg2  # library for connecting to postgresql via python. Had to addd this to lambda layers
import os
from datetime import datetime, date
from decimal import Decimal
from urllib.parse import parse_qs #Added for debugging helpers

# This helps helps handle the json request ensuring correct formats for API to handle them. 
def _json_safe(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return o

# ----------------------
# API Gateway helpers
# ----------------------
#---- This handles CORS (cross orgin) requests. As it gives a bit of flexibility from front end when OPTIONS are chosen and 
#---- Can be locked down the line past testing. 
def _ok_api(payload):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            # CORS (safe for testing for now, but I'd of course tighten later)
            "Access-Control-Allow-Origin": "*", #This for example is isn't tightened, but I'd change here for example.
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,OPTIONS"
        },
        "body": json.dumps(payload, default=_json_safe),
        "isBase64Encoded": False
    }

'''When you send an API call to backend you will get a response code (4xx, 5xx, etc). This weil look at body of response 
from backend and retun the error message.I have switched to using postman mostly for this, 
but keeping here as in different environment no postman exists.'''

def _err_api(status, msg):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"error": msg}),
        "isBase64Encoded": False
    }

def _get_http_method(event):
    # Works for both REST (v1) and HTTP API (v2) 
    m = event.get("httpMethod")
    if not m:
        m = (event.get("requestContext", {}).get("http", {}).get("method"))
    return (m or "").upper()

def _first_or_none(v):
    # If API sent multiValueQueryStringParameters, take first item
    if isinstance(v, list):
        return v[0] if v else None
    return v

def _extract_query_params(event):
    """
    Robustly pull query params across all my methods as it needed flexibility to troubleshoot against agent calls too:
    - REST API proxy: event.queryStringParameters (dict or None)
    - REST API multi-value: event.multiValueQueryStringParameters
    - HTTP API (v2): event.queryStringParameters
    - As my last resort, parse event.rawQueryString (and gets event info)
    """
    # 1) Normal single-value
    params = event.get("queryStringParameters")
    if params:
        return params

    # 2) Multi-value
    mv = event.get("multiValueQueryStringParameters")
    if mv:
        return {k: _first_or_none(v) for k, v in mv.items()}

    # 3) HTTP API raw query string fallback
    raw = event.get("rawQueryString")
    if raw:
        parsed = parse_qs(raw, keep_blank_values=True)
        # parse_qs values are lists; take first
        return {k: (v[0] if v else None) for k, v in parsed.items()}

    # 4) Nothing found
    return {}

# ----------------------
# Bedrock Agent helpers (Lambda action group) 
# ----------------------
def _ok_agent_lambda(event, payload_dict):
    """
    Bedrock Agents (Lambda action group) expect:
    {
      "messageVersion": "1.0",
      "response": {
        "actionGroup": "<from event>",
        "function":    "<from event>",  # for Lambda-type actions
        "responseBody": {
          "application/json": {
            "body": "<STRING>"         # body MUST be a string
          }
        }
      }
    }
    
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup", ""),
            "function": event.get("function", ""),
            "responseBody": {
                "application/json": {
                    "body": json.dumps(payload_dict, default=_json_safe)
                }
            }
        }
    }"""

def _err_agent_lambda(event, message, code="BadRequest"):
    return _ok_agent_lambda(event, {"error": message, "code": code})

# ----------------------
# Bedrock Agent helpers (API schema for hooks)
# ----------------------
def _to_str_body(b):
    if b is None:
        return ""
    if isinstance(b, (dict, list)):
        return json.dumps(b, default=_json_safe)
    return str(b)

def _ok_agent_api_hook_response(event, status_code, headers, body_any):
    """
    API action group RESPONSE hook envelope:
    {
      "messageVersion": "1.0",
      "response": {
        "apiPath": "<echo from event>",
        "httpMethod": "<echo from event>",
        "httpStatusCode": 200,
        "httpHeaders": {...},
        "responseBody": {
          "application/json": { "body": "<STRING>" }
        }
      }
    }
    """
    return {
        "messageVersion": "1.0",
        "response": {
            "apiPath": event.get("apiPath", ""),
            "httpMethod": event.get("httpMethod", ""),
            "httpStatusCode": status_code,
            "httpHeaders": headers or {},
            "responseBody": {
                "application/json": {
                    "body": _to_str_body(body_any)
                }
            }
        }
    }

def _ok_agent_api_hook_request(event, query=None, headers=None, body_any=None):
    """
    API action group REQUEST hook envelope (pass-through unless you need to edit):
    {
      "messageVersion": "1.0",
      "response": {
        "apiPath": "<echo>",
        "httpMethod": "<echo>",
        "queryStringParameters": {...},
        "httpHeaders": {...},
        "requestBody": { "application/json": { "body": "<STRING>" } }
      }
    }
    """
    return {
        "messageVersion": "1.0",
        "response": {
            "apiPath": event.get("apiPath", ""),
            "httpMethod": event.get("httpMethod", ""),
            "queryStringParameters": query or (event.get("queryStringParameters") or {}),
            "httpHeaders": headers or (event.get("headers") or {}),
            "requestBody": {
                "application/json": {
                    "body": _to_str_body(
                        body_any if body_any is not None
                        else (event.get("requestBody", {}).get("application/json", {}).get("body"))
                    )
                }
            }
        }
    }

# ----------------------
# Invocation source detectors
# ----------------------
def _is_agent_lambda_invocation(event):
    # Lambda action group: event has 'parameters' and ('actionGroup' or 'function')
    return ("parameters" in event) and ("actionGroup" in event or "function" in event)

def _is_agent_api_hook_response(event):
    # API action group RESPONSE hook: event has apiPath/httpMethod AND httpResponse
    return ("apiPath" in event) and ("httpMethod" in event) and ("httpResponse" in event)

def _is_agent_api_hook_request(event):
    # API action group REQUEST hook: event has apiPath/httpMethod; no httpResponse yet
    return ("apiPath" in event) and ("httpMethod" in event) and ("httpResponse" not in event)

# event: carries the incoming API request/query (with query parameters, headers, etc.)
# context: not used here

def lambda_handler(event, context):
    # ---------------- Agent API action group: RESPONSE HOOK ----------------
    if _is_agent_api_hook_response(event):
        # Simply repackage and echo apiPath/httpMethod to satisfy the platform
        http_resp = event.get("httpResponse", {}) or {}
        status_code = http_resp.get("statusCode", 200)
        headers = http_resp.get("headers", {}) or {}
        body = http_resp.get("body", "")

        # If backend returned a JSON string, you can parse the results here for dubugging. Nice to have varibale storing the json body for debugging.
        try:
            parsed = json.loads(body) if isinstance(body, str) else body
        except Exception:
            parsed = body

        return _ok_agent_api_hook_response(event, status_code, headers, parsed) # Respinses for agent differs so called added this to sopecify this relates to agent hooks.
    
    '''This entire section section below is was for tryin to get the aws bedrock agents to work correctly with it's
        action group parmeters. I failed so far so went a slightly different route for agentic tooling. Keeping mostly 
        because I may want/need to fix to use action groups later on. '''
    # ---------------- Agent API action group: REQUEST HOOK (this is optional) ----------------
    if _is_agent_api_hook_request(event):
        # No changes: pass through. If I need to add headers/auth, I'd add it it here later.
        return _ok_agent_api_hook_request(event)

    # ---------------- Agent Lambda action group (tool implemented in Lambda) ----------------
    #This actually failed initially as I'm working out why action groups failed. I fell back to using tooling from
    #front end for now instead of bedrock agent console.
    if _is_agent_lambda_invocation(event):
        conn = None
        cur = None
        try:
            # For an Agent Lambda action group with the input schema, parameters come under event["parameters"] and NOT querystingpamaters.
            params = event.get("parameters") or {}
            start = params.get("start")
            end = params.get("end")
            limit_raw = params.get("limit", 10)

            if not start or not end:
                return _err_agent_lambda(event, "Missing 'start'/'end' (YYYY-MM-DD).")

            try:
                limit = max(1, min(int(limit_raw), 500))
            except Exception:
                limit = 10

            conn = psycopg2.connect(
                host=os.environ["DB_HOST"],
                database=os.environ["DB_NAME"],
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                port=int(os.environ.get("DB_PORT", "5432")),
                connect_timeout=15
            )
            cur = conn.cursor()

            #Range mode not neccessary here, but added, because I originally had issues pulling with timestamp queries.
            # You'd set RANGE_MODE=end_exclusive if column is TIMESTAMP. It basically helps establish ranges properly.
            range_mode = os.environ.get("RANGE_MODE", "inclusive").lower()

            query = '''
                SELECT id, "timestamp", frequency, signal_strength, modulation,
                       bandwidth, location, device_type, antenna_type,
                       interference_type, latitude, longitude, altitude
                FROM rf_measurements
                WHERE 1=1
            '''
            values = []

            if range_mode == "end_exclusive":
                query += ' AND "timestamp" >= %s::date'
                values.append(start)
                query += ' AND "timestamp" < (%s::date + INTERVAL \'1 day\')'
                values.append(end)
            else:
                query += ' AND "timestamp" >= %s::date'
                values.append(start)
                query += ' AND "timestamp" <= %s::date'
                values.append(end)

            query += ' ORDER BY "timestamp" ASC LIMIT %s'
            values.append(limit)

            cur.execute(query, tuple(values))
            colnames = [c[0] for c in cur.description]
            rows = [dict(zip(colnames, r)) for r in cur.fetchall()]

            return _ok_agent_lambda(event, {
                "query": {"start": start, "end": end, "limit": limit},
                "count": len(rows),
                "rows": rows
            })

        except Exception as e:
            return _err_agent_lambda(event, f"Internal error: {str(e)}", code="InternalError")

        finally:
            try:
                if cur: cur.close()
            except Exception:
                pass
            try:
                if conn: conn.close()
            except Exception:
                pass

    # ---------------- API Gateway proxy (AKA current backend) ----------------
    # Handle CORS preflight (if browser or tools ever send OPTIONS.)
    if _get_http_method(event) == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
                "Access-Control-Allow-Methods": "GET,OPTIONS"
            },
            "body": "",
            "isBase64Encoded": False
        }

    conn = None
    cur = None
    try:
        # Pull query params from API Gateway proxy event (robust across REST/HTTP API)
        params = _extract_query_params(event)
        start = params.get("start")   # e.g. "2023-05-05" (DATE) or ISO; server will cast to ::date
        end = params.get("end")       # e.g. "2023-05-06"
        limit_raw = params.get("limit", 50)

        # ADded optional debug to CloudWatch (enable by setting env DEBUG=1 in env variables). 
        if os.environ.get("DEBUG") == "1":
            print("RAW EVENT (truncated):", json.dumps(event)[:1500])
            print("EXTRACTED PARAMS:", params)

        # Validate presence
        if not start or not end:
            return _err_api(400, "Missing required 'start' and/or 'end' query parameters (YYYY-MM-DD).")

        # Parse limit safely and clamp
        try:
            limit = max(1, min(int(limit_raw), 500))
        except Exception:
            limit = 50

        # Connect to Postgres (consider Secrets Manager in prod)
        conn = psycopg2.connect(
            host=os.environ["DB_HOST"],
            database=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            port=int(os.environ.get("DB_PORT", "5432")),
            connect_timeout=8
        )
        cur = conn.cursor()

        # Dynamic base query
        # If your "timestamp" column is DATE, <= end::date is fine.
        # If your column is TIMESTAMP and you want to include the entire end day,
        # set env RANGE_MODE=end_exclusive to use: >= start::date AND < (end::date + 1 day)
        range_mode = os.environ.get("RANGE_MODE", "inclusive").lower()

        query = '''
            SELECT id, "timestamp", frequency, signal_strength, modulation,
                   bandwidth, location, device_type, antenna_type,
                   interference_type, latitude, longitude, altitude
            FROM rf_measurements
            WHERE 1=1
        '''
        values = []

        if range_mode == "end_exclusive":
            # Good for TIMESTAMP columns: include entire end day
            query += ' AND "timestamp" >= %s::date'
            values.append(start)
            query += ' AND "timestamp" < (%s::date + INTERVAL \'1 day\')'
            values.append(end)
        else:
            # Good for DATE columns: inclusive range
            query += ' AND "timestamp" >= %s::date'
            values.append(start)
            query += ' AND "timestamp" <= %s::date'
            values.append(end)

        query += ' ORDER BY "timestamp" DESC LIMIT %s'
        values.append(limit)

        # Optional debug of SQL 
        if os.environ.get("DEBUG") == "1":
            print("SQL:", query)
            print("VALUES:", values)

        # Execute queriers and pull out the results in key pairs (columns:rows) along side count of each result
        cur.execute(query, tuple(values))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        results = [dict(zip(colnames, row)) for row in rows]

        payload = {"results": results, "count": len(results)}
        return _ok_api(payload)

    except Exception as e:
        # Log for CloudWatch and return safe error as I needed to debug with cloudwatch
        if os.environ.get("DEBUG") == "1":
            try:
                print("ERROR:", str(e))
            except Exception:
                pass
        return _err_api(500, f"Internal error: {str(e)}")

    finally:
        # Always close DB resources after each call
        try:
            if cur: cur.close()
        except Exception:
            pass
        try:
            if conn: conn.close()
        except Exception:
            pass
