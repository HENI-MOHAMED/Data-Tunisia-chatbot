import os
import json
import requests
import urllib3
from g4f import Client
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
from pdfminer.high_level import extract_text

client = Client()
# -----------------------------
# TOOL: Query data.gov.tn
# -----------------------------
def query_open_data(keyword: str):

    search_url = "https://catalog.data.gov.tn/api/3/action/package_search"
    MAX_DATASETS = 5

    try:
        r = requests.get(search_url, params={"q": keyword}, verify=False, timeout=15)
    except requests.exceptions.RequestException as e:
        return [{"error": f"Could not connect to data.gov.tn: {e}"}]

    if r.status_code != 200:
        return [{"error": f"data.gov.tn returned status {r.status_code}: {r.text[:500]}"}]

    try:
        data = r.json()
    except Exception:
        return [{"error": f"data.gov.tn returned non-JSON response: {r.text[:500]}"}]

    if not data.get("success"):
        return [{"error": f"API returned an error: {data}"}]

    results = []

    for dataset in data.get("result", {}).get("results", [])[:MAX_DATASETS]:

        for resource in dataset.get("resources", []):

            url = resource.get("url")

            if not url:
                continue

            try:

                file_content = download_and_extract(url)

                results.append({
                    "dataset": dataset["title"],
                    "file": url,
                    "content_preview": file_content[:1000]
                })

            except Exception as e:
                results.append({
                    "dataset": dataset["title"],
                    "file": url,
                    "error": str(e)
                })

    if not results:
        return [{"info": f"No datasets found for keyword '{keyword}'"}]
    print(results)
    return results


# -----------------------------
# DOWNLOAD + EXTRACT
# -----------------------------
def download_and_extract(url):

    r = requests.get(url, timeout=30, verify=False)

    if url.endswith(".pdf"):

        with open("temp.pdf", "wb") as f:
            f.write(r.content)

        return extract_text("temp.pdf")

    elif url.endswith(".csv"):

        from io import StringIO
        df = pd.read_csv(StringIO(r.text), sep=None, engine="python")
        return df.head(20).to_string()

    elif url.endswith(".xlsx") or url.endswith(".xls"):

        ext = ".xls" if url.endswith(".xls") else ".xlsx"
        temp_name = "temp" + ext
        with open(temp_name, "wb") as f:
            f.write(r.content)

        engine = "xlrd" if ext == ".xls" else "openpyxl"
        df = pd.read_excel(temp_name, engine=engine)
        return df.head(20).to_string()

    else:
        return r.text[:2000]


# -----------------------------
# TOOL SCHEMA FOR AI
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_open_data",
            "description": "Search Tunisia open government datasets and extract content",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "keyword to search datasets"
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]


# -----------------------------
# FLASK & TWILIO WEBHOOK
# -----------------------------
app = Flask(__name__)

# Dictionary to store conversation history per phone number
# Note: For production, use a database (e.g. SQLite, PostgreSQL, Redis)
user_sessions = {}

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')

    if not incoming_msg:
        return 'OK', 200

    # Initialize a session for new users
    if sender not in user_sessions:
        user_sessions[sender] = [
            {
                "role": "system",
                "content": """  You are a research assistant that can search Tunisia open data. Respond in a friendly and concise manner suitable for WhatsApp.(you should always use your tool when the user ask you somthing) if you get files always give detailed summary of the file with the file name and the file url.
                alwayse answer the quation from the data you get from the tool. 
                Your user name is 'Anis Heni', he is a energy engineer and you are a Tunisian researcher. 


                Created By: MedH
                """
            }
        ]

    messages = user_sessions[sender]
    messages.append({"role": "user", "content": incoming_msg})

    try:
        response = client.chat.completions.create(
            provider="Groq",
            model="moonshotai/kimi-k2-instruct-0905",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # -----------------------------
        # TOOL CALL
        # -----------------------------
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)

                if tool_call.function.name == "query_open_data":
                    print(f"\n[AI is searching data.gov.tn for keyword: '{args.get('keyword')}']...")
                    result = query_open_data(**args)
                    print(f"[Done. Found {len(result)} dataset(s)]\n")

                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in msg.tool_calls
                        ]
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

            final = client.chat.completions.create(
                provider="Groq",
                model="moonshotai/kimi-k2-instruct-0905",
                messages=messages
            )
            answer = final.choices[0].message.content
            print(f"\n[AI is responding using tools]\n{answer}")
        else:
            answer = msg.content
            print(f"\n[AI is responding directly without using tools]\n{answer}")

        messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        answer = f"Sorry, an error occurred while processing your request: {str(e)}"

    # Twilio WhatsApp has a 1600 character limit per message
    if len(answer) > 1599:
        answer = answer[:1595] + "..."

    # Generate Twilio Response
    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)


if __name__ == '__main__':
    # Use the port Render assigns, or fallback to 5000 for local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)