

import os
import json
import requests

WHITE_AGENT_URL = os.getenv("WHITE_AGENT_URL", "http://localhost:9002/")

def ask_white_agent(question: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": "ui-1",
        "method": "agent:perform",
        "params": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "name": "question",
                            "value": question,
                        }
                    ],
                }
            ]
        },
    }

    try:
        resp = requests.post(WHITE_AGENT_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return f"[UI ERROR] Cannot reach White Agent: {e}"

    try:
        data = resp.json()
    except Exception:
        return "White Agent returned non-JSON\n" + resp.text

    events = data.get("result", {}).get("events") or []
    output = []

    for ev in events:
        if ev.get("type") == "artifact":
            art = ev.get("artifact", {})
            if art.get("media_type") in ("text/plain", "text/markdown"):
                output.append(art.get("text", ""))

    if output:
        return "\n\n".join(output)

    return json.dumps(data, indent=2)
