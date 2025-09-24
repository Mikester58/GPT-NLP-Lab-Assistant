import os
import requests
import sys

API_KEY = os.getenv("TAMU_API_KEY")
API_ENDPOINT = os.getenv("TAMU_API_ENDPOINT", "https://chat-api.tamu.ai")
# $env:TAMU_API_KEY="<apikey>"
# $env:TAMU_API_ENDPOINT="https://chat-api.tamu.ai"

if not API_KEY:
    print("Invalid API key/key wasnt set properly")
    sys.exit(1)

headers = {"Authorization": f"Bearer {API_KEY}"}

chat_url = f"{API_ENDPOINT}/api/chat/completions"
body = {
    "model": "protected.llama3.2",
    "stream": False,
    "messages": [
        {"role": "user", "content": "What are florida beaches"}
    ],
}

chat_headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

resp2 = requests.post(chat_url, headers=chat_headers, json=body, timeout=30)

print("\n=== Chat Completion ===")
print("Status:", resp2.status_code)
print(resp2.json()["choices"][0]["message"]["content"])