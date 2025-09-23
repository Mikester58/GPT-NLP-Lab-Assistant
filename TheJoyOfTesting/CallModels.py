import requests
import json
import os

# To create a requirements file:
# pip freeze > requirements.txt
#pip install openai faiss-cpu sentence-transformers

API_KEY = os.getenv("TAMU_API_KEY")
# $env:TAMU_API_KEY="<apikey>"

def call_models_api():
    url = 'https://chat-api.tamu.ai/openai/models'

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"  # Replace with your API key
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an exception for bad status codes
    
    return response.json()

# Usage
try:
    result = call_models_api()
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error calling API: {e}")