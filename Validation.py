#Using the A&M API key, build an application to benchmark the LLM.
import os
import requests
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

#Need to build evaluation suite at soonish date (IE before weekend)

#APIkey suite
API_KEY = os.getenv("TAMU_API_KEY")
API_ENDPOINT = os.getenv("TAMU_API_ENDPOINT", "https://chat-api.tamu.ai")
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-key"
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tamu-llm-validation"

client = Client()

#Using traceable
@traceable(name="tamu_chat_completion")
def call_tamu_api(messages, model="protected.llama3.2", temperature=0.7):
    """Call TAMU API with LangSmith tracing"""
    chat_url = f"{API_ENDPOINT}/api/chat/completions"
    
    body = {"model": model, "stream": False, "messages": messages, "temperature": temperature}
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(chat_url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        response_data = resp.json()
        return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

#Wrapper for evaluation (takes dict input/output)
def tamu_application(inputs: dict) -> dict:
    """Application wrapper for LangSmith evaluation"""
    messages = inputs.get("messages", [])
    model = inputs.get("model", "protected.llama3.2")
    
    response = call_tamu_api(messages, model)
    return {"answer": response}
