"""
Central project application, can be ran locally using inputs from console or
on the AURA with inputs from Voice to Text module.
Code for this application based on LocalLLM with RAG project:
https://github.com/amscotti/local-LLM-with-RAG?tab=readme-ov-file
"""

import argparse
import sys
from langchain_ollama import ChatOllama

#pull funcs from local files
from model import PullModel
from database_bridge import PullDocuments
from llm import GetChatHistory
from lightrag import LightRAG
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, DEFAULT_DOCS_PATH

def main(model_name: str, embedding_model: str, doc_path: str) -> None:
    try:
        PullModel(model_name)
        PullModel(embedding_model)
    except Exception as e:
        print(f"couldnt pull model: {e}")
        sys.exit(1)
    
    try:
        db = PullDocuments(embedding_model, doc_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    llm = ChatOllama(model=model_name)
    chat = GetChatHistory(llm, db)

    lightrag = LightRAG(llm=llm, db=db, retriever_k=8)

    while True:
        try:
            user_input = input("\n\nPlease ask a question relating to a Lab in TAMU's ECEN 214 Lab or press 'q' to end: ").strip()
            if user_input.lower == "q":
                break
            if user_input.lower().startswith("rag:"):
                q = user_input[len("rag:"):].strip()
                out = lightrag.generate(q)
                print("\n=== LightRAG ANSWER ===\n")
                print(out["answer"])
                print("\n--- Evidence / overlap scores ---")
                for e in out["evidence"]:
                    print(e)
                print("\n--- Sources used ---")
                for s in out["docs_used"]:
                    print(s)
            else:
                chat(user_input)
        except KeyboardInterrupt:
            break

def parse() -> argparse.Namespace:
    parseHolder = argparse.ArgumentParser(description="Run LLM using Ollama & LightRAG.")
    parseHolder.add_argument(
        '-m', '--model', default=DEFAULT_MODEL, help="The name of the LLM to use (Ollama model)"
    )
    parseHolder.add_argument(
        '-e', '--embedding_model', default=DEFAULT_EMBEDDING_MODEL, help="name of embedding model (Ollama embeddings)."
    )
    parseHolder.add_argument(
        '-p', '--path', default=DEFAULT_DOCS_PATH, help="Path to the directory containing documents to be loaded."
    )

    return parseHolder.parse_args()

if __name__ == "__main__":
    arg = parse()
    main(arg.model, arg.embedding_model, arg.path)