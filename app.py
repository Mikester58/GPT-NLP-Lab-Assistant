"""
Central project application, can be ran locally using inputs from console or
on the AURA with inputs from Voice to Text module.
"""

import argparse
import sys
from langchain_ollama import ChatOllama

#pull funcs from local files
from model import PullModel
from database_bridge import PullDocuments
from llm import GetChatHistory

def main(model_name: str, embedding_model: str, doc_path: str) -> None:
    db = PullDocuments(embedding_model, doc_path)
    llm = ChatOllama(model=model_name)
    chat = GetChatHistory(llm, PullModel)

    while True:
        try:
            user_input = input("\n\nPlease ask a question relating to a Lab in TAMU's ECEN 214 Lab or press 'q' to end: ").strip()
            if user_input.lower == "q":
                break
            else:
                chat(user_input)
        except KeyboardInterrupt:
            break

def parse() -> argparse.Namespace:
    parseHolder = argparse.ArgumentParser()

    return parseHolder

if __name__ == "__main__":
    arg = parse()
    main(arg.model, arg.embedding_model, arg.path)