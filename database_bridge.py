"""
File to handle connecting documents from database to LightRAG
system on-device. Will also log sessions to be remotely stored & accessed
later for feedback/record keeping.
- CombineDocuments(docs) -> str
- PullDocuments(documentPath) -> List[Document]
- PushDocuments(model_name, documentPath, reload=False) -> Chroma
- SaveSession(session_data, session_id=None) -> str (outputs file name)
- PullSession(session_id=None) -> dict | list[dict]
- ListSessions() -> list[str]
"""

import os
import json
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from uuid import uuid4

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
PERSISTENT_DIRECTORY = os.path.join("storage", "chroma")
SESSIONS_DIRECTORY = os.path.join("storage", "sessions")

def CombineDocuments(docs: List[Document], seperator: str = "\n\n") -> str:
    #Needed for storing our precombine caches
    pieces = []
    for doc in docs:
        content = getattr(doc, "page_content", "")
        if hasattr(doc, "metadata"):
            src = doc.metadata.get("source")
            pieces.append(f"Source: {src}\n{content}")
        else:
            src = None
            pieces.append(content)
    return seperator.join(pieces)
    
def PullDocuments(documentPath: str) -> List[Document]:
    if not os.path.exists(documentPath):
        raise FileNotFoundError(f"File path '{documentPath}' does not exist.")
    
    loaders = { #WE love json
        ".pdf": DirectoryLoader(documentPath, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True),
        ".md": DirectoryLoader(documentPath, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        ".txt": DirectoryLoader(documentPath, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        ".docx": DirectoryLoader(documentPath, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    }


def PushDocuments(model_name: str, documentPath: str, reload: bool = False) -> Chroma:



def SaveSession():


def PullSession():

