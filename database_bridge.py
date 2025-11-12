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
from langchain.schema import format_document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DEFAULT_DOC_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, SESSIONS_DIR, STORAGE_DIR

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def CombineDocuments(docs: List[Document], document_prompt=DEFAULT_DOC_PROMPT, document_separator="\n\n") -> str:
    """
    Formats and concatenates a list of documents into a single string for LLM input.
    """
    formatted = [
        format_document(doc, document_prompt) 
        for doc in docs
    ]
    return document_separator.join(formatted)
    
def PullDocuments(documentPath: str) -> List[Document]:
    #Load from documentPath with support for various file types.
    #returns a list of langchain_core.documents.Document objects.
    if not os.path.exists(documentPath):
        raise FileNotFoundError(f"File path '{documentPath}' does not exist.")
    
    loaders = { #WE love json
        ".pdf": DirectoryLoader(documentPath, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True),
        ".md": DirectoryLoader(documentPath, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        ".txt": DirectoryLoader(documentPath, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        ".docx": DirectoryLoader(documentPath, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    }

    docs: List[Document] = []
    for file_type, loader in loaders.items():
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"loader for {file_type} failed: {e} (non-fatal)")

    return docs

def PushDocuments(model_name: str, documentPath: str, reload: bool = False) -> Chroma:
    #Load docs, split them, build embeddings & persist a chroma vectorstore to PERSIST_DIRECTORY.

    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if reload:
        print("Reload requested")

        rawDocs = PullDocuments(documentPath=documentPath)
        docs = SPLITTER.split_documents(rawDocs)
        print("Building embeddings...")

        db = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model=model_name), persist_directory=CHROMA_DIR)
        return db
    else: #either create a persist or return the chroma instance
        if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            print(f"Using existing persisted Chroma at {CHROMA_DIR}")
            return Chroma(embedding_function=OllamaEmbeddings(model=model_name), persist_directory=CHROMA_DIR)
        else:
            print("No existing persisted Chroma, building from docs now...")
            rawDocs = PullDocuments(documentPath=documentPath)
            docs = SPLITTER.split_documents(rawDocs)
            db = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model=model_name), persist_directory=CHROMA_DIR)
            return db
        

def SaveSession(session_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    #build id if not already
    if session_id is None:
        session_id = str(uuid4())
    
    #timestamp session & add relevant data
    session_data['timestamp'] = datetime.now().isoformat()
    session_data['session_id'] = session_id
    
    filename = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    
    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print(f"Session saved: {filename}")
    return session_id


def PullSession(session_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if session_id is None:
        # Return all sessions
        sessions = []
        if os.path.exists(SESSIONS_DIR):
            for filename in os.listdir(SESSIONS_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(SESSIONS_DIR, filename)
                    with open(filepath, 'r') as f:
                        sessions.append(json.load(f))
        return sessions
    else:
        # Return specific session
        filename = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Session {session_id} not found")
        
        with open(filename, 'r') as f:
            return json.load(f)


def ListSessions() -> List[str]:
    #If no sessions found dont list anything
    if not os.path.exists(SESSIONS_DIR):
        return []
    
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith('.json'):
            sessions.append(filename[:-5])  #Remove .json extension
    
    return sorted(sessions)