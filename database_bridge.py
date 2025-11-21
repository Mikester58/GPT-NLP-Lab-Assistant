"""
File to handle connecting documents from database to LightRAG
system on-device. Will also log sessions to be remotely stored & accessed
later for feedback/record keeping.

Provides:
- CombineDocuments(docs) -> str
- PullDocuments(documentPath) -> List[Document]
- PushDocuments(model_name, documentPath, reload=False) -> Chroma
- SaveSession(session_data, session_id=None) -> str (outputs file name)
- PullSession(session_id=None) -> dict | list[dict]
- ListSessions() -> list[str]
"""

import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

from langchain_core.documents import Document
from langchain_core.prompts import format_document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_DOC_PROMPT, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, STORAGE_DIR

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def CombineDocuments(docs: List[Document]) -> str:
    """Combine documents into single string for context"""
    formatted = [format_document(doc, DEFAULT_DOC_PROMPT) for doc in docs]
    return "\n\n".join(formatted)


def LoadDocuments(path: str) -> List[Document]:
    """Load documents from directory"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    loaders = {
        ".pdf": DirectoryLoader(
            path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        ),
        ".txt": DirectoryLoader(
            path, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            show_progress=True
        ),
        ".md": DirectoryLoader(
            path, 
            glob="**/*.md", 
            loader_cls=TextLoader,
            show_progress=True
        )
    }

    docs = []
    for file_type, loader in loaders.items():
        try:
            loaded = loader.load()
            docs.extend(loaded)
            if loaded:
                print(f"Loaded {len(loaded)} {file_type} files")
        except Exception as e:
            print(f"Warning: Could not load {file_type} files: {e}")

    if not docs:
        print(f"Warning: No documents found in {path}")
    
    return docs


def InitializeDatabase(embedding_model: str, docs_path: str, force_reload: bool = False) -> Chroma:
    """Initialize or load vector database"""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    if force_reload or not os.listdir(CHROMA_DIR):
        print("Building vector database from documents...")
        
        raw_docs = LoadDocuments(docs_path)
        if not raw_docs:
            raise ValueError(f"No documents found in {docs_path}")
        
        chunks = SPLITTER.split_documents(raw_docs)
        print(f"Created {len(chunks)} text chunks")
        print("Generating embeddings (this may take a few minutes)...")
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print("Database created and saved")
        return db
    else:
        print(f"Loading existing database from {CHROMA_DIR}")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )