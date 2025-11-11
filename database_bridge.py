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
from typing import List
from config import DEFAULT_DOC_PROMPT

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
PERSISTENT_DIRECTORY = os.path.join("storage", "chroma")
SESSIONS_DIRECTORY = os.path.join("storage", "sessions")

def CombineDocuments(docs: List[Document], document_prompt=DEFAULT_DOC_PROMPT, document_separator="\n\n") -> str:
    """
    Formats and concatenates a list of documents into a single string for LLM input.
    """
    formatted = [document_prompt.format(source=doc.metadata.get("source", "Unknown"),
                                        page=doc.metadata.get("page", "?"),
                                        page_content=doc.page_content)
                 for doc in docs]
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

    os.makedirs("storage", exist_ok=True)
    os.makedirs(PERSISTENT_DIRECTORY, exist_ok=True)

    if reload:
        print("Reload requested")

        rawDocs = PullDocuments(documentPath=documentPath)
        docs = SPLITTER.split_documents(rawDocs)
        print("Building embeddings...")

        db = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model=model_name), persist_directory=PERSISTENT_DIRECTORY)
        return db
    else: #either create a persist or return the chroma instance
        if os.path.isdir(PERSISTENT_DIRECTORY) & os.listdir(PERSISTENT_DIRECTORY):
            print(f"Using existing persisted Chroma at {PERSISTENT_DIRECTORY}")
            return Chroma(embedding_function=OllamaEmbeddings(model=model_name), persist_directory=PERSISTENT_DIRECTORY)
        else:
            print("No existing persisted Chroma, building from docs now...")
            rawDocs = PullDocuments(documentPath=documentPath)
            docs = SPLITTER.split_documents(rawDocs)
            db = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model=model_name), persist_directory=PERSISTENT_DIRECTORY)
            return db
        

#def PushSession(session_name: str, history: List[Dict[str, Any]]) -> None:


#def PullSession(session_name: str) -> Dict[str, Any]:

