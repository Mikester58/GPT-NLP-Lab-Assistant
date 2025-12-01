"""
Configuration file for the AURA project.
This file contains any prompts, settings, & constants to be used across the app.
"""

from langchain_core.prompts import PromptTemplate

############################
###  Formatting Prompts  ###
############################

DEFAULT_DOC_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)

#For standard RAG
ANSWER_PROMPT = """You are a helpful lab assistant for Electrical Engineering labs.
Use the context below to answer the question accurately.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

#LightRAG vectorization requests a different type of prompting
LIGHTRAG_PROMPT = """You are a lab assistant for Electrical Engineering labs.
Analyze the evidence below and provide a comprehensive answer.
If there are any equations in the answer please state them in the form which they appear.
Reference evidence numbers when relevant (e.g., "According to Evidence 1...").

{evidence}

Question: {question}

Answer:"""

#####################################
###  Database & Storage Settings  ###
#####################################

#Chunking for RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#Adjust based on how many docs each should retrieve
RETRIEVER_K = 5
LIGHTRAG_K = 4

#Models Used
DEFAULT_MODEL = "llama3.2:1b"  # Use Llama3.2 3B model per Vishuam, ensure the parameters
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

#Make sure these always align with folders in local/remote DB
DEFAULT_DOCS_PATH = "ECEN_214_Docs"
STORAGE_DIR = "storage"
CHROMA_DIR = "storage/chroma"
SESSIONS_DIR = "storage/sessions"