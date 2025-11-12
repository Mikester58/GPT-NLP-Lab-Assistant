"""
Configuration file for the AURA project.
This file contains any prompts, settings, & constants to be used across the app.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

############################
###  Formatting Prompts  ###
############################

DEFAULT_DOC_PROMPT = PromptTemplate.from_template(template="Source Document: {source}, Page {page}:\n{page_content}")

CONDENSED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
    """Given the following conversation and a follow-up question, 
    rephrase the follow-up question to be a standalone question that includes 
    relevant context from the chat history. If the question is already standalone, 
    return it as is.

    Chat History:
    {chat_history}

    Follow-up Question: {question}

    Standalone Question:""")
    ])

#For standard RAG
ANSWER_PROMPT_TMP = """You are a helpful AI lab assistant assisting with Lab related questions.
Use the following context from lab documents to answer the question. Be specific and cite sources when possible.

If you cannot answer the question based on the provided context, say so clearly. 
Do not make up information that isn't in the documents.

If you're unable to answer the question, do not list sources.

## Context from Lab Documents:
{context}

## Chat History:
{chat_history}

## Question:
{question}

## Answer:"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_PROMPT_TMP)

# LightRags vectorization requests a different type of prompting
LIGHTRAG_PROMPT = """You are an expert AI assistant assisting with Lab related questions.
Analyze the provided evidence carefully and synthesize a comprehensive answer.

Focus on:
1. Accuracy - Only use information from the provided evidence
2. Clarity - Explain concepts in a student-friendly manner
3. Completeness - Address all parts of the question
4. Citations - Reference which documents you're using

If the evidence doesn't contain enough information, acknowledge the limitations."""

#####################################
###  Database & Storage Settings  ###
#####################################

#Chunking for RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

#Adjust based on how many docs each should retrieve
RETRIEVER_K = 10
LIGHTRAG_K = 8

#Models Used
DEFAULT_MODEL = "mistral" #Use Llama3.2 3B model per Vishuam, ensure the parameters
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

#Make sure these always align with folders in local/remote DB
DEFAULT_DOCS_PATH = "Docs"
STORAGE_DIR = "storage"
CHROMA_DIR = "storage/chroma"
SESSIONS_DIR = "storage/sessions"
LIGHTRAG_DIR = "storage/lightrag"