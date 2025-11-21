"""
Streamlit UI for ECEN 214 Lab Assistant
Simple interface for local Jetson operation
"""

import streamlit as st
import os
from langchain_ollama import ChatOllama

from database_bridge import InitializeDatabase
from llm import BuildChain, GetSession
from lightrag import LightRAG
from model import GetListOfModels
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_DOCS_PATH, DEFAULT_MODEL

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from database_bridge import CombineDocuments
from config import ANSWER_PROMPT

st.set_page_config(
    page_title="ECEN 214 Lab Assistant",
    layout="centered"
)

st.title("ECEN 214 Lab Assistant")

######################
### SIDEBAR CONFIG ###
######################

with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    if "models" not in st.session_state:
        try:
            st.session_state.models = GetListOfModels()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.session_state.models = []
    
    if st.session_state.models:
        try:
            default_index = st.session_state.models.index(DEFAULT_MODEL)
        except ValueError:
            default_index = 0
        
        selected_model = st.selectbox(
            "Model",
            st.session_state.models,
            index=default_index
        )
    else:
        st.error("No models found. Ensure Ollama is running.")
        st.info("Run: ollama pull llama3.2:1b")
        selected_model = DEFAULT_MODEL
    
    # Update model if changed
    if st.session_state.get("current_model") != selected_model:
        st.session_state.current_model = selected_model
        st.session_state.llm = ChatOllama(model=selected_model)
    
    st.divider()
    
    # Document settings
    docs_path = st.text_input("Documents Path", DEFAULT_DOCS_PATH)
    
    if os.path.isdir(docs_path):
        doc_count = len([f for f in os.listdir(docs_path) 
                        if f.endswith(('.pdf', '.txt', '.md', '.docx'))])
        st.info(f"{doc_count} documents found")
    
    if st.button("Index Documents", use_container_width=True):
        if os.path.isdir(docs_path):
            with st.spinner("Indexing documents..."):
                try:
                    st.session_state.db = InitializeDatabase(
                        DEFAULT_EMBEDDING_MODEL, 
                        docs_path,
                        force_reload=True
                    )
                    st.success("Documents indexed")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Invalid path")
    
    st.divider()
    
    # Query mode
    query_mode = st.radio(
        "Mode",
        ["Normal", "Enhanced (LightRAG)"]
    )
    
    st.divider()
    
    # Session controls
    session_id = st.text_input("Session ID", "main")
    
    if st.button("Clear History", use_container_width=True):
        session = GetSession(session_id)
        session.clear()
        st.session_state.messages = []
        st.success("History cleared")
        st.rerun()

# Initialize session state
if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model=selected_model)

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main interface
if st.session_state.db is None:
    st.warning("Please index documents first")
    st.info("1. Set documents path in sidebar\n2. Click 'Index Documents'\n3. Start asking questions")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.text(source)
    
    # Chat input
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                if query_mode == "Normal":
                    # Build chain and get response
                    chain = BuildChain(st.session_state.llm, st.session_state.db, session_id)
                    
                    # Get retriever for sources
                    from config import RETRIEVER_K
                    retriever = st.session_state.db.as_retriever(
                        search_kwargs={"k": RETRIEVER_K}
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", ANSWER_PROMPT),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])
                    
                    def get_context(question):
                        docs = retriever.invoke(question)
                        return CombineDocuments(docs)
                    
                    chain = (
                        RunnablePassthrough.assign(
                            context=lambda x: get_context(x["question"])
                        )
                        | prompt_template
                        | st.session_state.llm
                    )
                    
                    chain_with_history = RunnableWithMessageHistory(
                        chain,
                        GetSession,
                        input_messages_key="question",
                        history_messages_key="history"
                    )
                    
                    # Stream and display
                    response_text = ""
                    response_placeholder = st.empty()
                    
                    for chunk in chain_with_history.stream(
                        {"question": prompt},
                        config={"configurable": {"session_id": session_id}}
                    ):
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        response_text += content
                        response_placeholder.write(response_text)
                    
                    # Get sources
                    docs = retriever.invoke(prompt)
                    sources = [
                        f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})"
                        for doc in docs[:3]
                    ]
                    
                    with st.expander("Sources"):
                        for source in sources:
                            st.text(source)
                    
                else:
                    # LightRAG mode
                    if "lightrag" not in st.session_state:
                        st.session_state.lightrag = LightRAG(
                            st.session_state.llm,
                            st.session_state.db
                        )
                    
                    with st.spinner("Generating enhanced response..."):
                        result = st.session_state.lightrag.generate(prompt)
                        response_text = result["answer"]
                        
                        st.write(response_text)
                        
                        with st.expander("Evidence"):
                            for ev in result["evidence"][:3]:
                                st.write(f"[{ev['id']}] {ev['source']} (Page {ev['page']})")
                                st.write(f"Retrieval: {ev['retrieval_score']:.3f} | Overlap: {ev['overlap_score']:.3f}")
                                st.write("")
                        
                        sources = result["sources"]
                        with st.expander("Sources"):
                            for source in sources:
                                st.text(source)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error: {e}")
                response_text = "An error occurred processing your question."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": []
                })