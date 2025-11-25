"""
Streamlit UI for lab assistant demo
run using python -m streamlit run ui.py
"""
import streamlit as st
import os
from langchain_ollama import ChatOllama

from database_bridge import InitializeDatabase, SaveSession, ListSessions, CombineDocuments
from llm import GetSession, BuildChain, ClearSession
from lightrag import LightRAG
from model import GetListOfModels
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_DOCS_PATH, DEFAULT_MODEL, CHROMA_DIR, ANSWER_PROMPT, RETRIEVER_K

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

st.title("AURA")

st.set_page_config(page_title="AURA", layout="centered")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        try:
            st.session_state.db = InitializeDatabase(
                DEFAULT_EMBEDDING_MODEL,
                DEFAULT_DOCS_PATH,
                force_reload=False
            )
        except:
            st.session_state.db = None
    else:
        st.session_state.db = None

if "llm" not in st.session_state:
    st.session_state.llm = ChatOllama(model=DEFAULT_MODEL)
    st.session_state.current_model = DEFAULT_MODEL

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    models = GetListOfModels()
    if models:
        selected_model = st.selectbox("Model", models)
    else:
        st.warning("No models found")
        selected_model = DEFAULT_MODEL
    
    if st.session_state.get("current_model") != selected_model:
        st.session_state.current_model = selected_model
        st.session_state.llm = ChatOllama(model=selected_model)
    
    st.divider()
    
    # Documents
    docs_path = st.text_input("Documents Path", DEFAULT_DOCS_PATH)
    
    if st.button("Index Documents"):
        if os.path.isdir(docs_path):
            with st.spinner("Indexing..."):
                try:
                    st.session_state.db = InitializeDatabase(
                        DEFAULT_EMBEDDING_MODEL, 
                        docs_path,
                        force_reload=True
                    )
                    st.success("Indexed")
                except Exception as e:
                    st.error(str(e))
        else:
            st.error("Invalid path")
    
    st.divider()
    
    # Mode
    query_mode = st.radio("Mode", ["Normal", "Enhanced"])
    
    st.divider()
    
    # Session management
    session_id = st.text_input("Session ID", "main")
    
    # Show existing sessions
    saved_sessions = ListSessions()
    if saved_sessions:
        with st.expander(f"Saved Sessions ({len(saved_sessions)})"):
            for sess in saved_sessions:
                st.text(sess)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear"):
            ClearSession(session_id)
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Save"):
            try:
                session = GetSession(session_id)
                messages = session.messages
                
                if messages:
                    session_data = {
                        "messages": [
                            {
                                "role": getattr(msg, "type", "unknown"),
                                "content": getattr(msg, "content", "")
                            }
                            for msg in messages
                        ]
                    }
                    
                    saved_id = SaveSession(session_data, session_id)
                    st.success(f"Saved: {saved_id}")
                else:
                    st.warning("No messages")
            except Exception as e:
                st.error(str(e))

# Main chat
if st.session_state.db is None:
    st.warning("Index documents first")
else:
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.text(s)
    
    # Input
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                sources = []
                
                if query_mode == "Normal":
                    
                    retriever = st.session_state.db.as_retriever(
                        search_kwargs={"k": RETRIEVER_K}
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", ANSWER_PROMPT),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])
                    
                    def get_context(q):
                        docs = retriever.invoke(q)
                        return CombineDocuments(docs)
                    
                    chain = (
                        RunnablePassthrough.assign(
                            context=lambda x: get_context(x["question"])
                        )
                        | prompt_template
                        | st.session_state.llm
                        | StrOutputParser()
                    )
                    
                    chain_with_history = RunnableWithMessageHistory(
                        chain,
                        GetSession,
                        input_messages_key="question",
                        history_messages_key="history"
                    )
                    
                    response_text = ""
                    placeholder = st.empty()
                    
                    for chunk in chain_with_history.stream(
                        {"question": prompt},
                        config={"configurable": {"session_id": session_id}}
                    ):
                        if isinstance(chunk, str):
                            response_text += chunk
                        else:
                            response_text += chunk
                        placeholder.write(response_text)
                    
                    docs = retriever.invoke(prompt)
                    sources = [
                        f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', '?')})" 
                        for d in docs[:3]
                    ]
                    
                    with st.expander("Sources"):
                        for s in sources:
                            st.text(s)
                
                else:
                    # Use LightRAG class
                    if "lightrag" not in st.session_state:
                        st.session_state.lightrag = LightRAG(
                            st.session_state.llm,
                            st.session_state.db
                        )
                    
                    result = st.session_state.lightrag.generate(prompt)
                    response_text = result["answer"]
                    
                    st.write(response_text)
                    
                    with st.expander("Evidence"):
                        for ev in result["evidence"][:3]:
                            st.write(f"[{ev['id']}] {ev['source']} (Page {ev['page']})")
                            st.write(f"Retrieval: {ev['retrieval_score']:.3f} | Overlap: {ev['overlap_score']:.3f}")
                    
                    sources = result["sources"]
                    with st.expander("Sources"):
                        for s in sources:
                            st.text(s)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(str(e))
                response_text = "Error occurred"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": []
                })