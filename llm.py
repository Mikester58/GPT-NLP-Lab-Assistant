"""
File to handle the Large Language Model being ran on-device.

Provides:
- GetChatHistory(llm, db, session_id) -> Callable
- GetStreamHistory(llm, db, session_id) -> Callable
- ClearMemory(session_id) -> None
- GetMemoryMessages(session_id) -> List[BaseMessage]
- ListSessions -> List[str]
- GetSessionHistory(session_id) -> BaseChatMessageHistory
"""
from typing import Dict, List
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from database_bridge import CombineDocuments
from config import ANSWER_PROMPT_TMP, DEFAULT_DOC_PROMPT, RETRIEVER_K

#Store for conversation history
store: Dict[str, InMemoryChatMessageHistory] = {}

def GetChatHistory(llm, db, session_id: str = "default"):
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    prompt = ChatPromptTemplate.from_messages([("system", ANSWER_PROMPT_TMP), MessagesPlaceholder(variable_name="chat_history"), ("human", "{question}")])
    
    #define retrieval
    def format_docs_with_context(question):
        docs = retriever.invoke(question)
        return CombineDocuments(docs, DEFAULT_DOC_PROMPT)
    
    chain = RunnablePassthrough.assign(context=lambda x: format_docs_with_context(x["question"])) | prompt | llm | StrOutputParser()
    
    #Build message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        GetSessionHistory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    
    def chat(user_input: str) -> None:
        result = chain_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(result)
        
        #Provide retrieved docs ranked
        docs = retriever.invoke(user_input)
        for i, doc in enumerate(docs[:3], 1):  #Show top 3 docs
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"{i}. {source} (Page {page})")
    return chat

def GetStreamHistory(llm, db, session_id: str = "default"):
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    #Build prompt w/ chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_PROMPT_TMP),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    #build retriever
    def format_docs_with_context(question):
        docs = retriever.invoke(question)
        return CombineDocuments(docs, DEFAULT_DOC_PROMPT)
    chain = RunnablePassthrough.assign(context=lambda x: format_docs_with_context(x["question"])) | prompt | llm
    chain_with_history = RunnableWithMessageHistory(chain, GetSessionHistory, input_messages_key="question", history_messages_key="chat_history")
    
    def stream_chat(user_input: str) -> str:
        full_response = ""
        
        for chunk in chain_with_history.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        ):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += content
        
        return full_response
    
    return stream_chat

#Helper functions

#Clear conversation memory for a specific session
def ClearMemory(session_id: str = "default") -> None:
    if session_id in store:
        store[session_id].clear()
    return None

#Return messages from conversation session
def GetMemoryMessages(session_id: str = "default") -> List[BaseMessage]:
    if session_id in store:
        return store[session_id].messages
    else:
        return []

#Return a list of active session IDs
def ListSessions() -> List[str]:
    return list(store.keys())

#Return/Create & Return a chat message history for a given chat
def GetSessionHistory(session_id: str) -> BaseChatMessageHistory:
    if session_id in store:
        return store[session_id]
    else:
        store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]