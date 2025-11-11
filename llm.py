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
    

def GetStreamHistory(llm, db, session_id: str = "default"):


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