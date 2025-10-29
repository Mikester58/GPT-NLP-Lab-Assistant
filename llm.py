"""
File to handle the Large Language Model being ran on-device.

Provides:
- GetChatHistory(llm, db) -> Callable
- GetStreamHistory(llm, db) -> Callable
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

store: Dict[str, InMemoryChatMessageHistory] = {}
