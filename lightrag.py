""""
Retrieval augmented Generation system based on LightRAG paper:
https://arxiv.org/abs/2410.05779

Provides:
- LightRAG Class

This is a fairly simplified implementation of the above research paper + github focusing on
Enhancing retrieval via scoring, simple reranking based on relevance scoring, Evidence-based answer
generation, & overlap scoring for transparency.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from config import LIGHTRAG_DIR, LIGHTRAG_K, LIGHTRAG_PROMPT

class LightRAG:
    """
    Lightweight RAG grooming wrapper:
    - retrieve top-K documents via db retriever
    - simple rerank (heuristic on score + doc length)
    - assemble evidence-first prompt and call llm
    - compute cheap overlap evidence scores and return structured output
    """
    
    