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
from config import LIGHTRAG_K, LIGHTRAG_PROMPT

class LightRAG:
    """
    Lightweight RAG wrapper enhancing standard RAG with:
    - retrieve top-K documents via db retriever
    - simple rerank (heuristic on score + doc length)
    - assemble evidence-first prompt and call llm
    - compute cheap overlap evidence scores and return structured output
    """

    def __init__(self, llm, db, retriever_k: int = LIGHTRAG_K):
        self.llm = llm
        self.db = db  #Chroma Vector
        self.retriever_k = retriever_k
        self.retriever = db.as_retriever(
            search_type = "similarity_score_threshould",
            search_kwargs = {
                'k': retriever_k
                'score_threshould': 0.0 #will filter myself
            }
        )
    
    def RetrieveDocs(self, query: str) -> List[tuple[Document, float]]:
        """
        Retrieve top K docs based on user query
        """
        result = self.db.similarity_search_with_relevance_scores(query, k=self.retriever_k)
        return result
    
    def RerankDocs(self, scoredDocs: List[tuple[Document, float]], query:str) -> List[tuple[Document, float]]:

    
    def BuildEvidencePrompt(self, query: str, docs_with_scores: List[tuple[Document, float]]) -> str:

    
    def ComputeOverlapScores(self, answer: str, docs_with_scores: List[tuple[Document, float]]) -> List[Dict[str, Any]]:

    
    def generate(self, query: str) -> Dict[str, Any]:
        

    def query(self, question: str) -> str:
        result = self.generate(question)
        return result
