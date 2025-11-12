""""
Retrieval augmented Generation system based on LightRAG paper:
https://arxiv.org/abs/2410.05779

Provides:
- LightRAG Class

This is a fairly simplified implementation of the above research paper + github focusing on
Enhancing retrieval via scoring, simple reranking based on relevance scoring, Evidence-based answer
generation, & overlap scoring for transparency.
"""

from typing import List, Dict, Any
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
            search_type = "similarity_score_threshold",
            search_kwargs = {
                'k': retriever_k,
                'score_threshold': 0.0 #will filter myself
            }
        )
    
    def RetrieveDocs(self, query: str) -> List[tuple[Document, float]]:
        """
        Retrieve top K docs based on user query
        """
        result = self.db.similarity_search_with_relevance_scores(query, k=self.retriever_k)
        return result
    
    def RerankDocs(self, scoredDocs: List[tuple[Document, float]], query:str) -> List[tuple[Document, float]]:
        #Boost scores a little for long docs but not so much that it only prefers
        reranked = []
        
        for doc, score in scoredDocs:
            content_length = len(doc.page_content)
            length_bonus = min(0.1, content_length / 10000 * 0.1) #Normalize
            adjusted_score = score + length_bonus
            reranked.append((doc, adjusted_score))
        
        #sort by descending scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def BuildEvidencePrompt(self, query: str, docs_with_scores: List[tuple[Document, float]]) -> str:
        evidence_sections = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content
            
            evidence_sections.append(
                f"[Evidence {i}] (Relevance: {score:.3f})\n"
                f"Source: {source}, Page: {page}\n"
                f"{content}\n"
            )
        
        evidence_text = "\n---\n".join(evidence_sections)
        prompt = f"""{LIGHTRAG_PROMPT}

                ## Evidence from Lab Documents:
                {evidence_text}

                ## Student Question:
                {query}

                ## Your Answer (cite evidence numbers when relevant):
                """
        return prompt
    
    def ComputeOverlapScores(self, answer: str, docs_with_scores: List[tuple[Document, float]]) -> List[Dict[str, Any]]:
        #word based tokenization
        answer_words = set(answer.lower().split())
        
        evidence_list = []
        
        for i, (doc, retrieval_score) in enumerate(docs_with_scores, 1):
            doc_words = set(doc.page_content.lower().split())
            overlap = len(answer_words & doc_words)
            total = len(answer_words | doc_words)
            overlap_score = overlap / total if total > 0 else 0
            
            evidence_list.append({
                "evidence_id": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "retrieval_score": retrieval_score,
                "overlap_score": overlap_score,
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        #Sort overlap Score
        evidence_list.sort(key=lambda x: x["overlap_score"], reverse=True)
        return evidence_list
    
    def generate(self, query: str) -> Dict[str, Any]:
        
        #retrieve docs
        docs_with_scores = self.RetrieveDocs(query)
        
        if not docs_with_scores:
            return {
                "answer": "I couldn't find any relevant information in the lab documents to answer your question.",
                "evidence": [],
                "docs_used": []
            }
        
        reranked_docs = self.RerankDocs(docs_with_scores, query)
        prompt = self.BuildEvidencePrompt(query, reranked_docs)
        
        #Build answer
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        evidence = self.ComputeOverlapScores(answer, reranked_docs)
        
        #Prep Docs List
        docs_used = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": score
            }
            for doc, score in reranked_docs
        ]
        
        return {
            "answer": answer,
            "evidence": evidence,
            "docs_used": docs_used
        }

    def query(self, question: str) -> str:
        result = self.generate(question)
        return result["answer"]
