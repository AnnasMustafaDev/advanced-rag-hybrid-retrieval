import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List
from app.config import settings

class HybridRetriever:
    def __init__(self, documents: List[Document] = None):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.documents = documents
        self.vector_store = None
        self.bm25 = None
        
        if documents:
            self.build_indices(documents)

    def build_indices(self, documents: List[Document]):
        """Builds both Dense (FAISS) and Sparse (BM25) indices."""
        print(f"Indexing {len(documents)} chunks...")
        
        # 1. Dense Index (FAISS)
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # 2. Sparse Index (BM25)
        tokenized_corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.documents = documents  # Store reference for BM25 retrieval

    def _sparse_search(self, query: str, k: int) -> List[Document]:
        tokenized_query = query.split()
        # Get top-k scores
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_n_indices]

    def _dense_search(self, query: str, k: int) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Document]:
        """
        Weighted Hybrid Search.
        Alpha relates to vector weight. 1.0 = Pure Vector, 0.0 = Pure Keyword.
        """
        # Get Dense Results
        dense_docs = self.vector_store.similarity_search_with_score(query, k=k)
        # Normalize Dense Scores (L2 distance, lower is better -> invert for similarity)
        # Note: This is a simplified normalization for demonstration
        
        # Get Sparse Results
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(sparse_scores)[::-1][:k]
        
        # Reciprocal Rank Fusion (RRF) or Simple Weighted Map
        # Here we use a deduplicated list approach for simplicity in the showcase
        
        combined_results = {}
        
        for doc, score in dense_docs:
            combined_results[doc.page_content] = {"doc": doc, "score": (1 / (score + 1)) * alpha} # approximate conversion
            
        for idx in top_indices:
            doc = self.documents[idx]
            current_score = combined_results.get(doc.page_content, {"score": 0})["score"]
            # Add sparse contribution
            combined_results[doc.page_content] = {
                "doc": doc, 
                "score": current_score + (sparse_scores[idx] * (1 - alpha))
            }
            
        # Sort by final score
        sorted_docs = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:k]]