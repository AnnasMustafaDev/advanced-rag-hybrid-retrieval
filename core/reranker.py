from typing import List
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes a Cross-Encoder for high-precision reranking.
        This model effectively acts as a 'judge' comparing query vs document directly.
        """
        self.model = HuggingFaceCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        if not documents:
            return []
            
        # Create pairs of (query, doc_content)
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N documents
        return [doc for doc, score in doc_score_pairs[:top_n]]