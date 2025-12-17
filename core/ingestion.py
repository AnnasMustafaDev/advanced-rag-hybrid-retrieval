from typing import List, Dict
from langchain_core.documents import Document
from core.chunking import AdvancedChunker
from core.retriever import HybridRetriever

class IngestionPipeline:
    def __init__(self, retriever: HybridRetriever):
        """
        The pipeline needs access to the retriever to update the index 
        after processing new documents.
        """
        self.chunker = AdvancedChunker()
        self.retriever = retriever

    def process_document(self, text: str, filename: str, doc_type: str = "general") -> Dict:
        """
        Full ingestion flow:
        1. Prepare metadata
        2. Chunk text (Structure-Aware)
        3. Add chunks to Retrieval System
        """
        
        # 1. Prepare Base Metadata
        base_metadata = {
            "source": filename,
            "doc_type": doc_type,
            "processed": True
        }

        # 2. Advanced Chunking
        # We use structure_aware_chunking to respect headers
        chunks = self.chunker.structure_aware_chunking(text, base_metadata)
        
        print(f"Generated {len(chunks)} chunks from {filename}")

        # 3. Update Retriever Indices
        # In a production DB (Pinecone/Weaviate), you would just .add_documents()
        # Since we use local FAISS/BM25 in memory, we might need to rebuild or extend.
        
        # We assume the retriever has a method to add documents dynamically.
        # If using the 'simple' list approach from the showcase, we extend the list and rebuild.
        
        if self.retriever.documents is None:
            self.retriever.documents = []
            
        self.retriever.documents.extend(chunks)
        
        # Re-build indices to include new data (Dense + Sparse)
        # Note: In massive production apps, you wouldn't rebuild the whole index every time.
        self.retriever.build_indices(self.retriever.documents)

        return {
            "chunks_count": len(chunks),
            "status": "success"
        }

    def process_batch(self, files: List[Dict]):
        """
        Handle multiple files at once.
        files = [{'text': '...', 'filename': '...'}, ...]
        """
        total_chunks = 0
        for file in files:
            result = self.process_document(file['text'], file['filename'])
            total_chunks += result['chunks_count']
            
        return total_chunks