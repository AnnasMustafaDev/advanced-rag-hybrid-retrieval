from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict

class AdvancedChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def semantic_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Standard recursive chunking with metadata preservation.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.create_documents([text], metadatas=[metadata])
        return chunks

    def structure_aware_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Splits by Markdown headers first, then recursively chunks.
        Ideal for technical documentation or structured reports.
        """
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)
        
        # Further split efficiently using recursive splitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        final_chunks = recursive_splitter.split_documents(md_header_splits)
        
        # Merge original metadata
        for chunk in final_chunks:
            chunk.metadata.update(metadata)
            
        return final_chunks