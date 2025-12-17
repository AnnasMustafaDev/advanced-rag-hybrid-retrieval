from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from typing import Literal

class EmbeddingService:
    def __init__(self, provider: Literal["openai", "huggingface"] = "openai"):
        self.provider = provider
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the embedding model based on the configuration.
        """
        if self.provider == "openai":
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
        
        elif self.provider == "huggingface":
            # Useful for local/offline usage or cost saving
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_embedding_function(self):
        """Returns the embedding function compatible with LangChain vector stores."""
        return self.model
    
    def embed_query(self, text: str):
        """Directly embed a single query string."""
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]):
        """Directly embed a list of documents."""
        return self.model.embed_documents(texts)