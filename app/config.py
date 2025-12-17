import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Paths
    DATA_DIR = "data"
    FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
    DOCS_PATH = os.path.join(DATA_DIR, "documents")
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    
    # RAG Parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 5
    
    # Hybrid Search Weights (Alpha: 1.0 = Dense only, 0.0 = Sparse only)
    HYBRID_ALPHA = 0.7

settings = Config()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.DOCS_PATH, exist_ok=True)