from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

# --- Shared Models ---

class DocumentMetadata(BaseModel):
    source: str
    section: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    score: Optional[float] = None  # For retrieval scores

class DocumentChunk(BaseModel):
    page_content: str
    metadata: DocumentMetadata

# --- Request Models ---

class IngestRequest(BaseModel):
    text: str = Field(..., description="The raw text content to ingest.")
    filename: str = Field(..., description="Name of the source file (e.g., 'contract_v1.md').")
    document_type: Optional[str] = Field("general", description="Category of doc (e.g., 'report', 'email').")

class QueryRequest(BaseModel):
    question: str = Field(..., description="User query to process.")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (Dense + Sparse).")

# --- Response Models ---

class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    total_documents_in_index: int

class QueryResponse(BaseModel):
    question: str
    answer: str
    context_used: List[DocumentChunk]
    latency_seconds: Optional[float] = None