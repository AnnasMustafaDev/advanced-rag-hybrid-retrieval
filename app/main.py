from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.graph import RAGWorkflow
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.chunking import AdvancedChunker
from langchain_core.documents import Document

app = FastAPI(title="Advanced RAG Showcase")

# --- Global State Simulation ---
# In a real app, this would be a proper vector DB connection
docs_store = []
retriever_instance = None
reranker_instance = Reranker()
rag_workflow = None

class IngestRequest(BaseModel):
    text: str
    filename: str

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global retriever_instance, rag_workflow
    # Initialize with dummy data or load from disk
    dummy_doc = Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", metadata={"source": "manual"})
    docs_store.append(dummy_doc)
    
    retriever_instance = HybridRetriever(docs_store)
    rag_workflow = RAGWorkflow(retriever_instance, reranker_instance)

@app.post("/ingest")
async def ingest_document(payload: IngestRequest):
    global retriever_instance, rag_workflow
    
    chunker = AdvancedChunker()
    # Use structure-aware chunking
    new_chunks = chunker.structure_aware_chunking(payload.text, {"source": payload.filename})
    
    docs_store.extend(new_chunks)
    
    # Rebuild index (simplified for showcase)
    retriever_instance.build_indices(docs_store)
    # Re-initialize workflow with updated retriever
    rag_workflow = RAGWorkflow(retriever_instance, reranker_instance)
    
    return {"message": f"Ingested {len(new_chunks)} chunks", "total_docs": len(docs_store)}

@app.post("/query")
async def query_rag(payload: QueryRequest):
    if not rag_workflow:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    inputs = {"question": payload.question}
    result = rag_workflow.workflow.invoke(inputs)
    
    return {
        "question": payload.question,
        "answer": result["answer"],
        "retrieved_docs": [d.page_content[:200] + "..." for d in result["documents"]]
    }