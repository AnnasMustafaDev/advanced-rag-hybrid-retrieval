from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document

# Import schemas to ensure data validation matches your Pydantic models
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

# Import Core Logic
from core.graph import RAGWorkflow
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.ingestion import IngestionPipeline

app = FastAPI(title="Advanced RAG Showcase")

# --- Global State ---
# In production, these would be managed by dependency injection or singletons
docs_store = []
retriever_instance = None
reranker_instance = Reranker()  # Load model once at module level
rag_workflow = None
ingestion_pipeline = None

@app.on_event("startup")
async def startup_event():
    global retriever_instance, rag_workflow, ingestion_pipeline, docs_store
    
    # 1. Initialize Dummy Data
    dummy_doc = Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.", 
        metadata={"source": "manual", "doc_type": "documentation"}
    )
    docs_store.append(dummy_doc)
    
    # 2. Initialize Retriever with existing docs
    retriever_instance = HybridRetriever(docs_store)
    
    # 3. Initialize Ingestion Pipeline (needs the retriever to update index)
    ingestion_pipeline = IngestionPipeline(retriever_instance)

    # 4. Initialize RAG Workflow (connects retriever + reranker + LLM)
    rag_workflow = RAGWorkflow(retriever_instance, reranker_instance)
    
    print("System initialized with 1 dummy document.")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(payload: IngestRequest):
    global rag_workflow, ingestion_pipeline
    
    if not ingestion_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # 1. Run the Ingestion Pipeline
    # This handles chunking and updating the retriever's index internally
    result = ingestion_pipeline.process_document(
        text=payload.text, 
        filename=payload.filename, 
        doc_type=payload.document_type
    )
    
    # 2. Re-initialize Workflow
    # Important: Because the retriever's index changed, we ensure the workflow uses the updated state.
    # (In a real vector DB, this step isn't needed as the DB is external, but for local FAISS it helps safety)
    rag_workflow = RAGWorkflow(retriever_instance, reranker_instance)
    
    return IngestResponse(
        message="Ingestion successful",
        chunks_created=result["chunks_count"],
        total_documents_in_index=len(retriever_instance.documents)
    )

@app.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    global rag_workflow
    
    if not rag_workflow:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # 1. Run the LangGraph Workflow
    inputs = {"question": payload.question}
    result = rag_workflow.workflow.invoke(inputs)
    
    # 2. Format Response
    # Extract just the top chunks for the API response
    context_used = [
        {
            "page_content": d.page_content,
            "metadata": d.metadata
        } 
        for d in result["documents"]
    ]
    
    return QueryResponse(
        question=payload.question,
        answer=result["answer"],
        context_used=context_used
    )