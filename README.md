# Advanced RAG System: Hybrid Retrieval, Reranking & LangGraph Orchestration

This project is a production-oriented implementation of a **Retrieval-Augmented Generation (RAG)** system. It moves beyond basic vector search to address common failure modes in RAG (e.g., poor recall, loss of context, hallucinations) by implementing a multi-stage pipeline.

 
*(Note: A diagram showing Query -> Hybrid Search -> Reranking -> LLM would be placed here)*

## 🚀 Key Features

1.  **Hybrid Retrieval**: Combines **BM25 (Sparse)** keyword search with **FAISS (Dense)** vector search. This captures both exact term matches (technical jargon) and semantic meaning.
2.  **Advanced Chunking**: Implements **Structure-Aware Chunking** (respecting Markdown headers) to keep related concepts together, preventing context fragmentation.
3.  **Multi-Stage Reranking**: Uses a **Cross-Encoder** to re-score retrieved documents. This allows retrieving a large pool of documents (Recall) and filtering them down to the most relevant ones (Precision) before the LLM step.
4.  **Agentic Workflow**: Orchestrated using **LangGraph**, creating a deterministic, debuggable flow where the state is passed clearly between retrieval, reranking, and generation nodes.
5.  **Evaluation hooks**: Built-in modules to test retrieval quality and hallucination (LLM-as-a-judge).

## 🛠 Tech Stack

* **Orchestration**: LangChain, LangGraph
* **Vector DB**: FAISS (Facebook AI Similarity Search)
* **Retrieval**: BM25 (Rank_BM25), OpenAI Embeddings
* **Reranking**: HuggingFace Cross-Encoders (`ms-marco-MiniLM-L-6-v2`)
* **API**: FastAPI
* **Language**: Python 3.10+

## 📂 Project Structure

```text
├── app/                  # API Layer
├── core/                 # RAG Logic (Retriever, Reranker, Graph)
├── data/                 # Local storage
└── notebooks/            # Experiments