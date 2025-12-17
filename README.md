# Advanced RAG System

## Hybrid Retrieval, Reranking, and LangGraph Orchestration

This repository contains a **production-oriented implementation of a Retrieval-Augmented Generation (RAG) system** designed to solve common real-world RAG failure modes such as low recall, context fragmentation, and hallucinations. Instead of relying on a single vector search step, the system implements a **multi-stage, agentic retrieval and generation pipeline** with explicit evaluation hooks.

The project is intended as a **portfolio-grade reference architecture** for building reliable RAG systems used in legal tech, enterprise knowledge bases, and domain-specific AI assistants.

---

## System Overview

High-level pipeline:

1. User query ingestion
2. Hybrid retrieval (sparse + dense)
3. Metadata-aware filtering and chunk selection
4. Multi-stage reranking for precision
5. Context assembly and validation
6. LLM answer generation
7. Evaluation and observability hooks

*(A diagram showing Query → Hybrid Search → Reranking → LLM Generation can be added here.)*

---

## Key Features

### 1. Hybrid Retrieval (Sparse + Dense)

Combines **BM25 keyword-based retrieval** with **FAISS-based semantic vector search**. This approach improves recall by capturing both exact term matches (legal or technical jargon) and semantic similarity.

### 2. Advanced Chunking Strategies

Implements **structure-aware chunking** that respects document boundaries such as Markdown headers and sections. This preserves semantic coherence and prevents loss of contextual meaning during retrieval.

### 3. Multi-Stage Reranking

Uses a **cross-encoder reranker** to rescore retrieved chunks. A larger candidate set is retrieved initially (high recall) and then filtered down to the most relevant context (high precision) before generation.

### 4. Agentic RAG Orchestration

The pipeline is orchestrated using **LangGraph**, enabling a deterministic, debuggable workflow. Each stage (retrieval, reranking, generation) is represented as a node with explicit state passing, making the system easy to inspect and extend.

### 5. Evaluation and Observability Hooks

Includes evaluation utilities for:

* Retrieval quality comparison (semantic vs sparse vs hybrid)
* Chunking strategy experiments
* Reranking impact analysis
* Hallucination checks using LLM-as-a-judge

---

## Tech Stack

### Core AI and Orchestration

* **LangChain** – LLM abstraction and tooling
* **LangGraph** – Agentic workflow orchestration

### Retrieval and Ranking

* **FAISS** – Dense vector similarity search
* **BM25 / Rank-BM25** – Sparse keyword retrieval
* **HuggingFace Cross-Encoders** – Document reranking (e.g., `ms-marco-MiniLM-L-6-v2`)

### Backend and Infrastructure

* **FastAPI** – API layer
* **Python 3.10+** – Core language
* **Pydantic** – Request and response schemas

---

## Project Structure

```
advanced-rag-system/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings (API keys, parameters)
│   ├── main.py             # FastAPI application entry point
│   ├── schemas.py          # Pydantic models for API requests and responses
├── core/
│   ├── __init__.py
│   ├── chunking.py         # Advanced and structure-aware chunking logic
│   ├── embedding.py        # Embedding model wrappers
│   ├── evaluation.py       # Retrieval and generation evaluation utilities
│   ├── graph.py            # LangGraph workflow definition
│   ├── ingestion.py        # Document loading and preprocessing
│   ├── reranker.py         # Cross-encoder and LLM-based reranking
│   ├── retriever.py        # Hybrid dense and sparse retrieval logic
├── data/                   # FAISS indexes and raw documents
├── notebooks/              # Experimental notebooks (optional)
├── tests/                  # Unit and integration tests
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Setup and Installation

1. Clone the repository
2. Create a virtual environment and install dependencies
3. Copy `.env.example` to `.env` and configure API keys
4. Ingest documents and build indexes
5. Start the FastAPI server

Detailed setup steps can be added based on deployment needs.

---

## Intended Use Cases

* Legal and compliance document assistants
* Enterprise internal knowledge search
* Technical documentation Q&A systems
* High-precision, low-hallucination RAG applications

---

## Design Philosophy

This project prioritizes:

* Reliability over demos
* Measurable retrieval quality
* Clear separation of concerns
* Debuggable and extensible AI workflows

It is designed to serve as both a **learning reference** and a **foundation for production RAG systems**.
