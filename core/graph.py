from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from core.retriever import HybridRetriever
from core.reranker import Reranker

# 1. Define State
class GraphState(TypedDict):
    question: str
    documents: List[str]  # Content of retrieved docs
    answer: str

class RAGWorkflow:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.workflow = self._build_graph()

    def retrieve(self, state: GraphState):
        """Retrieve documents using Hybrid Search."""
        question = state["question"]
        docs = self.retriever.hybrid_search(question, k=settings.TOP_K_RETRIEVAL)
        return {"documents": docs}

    def rerank(self, state: GraphState):
        """Rerank retrieved documents."""
        question = state["question"]
        # 'documents' in state might be objects here, need careful handling in TypedDict
        # For simplicity, we assume the previous step passed Document objects
        # In a strict TypedDict, we'd define a custom type, but Python is flexible here at runtime.
        docs = state["documents"] 
        reranked_docs = self.reranker.rerank(question, docs, top_n=settings.TOP_K_RERANK)
        return {"documents": reranked_docs}

    def generate(self, state: GraphState):
        """Generate answer using Reranked Context."""
        question = state["question"]
        docs = state["documents"]
        
        context_str = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant. Use the following context to answer the question.
            If the answer is not in the context, say you don't know.
            
            Context:
            {context}
            
            Question: 
            {question}
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": question})
        return {"answer": answer}

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        # Add Nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        
        # Define Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()