from langchain_openai import ChatOpenAI
from typing import List

class Evaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def evaluate_retrieval(self, retrieved_docs: List[str], expected_concept: str):
        """
        Simple LLM-as-a-judge to check if retrieved context contains the expected answer.
        """
        context = "\n".join(retrieved_docs)
        prompt = f"""
        Does the following text contain information about "{expected_concept}"?
        Answer only 'YES' or 'NO'.
        
        Text:
        {context}
        """
        response = self.llm.invoke(prompt).content.strip()
        return 1 if "YES" in response.upper() else 0

    def evaluate_hallucination(self, answer: str, context: List[str]):
        """
        Checks if the answer is supported by the context (Faithfulness).
        """
        context_text = "\n".join([d.page_content for d in context])
        prompt = f"""
        You are a grader assessing faithfulness. 
        Does the Answer rely ONLY on the provided Context? 
        
        Context: {context_text}
        Answer: {answer}
        
        Reply 'YES' if faithful, 'NO' if it contains hallucinations.
        """
        response = self.llm.invoke(prompt).content.strip()
        return response