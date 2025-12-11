import os
import logging
from typing import Dict, Any, List, Optional
from src.simple_vector_db import SimpleTFIDFVectorDB
from src.utils.llm_client import LLMClient
from src.config import config

logger = logging.getLogger(__name__)

class SafetyDatasetsRAG:
    """
    RAG system for safety datasets using TF-IDF vector database.
    """
    
    def __init__(self, vector_db_path: str, model: str = config.DEFAULT_MODEL):
        """
        Initialize the RAG system.
        
        Args:
            vector_db_path: Path to the vector database
            model: LLM model to use
        """
        self.vector_db_path = vector_db_path
        self.model = model
        self.vector_db = SimpleTFIDFVectorDB()
        self.llm_client = None
        
        # Initialize LLM client
        try:
            # Determine provider based on model name or config
            provider = config.DEFAULT_PROVIDER
            if "deepseek" in model:
                provider = "deepseek"
            elif "gpt" in model:
                provider = "openai"
            elif "claude" in model:
                provider = "anthropic"
                
            self.llm_client = LLMClient(provider=provider, model=model)
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    def load_vector_db(self) -> bool:
        """Load the vector database from disk."""
        if not os.path.exists(self.vector_db_path):
            logger.error(f"Vector database not found at {self.vector_db_path}")
            return False
            
        try:
            self.vector_db.load(self.vector_db_path)
            return True
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    async def complete_rag_query(self, question: str, top_k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve -> Generate.
        """
        # 1. Retrieve
        results = self.vector_db.search(question, top_k=top_k)
        
        if not results:
            return {"error": "No relevant documents found"}
            
        # 2. Generate Context
        context_parts = []
        for i, res in enumerate(results, 1):
            context_parts.append(f"Document {i} (ID: {res['id']}):\n{res['text']}\n")
        context = "\n".join(context_parts)
        
        if not use_llm or not self.llm_client:
            return {
                "question": question,
                "context": context,
                "retrieved_chunks": results
            }
            
        # 3. Generate Answer
        try:
            prompt = f"""Based on the following safety datasets information, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            response = await self.llm_client.generate_response(prompt)
            
            return {
                "question": question,
                "generated_response": response,
                "context": context,
                "retrieved_chunks": results
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}
