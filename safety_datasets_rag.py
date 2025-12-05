#!/usr/bin/env python3
"""
Safety Datasets RAG Pipeline
Core RAG system with retrieval, augmentation, and generation capabilities.
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from simple_vector_db import SimpleTFIDFVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafetyDatasetsRAG:
    """
    Core RAG system for safety datasets.
    Handles retrieval, augmentation, and generation.
    """
    
    def __init__(self, vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl", 
                 api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        Initialize the RAG system.
        
        Args:
            vector_db_path: Path to the vector database
            api_key: DeepSeek API key
            model: DeepSeek model to use
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model = model
        self.vector_db = None
        self.llm_client = None
        
        if self.api_key:
            self._initialize_llm_client()
        else:
            logger.warning("No DeepSeek API key provided. LLM generation will be disabled.")
    
    def _initialize_llm_client(self):
        """Initialize the DeepSeek LLM client."""
        try:
            from openai import AsyncOpenAI
            # Check if using OpenRouter (API key starts with sk-or-)
            if self.api_key and self.api_key.startswith("sk-or-"):
                self.llm_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                # Update model name for OpenRouter if needed
                self.model = "mistralai/mistral-7b-instruct"
                logger.info(f"Initialized OpenRouter LLM client with model: {self.model}")
            else:
                # Direct DeepSeek API
                self.llm_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info(f"Initialized DeepSeek LLM client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    def load_vector_db(self) -> bool:
        """Load the vector database from disk."""
        try:
            if not os.path.exists(self.vector_db_path):
                logger.error(f"Vector database not found at: {self.vector_db_path}")
                return False
            
            self.vector_db = SimpleTFIDFVectorDB()
            self.vector_db.load(self.vector_db_path)
            logger.info(f"Loaded TF-IDF vector database with {len(self.vector_db.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Step 1: RETRIEVAL - Find relevant datasets.
        
        Args:
            query: User question
            top_k: Number of results to return
            
        Returns:
            List of relevant datasets with metadata
        """
        if not self.vector_db:
            logger.error("Vector database not loaded")
            return []
        
        logger.info(f"Retrieving relevant datasets for: '{query}'")
        results = self.vector_db.search(query, top_k)
        logger.info(f"Retrieved {len(results)} relevant datasets")
        return results
    
    def augment(self, results: List[Dict[str, Any]]) -> str:
        """
        Step 2: AUGMENTATION - Prepare context for LLM.

        Args:
            results: Retrieved datasets

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant datasets found."

        context_parts = ["Based on the following documents:\n"]

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            # Include the FULL TEXT content for the LLM
            context_parts.append(f"\n--- Document {i}: {metadata.get('dataset_name', 'Unknown Dataset')} ---")
            context_parts.append(result.get('text', 'No text available'))
            context_parts.append("")

        return "\n".join(context_parts)
    
    async def generate(self, context: str, question: str) -> str:
        """
        Steps 3 & 4: GENERATION - Generate natural language response.
        
        Args:
            context: Augmented context from retrieved datasets
            question: Original user question
            
        Returns:
            Generated natural language response
        """
        if not self.llm_client:
            return "LLM generation is not available. Please provide a valid DeepSeek API key."
        
        try:
            prompt = f"""{context}

Question: {question}

Please provide a direct, concise answer to the question based ONLY on the information provided in the documents above. Extract the specific answer from the text and state it clearly.

Answer:"""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions directly and concisely based on the provided document content. Extract specific information from the text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    async def complete_rag_query(self, question: str, top_k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieval + augmentation + generation.
        
        Args:
            question: User question
            top_k: Number of datasets to retrieve
            use_llm: Whether to generate LLM response
            
        Returns:
            Dictionary with all RAG components
        """
        if not self.vector_db:
            return {"error": "Vector database not loaded"}
        
        # Step 1: RETRIEVAL
        retrieved_datasets = self.retrieve(question, top_k)
        
        if not retrieved_datasets:
            return {"error": "No relevant datasets found"}
        
        # Step 2: AUGMENTATION
        context = self.augment(retrieved_datasets)
        
        result = {
            "question": question,
            "retrieved_datasets": retrieved_datasets,
            "context": context,
            "generated_response": None
        }
        
        # Steps 3 & 4: GENERATION (if requested and available)
        if use_llm and self.llm_client:
            result["generated_response"] = await self.generate(context, question)
        else:
            result["generated_response"] = "LLM generation disabled or unavailable"
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system."""
        return {
            "vector_db_loaded": self.vector_db is not None,
            "vector_db_size": len(self.vector_db.documents) if self.vector_db else 0,
            "llm_available": self.llm_client is not None,
            "model": self.model if self.llm_client else None,
            "api_key_provided": bool(self.api_key)
        }
