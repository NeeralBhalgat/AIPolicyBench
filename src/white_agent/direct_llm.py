"""
Direct LLM White Agent - No RAG, direct model responses.

This agent directly queries the LLM without web search or RAG,
useful for comparing model knowledge vs RAG-enhanced responses.
"""

import os
import logging
from typing import Optional
from src.config import config
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class DirectLLM:
    """
    Direct LLM system that queries models without RAG augmentation.
    """

    def __init__(self,
                 llm_provider: str = "openrouter",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the Direct LLM system.

        Args:
            llm_provider: LLM provider name
            model: Model identifier (e.g., "mistral/mistral-7b-instruct")
            api_key: API key for the provider
        """
        self.llm_provider = llm_provider
        self.model = model or config.DEFAULT_MODEL

        # Select API key
        if api_key:
            self.api_key = api_key
        elif llm_provider == "openai":
            self.api_key = config.OPENAI_API_KEY
        elif llm_provider == "anthropic":
            self.api_key = config.ANTHROPIC_API_KEY
        elif llm_provider == "openrouter":
            self.api_key = config.OPENROUTER_API_KEY
        else:
            self.api_key = config.OPENROUTER_API_KEY

        # Initialize LLM Client
        try:
            logger.info(f"Initializing Direct LLM client for provider: {llm_provider}, model: {self.model}")
            self.llm_client = LLMClient(
                provider=llm_provider,
                api_key=self.api_key,
                model=self.model
            )
            logger.info(f"Initialized Direct LLM client successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    async def answer_question(self, question: str) -> str:
        """
        Answer a question directly using the LLM without any retrieval.

        Args:
            question: User's question

        Returns:
            Direct answer from the LLM
        """
        if not self.llm_client:
            return "Error: LLM client not initialized"

        try:
            logger.info(f"Answering question directly with {self.model}: {question}")

            # Create a prompt that asks the model to rely on its training data
            prompt = f"""You are an expert on AI policy, safety, and regulation.
Answer the following question based ONLY on your training data and knowledge.

IMPORTANT RULES:
1. If you are not ABSOLUTELY CERTAIN about the answer, respond with: "I don't have enough information to answer this question."
2. Only provide specific facts if you are confident they are accurate.
3. Do NOT speculate or guess.
4. Keep your answer concise (1-2 sentences).

Question: {question}

Answer:"""

            response = await self.llm_client.generate_response(
                prompt=prompt,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=800
            )

            logger.info(f"Generated direct response ({len(response)} chars)")
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating direct LLM response: {e}")
            return f"Error: Failed to generate response - {str(e)}"

    async def query(self, question: str) -> dict:
        """
        Query the direct LLM (for compatibility with RAG interface).

        Args:
            question: User's question

        Returns:
            Dictionary with answer and metadata
        """
        answer = await self.answer_question(question)

        return {
            "answer": answer,
            "model": self.model,
            "provider": self.llm_provider,
            "mode": "direct_llm",  # No RAG
            "sources": []  # No sources for direct LLM
        }
