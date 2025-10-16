"""
LLM Client

This module provides a unified interface for interacting with different LLM providers
including OpenAI, Anthropic, and local models.
"""

import openai
import asyncio
from typing import Dict, Any, List, Optional
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from a chat conversation."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # OpenAI API key is now passed to the client directly
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI."""
        try:
            # Use the new OpenAI client format
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using OpenAI."""
        try:
            # Use the new OpenAI client format
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating OpenAI chat response: {e}")
            raise


class DeepSeekClient(BaseLLMClient):
    """DeepSeek LLM client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key
            model: Model to use
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        # Use OpenAI-compatible API for DeepSeek
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using DeepSeek."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating DeepSeek response: {e}")
            raise
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using DeepSeek."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating DeepSeek chat response: {e}")
            raise


    """Anthropic LLM client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Import anthropic here to avoid dependency issues
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": prompt}],
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            raise
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=messages,
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating Anthropic chat response: {e}")
            raise


class LocalLLMClient(BaseLLMClient):
    """Local LLM client using transformers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize local LLM client.
        
        Args:
            model_name: Name of the local model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError:
            raise ImportError("Transformers package is required. Install with: pip install transformers torch")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using local model."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            if hasattr(self.model, 'cuda') and next(self.model.parameters()).is_cuda:
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=kwargs.get('max_length', 100),
                    num_return_sequences=1,
                    temperature=kwargs.get('temperature', 0.7),
                    do_sample=kwargs.get('do_sample', True),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating local model response: {e}")
            raise
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response using local model."""
        # Convert messages to a single prompt
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            prompt += f"{role}: {content}\n"
        
        prompt += "assistant: "
        return await self.generate_response(prompt, **kwargs)


class LLMClient:
    """Unified LLM client that can work with different providers."""
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'local')
            **kwargs: Provider-specific arguments
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = OpenAIClient(**kwargs)
        elif self.provider == "deepseek":
            self.client = DeepSeekClient(**kwargs)
        elif self.provider == "anthropic":
            self.client = AnthropicClient(**kwargs)
        elif self.provider == "local":
            self.client = LocalLLMClient(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info(f"Initialized LLM client with provider: {self.provider}")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        return await self.client.generate_response(prompt, **kwargs)
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat response from the LLM."""
        return await self.client.generate_chat_response(messages, **kwargs)
    
    async def generate_policy_response(self, 
                                     question: str, 
                                     context: str, 
                                     **kwargs) -> str:
        """
        Generate a policy-focused response.
        
        Args:
            question: Policy question
            context: Relevant context from documents
            **kwargs: Additional arguments
            
        Returns:
            Policy response
        """
        prompt = f"""You are a policy expert assistant. Based on the provided context from policy documents, please answer the following question.

Context:
{context}

Question: {question}

Please provide a comprehensive, well-structured answer that:
1. Directly addresses the question
2. Uses information from the provided context
3. Cites specific sources when referencing information
4. Provides balanced analysis when appropriate
5. Acknowledges limitations if the context is insufficient

Answer:"""

        return await self.generate_response(prompt, **kwargs)
    
    async def generate_summary(self, text: str, summary_type: str = "brief", **kwargs) -> str:
        """
        Generate a summary of text.
        
        Args:
            text: Text to summarize
            summary_type: Type of summary ('brief', 'detailed', 'bullet_points')
            **kwargs: Additional arguments
            
        Returns:
            Summary text
        """
        if summary_type == "brief":
            prompt = f"Please provide a brief 2-3 sentence summary of the following text:\n\n{text}"
        elif summary_type == "detailed":
            prompt = f"Please provide a detailed summary of the following text, covering all main points:\n\n{text}"
        elif summary_type == "bullet_points":
            prompt = f"Please provide a bullet-point summary of the following text:\n\n{text}"
        else:
            prompt = f"Please summarize the following text:\n\n{text}"
        
        return await self.generate_response(prompt, **kwargs)
    
    async def generate_analysis(self, 
                              topic: str, 
                              context: str, 
                              analysis_type: str = "general", 
                              **kwargs) -> str:
        """
        Generate an analysis of a topic.
        
        Args:
            topic: Topic to analyze
            context: Relevant context
            analysis_type: Type of analysis ('general', 'policy', 'technical', 'stakeholder')
            **kwargs: Additional arguments
            
        Returns:
            Analysis text
        """
        if analysis_type == "policy":
            prompt = f"""Please provide a policy analysis of: {topic}

Context:
{context}

Please analyze:
1. Current policy landscape
2. Key stakeholders and their positions
3. Challenges and opportunities
4. Potential policy implications
5. Recommendations

Analysis:"""
        elif analysis_type == "technical":
            prompt = f"""Please provide a technical analysis of: {topic}

Context:
{context}

Please analyze:
1. Technical aspects and requirements
2. Implementation challenges
3. Technical risks and mitigation
4. Best practices and standards
5. Future technical considerations

Analysis:"""
        elif analysis_type == "stakeholder":
            prompt = f"""Please provide a stakeholder analysis of: {topic}

Context:
{context}

Please analyze:
1. Key stakeholders and their interests
2. Stakeholder relationships and dynamics
3. Potential conflicts and alignments
4. Stakeholder influence and power
5. Engagement strategies

Analysis:"""
        else:
            prompt = f"""Please provide a comprehensive analysis of: {topic}

Context:
{context}

Please provide a well-structured analysis covering the key aspects, implications, and considerations related to this topic.

Analysis:"""
        
        return await self.generate_response(prompt, **kwargs)
