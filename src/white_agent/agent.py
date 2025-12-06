"""White agent implementation - the target RAG agent being tested."""

import uvicorn
import sys
import logging
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

from src.config import config
from src.white_agent.pipeline import WebSearchRAG

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_white_agent_card(url: str, model: str = config.DEFAULT_MODEL) -> AgentCard:
    """
    Prepare the agent card for the white agent.
    """
    skill = AgentSkill(
        id="ai_policy_rag",
        name="AI Policy RAG",
        description=f"Answers questions about AI safety and policy using Web Search RAG (Model: {model})",
        tags=["rag", "ai-safety", "policy", f"model:{model}", "web-search"],
        examples=[
            "What is the AI Action Plan?",
            "What are the recent executive orders on AI?",
            "Explain the EU AI Act key provisions."
        ],
    )
    card = AgentCard(
        name="aipolicybench_rag_agent",
        description=f"Web-Search RAG agent for AI safety and policy questions using {model}",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AIPolityRAGAgentExecutor(AgentExecutor):
    """Executor that handles RAG queries for AI policy questions using Web Search."""

    def __init__(self, model: str = config.DEFAULT_MODEL):
        """
        Initialize the RAG agent executor.
        """
        self.model = model
        self.rag_system = WebSearchRAG(model=model)
        logger.info("White agent Web Search RAG system initialized")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute a RAG query from the user.
        """
        try:
            # Get user input
            user_query = context.get_user_input()
            logger.info(f"White agent received query: {user_query}")

            # Execute Web RAG query
            result = await self.rag_system.answer_query(
                query=user_query,
                top_k=5
            )

            # Extract response
            if "error" in result:
                response_text = f"Error: {result['error']}"
            else:
                response_text = result.get("response", "No response generated")

            logger.info(f"White agent responding with: {response_text[:100]}...")

            # Send response
            await event_queue.enqueue_event(
                new_agent_text_message(
                    response_text,
                    context_id=context.context_id
                )
            )

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            await event_queue.enqueue_event(
                new_agent_text_message(
                    error_msg,
                    context_id=context.context_id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution (not implemented)."""
        raise NotImplementedError("Cancel operation not supported")


def start_white_agent(
    model: str = config.DEFAULT_MODEL,
    host: str = config.HOST,
    port: int = config.PORT
):
    """
    Start the white agent server.
    """
    logger.info(f"Starting white agent with model: {model}")
    url = f"http://{host}:{port}"
    
    # Use environment variable for public URL if set (for hosting)
    public_url = os.getenv("AGENT_URL", url)
    
    card = prepare_white_agent_card(public_url, model=model)

    request_handler = DefaultRequestHandler(
        agent_executor=AIPolityRAGAgentExecutor(model=model),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    logger.info(f"White agent listening on {host}:{port} (Public URL: {public_url})")
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == "__main__":
    start_white_agent()

