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
from src.white_agent.direct_llm import DirectLLM

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_white_agent_card(url: str, model: str = config.DEFAULT_MODEL, use_direct_llm: bool = False) -> AgentCard:
    """
    Prepare the agent card for the white agent.
    """
    mode = "Direct LLM" if use_direct_llm else "Web Search RAG"
    skill = AgentSkill(
        id="ai_policy_rag" if not use_direct_llm else "ai_policy_direct",
        name=f"AI Policy {mode}",
        description=f"Answers questions about AI safety and policy using {mode} (Model: {model})",
        tags=["ai-safety", "policy", f"model:{model}"] + (["web-search", "rag"] if not use_direct_llm else ["direct-llm"]),
        examples=[
            "What is the AI Action Plan?",
            "What are the recent executive orders on AI?",
            "Explain the EU AI Act key provisions."
        ],
    )
    card = AgentCard(
        name=f"aipolicybench_{'rag' if not use_direct_llm else 'direct'}_agent",
        description=f"{mode} agent for AI safety and policy questions using {model}",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AIPolityRAGAgentExecutor(AgentExecutor):
    """Executor that handles queries for AI policy questions (RAG or Direct LLM)."""
    
    # Hard timeout to ensure we respond before Cloudflare's 100s limit
    HARD_TIMEOUT = 95.0

    def __init__(self, model: str = config.DEFAULT_MODEL, use_direct_llm: bool = False):
        """
        Initialize the agent executor.

        Args:
            model: Model to use
            use_direct_llm: If True, use direct LLM without RAG; otherwise use WebSearch RAG
        """
        self.model = model
        self.use_direct_llm = use_direct_llm

        if use_direct_llm:
            self.system = DirectLLM(model=model)
            logger.info(f"White agent initialized in Direct LLM mode with model: {model}")
        else:
            self.rag_system = WebSearchRAG(model=model)
            logger.info(f"White agent initialized in Web Search RAG mode with model: {model}")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute a query from the user with a hard timeout guarantee.
        """
        user_query = context.get_user_input()
        logger.info(f"White agent received query: {user_query}")
        
        try:
            # Wrap the entire processing in a hard timeout
            response_text = await asyncio.wait_for(
                self._process_query(user_query),
                timeout=self.HARD_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Hard timeout hit - immediately return timeout response
            logger.warning(f"HARD TIMEOUT ({self.HARD_TIMEOUT}s) reached for query: {user_query[:50]}...")
            response_text = "[TIMEOUT] Query processing exceeded time limit."
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response_text = f"Error: {str(e)}"
        
        logger.info(f"White agent responding with: {response_text[:100]}...")
        
        # Send response - this MUST happen within the timeout
        await event_queue.enqueue_event(
            new_agent_text_message(
                response_text,
                context_id=context.context_id
            )
        )

    async def _process_query(self, user_query: str) -> str:
        """
        Internal query processing - can be cancelled by timeout.
        """
        if self.use_direct_llm:
            # Direct LLM mode
            result = await self.system.query(user_query)
            return result.get("answer", "No response generated")
        else:
            # Web RAG mode
            result = await self.rag_system.answer_query(
                query=user_query,
                top_k=5
            )

            # Extract response - prioritize timeout indicator
            if result.get("timed_out"):
                return result.get("response", "[TIMEOUT] Query processing exceeded time limit.")
            elif "error" in result and "response" not in result:
                return f"Error: {result['error']}"
            else:
                return result.get("response", "No response generated")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution (not implemented)."""
        raise NotImplementedError("Cancel operation not supported")


def start_white_agent(
    model: str = config.DEFAULT_MODEL,
    host: str = config.HOST,
    port: int = config.PORT,
    use_direct_llm: bool = False
):
    """
    Start the white agent server.

    Args:
        model: Model to use
        host: Host to bind to
        port: Port to bind to
        use_direct_llm: If True, use direct LLM mode without RAG
    """
    mode = "Direct LLM" if use_direct_llm else "Web Search RAG"
    logger.info(f"Starting white agent in {mode} mode with model: {model}")
    url = f"http://{host}:{port}"

    # Use environment variable for public URL if set (for hosting)
    public_url = os.getenv("AGENT_URL", url)

    # Fix double slashes in URL
    public_url = public_url.replace("//to_agent/", "/to_agent/")

    card = prepare_white_agent_card(public_url, model=model, use_direct_llm=use_direct_llm)

    request_handler = DefaultRequestHandler(
        agent_executor=AIPolityRAGAgentExecutor(model=model, use_direct_llm=use_direct_llm),
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

