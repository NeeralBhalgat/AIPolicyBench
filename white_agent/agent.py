"""White agent implementation - the target RAG agent being tested."""

import uvicorn
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from safety_datasets_rag import SafetyDatasetsRAG

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_white_agent_card(url: str, model: str = "deepseek-chat") -> AgentCard:
    """
    Prepare the agent card for the white agent.

    Args:
        url: The URL where the white agent will be hosted
        model: The LLM model used by the white agent

    Returns:
        AgentCard with agent metadata
    """
    skill = AgentSkill(
        id="ai_policy_rag",
        name="AI Policy RAG",
        description=f"Answers questions about AI safety and policy using RAG over safety datasets (Model: {model})",
        tags=["rag", "ai-safety", "policy", f"model:{model}"],
        examples=[
            "What datasets are available for AI safety research?",
            "Are there datasets about AI alignment?",
            "What resources exist for studying adversarial robustness?"
        ],
    )
    card = AgentCard(
        name="aipolicybench_rag_agent",
        description=f"RAG agent for AI safety and policy questions using {model}",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AIPolityRAGAgentExecutor(AgentExecutor):
    """Executor that handles RAG queries for AI policy questions."""

    def __init__(self, vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl",
                 model: str = "deepseek-chat"):
        """
        Initialize the RAG agent executor.

        Args:
            vector_db_path: Path to the vector database
            model: The LLM model to use for generation
        """
        self.model = model
        self.rag_system = SafetyDatasetsRAG(vector_db_path, model=model)
        self.initialized = False

    def _ensure_initialized(self):
        """Ensure the RAG system is initialized."""
        if not self.initialized:
            success = self.rag_system.load_vector_db()
            if not success:
                raise RuntimeError("Failed to load vector database")
            self.initialized = True
            logger.info("White agent RAG system initialized successfully")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute a RAG query from the user.

        Args:
            context: Request context containing the user query
            event_queue: Event queue for sending responses
        """
        try:
            # Ensure RAG system is initialized
            self._ensure_initialized()

            # Get user input
            user_query = context.get_user_input()
            logger.info(f"White agent received query: {user_query}")

            # Execute RAG query
            result = await self.rag_system.complete_rag_query(
                question=user_query,
                top_k=5,
                use_llm=True
            )

            # Extract response
            if "error" in result:
                response_text = f"Error: {result['error']}"
            else:
                response_text = result.get("generated_response", "No response generated")

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
    vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl",
    model: str = "deepseek-chat",
    host: str = "localhost",
    port: int = 9002
):
    """
    Start the white agent server.

    Args:
        vector_db_path: Path to the vector database
        model: The LLM model to use for generation
        host: Host to bind to
        port: Port to bind to
    """
    logger.info(f"Starting white agent with model: {model}")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url, model=model)

    request_handler = DefaultRequestHandler(
        agent_executor=AIPolityRAGAgentExecutor(vector_db_path, model=model),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    logger.info(f"White agent listening on {url}")
    uvicorn.run(app.build(), host=host, port=port)
