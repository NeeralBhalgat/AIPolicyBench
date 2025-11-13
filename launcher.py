"""Launcher module - initiates and coordinates the A2A evaluation process."""

import multiprocessing
import json
import asyncio
import logging
from pathlib import Path

from green_agent.a2a_evaluator import start_green_agent
from white_agent.agent import start_white_agent
from utils import a2a_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def launch_evaluation(
    queries_file: str = "data/predefined_queries.json",
    vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl",
    use_llm_judge: bool = False,
    green_host: str = "localhost",
    green_port: int = 9001,
    white_host: str = "localhost",
    white_port: int = 9002
):
    """
    Launch the complete evaluation workflow with green and white agents.

    Args:
        queries_file: Path to predefined queries JSON file
        vector_db_path: Path to the vector database
        use_llm_judge: Whether to use LLM-as-a-judge evaluation
        green_host: Host for green agent
        green_port: Port for green agent
        white_host: Host for white agent
        white_port: Port for white agent
    """
    logger.info("=" * 80)
    logger.info("🚀 Launching AIPolicyBench A2A Evaluation")
    logger.info("=" * 80)

    # Verify queries file exists
    if not Path(queries_file).exists():
        logger.error(f"❌ Queries file not found: {queries_file}")
        return

    # Verify vector DB exists
    if not Path(vector_db_path).exists():
        logger.error(f"❌ Vector database not found: {vector_db_path}")
        return

    # Start green agent
    logger.info("\n📗 Launching green agent (evaluator)...")
    green_url = f"http://{green_host}:{green_port}"
    p_green = multiprocessing.Process(
        target=start_green_agent,
        args=("green_agent", green_host, green_port)
    )
    p_green.start()

    # Wait for green agent to be ready
    is_ready = await a2a_client.wait_agent_ready(green_url, timeout=15)
    if not is_ready:
        logger.error("❌ Green agent not ready in time")
        p_green.terminate()
        p_green.join()
        return

    logger.info(f"✅ Green agent is ready at {green_url}")

    # Start white agent
    logger.info("\n📄 Launching white agent (RAG system)...")
    white_url = f"http://{white_host}:{white_port}"
    p_white = multiprocessing.Process(
        target=start_white_agent,
        args=(vector_db_path, white_host, white_port)
    )
    p_white.start()

    # Wait for white agent to be ready
    is_ready = await a2a_client.wait_agent_ready(white_url, timeout=15)
    if not is_ready:
        logger.error("❌ White agent not ready in time")
        p_green.terminate()
        p_green.join()
        p_white.terminate()
        p_white.join()
        return

    logger.info(f"✅ White agent is ready at {white_url}")

    # Send evaluation task to green agent
    logger.info("\n📤 Sending evaluation task to green agent...")
    task_text = f"""Your task is to evaluate the RAG agent located at:
<white_agent_url>
{white_url}
</white_agent_url>
Use the following configuration:
<queries_file>
{queries_file}
</queries_file>
<use_llm_judge>
{str(use_llm_judge).lower()}
</use_llm_judge>

Please evaluate all queries and provide a detailed report.
"""

    logger.info("Task description:")
    logger.info(task_text)

    try:
        logger.info("\n⏳ Running evaluation (this may take a while)...")
        response = await a2a_client.send_message(green_url, task_text, timeout=300.0)

        logger.info("\n" + "=" * 80)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("=" * 80)

        # Extract and display response
        from a2a.types import SendMessageSuccessResponse, Message
        from a2a.utils import get_text_parts

        res_root = response.root
        if isinstance(res_root, SendMessageSuccessResponse):
            res_result = res_root.result
            if isinstance(res_result, Message):
                text_parts = get_text_parts(res_result.parts)
                if text_parts:
                    logger.info(text_parts[0])
                else:
                    logger.warning("No text in response")
            else:
                logger.warning(f"Unexpected response type: {type(res_result)}")
        else:
            logger.error(f"Evaluation failed: {response}")

    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")

    # Cleanup
    logger.info("\n🧹 Cleaning up...")
    logger.info("Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    logger.info("✅ Agents terminated successfully")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 80)


def main():
    """Main entry point for the launcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AIPolicyBench A2A Evaluation Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--queries_file", default="data/predefined_queries.json",
                        help="Path to predefined queries JSON file")
    parser.add_argument("--vector_db", default="./vector_db/safety_datasets_tfidf_db.pkl",
                        help="Path to the vector database file")
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM-as-a-judge evaluation instead of rule-based")
    parser.add_argument("--green_host", default="localhost",
                        help="Host for green agent (default: localhost)")
    parser.add_argument("--green_port", type=int, default=9001,
                        help="Port for green agent (default: 9001)")
    parser.add_argument("--white_host", default="localhost",
                        help="Host for white agent (default: localhost)")
    parser.add_argument("--white_port", type=int, default=9002,
                        help="Port for white agent (default: 9002)")

    args = parser.parse_args()

    asyncio.run(launch_evaluation(
        queries_file=args.queries_file,
        vector_db_path=args.vector_db,
        use_llm_judge=args.use_llm_judge,
        green_host=args.green_host,
        green_port=args.green_port,
        white_host=args.white_host,
        white_port=args.white_port
    ))


if __name__ == "__main__":
    main()
