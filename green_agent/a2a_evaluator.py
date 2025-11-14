"""Green agent A2A evaluator - manages assessment and evaluation of RAG agents."""

import uvicorn
import sys
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Handle tomllib for different Python versions
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 and earlier

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.parsing import parse_tags
from utils import a2a_client
from green_agent.evaluation import RuleBasedEvaluator, LLMJudgeEvaluator

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_agent_card_toml(agent_name: str):
    """
    Load agent card configuration from TOML file.

    Args:
        agent_name: Name of the agent configuration file (without .toml extension)

    Returns:
        Dictionary containing agent card configuration
    """
    current_dir = Path(__file__).parent
    toml_path = current_dir / f"{agent_name}.toml"
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


async def evaluate_white_agent(
    white_agent_url: str,
    queries_file: str = "data/predefined_queries.json",
    use_llm_judge: bool = False,
    results_dir: str = "./results"
) -> dict:
    """
    Evaluate a white agent using predefined queries.

    Args:
        white_agent_url: URL of the white agent to evaluate
        queries_file: Path to predefined queries JSON file
        use_llm_judge: Whether to use LLM-as-a-judge evaluation
        results_dir: Directory to save evaluation results

    Returns:
        Dictionary containing evaluation results and statistics
    """
    logger.info(f"Starting evaluation of white agent at {white_agent_url}")

    # Extract white agent name from URL for results directory
    white_agent_name = white_agent_url.replace("http://", "").replace("https://", "").replace("/", "_").replace(":", "_")

    # Create results directory
    agent_results_dir = Path(results_dir) / white_agent_name
    agent_results_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_session_dir = agent_results_dir / timestamp
    eval_session_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {eval_session_dir}")

    # Load predefined queries
    try:
        with open(queries_file, 'r') as f:
            data = json.load(f)
            queries = data.get('queries', [])
    except Exception as e:
        logger.error(f"Error loading queries: {e}")
        return {"error": f"Failed to load queries: {e}"}

    if not queries:
        return {"error": "No queries loaded"}

    logger.info(f"Loaded {len(queries)} queries for evaluation")

    # Initialize evaluator
    if use_llm_judge:
        logger.info("Using LLM-as-a-judge evaluation with GPT-4o-mini")
        # Use GPT-4o-mini with OpenRouter (high quality, cost-effective)
        evaluator = LLMJudgeEvaluator(
            provider="deepseek",  # Uses OpenRouter if DEEPSEEK_API_KEY starts with sk-or-
            model="openai/gpt-4o-mini",  # GPT-4o-mini model on OpenRouter
            temperature=0.0,
            max_tokens=800  # Sufficient for detailed judgments
        )
    else:
        logger.info("Using rule-based evaluation")
        evaluator = RuleBasedEvaluator(case_sensitive=False)

    # Evaluate each query with progress bar
    results = []
    correct_count = 0
    miss_count = 0
    hallucination_count = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting evaluation of {len(queries)} queries")
    logger.info(f"{'='*80}\n")

    # Create progress bar
    pbar = tqdm(queries, desc="Evaluating queries", unit="query")

    for query_data in pbar:
        query_id = query_data['id']
        query = query_data['query']
        ground_truth = query_data['ground_truth']

        # Update progress bar description
        pbar.set_description(f"Query {query_id}/{len(queries)}")

        logger.info(f"\n{'='*80}")
        logger.info(f"[Query {query_id}] {query}")
        logger.info(f"[Ground Truth] {ground_truth}")
        logger.info(f"{'='*80}")

        try:
            # Send query to white agent
            response = await a2a_client.send_message(white_agent_url, query)
            res_root = response.root

            if not isinstance(res_root, SendMessageSuccessResponse):
                results.append({
                    "query_id": query_id,
                    "query": query,
                    "ground_truth": ground_truth,
                    "error": "Invalid response from white agent",
                    "evaluation_result": "error"
                })
                continue

            res_result = res_root.result
            if not isinstance(res_result, Message):
                results.append({
                    "query_id": query_id,
                    "query": query,
                    "ground_truth": ground_truth,
                    "error": "Expected Message response",
                    "evaluation_result": "error"
                })
                continue

            # Extract text from response
            text_parts = get_text_parts(res_result.parts)
            if not text_parts:
                results.append({
                    "query_id": query_id,
                    "query": query,
                    "ground_truth": ground_truth,
                    "error": "No text in response",
                    "evaluation_result": "error"
                })
                continue

            response_text = text_parts[0]
            logger.info(f"[White Agent Response]")
            logger.info(f"{response_text}\n")

            # Evaluate response
            logger.info(f"[Evaluating with {('LLM-as-a-judge' if use_llm_judge else 'Rule-based')}]")
            if use_llm_judge:
                # LLM judge requires async call with question parameter
                eval_result = await evaluator.evaluate(
                    response=response_text,
                    ground_truth=ground_truth,
                    question=query
                )
            else:
                # Rule-based is synchronous
                eval_result = evaluator.evaluate(response_text, ground_truth)

            # Log evaluation result
            result_symbol = {
                "correct": "✅",
                "miss": "⚠️",
                "hallucination": "❌",
                "error": "🔴"
            }.get(eval_result["result"], "❓")

            logger.info(f"\n[Judgment] {result_symbol} {eval_result['result'].upper()}")
            if "confidence" in eval_result:
                logger.info(f"[Confidence] {eval_result['confidence']:.2f}")
            if "reasoning" in eval_result:
                logger.info(f"[Reasoning] {eval_result['reasoning']}")

            # Track statistics
            if eval_result["result"] == "correct":
                correct_count += 1
            elif eval_result["result"] == "miss":
                miss_count += 1
            elif eval_result["result"] == "hallucination":
                hallucination_count += 1

            # Build result entry
            result_entry = {
                "query_id": query_id,
                "query": query,
                "response": response_text,
                "ground_truth": ground_truth,
                "evaluation_result": eval_result["result"],
                "evaluation_method": eval_result["method"],
                "timestamp": datetime.now().isoformat()
            }

            # Add LLM judge specific fields if available
            if "confidence" in eval_result:
                result_entry["confidence"] = eval_result["confidence"]
            if "reasoning" in eval_result:
                result_entry["reasoning"] = eval_result["reasoning"]
            if "provider" in eval_result:
                result_entry["provider"] = eval_result["provider"]

            results.append(result_entry)

            # Save individual result to file
            result_file = eval_session_dir / f"query_{query_id:03d}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_entry, f, indent=2, ensure_ascii=False)
            logger.info(f"[Saved] {result_file}")

            # Update progress bar with current stats
            pbar.set_postfix({
                "Correct": correct_count,
                "Miss": miss_count,
                "Halluc": hallucination_count
            })

        except Exception as e:
            logger.error(f"Error evaluating query {query_id}: {e}")
            results.append({
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "error": str(e),
                "evaluation_result": "error"
            })

    # Close progress bar
    pbar.close()

    # Calculate statistics
    total = len(results)
    correct_rate = (correct_count / total * 100) if total > 0 else 0.0
    miss_rate = (miss_count / total * 100) if total > 0 else 0.0
    hallucination_rate = (hallucination_count / total * 100) if total > 0 else 0.0
    factuality_rate = correct_rate + miss_rate  # Correct + Miss (not hallucinating)

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete: {correct_count}/{total} correct ({correct_rate:.2f}%)")
    logger.info(f"Correct: {correct_count} ({correct_rate:.2f}%)")
    logger.info(f"Miss: {miss_count} ({miss_rate:.2f}%)")
    logger.info(f"Hallucination: {hallucination_count} ({hallucination_rate:.2f}%)")
    logger.info(f"Factuality Rate: {factuality_rate:.2f}%")
    logger.info(f"{'='*80}\n")

    # Build return value
    result_dict = {
        "results": results,
        "statistics": {
            "total": total,
            "correct": correct_count,
            "miss": miss_count,
            "hallucination": hallucination_count,
            "correct_rate": correct_rate,
            "miss_rate": miss_rate,
            "hallucination_rate": hallucination_rate,
            "factuality_rate": factuality_rate
        },
        "method": "LLM-as-a-judge" if use_llm_judge else "Rule-based",
        "white_agent_url": white_agent_url,
        "queries_file": queries_file,
        "timestamp": timestamp,
        "results_dir": str(eval_session_dir)
    }

    # Add provider info for LLM judge
    if use_llm_judge:
        result_dict["provider"] = "openai/gpt-4o-mini"
        result_dict["model"] = "openai/gpt-4o-mini"

    # Save summary results
    summary_file = eval_session_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {summary_file}")

    # Save statistics in a separate file for easy reading
    stats_file = eval_session_dir / "statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"AIPolicyBench Evaluation Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"White Agent: {white_agent_url}\n")
        f.write(f"Evaluation Method: {result_dict['method']}\n")
        if use_llm_judge:
            f.write(f"Model: {result_dict['model']}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Statistics:\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total Queries: {total}\n")
        f.write(f"Correct: {correct_count} ({correct_rate:.2f}%)\n")
        f.write(f"Miss: {miss_count} ({miss_rate:.2f}%)\n")
        f.write(f"Hallucination: {hallucination_count} ({hallucination_rate:.2f}%)\n")
        f.write(f"Factuality Rate: {factuality_rate:.2f}%\n")
        f.write(f"\n{'='*80}\n")
    logger.info(f"Statistics saved to: {stats_file}")

    return result_dict


class GreenAgentExecutor(AgentExecutor):
    """Green agent executor for managing RAG agent assessments."""

    def __init__(self):
        """Initialize the green agent executor."""
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute an evaluation task.

        Args:
            context: Request context containing the evaluation task
            event_queue: Event queue for sending updates and results
        """
        logger.info("Green agent: Received evaluation task")

        # Parse the task
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        # Extract configuration
        white_agent_url = tags.get("white_agent_url")
        queries_file = tags.get("queries_file", "data/predefined_queries.json")
        use_llm_judge = tags.get("use_llm_judge", "false").lower() == "true"

        if not white_agent_url:
            error_msg = "Error: white_agent_url not provided in task"
            logger.error(error_msg)
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return

        logger.info(f"Evaluating white agent at: {white_agent_url}")
        logger.info(f"Using queries file: {queries_file}")
        logger.info(f"LLM judge: {use_llm_judge}")

        # Run evaluation
        try:
            result = await evaluate_white_agent(
                white_agent_url=white_agent_url,
                queries_file=queries_file,
                use_llm_judge=use_llm_judge
            )

            # Format response
            if "error" in result:
                response_text = f"❌ Evaluation failed: {result['error']}"
            else:
                stats = result["statistics"]
                response_text = f"""✅ Evaluation Complete!

📊 Results Summary:
- Total Queries: {stats['total']}
- Correct: {stats['correct']} ({stats['correct_rate']:.2f}%)
- Miss: {stats['miss']} ({stats['miss_rate']:.2f}%)
- Hallucination: {stats['hallucination']} ({stats['hallucination_rate']:.2f}%)
- Factuality Rate: {stats['factuality_rate']:.2f}%

Evaluation Method: {result['method']}"""

                # Add provider info for LLM judge
                if "provider" in result:
                    response_text += f"\nLLM Provider: {result['provider']}"

                response_text += f"\n\nDetailed results: {json.dumps(result['results'], indent=2)}\n"

            logger.info("Sending evaluation results")
            await event_queue.enqueue_event(new_agent_text_message(response_text))

        except Exception as e:
            error_msg = f"❌ Evaluation error: {str(e)}"
            logger.error(error_msg)
            await event_queue.enqueue_event(new_agent_text_message(error_msg))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution (not implemented)."""
        raise NotImplementedError("Cancel operation not supported")


def start_green_agent(
    agent_name: str = "green_agent",
    host: str = "localhost",
    port: int = 9001
):
    """
    Start the green agent A2A server.

    Args:
        agent_name: Name of the agent configuration file (without .toml)
        host: Host to bind to
        port: Port to bind to
    """
    logger.info("Starting green agent...")

    # Load agent card
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    logger.info(f"Green agent listening on {url}")
    uvicorn.run(app.build(), host=host, port=port)
