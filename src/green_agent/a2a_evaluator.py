"""Green agent A2A evaluator - manages assessment and evaluation of RAG agents."""

import uvicorn
import sys
import json
import logging
import os
import asyncio
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

from src.utils.parsing import parse_tags
from src.utils import a2a_client
from .evaluation import RuleBasedEvaluator, LLMJudgeEvaluator

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global URL to identifier mapping
URL_IDENTIFIER_MAP = {}


def load_white_agents_config(config_file: str = "white_agents_config.json") -> dict:
    """
    Load white agents configuration mapping URLs to identifiers.

    Args:
        config_file: Path to the configuration file (default: white_agents_config.json in project root)

    Returns:
        Dictionary mapping URLs to agent identifiers
    """
    global URL_IDENTIFIER_MAP

    # Try project root first
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / config_file

    if not config_path.exists():
        logger.warning(f"White agents config file not found: {config_path}")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Build URL mapping
        url_map = {}
        for agent in config.get('agents', []):
            url = agent.get('url', '')
            identifier = agent.get('identifier', '')

            if url and identifier:
                # Store both the exact URL and variations
                url_map[url] = identifier
                url_map[url.rstrip('/')] = identifier

                # Also map by base URL (without /to_agent path)
                if '/to_agent' in url:
                    base_url = url.split('/to_agent')[0]
                    url_map[base_url] = identifier
                    url_map[base_url.rstrip('/')] = identifier

        URL_IDENTIFIER_MAP = url_map
        logger.info(f"Loaded {len(config.get('agents', []))} white agent configurations")
        return url_map

    except Exception as e:
        logger.error(f"Error loading white agents config: {e}")
        return {}


def get_identifier_for_url(url: str) -> str:
    """
    Look up identifier for a given URL from the loaded configuration.

    Args:
        url: White agent URL

    Returns:
        Identifier string (e.g., 'mistralai-mistral-7b-instruct_rag') or None if not found
    """
    # Try exact match first
    if url in URL_IDENTIFIER_MAP:
        return URL_IDENTIFIER_MAP[url]

    # Try without trailing slash
    url_stripped = url.rstrip('/')
    if url_stripped in URL_IDENTIFIER_MAP:
        return URL_IDENTIFIER_MAP[url_stripped]

    # Try base URL (without /to_agent path)
    if '/to_agent' in url:
        base_url = url.split('/to_agent')[0]
        if base_url in URL_IDENTIFIER_MAP:
            return URL_IDENTIFIER_MAP[base_url]
        if base_url.rstrip('/') in URL_IDENTIFIER_MAP:
            return URL_IDENTIFIER_MAP[base_url.rstrip('/')]

    return None


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


async def fetch_agent_info(white_agent_url: str, max_retries: int = 3) -> dict:
    """
    Fetch agent card to extract model and mode information with retries.

    Args:
        white_agent_url: URL of the white agent
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Dictionary with 'model' and 'mode' keys
    """
    import re

    for attempt in range(max_retries):
        try:
            # Fetch agent card using a2a_client method (which has its own retries)
            agent_card = await a2a_client.get_agent_card(white_agent_url)

            if not agent_card:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Agent card empty, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.warning("Failed to fetch agent card after retries, using defaults")
                    return {"model": "unknown", "mode": "unknown"}

            # Extract model and mode from agent card
            agent_name = agent_card.name if hasattr(agent_card, 'name') else "unknown"
            description = agent_card.description if hasattr(agent_card, 'description') else ""
            skills = agent_card.skills if hasattr(agent_card, 'skills') else []

            # Determine mode
            mode = "direct" if "direct" in agent_name.lower() or "direct" in description.lower() else "rag"

            # Extract model from description or tags
            model = "unknown"
            if skills:
                skill_desc = skills[0].description if hasattr(skills[0], 'description') else ""
                # Try to extract model from description like "Model: deepseek/deepseek-chat"
                model_match = re.search(r'Model:\s*([^\)]+)', skill_desc)
                if model_match:
                    model = model_match.group(1).strip()
                else:
                    # Try to find model in tags
                    tags = skills[0].tags if hasattr(skills[0], 'tags') else []
                    for tag in tags:
                        if tag.startswith("model:"):
                            model = tag.replace("model:", "").strip()
                            break

            logger.info(f"✓ Agent info - Model: {model}, Mode: {mode}")
            return {"model": model, "mode": mode}

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"Failed to fetch agent card after {max_retries} attempts, using defaults")
                return {"model": "unknown", "mode": "unknown"}

    # Fallback
    return {"model": "unknown", "mode": "unknown"}


async def process_single_query(
    white_agent_url: str,
    query_data: dict,
    evaluator,
    use_llm_judge: bool
) -> dict:
    """
    Process a single query and return the evaluation result.

    Args:
        white_agent_url: URL of the white agent
        query_data: Query data with id, query, ground_truth
        evaluator: Evaluator instance (RuleBasedEvaluator or LLMJudgeEvaluator)
        use_llm_judge: Whether using LLM judge

    Returns:
        Dictionary with evaluation result
    """
    query_id = query_data['id']
    query = query_data['query']
    ground_truth = query_data['ground_truth']

    try:
        # Send query to white agent
        response = await a2a_client.send_message(white_agent_url, query, timeout=180.0)
        res_root = response.root

        if not isinstance(res_root, SendMessageSuccessResponse):
            return {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "error": "Invalid response from white agent",
                "evaluation_result": "error"
            }

        res_result = res_root.result
        if not isinstance(res_result, Message):
            return {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "error": "Expected Message response",
                "evaluation_result": "error"
            }

        # Extract text from response
        text_parts = get_text_parts(res_result.parts)
        if not text_parts:
            return {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "error": "No text in response",
                "evaluation_result": "error"
            }

        response_text = text_parts[0]

        # Check for timeout response (empty or contains timeout indicator)
        is_timeout = (
            not response_text or 
            response_text.strip() == "" or
            response_text.startswith("[TIMEOUT]") or
            "timeout" in response_text.lower() or
            "timed out" in response_text.lower()
        )
        
        if is_timeout:
            return {
                "query_id": query_id,
                "query": query,
                "response": response_text,
                "ground_truth": ground_truth,
                "evaluation_result": "timeout",
                "evaluation_method": "auto-detect",
                "timestamp": datetime.now().isoformat(),
                "reason": "White agent response timed out or was empty"
            }

        # Evaluate response
        if use_llm_judge:
            eval_result = await evaluator.evaluate(
                response=response_text,
                ground_truth=ground_truth,
                question=query
            )
        else:
            eval_result = evaluator.evaluate(response_text, ground_truth)

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

        return result_entry

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
        return {
            "query_id": query_id,
            "query": query,
            "ground_truth": ground_truth,
            "error": str(e),
            "evaluation_result": "error"
        }


async def evaluate_white_agent(
    white_agent_url: str,
    queries_file: str = "data/predefined_queries.json",
    use_llm_judge: bool = False,
    max_queries: int = None,
    results_dir: str = "results/new_white_agent_design",
    agent_identifier: str = None,
    batch_size: int = 4
) -> dict:
    """
    Evaluate a white agent using predefined queries with batch processing.

    Args:
        white_agent_url: URL of the white agent to evaluate
        queries_file: Path to predefined queries JSON file
        use_llm_judge: Whether to use LLM-as-a-judge evaluation
        max_queries: Maximum number of queries to evaluate (None for all)
        results_dir: Directory to save evaluation results
        agent_identifier: Custom identifier for result directory naming
        batch_size: Number of concurrent requests to send (default: 16)

    Returns:
        Dictionary containing evaluation results and statistics
    """
    logger.info(f"Starting evaluation of white agent at {white_agent_url}")

    # Use agent_identifier if provided, otherwise extract from agent card
    if agent_identifier:
        white_agent_model = agent_identifier
        logger.info(f"Using provided agent identifier: {white_agent_model}")
    else:
        # Get white agent card to extract model name
        white_agent_model = "unknown"
        try:
            white_agent_card = await a2a_client.get_agent_card(white_agent_url)
            if white_agent_card and white_agent_card.skills:
                # Extract model from tags (format: "model:deepseek-chat")
                for skill in white_agent_card.skills:
                    for tag in skill.tags:
                        if tag.startswith("model:"):
                            white_agent_model = tag.replace("model:", "")
                            break
                    if white_agent_model != "unknown":
                        break
        except Exception as e:
            logger.warning(f"Could not get white agent model name: {e}")

        logger.info(f"White agent model: {white_agent_model}")

    # Create results directory based on model name or identifier (no timestamp in path)
    eval_session_dir = Path(results_dir) / white_agent_model
    eval_session_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {eval_session_dir}")

    # Keep timestamp for metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    # Limit queries if max_queries is specified
    total_queries_available = len(queries)
    if max_queries is not None and max_queries > 0:
        queries = queries[:max_queries]
        logger.info(f"Limiting to first {max_queries} queries out of {total_queries_available} total")

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

    # Evaluate queries with batched processing
    results = []
    correct_count = 0
    miss_count = 0
    hallucination_count = 0
    timeout_count = 0

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting evaluation of {len(queries)} queries with batch size {batch_size}")
    logger.info(f"{'='*80}\n")

    # Split queries into batches
    total_queries = len(queries)
    num_batches = (total_queries + batch_size - 1) // batch_size

    # Create progress bar for batches
    pbar = tqdm(total=total_queries, desc="Evaluating queries", unit="query")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_queries)
        batch_queries = queries[start_idx:end_idx]

        logger.info(f"\n{'='*80}")
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (queries {start_idx + 1}-{end_idx})")
        logger.info(f"{'='*80}\n")

        # Process batch concurrently
        batch_tasks = [
            process_single_query(white_agent_url, query_data, evaluator, use_llm_judge)
            for query_data in batch_queries
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process batch results
        for query_data, result in zip(batch_queries, batch_results):
            query_id = query_data['id']

            # Handle exceptions
            if isinstance(result, Exception):
                logger.error(f"Exception processing query {query_id}: {result}")
                result = {
                    "query_id": query_id,
                    "query": query_data['query'],
                    "ground_truth": query_data['ground_truth'],
                    "error": str(result),
                    "evaluation_result": "error"
                }

            # Log result
            eval_result_str = result.get("evaluation_result", "error")
            result_symbol = {
                "correct": "✅",
                "miss": "⚠️",
                "hallucination": "❌",
                "timeout": "⏱️",
                "error": "🔴"
            }.get(eval_result_str, "❓")

            logger.info(f"[Query {query_id}] {result_symbol} {eval_result_str.upper()}")

            # Track statistics
            if eval_result_str == "correct":
                correct_count += 1
            elif eval_result_str == "miss":
                miss_count += 1
            elif eval_result_str == "hallucination":
                hallucination_count += 1
            elif eval_result_str == "timeout":
                timeout_count += 1

            results.append(result)

            # Save individual result to file
            result_file = eval_session_dir / f"query_{query_id:03d}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "Correct": correct_count,
                "Miss": miss_count,
                "Halluc": hallucination_count,
                "Timeout": timeout_count
            })

    # Close progress bar
    pbar.close()

    # Calculate statistics
    # Exclude timeouts from factuality calculation
    total = len(results)
    evaluated_total = total - timeout_count  # Only count non-timeout responses
    
    # Rates based on total (for reference)
    timeout_rate = (timeout_count / total * 100) if total > 0 else 0.0
    
    # Rates based on evaluated (excluding timeouts) - for factuality calculation
    correct_rate = (correct_count / evaluated_total * 100) if evaluated_total > 0 else 0.0
    miss_rate = (miss_count / evaluated_total * 100) if evaluated_total > 0 else 0.0
    hallucination_rate = (hallucination_count / evaluated_total * 100) if evaluated_total > 0 else 0.0
    factuality_rate = correct_rate + miss_rate - hallucination_rate  # c=1 for miss

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete: {correct_count}/{evaluated_total} correct ({correct_rate:.2f}%)")
    logger.info(f"Total queries: {total}, Evaluated: {evaluated_total}, Timeouts: {timeout_count}")
    logger.info(f"Correct: {correct_count} ({correct_rate:.2f}%)")
    logger.info(f"Miss: {miss_count} ({miss_rate:.2f}%)")
    logger.info(f"Hallucination: {hallucination_count} ({hallucination_rate:.2f}%)")
    logger.info(f"Timeout: {timeout_count} ({timeout_rate:.2f}%) - excluded from factuality")
    logger.info(f"Factuality Rate: {factuality_rate:.2f}%")
    logger.info(f"{'='*80}\n")

    # Build return value
    result_dict = {
        "results": results,
        "statistics": {
            "total": total,
            "evaluated": evaluated_total,
            "correct": correct_count,
            "miss": miss_count,
            "hallucination": hallucination_count,
            "timeout": timeout_count,
            "correct_rate": correct_rate,
            "miss_rate": miss_rate,
            "hallucination_rate": hallucination_rate,
            "timeout_rate": timeout_rate,
            "factuality_rate": factuality_rate
        },
        "method": "LLM-as-a-judge" if use_llm_judge else "Rule-based",
        "white_agent_url": white_agent_url,
        "white_agent_model": white_agent_model,
        "queries_file": queries_file,
        "timestamp": timestamp,
        "results_dir": str(eval_session_dir)
    }

    # Add provider info for LLM judge
    if use_llm_judge:
        result_dict["llm_judge_provider"] = "openai/gpt-4o-mini"
        result_dict["llm_judge_model"] = "openai/gpt-4o-mini"

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
        f.write(f"White Agent Model: {white_agent_model}\n")
        f.write(f"Evaluation Method: {result_dict['method']}\n")
        if use_llm_judge:
            f.write(f"LLM Judge Model: {result_dict['llm_judge_model']}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Statistics:\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total Queries: {total}\n")
        f.write(f"Evaluated (excl. timeout): {evaluated_total}\n")
        f.write(f"Timeout: {timeout_count} ({timeout_rate:.2f}%) - excluded from factuality\n\n")
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
        self._background_tasks = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute an evaluation task for one or more white agents.
        
        IMPORTANT: To avoid Cloudflare 524 timeout (100s limit), this method
        sends an immediate acknowledgment and processes evaluation in background.

        Args:
            context: Request context containing the evaluation task
            event_queue: Event queue for sending updates and results

        Expected input format:
            Simple URL tags only - green agent looks up identifiers from config:
            "Your task is to assess the agents located at:
            <white_agent_url>URL1</white_agent_url>
            <white_agent_url>URL2</white_agent_url>
            <white_agent_url>URL3</white_agent_url>..."

        Default configuration (can be overridden with tags):
            - use_llm_judge: true (GPT-4o-mini for evaluation)
            - max_queries: 5 (per agent)
            - batch_size: 4 (concurrent requests per batch)
            - queries_file: data/predefined_queries.json
        """
        logger.info("Green agent: Received evaluation task")

        # Load white agents configuration
        load_white_agents_config()

        # Parse the task
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        # Extract ALL white_agent_url tags (simple URLs only)
        # Format: <white_agent_url>URL</white_agent_url>
        import re
        url_matches = re.findall(r'<white_agent_url>\s*([^\s<]+)\s*</white_agent_url>', user_input)

        if not url_matches:
            error_msg = "Error: No white_agent_url tags found in task"
            logger.error(error_msg)
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return

        # Parse URLs and look up identifiers from configuration
        white_agent_configs = []
        for url in url_matches:
            url = url.strip()

            # Fix double slashes in URL if present
            url = url.replace("//to_agent/", "/to_agent/")

            # Look up identifier from configuration
            identifier = get_identifier_for_url(url)

            if identifier:
                logger.info(f"✓ Mapped URL to identifier: {identifier}")
            else:
                logger.warning(f"⚠️ No identifier found for URL: {url} - will try to fetch from agent card")

            white_agent_configs.append({"url": url, "identifier": identifier})

        logger.info(f"Found {len(white_agent_configs)} white agent(s) to evaluate")

        # Use default configuration (optimized for full dataset evaluation)
        queries_file = tags.get("queries_file", "data/predefined_queries.json")

        # Default to LLM judge for more accurate evaluation
        use_llm_judge_str = tags.get("use_llm_judge", "true")
        use_llm_judge = use_llm_judge_str.lower() == "true"

        # Default to 5 queries per agent
        max_queries_str = tags.get("max_queries", "5")
        max_queries = None if max_queries_str == "all" else int(max_queries_str)

        # Batch size for concurrent requests (default: 4)
        batch_size_str = tags.get("batch_size", "4")
        batch_size = int(batch_size_str)

        logger.info(f"Using queries file: {queries_file}")
        logger.info(f"LLM judge: {use_llm_judge}")
        logger.info(f"Max queries: {max_queries if max_queries is not None else 'all'}")
        logger.info(f"Batch size: {batch_size}")

        # ============================================================
        # SYNCHRONOUS EVALUATION: Wait for completion before responding
        # This ensures agentbeats workflow doesn't end prematurely
        # ============================================================
        logger.info("Starting synchronous evaluation (will wait for completion)")
        
        # Run evaluation synchronously and collect results
        all_results = await self._run_evaluation_sync(
            white_agent_configs=white_agent_configs,
            queries_file=queries_file,
            use_llm_judge=use_llm_judge,
            max_queries=max_queries,
            batch_size=batch_size
        )
        
        # Build final response message
        agent_list = "\n".join([f"  - {c['identifier'] or c['url']}" for c in white_agent_configs])
        
        # Build results summary
        results_summary = ""
        for i, result in enumerate(all_results, 1):
            agent_id = result.get("agent_identifier", f"agent_{i}")
            if "error" in result:
                results_summary += f"\n{i}. ❌ {agent_id}: Failed - {result['error']}"
            else:
                stats = result["statistics"]
                timeout_info = f", ⏱️{stats.get('timeout', 0)} timeout" if stats.get('timeout', 0) > 0 else ""
                results_summary += f"\n{i}. ✅ {agent_id}:"
                results_summary += f"\n   - Factuality: {stats['factuality_rate']:.2f}%"
                results_summary += f"\n   - Correct: {stats['correct']}/{stats.get('evaluated', stats['total'])} ({stats['correct_rate']:.2f}%){timeout_info}"
                results_summary += f"\n   - Miss: {stats['miss']} ({stats['miss_rate']:.2f}%)"
                results_summary += f"\n   - Hallucination: {stats['hallucination']} ({stats['hallucination_rate']:.2f}%)"
        
        final_message = f"""✅ EVALUATION COMPLETE

📋 Configuration:
  - Agents evaluated: {len(white_agent_configs)}
  - Questions per agent: {max_queries if max_queries else 'all'}
  - LLM Judge: {use_llm_judge}
  - Batch size: {batch_size}

🎯 Target agents:
{agent_list}

📊 Results:
{results_summary}

📂 Results saved to: results/new_white_agent_design/"""

        await event_queue.enqueue_event(new_agent_text_message(final_message))
        logger.info("Evaluation complete, response sent")

    async def _run_evaluation_sync(
        self,
        white_agent_configs: list,
        queries_file: str,
        use_llm_judge: bool,
        max_queries: int,
        batch_size: int
    ) -> list:
        """
        Run the actual evaluation synchronously.
        Results are saved to files and returned to caller.
        
        Returns:
            List of evaluation results for each agent
        """
        logger.info("Synchronous evaluation started")
        all_results = []
        
        for idx, config in enumerate(white_agent_configs, 1):
            white_agent_url = config["url"]
            provided_identifier = config["identifier"]

            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating white agent {idx}/{len(white_agent_configs)}")
            logger.info(f"URL: {white_agent_url}")
            logger.info(f"{'='*80}\n")

            try:
                # Use provided identifier if available, otherwise fetch from agent card
                if provided_identifier:
                    agent_identifier = provided_identifier
                    logger.info(f"✓ Using provided identifier: {agent_identifier}")
                else:
                    # Fetch agent info to create identifier
                    logger.info("⚠️ No identifier provided, fetching from agent card...")
                    agent_info = await fetch_agent_info(white_agent_url)
                    model = agent_info["model"]
                    mode = agent_info["mode"]

                    # Create agent identifier: model_mode (e.g., "deepseek-chat_rag")
                    # Clean up model name for filename
                    model_clean = model.replace("/", "-").replace(":", "-")
                    agent_identifier = f"{model_clean}_{mode}"
                    logger.info(f"Agent identifier: {agent_identifier}")

                logger.info(f"🔄 Evaluating agent {idx}/{len(white_agent_configs)}: {agent_identifier}...")

                # Run evaluation
                result = await evaluate_white_agent(
                    white_agent_url=white_agent_url,
                    queries_file=queries_file,
                    use_llm_judge=use_llm_judge,
                    max_queries=max_queries,
                    agent_identifier=agent_identifier,
                    batch_size=batch_size
                )

                # Store result
                result["agent_identifier"] = agent_identifier
                result["agent_url"] = white_agent_url
                all_results.append(result)

                # Log individual result
                if "error" in result:
                    logger.error(f"❌ {agent_identifier}: Evaluation failed - {result['error']}")
                else:
                    stats = result["statistics"]
                    logger.info(f"""✅ {agent_identifier}: Complete
- Correct: {stats['correct']} ({stats['correct_rate']:.2f}%)
- Miss: {stats['miss']} ({stats['miss_rate']:.2f}%)
- Hallucination: {stats['hallucination']} ({stats['hallucination_rate']:.2f}%)
- Factuality: {stats['factuality_rate']:.2f}%""")

            except Exception as e:
                error_msg = f"❌ Error evaluating {white_agent_url}: {str(e)}"
                logger.error(error_msg)
                all_results.append({
                    "agent_url": white_agent_url,
                    "error": str(e)
                })

        # Log final summary
        logger.info("\n" + "="*80)
        logger.info("ALL EVALUATIONS COMPLETE!")
        logger.info("="*80)

        summary = f"\n{'='*80}\n✅ ALL EVALUATIONS COMPLETE!\n{'='*80}\n\n"
        summary += f"Evaluated {len(white_agent_configs)} white agent(s)\n\n"

        for i, result in enumerate(all_results, 1):
            agent_id = result.get("agent_identifier", f"agent_{i}")
            if "error" in result:
                summary += f"{i}. ❌ {agent_id}: Failed\n"
            else:
                stats = result["statistics"]
                timeout_info = f", ⏱️{stats.get('timeout', 0)} timeout" if stats.get('timeout', 0) > 0 else ""
                summary += f"{i}. ✅ {agent_id}:\n"
                summary += f"   - Factuality: {stats['factuality_rate']:.2f}%\n"
                summary += f"   - Correct: {stats['correct']}/{stats.get('evaluated', stats['total'])} ({stats['correct_rate']:.2f}%){timeout_info}\n"

        summary += f"\n{'='*80}\n"
        logger.info(summary)
        
        return all_results

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

    # Use environment variable for public URL if set (for hosting)
    public_url = os.getenv("AGENT_URL", url)

    # Fix double slashes in URL (e.g., "https://domain.com//to_agent/id" -> "https://domain.com/to_agent/id")
    public_url = public_url.replace("//to_agent/", "/to_agent/")

    agent_card_dict["url"] = public_url

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

    logger.info(f"Green agent listening on {host}:{port} (Public URL: {public_url})")
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == "__main__":
    """Main entry point to start the green agent server."""
    start_green_agent()