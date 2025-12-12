#!/usr/bin/env python3
"""
Send evaluation task to a running green agent for all white agents.
Automatically reads:
- Green agent URL from .ab/agents/*/agent_card
- White agent URLs from white_agents_config.json

Note: Evaluation parameters (max_queries, batch_size, etc.) are configured
in the green agent's a2a_evaluator.py, not passed via this script.
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path
from src.utils import a2a_client
from a2a.types import SendMessageSuccessResponse, Message
from a2a.utils import get_text_parts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_green_agent_url_from_ab() -> str | None:
    """
    Get the green agent URL from .ab/agents/*/agent_card file.
    
    Looks for an agent with name containing 'green' or description 
    containing 'Assessment manager' or 'evaluation'.
    
    Returns:
        Green agent URL if found, None otherwise
    """
    ab_agents_dir = Path(".ab/agents")
    
    if not ab_agents_dir.exists():
        logger.warning(f"⚠️ .ab/agents directory not found")
        return None
    
    # Iterate through all agent directories
    for agent_dir in ab_agents_dir.iterdir():
        if not agent_dir.is_dir():
            continue
            
        agent_card_path = agent_dir / "agent_card"
        if not agent_card_path.exists():
            continue
        
        try:
            with open(agent_card_path, 'r') as f:
                agent_card = json.load(f)
            
            name = agent_card.get('name', '').lower()
            description = agent_card.get('description', '').lower()
            url = agent_card.get('url', '')
            
            # Check if this is a green agent
            is_green = (
                'green' in name or
                'assessment manager' in description or
                'evaluating rag agents' in description
            )
            
            if is_green and url:
                logger.info(f"✅ Found green agent from .ab: {agent_card.get('name')}")
                logger.info(f"   URL: {url}")
                return url
                
        except Exception as e:
            logger.warning(f"⚠️ Error reading agent_card in {agent_dir}: {e}")
            continue
    
    logger.warning("⚠️ No green agent found in .ab/agents/")
    return None


def load_white_agent_urls(config_file: str = "white_agents_config.json") -> list:
    """
    Load white agent URLs from configuration file.

    Args:
        config_file: Path to configuration file

    Returns:
        List of white agent URLs
    """
    config_path = Path(config_file)

    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_file}")
        logger.error("Please run ./start_multi_agents_cloudflare.sh first to generate the config")
        return []

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        agents = config.get('agents', [])
        urls = [agent.get('url') for agent in agents if agent.get('url')]

        logger.info(f"✅ Loaded {len(urls)} white agent URLs from {config_file}")
        return urls

    except Exception as e:
        logger.error(f"❌ Error loading config file: {e}")
        return []


async def send_evaluation_task(
    config_file: str = "white_agents_config.json"
):
    """
    Send evaluation task to green agent for all white agents from config.
    
    Green agent URL is automatically obtained from .ab/agents/*/agent_card.

    Args:
        config_file: Path to white agents configuration file
    """
    logger.info("=" * 80)
    logger.info("📤 Sending evaluation task to green agent")
    logger.info("=" * 80)

    # Get green agent URL from .ab directory
    green_url = get_green_agent_url_from_ab()
    
    if not green_url:
        logger.error("❌ Could not find green agent URL from .ab directory")
        logger.error("Please start the green agent first:")
        logger.error("  ./start_multi_agents_cloudflare.sh")
        return
    
    # Load white agent URLs from config
    white_urls = load_white_agent_urls(config_file)

    if not white_urls:
        logger.error("❌ No white agent URLs found in config file")
        return

    logger.info(f"\n📋 White agents to evaluate:")
    for idx, url in enumerate(white_urls, 1):
        logger.info(f"  {idx}. {url}")

    # Check if green agent is running
    logger.info(f"\n🔍 Checking green agent at {green_url}...")
    try:
        green_card = await a2a_client.get_agent_card(green_url)
        if not green_card:
            logger.error(f"❌ Green agent not accessible at {green_url}")
            logger.error("Please start the green agent first:")
            logger.error("  ./start_multi_agents_cloudflare.sh")
            return

        logger.info(f"✅ Green agent is running: {green_card.name}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to green agent: {e}")
        logger.error("Please start the green agent first:")
        logger.error("  ./start_multi_agents_cloudflare.sh")
        return

    # Build task text with all white agent URLs
    white_agent_tags = "\n".join([f"<white_agent_url>{url}</white_agent_url>" for url in white_urls])

    task_text = f"""{white_agent_tags}

Please evaluate all white agents.
"""

    logger.info("\n" + "=" * 80)
    logger.info("📝 Task configuration:")
    logger.info(f"  Number of white agents: {len(white_urls)}")
    logger.info("  (Evaluation params configured in green agent's a2a_evaluator.py)")
    logger.info("=" * 80)

    try:
        logger.info("\n⏳ Running evaluation (this may take a while)...")
        logger.info("")

        response = await a2a_client.send_message(green_url, task_text, timeout=1800.0)

        logger.info("\n" + "=" * 80)
        logger.info("📊 EVALUATION RESULTS")
        logger.info("=" * 80)

        # Extract and display response
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
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("✅ Task complete!")
    logger.info("=" * 80)
    logger.info("\n📂 Results saved to: results/new_white_agent_design/")
    logger.info("\nView results:")
    logger.info("  ls -la results/new_white_agent_design/")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Send evaluation task to green agent for all white agents"
    )
    parser.add_argument(
        "--config",
        default="white_agents_config.json",
        help="Path to white agents config file (default: white_agents_config.json)"
    )

    args = parser.parse_args()

    asyncio.run(send_evaluation_task(
        config_file=args.config
    ))


if __name__ == "__main__":
    main()
