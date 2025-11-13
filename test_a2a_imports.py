#!/usr/bin/env python3
"""Test script to verify A2A imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing A2A implementation imports...")
print("=" * 80)

try:
    print("✓ Importing utils.a2a_client...")
    from utils import a2a_client
    print(f"  - get_agent_card: {hasattr(a2a_client, 'get_agent_card')}")
    print(f"  - send_message: {hasattr(a2a_client, 'send_message')}")
    print(f"  - wait_agent_ready: {hasattr(a2a_client, 'wait_agent_ready')}")
except Exception as e:
    print(f"✗ Error importing utils.a2a_client: {e}")
    sys.exit(1)

try:
    print("\n✓ Importing utils.parsing...")
    from utils.parsing import parse_tags
    print(f"  - parse_tags: {callable(parse_tags)}")
except Exception as e:
    print(f"✗ Error importing utils.parsing: {e}")
    sys.exit(1)

try:
    print("\n✓ Importing white_agent.agent...")
    from white_agent.agent import start_white_agent, AIPolityRAGAgentExecutor
    print(f"  - start_white_agent: {callable(start_white_agent)}")
    print(f"  - AIPolityRAGAgentExecutor: {AIPolityRAGAgentExecutor is not None}")
except Exception as e:
    print(f"✗ Error importing white_agent.agent: {e}")
    sys.exit(1)

try:
    print("\n✓ Importing green_agent.a2a_evaluator...")
    from green_agent.a2a_evaluator import start_green_agent, GreenAgentExecutor
    print(f"  - start_green_agent: {callable(start_green_agent)}")
    print(f"  - GreenAgentExecutor: {GreenAgentExecutor is not None}")
except Exception as e:
    print(f"✗ Error importing green_agent.a2a_evaluator: {e}")
    sys.exit(1)

try:
    print("\n✓ Importing launcher...")
    from launcher import launch_evaluation
    print(f"  - launch_evaluation: {callable(launch_evaluation)}")
except Exception as e:
    print(f"✗ Error importing launcher: {e}")
    sys.exit(1)

try:
    print("\n✓ Importing main CLI...")
    import main
    print(f"  - app: {hasattr(main, 'app')}")
except Exception as e:
    print(f"✗ Error importing main: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ All A2A imports successful!")
print("\nYou can now use:")
print("  - python main.py --help")
print("  - python main.py info")
print("  - python main.py launch")
print("=" * 80)
