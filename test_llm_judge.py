#!/usr/bin/env python3
"""Test script for LLM-as-a-judge evaluation."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from green_agent.evaluation import LLMJudgeEvaluator, RuleBasedEvaluator


async def test_evaluators():
    """Test both rule-based and LLM judge evaluators."""

    print("=" * 80)
    print("Testing AIPolicyBench Evaluators")
    print("=" * 80)

    # Test data
    test_cases = [
        {
            "question": "Are there datasets available for AI safety research?",
            "response": "Yes, there are several datasets available for AI safety research, including datasets on adversarial robustness, fairness, and alignment.",
            "ground_truth": "yes",
            "expected": "correct"
        },
        {
            "question": "Who leads the evaluation team?",
            "response": "I'm not sure who leads the evaluation team.",
            "ground_truth": "Charles",
            "expected": "miss"
        },
        {
            "question": "How many members are in the team?",
            "response": "There are 5 members in the team.",
            "ground_truth": "4",
            "expected": "hallucination"
        }
    ]

    # Test Rule-Based Evaluator
    print("\n" + "=" * 80)
    print("1. Rule-Based Evaluation")
    print("=" * 80)

    rule_evaluator = RuleBasedEvaluator(case_sensitive=False)

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Question: {test['question']}")
        print(f"Response: {test['response']}")
        print(f"Ground Truth: {test['ground_truth']}")

        result = rule_evaluator.evaluate(
            response=test['response'],
            ground_truth=test['ground_truth']
        )

        status = "✅" if result['result'] == test['expected'] else "❌"
        print(f"\n{status} Result: {result['result']} (expected: {test['expected']})")
        print(f"Method: {result['method']}")

    # Test LLM Judge (optional - requires API key)
    print("\n" + "=" * 80)
    print("2. LLM-as-a-Judge Evaluation")
    print("=" * 80)

    try:
        llm_evaluator = LLMJudgeEvaluator(
            provider="deepseek",
            model="mistralai/mistral-7b-instruct",  # Use Mistral via OpenRouter
            temperature=0.0,
            max_tokens=500  # Limit tokens to avoid credit issues
        )

        print("\n✅ LLM Judge initialized successfully")
        print("Note: This requires DEEPSEEK_API_KEY (OpenRouter) in .env")
        print("Using Mistral model with max_tokens=500\n")

        # Test just the first case to verify it works
        test = test_cases[0]
        print(f"--- Test 1 (LLM Judge) ---")
        print(f"Question: {test['question']}")
        print(f"Response: {test['response']}")
        print(f"Ground Truth: {test['ground_truth']}")

        result = await llm_evaluator.evaluate(
            response=test['response'],
            ground_truth=test['ground_truth'],
            question=test['question']
        )

        if result['result'] != 'error':
            print(f"\n✅ Result: {result['result']}")
            print(f"Method: {result['method']}")
            print(f"Provider: {result.get('provider', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        else:
            print(f"\n⚠️ LLM Judge returned error (likely no API key)")
            print(f"Reason: {result.get('reason', 'Unknown')}")

    except Exception as e:
        print(f"\n⚠️ LLM Judge test skipped: {e}")
        print("This is expected if DEEPSEEK_API_KEY is not set in .env")

    print("\n" + "=" * 80)
    print("✅ Evaluator tests complete!")
    print("=" * 80)
    print("\nTo use LLM judge in evaluation:")
    print("  conda run -n aipolicy python main.py launch --llm-judge")
    print("\=" * 80)


if __name__ == "__main__":
    asyncio.run(test_evaluators())
