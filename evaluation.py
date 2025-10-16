#!/usr/bin/env python3
"""
Evaluation Module for AIPolicyBench
Implements rule-based evaluation using substring matching to compare model responses with ground truth.
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RuleBasedEvaluator:
    """
    Rule-based evaluator using substring matching.
    Checks if ground truth appears as a substring in the model's response.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the rule-based evaluator.

        Args:
            case_sensitive: Whether to perform case-sensitive matching (default: False)
        """
        self.case_sensitive = case_sensitive
        logger.info(f"Initialized Rule-based Evaluator (case_sensitive={case_sensitive})")

    def evaluate(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a model response against ground truth using substring matching.

        Args:
            response: The model's generated response
            ground_truth: The expected ground truth answer

        Returns:
            Dictionary with evaluation result:
            {
                "result": "correct" or "incorrect",
                "method": "Rule-based",
                "ground_truth": ground_truth,
                "response_preview": first 200 chars of response
            }
        """
        if not response or not ground_truth:
            logger.warning("Empty response or ground truth provided")
            return {
                "result": "incorrect",
                "method": "Rule-based",
                "ground_truth": ground_truth,
                "response_preview": response[:200] if response else "",
                "reason": "Empty response or ground truth"
            }

        # Perform substring matching
        if self.case_sensitive:
            match_found = ground_truth in response
        else:
            match_found = ground_truth.lower() in response.lower()

        result = "correct" if match_found else "incorrect"

        logger.info(f"Evaluation result: {result} (Ground truth: '{ground_truth}')")

        return {
            "result": result,
            "method": "Rule-based",
            "ground_truth": ground_truth,
            "response_preview": response[:200] if len(response) > 200 else response,
            "match_found": match_found
        }

    def evaluate_batch(self, responses: list, ground_truths: list) -> Dict[str, Any]:
        """
        Evaluate multiple responses against their ground truths.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers

        Returns:
            Dictionary with batch evaluation results and statistics
        """
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")

        results = []
        correct_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            eval_result = self.evaluate(response, ground_truth)
            results.append(eval_result)

            if eval_result["result"] == "correct":
                correct_count += 1

        total = len(responses)
        accuracy = (correct_count / total * 100) if total > 0 else 0.0

        logger.info(f"Batch evaluation complete: {correct_count}/{total} correct ({accuracy:.2f}%)")

        return {
            "results": results,
            "statistics": {
                "total": total,
                "correct": correct_count,
                "incorrect": total - correct_count,
                "accuracy": accuracy
            },
            "method": "Rule-based"
        }


def evaluate_single(response: str, ground_truth: str, case_sensitive: bool = False) -> str:
    """
    Convenience function to evaluate a single response.

    Args:
        response: The model's generated response
        ground_truth: The expected ground truth answer
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        "correct" or "incorrect"
    """
    evaluator = RuleBasedEvaluator(case_sensitive=case_sensitive)
    result = evaluator.evaluate(response, ground_truth)
    return result["result"]


def main():
    """Demo of the evaluation module."""
    print("=" * 80)
    print("Rule-based Evaluator Demo")
    print("=" * 80)

    # Create evaluator
    evaluator = RuleBasedEvaluator(case_sensitive=False)

    # Test cases
    test_cases = [
        {
            "response": "The admin work lead of the AIPolicyBench team is Isabella.",
            "ground_truth": "Isabella",
            "expected": "correct"
        },
        {
            "response": "There are 4 members in the AIPolicyBench teams.",
            "ground_truth": "4",
            "expected": "correct"
        },
        {
            "response": "Charles is the evaluation part leader of the AIPolicyBench team.",
            "ground_truth": "Charles",
            "expected": "correct"
        },
        {
            "response": "The team has multiple members working on different aspects.",
            "ground_truth": "Isabella",
            "expected": "incorrect"
        },
        {
            "response": "John leads the evaluation part.",
            "ground_truth": "Charles",
            "expected": "incorrect"
        }
    ]

    print("\nTest Cases:")
    print("-" * 80)

    for i, test in enumerate(test_cases, 1):
        result = evaluator.evaluate(test["response"], test["ground_truth"])
        status = "✓" if result["result"] == test["expected"] else "✗"

        print(f"\n{status} Test {i}:")
        print(f"  Response: {test['response']}")
        print(f"  Ground Truth: {test['ground_truth']}")
        print(f"  Result: {result['result']} (expected: {test['expected']})")

    print("\n" + "=" * 80)
    print("Batch Evaluation Test:")
    print("=" * 80)

    responses = [tc["response"] for tc in test_cases[:3]]
    ground_truths = [tc["ground_truth"] for tc in test_cases[:3]]

    batch_result = evaluator.evaluate_batch(responses, ground_truths)

    print(f"\nStatistics:")
    print(f"  Total: {batch_result['statistics']['total']}")
    print(f"  Correct: {batch_result['statistics']['correct']}")
    print(f"  Incorrect: {batch_result['statistics']['incorrect']}")
    print(f"  Accuracy: {batch_result['statistics']['accuracy']:.2f}%")
    print(f"  Method: {batch_result['method']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
