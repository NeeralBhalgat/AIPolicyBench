#!/usr/bin/env python3
"""
Evaluation Module for AIPolicyBench
Implements rule-based evaluation and LLM-as-a-judge evaluation.

Evaluation Paradigm: Question-Answering (QA)

Evaluation Methods:
1. Rule-based Evaluation: Direct string matching and pattern detection
2. LLM-as-a-Judge: Uses a trusted LLM to evaluate responses

Evaluation Metrics:
- Correct Rate: Proportion of questions answered correctly (Higher is better)
- Hallucination Rate: Proportion of factually incorrect responses (Lower is better)
- Miss Rate: Proportion of uncertain/don't know responses (Rewards uncertainty over wrong answers)
- Factuality Rate: Correct Rate - Hallucination Rate + c * Miss Rate (Higher is better)
"""

import logging
import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from utils.llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common uncertainty phrases
UNCERTAINTY_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i am not sure",
    "not certain",
    "unclear",
    "cannot determine",
    "can't determine",
    "unable to answer",
    "don't have enough information",
    "insufficient information",
    "i cannot answer",
    "i can't answer"
]


class RuleBasedEvaluator:
    """
    Rule-based evaluator using substring matching and pattern detection.

    Classifies responses into three categories:
    - correct: Response contains the ground truth
    - miss: Response expresses uncertainty or admits not knowing
    - hallucination: Response attempts an answer but is incorrect
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the rule-based evaluator.

        Args:
            case_sensitive: Whether to perform case-sensitive matching (default: False)
        """
        self.case_sensitive = case_sensitive
        logger.info(f"Initialized Rule-based Evaluator (case_sensitive={case_sensitive})")

    def _is_uncertain(self, response: str) -> bool:
        """
        Check if the response expresses uncertainty or admits not knowing.

        Args:
            response: The model's response

        Returns:
            True if response contains uncertainty phrases
        """
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in UNCERTAINTY_PHRASES)

    def evaluate(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a model response against ground truth using substring matching.

        Classifies responses as:
        - correct: Response contains the ground truth
        - miss: Response expresses uncertainty
        - hallucination: Response is incorrect

        Args:
            response: The model's generated response
            ground_truth: The expected ground truth answer

        Returns:
            Dictionary with evaluation result:
            {
                "result": "correct", "miss", or "hallucination",
                "method": "Rule-based",
                "ground_truth": ground_truth,
                "response_preview": first 200 chars of response
            }
        """
        if not response:
            logger.warning("Empty response provided")
            return {
                "result": "miss",
                "method": "Rule-based",
                "ground_truth": ground_truth,
                "response_preview": "",
                "reason": "Empty response"
            }

        if not ground_truth:
            logger.warning("Empty ground truth provided")
            return {
                "result": "hallucination",
                "method": "Rule-based",
                "ground_truth": "",
                "response_preview": response[:200] if response else "",
                "reason": "No ground truth to compare"
            }

        # Check for uncertainty first
        is_uncertain = self._is_uncertain(response)

        # Perform substring matching
        if self.case_sensitive:
            match_found = ground_truth in response
        else:
            match_found = ground_truth.lower() in response.lower()

        # Determine result based on matching and uncertainty
        if match_found:
            result = "correct"
        elif is_uncertain:
            result = "miss"
        else:
            result = "hallucination"

        logger.info(f"Evaluation result: {result} (Ground truth: '{ground_truth}')")

        return {
            "result": result,
            "method": "Rule-based",
            "ground_truth": ground_truth,
            "response_preview": response[:200] if len(response) > 200 else response,
            "match_found": match_found,
            "is_uncertain": is_uncertain
        }

    def evaluate_batch(
        self,
        responses: list,
        ground_truths: list,
        miss_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate multiple responses against their ground truths.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers
            miss_weight: Weight for miss rate in factuality calculation (default: 0.5)

        Returns:
            Dictionary with batch evaluation results and statistics including:
            - Correct Rate
            - Hallucination Rate
            - Miss Rate
            - Factuality Rate
        """
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")

        results = []
        correct_count = 0
        hallucination_count = 0
        miss_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            eval_result = self.evaluate(response, ground_truth)
            results.append(eval_result)

            result_type = eval_result["result"]
            if result_type == "correct":
                correct_count += 1
            elif result_type == "hallucination":
                hallucination_count += 1
            elif result_type == "miss":
                miss_count += 1

        total = len(responses)

        # Calculate rates (as percentages)
        correct_rate = (correct_count / total * 100) if total > 0 else 0.0
        hallucination_rate = (hallucination_count / total * 100) if total > 0 else 0.0
        miss_rate = (miss_count / total * 100) if total > 0 else 0.0

        # Calculate factuality rate: Correct - Hallucination + c * Miss
        factuality_rate = correct_rate - hallucination_rate + (miss_weight * miss_rate)

        logger.info(
            f"Batch evaluation complete: "
            f"{correct_count} correct, "
            f"{hallucination_count} hallucination, "
            f"{miss_count} miss "
            f"(Factuality: {factuality_rate:.2f}%)"
        )

        return {
            "results": results,
            "statistics": {
                "total": total,
                "correct": correct_count,
                "hallucination": hallucination_count,
                "miss": miss_count,
                "correct_rate": correct_rate,
                "hallucination_rate": hallucination_rate,
                "miss_rate": miss_rate,
                "factuality_rate": factuality_rate,
                "miss_weight": miss_weight
            },
            "method": "Rule-based"
        }


class LLMJudgeEvaluator:
    """
    LLM-as-a-judge evaluator that uses a trusted LLM (Green Agent) to evaluate
    White Agent responses.

    Classifies responses into three categories:
    - correct: Response correctly answers the question
    - miss: Response expresses uncertainty or admits not knowing
    - hallucination: Response is factually incorrect
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """
        Initialize the LLM-as-a-judge evaluator.

        Args:
            provider: LLM provider to use ('openai', 'deepseek', 'anthropic')
            model: Specific model to use (provider-specific)
            api_key: API key for the LLM provider
            temperature: Temperature for LLM generation (0.0 for deterministic)
            max_tokens: Maximum tokens for LLM response (default: 1000)
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM client
        client_kwargs = {}
        if model:
            client_kwargs['model'] = model
        if api_key:
            client_kwargs['api_key'] = api_key

        self.llm_client = LLMClient(provider=provider, **client_kwargs)
        logger.info(f"Initialized LLM Judge Evaluator (provider={provider}, model={model}, max_tokens={max_tokens})")

    def _create_judge_prompt(
        self,
        question: str,
        response: str,
        ground_truth: str,
        context: Optional[str] = None
    ) -> str:
        """
        Create the prompt for the LLM judge (Green Agent).

        Args:
            question: The original question asked
            response: The White Agent's response to evaluate
            ground_truth: The ground truth answer
            context: Optional context that was provided to the White Agent

        Returns:
            Formatted prompt for the judge
        """
        prompt = f"""You are the Green Agent, an expert evaluator tasked with judging the quality and correctness of a White Agent's response to a question about AI policy and safety datasets.

Your task is to evaluate the White Agent's response and classify it into one of three categories:
1. **correct**: The response correctly answers the question and contains the ground truth information
2. **miss**: The response expresses uncertainty or admits not knowing (e.g., "I don't know", "I'm not sure")
3. **hallucination**: The response attempts to answer but is factually incorrect

**Question:**
{question}

**White Agent's Response:**
{response}

**Ground Truth Answer:**
{ground_truth}
"""

        prompt += """
**Instructions:**
Carefully evaluate the White Agent's response against the ground truth. You must classify it as one of:
- "correct": The response contains the ground truth and is factually accurate
- "miss": The response expresses uncertainty or admits not knowing
- "hallucination": The response attempts an answer but is factually incorrect

**Output Format:**
You MUST respond with ONLY a valid JSON object in this exact format (no other text before or after):
{
    "classification": "correct",
    "confidence": 0.95,
    "reasoning": "Your brief explanation here"
}

Replace the values with your evaluation. The classification must be exactly one of: "correct", "miss", or "hallucination".
The confidence must be a number between 0.0 and 1.0.
"""

        return prompt

    async def evaluate(
        self,
        response: str,
        ground_truth: str,
        question: str = "",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a White Agent response using LLM-as-a-judge (Green Agent).

        Args:
            response: The White Agent's generated response
            ground_truth: The expected ground truth answer
            question: The original question (recommended)
            context: Context provided to the White Agent (optional)

        Returns:
            Dictionary with evaluation result including classification and reasoning
        """
        if not response:
            logger.warning("Empty response provided")
            return {
                "result": "miss",
                "method": "LLM-as-a-judge",
                "provider": self.provider,
                "ground_truth": ground_truth,
                "response_preview": "",
                "reason": "Empty response",
                "confidence": 1.0
            }

        if not ground_truth:
            logger.warning("Empty ground truth provided")
            return {
                "result": "hallucination",
                "method": "LLM-as-a-judge",
                "provider": self.provider,
                "ground_truth": "",
                "response_preview": response[:200] if response else "",
                "reason": "No ground truth to compare",
                "confidence": 1.0
            }

        try:
            # Create judge prompt
            judge_prompt = self._create_judge_prompt(
                question=question,
                response=response,
                ground_truth=ground_truth,
                context=context
            )

            # Get LLM judgment
            logger.info("Requesting LLM judgment from Green Agent...")

            llm_response = await self.llm_client.generate_response(
                judge_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Parse JSON response
            llm_response = llm_response.strip()

            # Try to extract JSON from response (handle markdown code blocks, extra text, etc.)
            json_str = None

            # Method 1: Try to find JSON in markdown code block
            if "```json" in llm_response.lower():
                start = llm_response.lower().find("```json") + 7
                end = llm_response.find("```", start)
                if end > start:
                    json_str = llm_response[start:end].strip()

            # Method 2: Try to find JSON in regular code block
            if not json_str and "```" in llm_response:
                start = llm_response.find("```") + 3
                end = llm_response.find("```", start)
                if end > start:
                    json_str = llm_response[start:end].strip()

            # Method 3: Try to find raw JSON object
            if not json_str:
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = llm_response[start_idx:end_idx]

            if not json_str:
                logger.error(f"Could not find JSON in response: {llm_response[:200]}")
                raise ValueError(f"No valid JSON found in LLM response. Response: {llm_response[:200]}")

            # Parse JSON
            try:
                judgment = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {json_str[:200]}")
                raise ValueError(f"Invalid JSON in response: {str(e)}")

            # Extract classification
            classification = judgment.get('classification', 'hallucination').lower()
            confidence = judgment.get('confidence', 0.5)
            reasoning = judgment.get('reasoning', 'No reasoning provided')

            # Validate classification
            if classification not in ['correct', 'miss', 'hallucination']:
                logger.warning(f"Invalid classification '{classification}', defaulting to 'hallucination'")
                classification = 'hallucination'

            logger.info(f"LLM Judge result: {classification} (confidence: {confidence:.2f})")

            return {
                "result": classification,
                "method": "LLM-as-a-judge",
                "provider": self.provider,
                "ground_truth": ground_truth,
                "response_preview": response[:200] if len(response) > 200 else response,
                "confidence": confidence,
                "reasoning": reasoning,
                "raw_judgment": judgment
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM judgment as JSON: {e}")
            logger.error(f"LLM Response: {llm_response}")
            return {
                "result": "error",
                "method": "LLM-as-a-judge",
                "provider": self.provider,
                "ground_truth": ground_truth,
                "response_preview": response[:200] if response else "",
                "reason": f"Failed to parse LLM judgment: {str(e)}",
                "confidence": 0.0,
                "llm_response": llm_response
            }
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            return {
                "result": "error",
                "method": "LLM-as-a-judge",
                "provider": self.provider,
                "ground_truth": ground_truth,
                "response_preview": response[:200] if response else "",
                "reason": f"LLM evaluation error: {str(e)}",
                "confidence": 0.0
            }

    async def evaluate_batch(
        self,
        responses: list,
        ground_truths: list,
        questions: list = None,
        contexts: list = None,
        miss_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate multiple responses using LLM-as-a-judge.

        Args:
            responses: List of White Agent responses
            ground_truths: List of ground truth answers
            questions: List of original questions (optional)
            contexts: List of contexts (optional)
            miss_weight: Weight for miss rate in factuality calculation (default: 0.5)

        Returns:
            Dictionary with batch evaluation results and statistics including:
            - Correct Rate
            - Hallucination Rate
            - Miss Rate
            - Factuality Rate
        """
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")

        if questions is None:
            questions = [""] * len(responses)
        if contexts is None:
            contexts = [None] * len(responses)

        results = []
        correct_count = 0
        hallucination_count = 0
        miss_count = 0
        error_count = 0

        for i, (response, ground_truth, question, context) in enumerate(
            zip(responses, ground_truths, questions, contexts)
        ):
            logger.info(f"Evaluating response {i+1}/{len(responses)}...")
            eval_result = await self.evaluate(
                response=response,
                ground_truth=ground_truth,
                question=question,
                context=context
            )
            results.append(eval_result)

            result_type = eval_result["result"]
            if result_type == "correct":
                correct_count += 1
            elif result_type == "hallucination":
                hallucination_count += 1
            elif result_type == "miss":
                miss_count += 1
            elif result_type == "error":
                error_count += 1

        total = len(responses)

        # Calculate rates (as percentages)
        correct_rate = (correct_count / total * 100) if total > 0 else 0.0
        hallucination_rate = (hallucination_count / total * 100) if total > 0 else 0.0
        miss_rate = (miss_count / total * 100) if total > 0 else 0.0
        error_rate = (error_count / total * 100) if total > 0 else 0.0

        # Calculate factuality rate: Correct - Hallucination + c * Miss
        factuality_rate = correct_rate - hallucination_rate + (miss_weight * miss_rate)

        logger.info(
            f"Batch LLM evaluation complete: "
            f"{correct_count} correct, "
            f"{hallucination_count} hallucination, "
            f"{miss_count} miss, "
            f"{error_count} error "
            f"(Factuality: {factuality_rate:.2f}%)"
        )

        return {
            "results": results,
            "statistics": {
                "total": total,
                "correct": correct_count,
                "hallucination": hallucination_count,
                "miss": miss_count,
                "error": error_count,
                "correct_rate": correct_rate,
                "hallucination_rate": hallucination_rate,
                "miss_rate": miss_rate,
                "error_rate": error_rate,
                "factuality_rate": factuality_rate,
                "miss_weight": miss_weight
            },
            "method": "LLM-as-a-judge",
            "provider": self.provider
        }


def evaluate_single(response: str, ground_truth: str, case_sensitive: bool = False) -> str:
    """
    Convenience function to evaluate a single response using rule-based evaluation.

    Args:
        response: The model's generated response
        ground_truth: The expected ground truth answer
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        "correct", "miss", or "hallucination"
    """
    evaluator = RuleBasedEvaluator(case_sensitive=case_sensitive)
    result = evaluator.evaluate(response, ground_truth)
    return result["result"]


def main():
    """Demo of the evaluation module."""
    print("=" * 80)
    print("AIPolicyBench Evaluation Module Demo")
    print("=" * 80)

    # Create evaluator
    evaluator = RuleBasedEvaluator(case_sensitive=False)

    # Test cases covering correct, miss, and hallucination scenarios
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
            "response": "I'm not sure who leads the evaluation part.",
            "ground_truth": "Charles",
            "expected": "miss"
        },
        {
            "response": "John leads the evaluation part.",
            "ground_truth": "Charles",
            "expected": "hallucination"
        },
        {
            "response": "I don't have enough information to answer that question.",
            "ground_truth": "Isabella",
            "expected": "miss"
        }
    ]

    print("\n" + "=" * 80)
    print("Rule-based Evaluation Test:")
    print("=" * 80)

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

    responses = [tc["response"] for tc in test_cases]
    ground_truths = [tc["ground_truth"] for tc in test_cases]

    batch_result = evaluator.evaluate_batch(responses, ground_truths, miss_weight=0.5)

    print(f"\nStatistics:")
    print(f"  Total: {batch_result['statistics']['total']}")
    print(f"  Correct: {batch_result['statistics']['correct']}")
    print(f"  Hallucination: {batch_result['statistics']['hallucination']}")
    print(f"  Miss: {batch_result['statistics']['miss']}")
    print(f"  Correct Rate: {batch_result['statistics']['correct_rate']:.2f}%")
    print(f"  Hallucination Rate: {batch_result['statistics']['hallucination_rate']:.2f}%")
    print(f"  Miss Rate: {batch_result['statistics']['miss_rate']:.2f}%")
    print(f"  Factuality Rate: {batch_result['statistics']['factuality_rate']:.2f}%")
    print(f"  Method: {batch_result['method']}")
    print("=" * 80)

    print("\nNote: LLM-as-a-judge evaluation requires API credentials.")
    print("Use LLMJudgeEvaluator for more nuanced evaluation with reasoning.")


if __name__ == "__main__":
    main()
