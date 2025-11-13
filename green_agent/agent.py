#!/usr/bin/env python3
"""
Predefined Query Interface for AIPolicyBench
Uses only predefined queries with ground truth answers for evaluation.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from safety_datasets_rag import SafetyDatasetsRAG
from evaluation import RuleBasedEvaluator, LLMJudgeEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredefinedQueryInterface:
    """Interface for predefined queries with ground truth evaluation."""

    def __init__(
        self,
        queries_file: str = "data/predefined_queries.json",
        vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl",
        use_llm_judge: bool = False,
        llm_provider: str = "deepseek"
    ):
        """
        Initialize the predefined query interface.

        Args:
            queries_file: Path to JSON file with predefined queries
            vector_db_path: Path to the saved vector database
            use_llm_judge: Whether to use LLM-as-a-judge evaluation (default: False)
            llm_provider: LLM provider for judge evaluation (default: "deepseek")
        """
        self.queries_file = queries_file
        self.predefined_queries = []
        self.rag_system = SafetyDatasetsRAG(vector_db_path)
        self.use_llm_judge = use_llm_judge

        # Initialize evaluator based on mode
        if use_llm_judge:
            self.evaluator = LLMJudgeEvaluator(provider=llm_provider, temperature=0.0)
            logger.info(f"Using LLM-as-a-judge evaluation with {llm_provider}")
        else:
            self.evaluator = RuleBasedEvaluator(case_sensitive=False)
            logger.info("Using rule-based evaluation")

        # Load predefined queries
        self._load_queries()

    def _load_queries(self):
        """Load predefined queries from JSON file."""
        try:
            if not os.path.exists(self.queries_file):
                logger.error(f"Queries file not found: {self.queries_file}")
                return

            with open(self.queries_file, 'r') as f:
                data = json.load(f)
                self.predefined_queries = data.get('queries', [])

            logger.info(f"Loaded {len(self.predefined_queries)} predefined queries")

        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            self.predefined_queries = []

    def initialize(self) -> bool:
        """Initialize the RAG system."""
        return self.rag_system.load_vector_db()

    async def evaluate_query(self, query_id: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate a single predefined query.

        Args:
            query_id: ID of the query to evaluate (1-indexed)
            top_k: Number of datasets to retrieve

        Returns:
            Dictionary with query, response, ground truth, and evaluation result
        """
        # Find query by ID
        query_data = None
        for q in self.predefined_queries:
            if q['id'] == query_id:
                query_data = q
                break

        if not query_data:
            return {
                "error": f"Query ID {query_id} not found",
                "available_ids": [q['id'] for q in self.predefined_queries]
            }

        query = query_data['query']
        ground_truth = query_data['ground_truth']

        logger.info(f"Evaluating Query {query_id}: {query}")

        # Get RAG response
        rag_result = await self.rag_system.complete_rag_query(query, top_k, use_llm=True)

        if "error" in rag_result:
            return {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "error": rag_result['error'],
                "evaluation_result": "incorrect"
            }

        response = rag_result.get("generated_response", "")
        context = rag_result.get("context", "")

        # Evaluate response
        if self.use_llm_judge:
            # LLM-as-a-judge requires question and context
            eval_result = await self.evaluator.evaluate(
                response=response,
                ground_truth=ground_truth,
                question=query,
                context=context
            )
        else:
            # Rule-based evaluation
            eval_result = self.evaluator.evaluate(response, ground_truth)

        return {
            "query_id": query_id,
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
            "evaluation_result": eval_result["result"],
            "evaluation_method": eval_result["method"],
            "retrieved_datasets": rag_result.get("retrieved_datasets", []),
            "context": rag_result.get("context", "")
        }

    async def evaluate_all_queries(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate all predefined queries.

        Args:
            top_k: Number of datasets to retrieve for each query

        Returns:
            Dictionary with all results and statistics
        """
        if not self.predefined_queries:
            return {"error": "No predefined queries loaded"}

        logger.info(f"Evaluating all {len(self.predefined_queries)} queries...")

        results = []
        correct_count = 0

        for query_data in self.predefined_queries:
            result = await self.evaluate_query(query_data['id'], top_k)
            results.append(result)

            if result.get("evaluation_result") == "correct":
                correct_count += 1

        total = len(results)
        accuracy = (correct_count / total * 100) if total > 0 else 0.0

        logger.info(f"Evaluation complete: {correct_count}/{total} correct ({accuracy:.2f}%)")

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

    def display_query_result(self, result: Dict[str, Any]):
        """Display a single query evaluation result."""
        print("\n" + "=" * 80)
        print(f"Query ID: {result.get('query_id', 'N/A')}")
        print("=" * 80)

        print(f"\n📝 Query: {result.get('query', 'N/A')}")
        print("-" * 80)

        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n💬 Model Response:")
            print(result.get('response', 'No response'))
            print(f"\n✓ Ground Truth: {result.get('ground_truth', 'N/A')}")

            eval_result = result.get('evaluation_result', 'unknown')
            if eval_result == "correct":
                print(f"\n✅ Evaluation: CORRECT")
            elif eval_result == "miss":
                print(f"\n⚠️  Evaluation: MISS (expressed uncertainty)")
            elif eval_result == "hallucination":
                print(f"\n❌ Evaluation: HALLUCINATION (incorrect)")
            else:
                print(f"\n❓ Evaluation: {eval_result.upper()}")

            print(f"📊 Method: {result.get('evaluation_method', 'Rule-based')}")

            # Show confidence and reasoning for LLM judge
            if 'confidence' in result:
                print(f"🎯 Confidence: {result['confidence']:.2f}")
            if 'reasoning' in result:
                print(f"💭 Reasoning: {result['reasoning']}")

            # Show retrieved datasets count
            datasets = result.get('retrieved_datasets', [])
            print(f"📚 Retrieved Datasets: {len(datasets)}")

        print("=" * 80)

    def display_all_results(self, batch_result: Dict[str, Any]):
        """Display all evaluation results with statistics."""
        print("\n" + "=" * 80)
        print("🎯 AIPOLICYBENCH EVALUATION RESULTS")
        print("=" * 80)

        if "error" in batch_result:
            print(f"\n❌ Error: {batch_result['error']}")
            return

        # Display individual results
        for result in batch_result.get('results', []):
            self.display_query_result(result)

        # Display statistics
        stats = batch_result.get('statistics', {})
        print("\n" + "=" * 80)
        print("📊 OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Queries: {stats.get('total', 0)}")
        print(f"Correct: {stats.get('correct', 0)}")
        print(f"Hallucination: {stats.get('hallucination', 0)}")
        print(f"Miss: {stats.get('miss', 0)}")
        print(f"")
        print(f"Correct Rate: {stats.get('correct_rate', 0.0):.2f}%")
        print(f"Hallucination Rate: {stats.get('hallucination_rate', 0.0):.2f}%")
        print(f"Miss Rate: {stats.get('miss_rate', 0.0):.2f}%")
        print(f"Factuality Rate: {stats.get('factuality_rate', 0.0):.2f}%")
        print(f"")
        print(f"Evaluation Method: {batch_result.get('method', 'Rule-based')}")
        if 'provider' in batch_result:
            print(f"LLM Provider: {batch_result['provider']}")
        print("=" * 80)

    def interactive_mode(self):
        """Run interactive mode with predefined queries."""
        if not self.initialize():
            print("❌ Failed to initialize RAG system")
            return

        if not self.predefined_queries:
            print("❌ No predefined queries loaded")
            return

        print("\n" + "=" * 80)
        print("🎯 AIPOLICYBENCH - PREDEFINED QUERY EVALUATION SYSTEM")
        print("=" * 80)
        print(f"Loaded {len(self.predefined_queries)} predefined queries")
        print("\nCommands:")
        print("  1-{}: Evaluate specific query".format(len(self.predefined_queries)))
        print("  all: Evaluate all queries")
        print("  list: Show all queries")
        print("  quit/exit: Exit the program")
        print("=" * 80)

        while True:
            try:
                command = input("\n🎯 Enter command: ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if command == 'list':
                    self.show_queries()
                    continue

                if command == 'all':
                    print("\n⏳ Evaluating all queries...")
                    result = asyncio.run(self.evaluate_all_queries())
                    self.display_all_results(result)
                    continue

                # Try to parse as query ID
                try:
                    query_id = int(command)
                    if 1 <= query_id <= len(self.predefined_queries):
                        print(f"\n⏳ Evaluating query {query_id}...")
                        result = asyncio.run(self.evaluate_query(query_id))
                        self.display_query_result(result)
                    else:
                        print(f"❌ Invalid query ID. Available: 1-{len(self.predefined_queries)}")
                except ValueError:
                    print("❌ Invalid command. Type 'list' to see available commands.")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")

    def show_queries(self):
        """Display all predefined queries."""
        print("\n" + "=" * 80)
        print("📋 PREDEFINED QUERIES")
        print("=" * 80)

        for q in self.predefined_queries:
            print(f"\nQuery {q['id']}:")
            print(f"  Question: {q['query']}")
            print(f"  Ground Truth: {q['ground_truth']}")

        print("=" * 80)


def main():
    """Main function to run the predefined query interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AIPolicyBench Predefined Query Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--queries_file", default="data/predefined_queries.json",
                        help="Path to predefined queries JSON file")
    parser.add_argument("--vector_db", default="./vector_db/safety_datasets_tfidf_db.pkl",
                        help="Path to the vector database file")
    parser.add_argument("--query_id", type=int,
                        help="Evaluate specific query by ID")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all queries")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of datasets to retrieve")
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM-as-a-judge evaluation instead of rule-based")
    parser.add_argument("--llm_provider", default="deepseek",
                        choices=["deepseek", "openai", "anthropic"],
                        help="LLM provider for judge evaluation (default: deepseek)")

    args = parser.parse_args()

    try:
        # Initialize interface
        interface = PredefinedQueryInterface(
            queries_file=args.queries_file,
            vector_db_path=args.vector_db,
            use_llm_judge=args.use_llm_judge,
            llm_provider=args.llm_provider
        )

        if not interface.initialize():
            print("❌ Failed to initialize RAG system")
            return

        # Evaluate specific query
        if args.query_id:
            print(f"⏳ Evaluating query {args.query_id}...")
            result = asyncio.run(interface.evaluate_query(args.query_id, args.top_k))
            interface.display_query_result(result)

        # Evaluate all queries
        elif args.all:
            print("⏳ Evaluating all queries...")
            result = asyncio.run(interface.evaluate_all_queries(args.top_k))
            interface.display_all_results(result)

        # Interactive mode
        else:
            interface.interactive_mode()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
