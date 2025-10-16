#!/usr/bin/env python3
"""
Safety Datasets Query Interface
Clean interface that calls the RAG system and provides natural language responses.
"""

import os
import sys
import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from safety_datasets_rag import SafetyDatasetsRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryInterface:
    """Clean interface that calls the RAG system and formats responses."""
    
    def __init__(self, vector_db_path: str = "./vector_db/safety_datasets_tfidf_db.pkl", api_key: str = None):
        """
        Initialize the query interface.
        
        Args:
            vector_db_path: Path to the saved vector database
            api_key: DeepSeek API key for LLM generation
        """
        self.rag_system = SafetyDatasetsRAG(vector_db_path, api_key)
    
    def initialize(self) -> bool:
        """Initialize the RAG system."""
        return self.rag_system.load_vector_db()
    
    async def ask_question(self, question: str, top_k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
        """
        Ask a question and get a natural language response.
        
        Args:
            question: User question
            top_k: Number of datasets to retrieve
            use_llm: Whether to generate LLM response
            
        Returns:
            Dictionary with question, answer, and metadata
        """
        result = await self.rag_system.complete_rag_query(question, top_k, use_llm)
        
        if "error" in result:
            return {
                "question": question,
                "answer": f"I'm sorry, I encountered an error: {result['error']}",
                "datasets_found": 0,
                "llm_used": False
            }
        
        return {
            "question": question,
            "answer": result["generated_response"],
            "datasets_found": len(result["retrieved_datasets"]),
            "llm_used": use_llm and self.rag_system.llm_client is not None,
            "retrieved_datasets": result["retrieved_datasets"]
        }
    
    def display_response(self, response: Dict[str, Any], show_datasets: bool = True):
        """Display the response in a clean, natural format."""
        print(f"\n🎯 Question: {response['question']}")
        print("=" * 80)
        
        if show_datasets and response.get("retrieved_datasets"):
            print(f"\n📊 Found {response['datasets_found']} relevant datasets:")
            print("-" * 40)
            for i, dataset in enumerate(response["retrieved_datasets"][:3], 1):  # Show top 3
                metadata = dataset['metadata']
                print(f"{i}. {metadata.get('dataset_name', 'Unknown')} (Score: {dataset['score']:.3f})")
                print(f"   Purpose: {metadata.get('purpose_stated', 'N/A')}")
                print(f"   Publication: {metadata.get('publication_venue', 'N/A')}")
                print()
        
        print(f"\n💬 Answer:")
        print("-" * 40)
        print(response['answer'])
        
        if response['llm_used']:
            print(f"\n✅ Generated using DeepSeek LLM with {response['datasets_found']} datasets")
        else:
            print(f"\nℹ️  Retrieved {response['datasets_found']} datasets (LLM generation disabled)")
        
        print("=" * 80)
    
    def interactive_mode(self):
        """Run the interactive query mode."""
        if not self.initialize():
            return
        
        print("\n" + "="*80)
        print("🎯 AI POLICY DATASET QUERY SYSTEM")
        print("="*80)
        print("Ask questions about AI safety datasets and get natural language answers.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'help' for example questions.")
        print("Type 'status' to see system status.")
        print("Type 'no-llm' to toggle LLM generation.")
        print("="*80)
        
        use_llm = True
        
        while True:
            try:
                question = input("\n🎯 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                if question.lower() == 'status':
                    self.show_status()
                    continue
                
                if question.lower() == 'no-llm':
                    use_llm = not use_llm
                    status = "enabled" if use_llm else "disabled"
                    print(f"LLM generation {status}")
                    continue
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                # Get number of datasets to retrieve
                try:
                    top_k = int(input("Number of datasets to retrieve (default 5): ") or "5")
                except ValueError:
                    top_k = 5
                
                # Ask question and display response
                response = asyncio.run(self.ask_question(question, top_k, use_llm))
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
    
    def show_help(self):
        """Show help with example questions."""
        print("\n" + "="*80)
        print("🎯 EXAMPLE QUESTIONS:")
        print("="*80)
        print("• 'What datasets evaluate AI safety in healthcare applications?'")
        print("• 'Which datasets are best for evaluating AI bias in decision-making?'")
        print("• 'What datasets assess AI compliance with government regulations?'")
        print("• 'Which datasets evaluate AI safety in multilingual contexts?'")
        print("• 'What datasets are available for evaluating AI toxicity and harm?'")
        print("• 'Which datasets assess AI privacy and security risks?'")
        print("• 'What datasets evaluate AI moral reasoning and ethics?'")
        print("• 'Which datasets test AI robustness against adversarial attacks?'")
        print("• 'What datasets evaluate AI fairness across different demographics?'")
        print("• 'Which datasets assess AI safety in Chinese language models?'")
        print("="*80)
        print("💡 You'll get natural language answers with specific dataset recommendations!")
        print("="*80)
    
    def show_status(self):
        """Show system status."""
        status = self.rag_system.get_system_status()
        print("\n" + "="*80)
        print("🔧 SYSTEM STATUS:")
        print("="*80)
        print(f"Vector Database: {'✅ Loaded' if status['vector_db_loaded'] else '❌ Not loaded'}")
        print(f"Datasets Available: {status['vector_db_size']}")
        print(f"LLM Available: {'✅ Yes' if status['llm_available'] else '❌ No'}")
        if status['llm_available']:
            print(f"Model: {status['model']}")
        print(f"API Key: {'✅ Provided' if status['api_key_provided'] else '❌ Missing'}")
        print("="*80)

def main():
    """Main function to run the query interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Policy Dataset Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python query_interface.py --api_key YOUR_DEEPSEEK_KEY
  
  # Single question
  python query_interface.py --question "What datasets evaluate AI bias?" --api_key YOUR_KEY
  
  # Without LLM (retrieval only)
  python query_interface.py --question "What datasets evaluate AI safety?" --no-llm
        """
    )
    
    parser.add_argument("--vector_db", default="./vector_db/safety_datasets_tfidf_db.pkl", 
                       help="Path to the vector database file")
    parser.add_argument("--api_key", type=str,
                       help="DeepSeek API key for LLM generation")
    parser.add_argument("--question", default=None, 
                       help="Single question to ask (non-interactive mode)")
    parser.add_argument("--top_k", type=int, default=5, 
                       help="Number of datasets to retrieve")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM generation (retrieval only)")
    
    args = parser.parse_args()
    
    try:
        # Initialize query interface
        interface = QueryInterface(vector_db_path=args.vector_db, api_key=args.api_key)
        
        if args.question:
            # Single question mode
            if not interface.initialize():
                return
            
            response = asyncio.run(interface.ask_question(args.question, args.top_k, use_llm=not args.no_llm))
            interface.display_response(response)
        else:
            # Interactive mode
            interface.interactive_mode()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()