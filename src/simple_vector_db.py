#!/usr/bin/env python3
"""
Safety Datasets Vector Database System
Reads JSON file and creates a searchable TF-IDF vector database for safety datasets.
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTFIDFVectorDB:
    """Simple vector database using TF-IDF vectors."""
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the TF-IDF vector database.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to TF-IDF vector database...")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Generate TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Store data
        self.documents.extend(documents)
        self.metadatas.extend(metadatas or [{}] * len(documents))
        self.ids.extend(ids or [f"doc_{i}" for i in range(len(documents))])
        
        logger.info(f"Successfully added {len(documents)} documents. TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with text, metadata, and similarity score
        """
        if self.tfidf_matrix is None or len(self.documents) == 0:
            logger.warning("No documents in database")
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
                'id': self.ids[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the vector database to disk."""
        logger.info(f"Saving TF-IDF vector database to {filepath}")
        
        data = {
            'tfidf_matrix': self.tfidf_matrix,
            'vectorizer': self.vectorizer,
            'documents': self.documents,
            'metadatas': self.metadatas,
            'ids': self.ids,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("TF-IDF vector database saved successfully")
    
    def load(self, filepath: str):
        """Load the vector database from disk."""
        logger.info(f"Loading TF-IDF vector database from {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.tfidf_matrix = data['tfidf_matrix']
        self.vectorizer = data['vectorizer']
        self.documents = data['documents']
        self.metadatas = data['metadatas']
        self.ids = data['ids']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        
        logger.info(f"Loaded TF-IDF vector database with {len(self.documents)} documents")

class SafetyDatasetsProcessor:
    """Main processor for safety datasets JSON to vector database conversion."""

    def __init__(self, json_file_path: str, vector_db_path: str = "./vector_db"):
        """
        Initialize the safety datasets processor.

        Args:
            json_file_path: Path to the JSON file
            vector_db_path: Path to store the vector database
        """
        self.json_file_path = json_file_path
        self.vector_db_path = vector_db_path
        self.vector_db = None
        
    def read_json(self) -> List[Dict[str, str]]:
        """Read and return the JSON file as a list of dictionaries."""
        logger.info(f"Reading JSON file: {self.json_file_path}")
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully read {len(data)} datasets from JSON file")
        return data
    
    def prepare_data_for_vector_db(self, json_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Prepare data from JSON for vector database.

        Args:
            json_data: List of dictionaries with 'id' and 'text' fields

        Returns:
            List of dictionaries ready for vector database
        """
        logger.info("Preparing data for vector database...")

        processed_data = []

        for item in json_data:
            # Data is already in the correct format: id and text
            processed_data.append({
                'id': item['id'],
                'text': item['text'],
                'metadata': {}  # No metadata in simple format
            })

        logger.info(f"Prepared {len(processed_data)} datasets for vector database")
        return processed_data


    def build_vector_database(self, data: List[Dict[str, Any]]) -> SimpleTFIDFVectorDB:
        """
        Build the vector database from processed data.
        
        Args:
            data: List of processed dataset dictionaries
            
        Returns:
            Initialized SimpleTFIDFVectorDB instance
        """
        logger.info("Building TF-IDF vector database...")
        
        # Initialize vector database
        self.vector_db = SimpleTFIDFVectorDB(max_features=5000, ngram_range=(1, 2))
        
        # Add documents to the database
        documents = [item['text'] for item in data]
        metadatas = [item['metadata'] for item in data]
        ids = [item['id'] for item in data]
        
        self.vector_db.add_documents(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully built TF-IDF vector database with {len(data)} documents")
        return self.vector_db
    
    def process_json_to_vector_db(self) -> SimpleTFIDFVectorDB:
        """
        Complete pipeline: Read JSON -> Process Data -> Build Vector DB.

        Returns:
            Built SimpleTFIDFVectorDB instance
        """
        logger.info("Starting JSON to Vector DB pipeline...")

        # Step 1: Read JSON
        json_data = self.read_json()

        # Step 2: Prepare data
        processed_data = self.prepare_data_for_vector_db(json_data)

        # Step 3: Build vector database
        vector_db = self.build_vector_database(processed_data)

        logger.info("Pipeline completed successfully!")
        return vector_db
    
    def test_vector_db(self, query: str = "safety evaluation", top_k: int = 5):
        """
        Test the vector database with a sample query.

        Args:
            query: Test query string
            top_k: Number of results to return
        """
        if not self.vector_db:
            logger.error("Vector database not built yet. Run process_json_to_vector_db() first.")
            return

        logger.info(f"Testing vector database with query: '{query}'")

        try:
            results = self.vector_db.search(query, top_k=top_k)

            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                logger.info(f"\n--- Result {i} ---")
                logger.info(f"ID: {result['id']}")
                logger.info(f"Score: {result['score']:.4f}")
                logger.info(f"Text preview: {result['text'][:200]}...")

        except Exception as e:
            logger.error(f"Error testing vector database: {e}")
    
    def save_vector_db(self, filepath: str = None):
        """Save the vector database to disk."""
        if not self.vector_db:
            logger.error("No vector database to save")
            return
        
        if filepath is None:
            filepath = os.path.join(self.vector_db_path, "safety_datasets_tfidf_db.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.vector_db.save(filepath)
        logger.info(f"Vector database saved to: {filepath}")

def main():
    """Main function to run the JSON to vector DB converter."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON file to TF-IDF vector database")
    parser.add_argument("--json_file", default="data/safety_datasets.json",
                       help="Path to the JSON file")
    parser.add_argument("--vector_db_path", default="./vector_db",
                       help="Path to store the vector database")
    parser.add_argument("--test_query", default="admin work lead",
                       help="Test query for the vector database")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of results to return for test query")
    parser.add_argument("--save", action="store_true",
                       help="Save the vector database to disk")

    args = parser.parse_args()

    # Check if JSON file exists
    if not os.path.exists(args.json_file):
        logger.error(f"JSON file not found: {args.json_file}")
        return

    try:
        # Initialize processor
        processor = SafetyDatasetsProcessor(
            json_file_path=args.json_file,
            vector_db_path=args.vector_db_path
        )

        # Process JSON to vector database
        vector_db = processor.process_json_to_vector_db()

        # Test the vector database
        processor.test_vector_db(query=args.test_query, top_k=args.top_k)

        # Save if requested
        if args.save:
            processor.save_vector_db()

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()
