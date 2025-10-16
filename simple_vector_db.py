#!/usr/bin/env python3
"""
Safety Datasets Vector Database System
Reads CSV file and creates a searchable TF-IDF vector database for safety datasets.
"""

import os
import sys
import pandas as pd
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
    """Main processor for safety datasets CSV to vector database conversion."""
    
    def __init__(self, csv_file_path: str, vector_db_path: str = "./vector_db"):
        """
        Initialize the safety datasets processor.
        
        Args:
            csv_file_path: Path to the CSV file
            vector_db_path: Path to store the vector database
        """
        self.csv_file_path = csv_file_path
        self.vector_db_path = vector_db_path
        self.vector_db = None
        
    def read_csv(self) -> pd.DataFrame:
        """Read and return the CSV file as a DataFrame."""
        logger.info(f"Reading CSV file: {self.csv_file_path}")
        df = pd.read_csv(self.csv_file_path)
        logger.info(f"Successfully read {len(df)} rows from CSV file")
        return df
    
    def prepare_data_for_vector_db(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare data from DataFrame for vector database.
        
        Args:
            df: DataFrame containing the safety datasets
            
        Returns:
            List of dictionaries with processed data
        """
        logger.info("Preparing data for vector database...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            # Create a comprehensive text representation of each dataset
            text_content = self._create_dataset_text(row)
            
            # Create metadata
            metadata = {
                'dataset_name': row.get('data_name', ''),
                'purpose_type': row.get('purpose_type', ''),
                'purpose_tags': row.get('purpose_tags', ''),
                'entries_type': row.get('entries_type', ''),
                'entries_languages': row.get('entries_languages', ''),
                'entries_n': row.get('entries_n', ''),
                'publication_date': row.get('publication_date', ''),
                'publication_affils': row.get('publication_affils', ''),
                'publication_sector': row.get('publication_sector', ''),
                'publication_name': row.get('publication_name', ''),
                'publication_venue': row.get('publication_venue', ''),
                'publication_url': row.get('publication_url', ''),
                'access_license': row.get('access_license', ''),
                'row_index': idx
            }
            
            processed_data.append({
                'id': f"dataset_{idx}",
                'text': text_content,
                'metadata': metadata
            })
            
        logger.info(f"Prepared {len(processed_data)} datasets for vector database")
        return processed_data
    
    def _create_dataset_text(self, row: pd.Series) -> str:
        """
        Create a comprehensive text representation of a dataset row.
        
        Args:
            row: Single row from the DataFrame
            
        Returns:
            Formatted text representation
        """
        text_parts = []
        
        # Basic information
        if row.get('data_name'):
            text_parts.append(f"Dataset Name: {row['data_name']}")
        
        if row.get('purpose_type'):
            text_parts.append(f"Purpose Type: {row['purpose_type']}")
            
        if row.get('purpose_tags') and pd.notna(row.get('purpose_tags')):
            text_parts.append(f"Purpose Tags: {row['purpose_tags']}")
            
        if row.get('purpose_stated'):
            text_parts.append(f"Purpose: {row['purpose_stated']}")
            
        if row.get('purpose_llmdev'):
            text_parts.append(f"LLM Development Purpose: {row['purpose_llmdev']}")
        
        # Dataset details
        if row.get('entries_type'):
            text_parts.append(f"Entries Type: {row['entries_type']}")
            
        if row.get('entries_languages'):
            text_parts.append(f"Languages: {row['entries_languages']}")
            
        if row.get('entries_n'):
            text_parts.append(f"Number of Entries: {row['entries_n']}")
            
        if row.get('entries_unit'):
            text_parts.append(f"Entry Unit: {row['entries_unit']}")
            
        if row.get('entries_detail'):
            text_parts.append(f"Entry Details: {row['entries_detail']}")
        
        # Creation information
        if row.get('creation_creator_type'):
            text_parts.append(f"Creator Type: {row['creation_creator_type']}")
            
        if row.get('creation_source_type') and pd.notna(row.get('creation_source_type')):
            text_parts.append(f"Source Type: {row['creation_source_type']}")
            
        if row.get('creation_detail'):
            text_parts.append(f"Creation Details: {row['creation_detail']}")
        
        # Access information
        if row.get('access_git_url'):
            text_parts.append(f"GitHub URL: {row['access_git_url']}")
            
        if row.get('access_hf_url') and row.get('access_hf_url') != 'not available':
            text_parts.append(f"HuggingFace URL: {row['access_hf_url']}")
            
        if row.get('access_license'):
            text_parts.append(f"License: {row['access_license']}")
        
        # Publication information
        if row.get('publication_date'):
            text_parts.append(f"Publication Date: {row['publication_date']}")
            
        if row.get('publication_affils'):
            text_parts.append(f"Affiliations: {row['publication_affils']}")
            
        if row.get('publication_sector'):
            text_parts.append(f"Sector: {row['publication_sector']}")
            
        if row.get('publication_name'):
            text_parts.append(f"Publication: {row['publication_name']}")
            
        if row.get('publication_venue'):
            text_parts.append(f"Venue: {row['publication_venue']}")
            
        if row.get('publication_url'):
            text_parts.append(f"Publication URL: {row['publication_url']}")
        
        # Additional notes
        if row.get('other_notes') and pd.notna(row.get('other_notes')):
            text_parts.append(f"Notes: {row['other_notes']}")
            
        if row.get('other_date_added'):
            text_parts.append(f"Date Added: {row['other_date_added']}")
        
        return "\n".join(text_parts)
    
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
    
    def process_csv_to_vector_db(self) -> SimpleTFIDFVectorDB:
        """
        Complete pipeline: Read CSV -> Process Data -> Build Vector DB.
        
        Returns:
            Built SimpleTFIDFVectorDB instance
        """
        logger.info("Starting CSV to Vector DB pipeline...")
        
        # Step 1: Read CSV
        df = self.read_csv()
        
        # Step 2: Prepare data
        processed_data = self.prepare_data_for_vector_db(df)
        
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
            logger.error("Vector database not built yet. Run process_csv_to_vector_db() first.")
            return
        
        logger.info(f"Testing vector database with query: '{query}'")
        
        try:
            results = self.vector_db.search(query, top_k=top_k)
            
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                logger.info(f"\n--- Result {i} ---")
                logger.info(f"Dataset: {result['metadata'].get('dataset_name', 'Unknown')}")
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
    """Main function to run the CSV to vector DB converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSV file to TF-IDF vector database")
    parser.add_argument("--csv_file", default="data/safety_datasets.csv", 
                       help="Path to the CSV file")
    parser.add_argument("--vector_db_path", default="./vector_db", 
                       help="Path to store the vector database")
    parser.add_argument("--test_query", default="safety evaluation", 
                       help="Test query for the vector database")
    parser.add_argument("--top_k", type=int, default=5, 
                       help="Number of results to return for test query")
    parser.add_argument("--save", action="store_true", 
                       help="Save the vector database to disk")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    try:
        # Initialize processor
        processor = SafetyDatasetsProcessor(
            csv_file_path=args.csv_file,
            vector_db_path=args.vector_db_path
        )
        
        # Process CSV to vector database
        vector_db = processor.process_csv_to_vector_db()
        
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
