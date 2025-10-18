#!/usr/bin/env python3
"""
PDF Parser and Chunker

This script parses PDF documents from ./docs directory, chunks them into suitable sizes,
and generates ./data/safety_datasets.json with id and text fields.

Features:
- Extracts text from PDF files
- Intelligent chunking with overlap for context preservation
- Generates unique IDs for each chunk
- Outputs JSON format with only 'id' and 'text' fields
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        import PyPDF2

        logger.info(f"Extracting text from: {pdf_path}")
        text = ""

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"

        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text

    except ImportError:
        logger.error("PyPDF2 not installed. Trying pdfplumber...")
        try:
            import pdfplumber

            logger.info(f"Extracting text from: {pdf_path}")
            text = ""

            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"PDF has {num_pages} pages")

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text

        except ImportError:
            logger.error("Neither PyPDF2 nor pdfplumber is installed.")
            logger.error("Install with: pip install PyPDF2 or pip install pdfplumber")
            raise


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    Args:
        text: Raw text from PDF

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks of suitable size.

    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size to keep

    Returns:
        List of text chunks
    """
    logger.info(f"Chunking text (size: {len(text)} chars, chunk_size: {chunk_size}, overlap: {overlap})")

    # Split by paragraphs first for better semantic chunking
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph itself is too long, split by sentences
        if len(paragraph) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap
                    words = current_chunk.split()
                    overlap_words = min(overlap // 5, len(words))  # Approximate word overlap
                    overlap_text = ' '.join(words[-overlap_words:]) if overlap_words > 0 else ""
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        else:
            # If adding this paragraph exceeds chunk size, save current chunk and start new one
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap
                words = current_chunk.split()
                overlap_words = min(overlap // 5, len(words))  # Approximate word overlap
                overlap_text = ' '.join(words[-overlap_words:]) if overlap_words > 0 else ""
                current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

    # Add the last chunk
    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def create_document_id(pdf_filename: str, chunk_index: int) -> str:
    """
    Create a unique ID for a document chunk.

    Args:
        pdf_filename: Name of the source PDF file
        chunk_index: Index of the chunk

    Returns:
        Unique document ID
    """
    # Remove file extension and clean filename
    base_name = Path(pdf_filename).stem
    # Remove leading numbers and clean special characters
    base_name = re.sub(r'^\d+\s*', '', base_name)
    base_name = re.sub(r'[^\w\s-]', '', base_name)
    base_name = re.sub(r'\s+', '_', base_name)

    return f"{base_name}_chunk_{chunk_index}"


def process_pdf_file(
    pdf_path: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, str]]:
    """
    Process a single PDF file into chunks.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of dictionaries with 'id' and 'text' fields
    """
    pdf_filename = Path(pdf_path).name
    logger.info(f"Processing PDF: {pdf_filename}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)

    # Clean text
    text = clean_text(text)

    # Chunk text
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Create documents with metadata
    documents = []
    for idx, chunk in enumerate(chunks):
        doc_id = create_document_id(pdf_filename, idx)

        # Add source information to the text
        formatted_text = f"Source: {pdf_filename}\n\n{chunk}"

        documents.append({
            "id": doc_id,
            "text": formatted_text
        })

    logger.info(f"Created {len(documents)} document chunks from {pdf_filename}")
    return documents


def parse_all_pdfs(
    docs_dir: str = "./docs",
    output_file: str = "./data/safety_datasets.json",
    chunk_size: int = 1000,
    overlap: int = 200
):
    """
    Parse all PDF files in the docs directory and generate JSON output.

    Args:
        docs_dir: Directory containing PDF files
        output_file: Output JSON file path
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Docs directory not found: {docs_dir}")
        return

    # Find all PDF files
    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process all PDFs
    all_documents = []

    for pdf_file in sorted(pdf_files):
        try:
            documents = process_pdf_file(
                str(pdf_file),
                chunk_size=chunk_size,
                overlap=overlap
            )
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            continue

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(all_documents)} documents to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully created {output_file}")
    logger.info(f"Total documents: {len(all_documents)}")

    # Print statistics
    total_chars = sum(len(doc['text']) for doc in all_documents)
    avg_chunk_size = total_chars / len(all_documents) if all_documents else 0

    logger.info(f"Statistics:")
    logger.info(f"  - Total documents: {len(all_documents)}")
    logger.info(f"  - Total characters: {total_chars}")
    logger.info(f"  - Average chunk size: {avg_chunk_size:.0f} characters")


def main():
    """Main function with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse PDF documents and generate chunked JSON dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs with default settings
  python scripts/parse_pdfs_to_json.py

  # Custom chunk size and overlap
  python scripts/parse_pdfs_to_json.py --chunk_size 1500 --overlap 300

  # Custom input/output paths
  python scripts/parse_pdfs_to_json.py --docs_dir ./documents --output ./output/data.json
        """
    )

    parser.add_argument(
        "--docs_dir",
        default="./docs",
        help="Directory containing PDF files (default: ./docs)"
    )
    parser.add_argument(
        "--output",
        default="./data/safety_datasets.json",
        help="Output JSON file path (default: ./data/safety_datasets.json)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PDF Parser and Chunker")
    logger.info("=" * 80)
    logger.info(f"Docs directory: {args.docs_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Overlap: {args.overlap}")
    logger.info("=" * 80)

    try:
        parse_all_pdfs(
            docs_dir=args.docs_dir,
            output_file=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

        logger.info("=" * 80)
        logger.info("Processing completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
