#!/usr/bin/env python3
"""
Web Search RAG Pipeline
Dynamically searches the web, builds an ephemeral vector DB, and answers questions.
"""

import os
import sys
import logging
import asyncio
import requests
import time
import random
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_vector_db import SimpleTFIDFVectorDB
from utils.llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Domain Configuration ---

# Base reputable TLDs
REPUTABLE_TLDS = ['.gov', '.edu', '.mil', '.org']

# High-priority domains (boost these to top)
PRIMARY_DOMAINS = [
    'federalregister.gov',
    'congress.gov',
    'whitehouse.gov',
    'nist.gov',
    'ai.gov',
    'state.gov',
    'defense.gov',
    'energy.gov',
    'hhs.gov',
    'transportation.gov',
    'justice.gov',
    'ftc.gov',
    'sec.gov',
    'regulations.gov'
]

THINK_TANK_DOMAINS = [
    'cset.georgetown.edu',
    'brookings.edu',
    'csis.org',
    'cnas.org',
    'rand.org',
    'carnegieendowment.org',
    'stanford.edu',  # HAI
    'mit.edu',       # CSAail
    'berkeley.edu',  # CHAI
    'ox.ac.uk'       # FHI
]

# General reputable domains (allow but don't boost)
GENERAL_ALLOWLIST = [
    'reuters.com',
    'apnews.com',
    'bloomberg.com',
    'npr.org',
    'bbc.com',
    'nytimes.com',
    'wsj.com',
    'washingtonpost.com',
    'ft.com',
    'economist.com',
    'nature.com',
    'science.org',
    'ieee.org',
    'acm.org',
    'arxiv.org',
    'wired.com',
    'theverge.com',
    'arstechnica.com',
    'techcrunch.com'
]

# Blocklist for low-quality or aggregator sites
BLOCKLIST_DOMAINS = [
    'reddit.com',
    'quora.com',
    'medium.com',
    'linkedin.com',
    'facebook.com',
    'twitter.com',
    'youtube.com',
    'pinterest.com',
    'tumblr.com',
    'instagram.com'
]

class WebSearchRAG:
    """
    RAG system that searches the web on-the-fly with policy-specific optimizations.
    """
    
    def __init__(self, 
                 llm_provider: str = "deepseek", 
                 model: str = "deepseek-chat",
                 api_key: Optional[str] = None):
        """
        Initialize the Web Search RAG system.
        """
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        
        # Initialize LLM Client
        try:
            self.llm_client = LLMClient(
                provider=llm_provider,
                api_key=self.api_key,
                model=model
            )
            logger.info(f"Initialized LLM client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

        self.vector_db = None

    async def rewrite_query(self, original_query: str) -> str:
        """
        Use LLM to rewrite the query into a targeted search string for policy documents.
        """
        if not self.llm_client:
            return original_query

        prompt = f"""You are an expert AI policy researcher. Rewrite the following user question into an effective search engine query to find official documents, executive orders, legislation, or reputable analysis.

User Question: "{original_query}"

Instructions:
1. Add specific terms like "text", "executive order", "bill", "regulation", "federal register", "white house", "congress" if applicable.
2. If the query is about a specific person or event (e.g. "Trump"), add the year (e.g. "2025") if it implies recent events.
3. Keep it keywords-based.
4. ONLY output the rewritten query string, nothing else.

Rewritten Query:"""

        try:
            rewritten = await self.llm_client.generate_response(prompt, temperature=0.0)
            rewritten = rewritten.strip().replace('"', '')
            logger.info(f"Rewrote query: '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return original_query

    def get_domain_score(self, url: str) -> int:
        """
        Score a URL based on its domain:
        3 = Primary Source (gov)
        2 = Think Tank
        1 = General Reputable
        0 = Unknown/Other
        -1 = Blocked
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check Blocklist
            for blocked in BLOCKLIST_DOMAINS:
                if blocked in domain:
                    return -1

            # Check Primary
            for d in PRIMARY_DOMAINS:
                if domain.endswith(d):
                    return 3
            
            # Check Think Tanks
            for d in THINK_TANK_DOMAINS:
                if domain.endswith(d):
                    return 2
            
            # Check TLDs (catch-all for .gov/.edu)
            if domain.endswith('.gov') or domain.endswith('.mil'):
                return 3
            if domain.endswith('.edu'):
                return 2
                
            # Check General Allowlist
            for d in GENERAL_ALLOWLIST:
                if domain.endswith(d):
                    return 1
            
            # Default
            return 0
        except:
            return 0

    def search_web(self, query: str, max_results: int = 15) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo with retries, robust backend, and domain boosting.
        """
        logger.info(f"Searching web for: '{query}'")
        results = []
        
        # Try backends in order of reliability/strictness
        backends = ['html', 'lite', 'api']
        
        for backend in backends:
            try:
                logger.info(f"Trying DDG backend: {backend}")
                with DDGS() as ddgs:
                    # Fetch 3x requested to allow for filtering/boosting
                    ddg_results = list(ddgs.text(query, max_results=max_results * 3, backend=backend))
                
                if ddg_results:
                    logger.info(f"Got {len(ddg_results)} raw results from {backend}")
                    
                    # Process and Score
                    scored_results = []
                    for r in ddg_results:
                        url = r.get('href', '')
                        score = self.get_domain_score(url)
                        
                        if score > 0:  # Only keep reputable sources (score 1, 2, 3)
                            scored_results.append({
                                'title': r.get('title', ''),
                                'url': url,
                                'snippet': r.get('body', ''),
                                'score': score
                            })
                    
                    # Sort by score descending
                    scored_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    results = scored_results[:max_results]
                    
                    if results:
                        logger.info(f"Found {len(results)} reputable results after filtering.")
                        for i, res in enumerate(results):
                            logger.info(f"[{i+1}] Score {res['score']}: {res['url']}")
                        return results
            
            except Exception as e:
                logger.warning(f"Search backend '{backend}' failed: {e}")
                time.sleep(1)  # Short backoff
                continue
        
        logger.error("All search backends failed or returned no reputable results.")
        return []

    def scrape_url(self, url: str, timeout: int = 5) -> str:
        """
        Fetch and extract text from a URL.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script.decompose()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean lines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def build_dynamic_index(self, search_results: List[Dict[str, str]]):
        """
        Scrape URLs, chunk content, and build an in-memory TF-IDF DB.
        """
        documents = []
        metadatas = []
        ids = []
        
        logger.info("Scraping and processing search results...")
        
        for i, res in enumerate(search_results):
            url = res['url']
            title = res['title']
            snippet = res['snippet']
            
            # Use snippet as fallback if scraping fails
            content = self.scrape_url(url)
            if len(content) < 200:  # Increased threshold for useful content
                content = snippet
                
            # Chunk content
            chunks = self.chunk_text(content)
            
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    'url': url,
                    'title': title,
                    'chunk_index': j,
                    'source_type': 'web_search'
                })
                ids.append(f"doc_{i}_chunk_{j}")
        
        if not documents:
            logger.warning("No content could be scraped/processed.")
            return False

        # Build Vector DB
        self.vector_db = SimpleTFIDFVectorDB(max_features=5000)
        self.vector_db.add_documents(documents, metadatas, ids)
        logger.info(f"Built dynamic index with {len(documents)} chunks.")
        return True

    async def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        End-to-end pipeline: Rewrite -> Search -> Scrape -> Index -> Retrieve -> Generate.
        """
        # 0. Rewrite Query
        search_query = await self.rewrite_query(query)
        
        # 1. Search
        search_results = self.search_web(search_query)
        if not search_results:
            return {"error": "No relevant reputable articles found."}
            
        # 2. Build Index
        success = self.build_dynamic_index(search_results)
        if not success:
            return {"error": "Failed to process search results into an index."}
            
        # 3. Retrieve (using original query for semantic matching against chunks)
        retrieved = self.vector_db.search(query, top_k=top_k)
        
        # 4. Augment
        context_parts = ["Based on the following reputable sources:\n"]
        unique_sources = set()
        
        for i, res in enumerate(retrieved, 1):
            meta = res['metadata']
            text = res['text']
            source = f"{meta['title']} ({meta['url']})"
            unique_sources.add(source)
            
            context_parts.append(f"\n--- Source {i}: {source} ---")
            context_parts.append(text)
            context_parts.append("")
            
        context = "\n".join(context_parts)
        
        # 5. Generate
        if not self.llm_client:
            return {
                "response": "LLM client not available (check API keys).",
                "context": context,
                "sources": list(unique_sources)
            }
            
        prompt = f"""{context}

Question: {query}

Instructions:
1. You MUST provide your answer in exactly two parts.

Part 1: A single, very concise sentence that directly answers the question. 
   - Try to integrate the specific text/phrases from the provided context that contain the answer. 
   - This sentence should closely match the explicit ground truth found in the text.

Part 2: A 3-4 sentence explanation providing more detail and reasoning based on the context.

Constraint:
- Do NOT cite sources (e.g., do not say "Source 1" or include URLs in the text).
- Just provide the information directly.

Answer:"""

        try:
            response = await self.llm_client.generate_response(prompt, temperature=0.3)
            return {
                "response": response,
                "context": context,
                "sources": list(unique_sources),
                "retrieved_chunks": retrieved,
                "search_query": search_query
            }
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return {"error": f"LLM Generation failed: {str(e)}"}

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Web-Search RAG")
    parser.add_argument("query", help="The question to answer")
    args = parser.parse_args()
    
    pipeline = WebSearchRAG()
    print(f"\n🔎 Running Web-Search RAG for: '{args.query}'\n")
    
    result = await pipeline.answer_query(args.query)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"\n🔄 Search Query Used: {result.get('search_query')}")
        print("\n💬 Answer:")
        print(result['response'])
        print("\n📚 Sources Used:")
        for source in result['sources']:
            print(f"- {source}")

if __name__ == "__main__":
    asyncio.run(main())
