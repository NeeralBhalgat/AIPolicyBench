
import os
import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor

from src.config import config
from src.simple_vector_db import SimpleTFIDFVectorDB
from src.utils.llm_client import LLMClient

# Thread pool for synchronous operations
_executor = ThreadPoolExecutor(max_workers=5)

# Configure logging
logger = logging.getLogger(__name__)

# --- Domain Configuration ---

# Base reputable TLDs
REPUTABLE_TLDS = ['.gov', '.edu', '.mil', '.org']

# High-priority domains (boost these to top)
PRIMARY_DOMAINS = [
    'federalregister.gov', 'congress.gov', 'whitehouse.gov', 'nist.gov', 'ai.gov',
    'state.gov', 'defense.gov', 'energy.gov', 'hhs.gov', 'transportation.gov',
    'justice.gov', 'ftc.gov', 'sec.gov', 'regulations.gov'
]

THINK_TANK_DOMAINS = [
    'cset.georgetown.edu', 'brookings.edu', 'csis.org', 'cnas.org', 'rand.org',
    'carnegieendowment.org', 'stanford.edu', 'mit.edu', 'berkeley.edu', 'ox.ac.uk'
]

# General reputable domains (allow but don't boost)
GENERAL_ALLOWLIST = [
    'reuters.com', 'apnews.com', 'bloomberg.com', 'npr.org', 'bbc.com', 'nytimes.com',
    'wsj.com', 'washingtonpost.com', 'ft.com', 'economist.com', 'nature.com',
    'science.org', 'ieee.org', 'acm.org', 'arxiv.org', 'wired.com', 'theverge.com',
    'arstechnica.com', 'techcrunch.com'
]

# Blocklist for low-quality or aggregator sites
BLOCKLIST_DOMAINS = [
    'reddit.com', 'quora.com', 'medium.com', 'linkedin.com', 'facebook.com',
    'twitter.com', 'youtube.com', 'pinterest.com', 'tumblr.com', 'instagram.com'
]

class WebSearchRAG:
    """
    RAG system that searches the web on-the-fly with policy-specific optimizations.
    """
    
    def __init__(self, 
                 llm_provider: str = config.DEFAULT_PROVIDER, 
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the Web Search RAG system.
        """
        self.llm_provider = llm_provider
        
        # Set default model based on provider if not specified
        if model:
            self.model = model
        elif llm_provider == "openai":
            self.model = "gpt-4o"
        elif llm_provider == "anthropic":
            self.model = "claude-3-sonnet-20240229"
        elif llm_provider == "openrouter":
            self.model = "deepseek/deepseek-chat"
        else:  # Legacy "deepseek" provider maps to openrouter
            self.model = "deepseek/deepseek-chat"

        # Select appropriate API key based on provider
        if api_key:
            self.api_key = api_key
        elif llm_provider == "openai":
            self.api_key = config.OPENAI_API_KEY
        elif llm_provider == "anthropic":
            self.api_key = config.ANTHROPIC_API_KEY
        elif llm_provider == "openrouter":
            self.api_key = config.OPENROUTER_API_KEY
        else:  # Legacy "deepseek" provider maps to openrouter
            self.api_key = config.OPENROUTER_API_KEY
        
        # Initialize LLM Client
        try:
            logger.info(f"Initializing LLM client for provider: {llm_provider}")
            self.llm_client = LLMClient(
                provider=llm_provider,
                api_key=self.api_key,
                model=self.model
            )
            logger.info(f"Initialized LLM client with model: {self.model}")
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
            
            for blocked in BLOCKLIST_DOMAINS:
                if blocked in domain:
                    return -1

            for d in PRIMARY_DOMAINS:
                if domain.endswith(d):
                    return 3
            
            for d in THINK_TANK_DOMAINS:
                if domain.endswith(d):
                    return 2
            
            if domain.endswith('.gov') or domain.endswith('.mil'):
                return 3
            if domain.endswith('.edu'):
                return 2
                
            for d in GENERAL_ALLOWLIST:
                if domain.endswith(d):
                    return 1
            
            return 0
        except:
            return 0

    def search_web(self, query: str, max_results: int = config.MAX_SEARCH_RESULTS) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo with retries, robust backend, and domain boosting.
        """
        logger.info(f"Searching web for: '{query}'")
        results = []
        backends = ['html', 'lite', 'api']
        
        for backend in backends:
            try:
                logger.info(f"Trying DDG backend: {backend}")
                with DDGS() as ddgs:
                    # Fetch 2x requested to allow for filtering/boosting
                    ddg_results = list(ddgs.text(query, max_results=max_results * 2, backend=backend))
                
                if ddg_results:
                    logger.info(f"Got {len(ddg_results)} raw results from {backend}")
                    
                    scored_results = []
                    for r in ddg_results:
                        url = r.get('href', '')
                        score = self.get_domain_score(url)
                        
                        # Allow score >= 0 (include unknown domains, just rank lower)
                        # Only filter out explicitly blocked domains (score == -1)
                        if score >= 0:
                            scored_results.append({
                                'title': r.get('title', ''),
                                'url': url,
                                'snippet': r.get('body', ''),
                                'score': score
                            })
                    
                    # Sort by score (reputable first) but keep all
                    scored_results.sort(key=lambda x: x['score'], reverse=True)
                    results = scored_results[:max_results]
                    
                    if results:
                        logger.info(f"Found {len(results)} results after filtering (top scores: {[r['score'] for r in results[:3]]})")
                        return results
            
            except Exception as e:
                logger.warning(f"Search backend '{backend}' failed: {e}")
                time.sleep(0.5)  # Reduced from 1s
                continue
        
        logger.error("All search backends failed or returned no results.")
        return []

    async def scrape_url_async(self, url: str, session: aiohttp.ClientSession, timeout: int = 3) -> str:
        """
        Async fetch and extract text from a URL.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    return ""
                content = await response.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script.decompose()
                
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit text size
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""
    
    def scrape_url(self, url: str, timeout: int = 3) -> str:
        """
        Sync fallback for scraping (used when async not available).
        """
        try:
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script.decompose()
                
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit text size
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
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

    async def build_dynamic_index_async(self, search_results: List[Dict[str, str]]):
        """
        Async scrape URLs in parallel, chunk content, and build an in-memory TF-IDF DB.
        """
        documents = []
        metadatas = []
        ids = []
        
        logger.info(f"Scraping {len(search_results)} URLs in parallel...")
        
        # Parallel scraping with aiohttp
        async with aiohttp.ClientSession() as session:
            tasks = [self.scrape_url_async(res['url'], session) for res in search_results]
            scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (res, content) in enumerate(zip(search_results, scraped_contents)):
            url = res['url']
            title = res['title']
            snippet = res['snippet']
            
            # Handle exceptions from scraping
            if isinstance(content, Exception):
                content = ""
            
            # Use snippet as fallback if scraping failed or got little content
            if not content or len(content) < 200:
                content = snippet
                logger.info(f"Using snippet for {url[:50]}...")
                
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

        self.vector_db = SimpleTFIDFVectorDB(max_features=5000)
        self.vector_db.add_documents(documents, metadatas, ids)
        logger.info(f"Built dynamic index with {len(documents)} chunks from {len(search_results)} sources.")
        return True
    
    def build_dynamic_index(self, search_results: List[Dict[str, str]]):
        """
        Sync fallback - scrape URLs and build index.
        """
        documents = []
        metadatas = []
        ids = []
        
        logger.info("Scraping and processing search results...")
        
        for i, res in enumerate(search_results):
            url = res['url']
            title = res['title']
            snippet = res['snippet']
            
            content = self.scrape_url(url)
            if len(content) < 200:
                content = snippet
                
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

        self.vector_db = SimpleTFIDFVectorDB(max_features=5000)
        self.vector_db.add_documents(documents, metadatas, ids)
        logger.info(f"Built dynamic index with {len(documents)} chunks.")
        return True

    async def answer_query(self, query: str, top_k: int = 5, timeout: float = 90.0) -> Dict[str, Any]:
        """
        End-to-end pipeline: Rewrite -> Search -> Scrape -> Index -> Retrieve -> Generate.
        
        Args:
            query: The question to answer
            top_k: Number of chunks to retrieve
            timeout: Maximum time in seconds before returning empty response (default: 95s for Cloudflare)
        """
        try:
            return await asyncio.wait_for(
                self._answer_query_internal(query, top_k),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Query timed out after {timeout}s")
            return {
                "response": "[TIMEOUT] Query processing exceeded time limit.",
                "error": f"Query processing exceeded {timeout}s timeout",
                "timed_out": True
            }
    
    async def _answer_query_internal(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Internal implementation of answer_query without timeout."""
        # Run search in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        search_query = await self.rewrite_query(query)
        
        search_results = await loop.run_in_executor(_executor, self.search_web, search_query)
        if not search_results:
            return {"error": "No relevant articles found.", "response": "I don't have enough information to answer this question."}
            
        # Use async parallel scraping
        success = await self.build_dynamic_index_async(search_results)
        if not success:
            return {"error": "Failed to process search results.", "response": "I don't have enough information to answer this question."}
            
        retrieved = self.vector_db.search(query, top_k=top_k)
        
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
        
        if not self.llm_client:
            return {
                "response": "LLM client not available (check API keys).",
                "context": context,
                "sources": list(unique_sources)
            }
            
        prompt = f"""{context}

Question: {query}

Instructions:
1. ONLY answer if you find clear, direct evidence in the sources above.
2. If the sources don't contain a clear answer, respond with: "I don't have enough information to answer this question."
3. If you do find the answer, provide a concise, direct response (1-2 sentences max).
4. Your answer should closely match specific phrases from the sources.
5. Do NOT make up information or speculate beyond what's in the sources.

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
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, deepseek, anthropic)")
    parser.add_argument("--raw", action="store_true", help="Print only the response text")
    args = parser.parse_args()
    
    if args.raw:
        logging.getLogger().setLevel(logging.ERROR)
    
    pipeline = WebSearchRAG(llm_provider=args.provider)
    
    if not args.raw:
        print(f"\n🔎 Running Web-Search RAG for: '{args.query}'\n")
    
    result = await pipeline.answer_query(args.query)
    
    if "error" in result:
        if args.raw:
            print(f"Error: {result['error']}")
        else:
            print(f"❌ Error: {result['error']}")
    else:
        if args.raw:
            print(result['response'])
        else:
            print(f"\n🔄 Search Query Used: {result.get('search_query')}")
            print("\n💬 Answer:")
            print(result['response'])
            print("\n📚 Sources Used:")
            for source in result['sources']:
                print(f"- {source}")

if __name__ == "__main__":
    asyncio.run(main())
