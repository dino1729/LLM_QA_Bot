"""
Specialized Daily News Researcher
Fetches fresh news using aggregator-light strategy + Firecrawl search

Multi-Model Strategy:
1. Fast Model: Keyword extraction, source ranking
2. Smart Model: Deep reasoning for news synthesis
3. Strategic Model: Final editorial polish and enhancement

Aggregator Sources (for trending topic discovery):
- TLDR Tech (https://tldr.tech/) - General tech and startup news
- Ben's Bites (https://www.bensbites.com/) - AI tools and builder-focused
- AINews (https://news.smol.ai/issues/) - Comprehensive AI news summaries

Model configuration is read from config.yml (litellm_fast_llm, litellm_smart_llm, litellm_strategic_llm).
This approach ensures high-quality, well-reasoned news summaries with optimal performance.

Performance Optimizations:
- Async/await with aiohttp for parallel URL scraping
- Aggressive timeouts (15s scrape, 10s connect) with max 1 retry
- URL tracking to prevent redundant scraping
- Semaphore-limited concurrent connections (max 5)
"""
import asyncio
import aiohttp
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from bs4 import BeautifulSoup
from config import config
from helper_functions.llm_client import get_client

logger = logging.getLogger(__name__)

# Configuration
firecrawl_server_url = config.firecrawl_server_url

# Async performance tuning constants (aggressive timeouts for speed)
SCRAPE_TIMEOUT = 15  # Reduced from 30s - aggressive timeout
CONNECT_TIMEOUT = 10  # Connection establishment timeout
SEARCH_TIMEOUT = 90   # Search API timeout (searches take longer)
MAX_RETRIES = 1       # Reduced from 2 - fail fast
MAX_CONCURRENT_SCRAPES = 5  # Semaphore limit for parallel scraping

# Category-specific query sets for diverse topic coverage
CATEGORY_QUERIES = {
    "technology": [
        "artificial intelligence AI breakthroughs",
        "technology startups funding IPO",
        "cybersecurity data privacy breach",
        "tech regulation antitrust policy"
    ],
    "financial": [
        "stock market equity indices",
        "cryptocurrency bitcoin ethereum",
        "economic indicators GDP inflation",
        "corporate earnings business results"
    ],
    "india": [
        "India business economy growth",
        "India politics government policy",
        "India technology startups innovation",
        "India international relations diplomacy"
    ]
}


def scrape_with_firecrawl(url: str, max_age: int = 0, timeout: int = 30) -> Optional[str]:
    """
    Scrape a URL using Firecrawl server with freshness control
    
    Args:
        url: The URL to scrape
        max_age: Cache age in milliseconds (0 = always fresh, 3600000 = 1 hour)
        timeout: Request timeout in seconds
    
    Returns:
        Scraped text content or None if failed
    """
    try:
        scrape_url = f"{firecrawl_server_url}/scrape"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "maxAge": max_age,  # Freshness control
            "includeTags": ["article", "main", "content", "p", "h1", "h2", "h3", "time"],
            "excludeTags": ["nav", "footer", "header", "aside", "script", "style"],
            "waitFor": 1000,
            "timeout": timeout * 1000
        }
        
        response = requests.post(scrape_url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "markdown" in result["data"]:
                return result["data"]["markdown"]
        else:
            logger.warning(f"Firecrawl scrape failed for {url}: status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping {url} with Firecrawl: {str(e)}")
        return None


async def scrape_with_firecrawl_async(
    session: aiohttp.ClientSession,
    url: str,
    scraped_urls: Set[str],
    max_age: int = 0,
    timeout: int = SCRAPE_TIMEOUT
) -> Optional[Dict[str, str]]:
    """
    Async version: Scrape a URL using Firecrawl server with URL tracking
    
    Args:
        session: aiohttp ClientSession for connection pooling
        url: The URL to scrape
        scraped_urls: Set of already-scraped URLs to avoid duplicates
        max_age: Cache age in milliseconds (0 = always fresh)
        timeout: Request timeout in seconds (default: 15s aggressive)
    
    Returns:
        Dict with {url, content, success} or None if already scraped/failed
    """
    # Skip if already attempted (prevents redundant scraping)
    if url in scraped_urls:
        logger.debug(f"Skipping already-scraped URL: {url[:60]}...")
        return None
    
    # Mark as attempted immediately (even before trying)
    scraped_urls.add(url)
    
    try:
        scrape_url = f"{firecrawl_server_url}/scrape"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "maxAge": max_age,
            "includeTags": ["article", "main", "content", "p", "h1", "h2", "h3", "time"],
            "excludeTags": ["nav", "footer", "header", "aside", "script", "style"],
            "waitFor": 1000,
            "timeout": timeout * 1000
        }
        
        # Use aggressive timeouts
        client_timeout = aiohttp.ClientTimeout(
            total=timeout,
            connect=CONNECT_TIMEOUT
        )
        
        async with session.post(scrape_url, json=payload, timeout=client_timeout) as response:
            if response.status == 200:
                result = await response.json()
                if "data" in result and "markdown" in result["data"]:
                    content = result["data"]["markdown"]
                    if content and len(content) > 100:
                        logger.debug(f"Successfully scraped {url[:60]}... ({len(content)} chars)")
                        return {
                            "url": url,
                            "content": content,
                            "success": True
                        }
            else:
                logger.warning(f"Firecrawl async scrape failed for {url[:60]}: status {response.status}")
        
        return {"url": url, "content": "", "success": False}
        
    except asyncio.TimeoutError:
        logger.warning(f"Async scrape timeout ({timeout}s) for {url[:60]}...")
        return {"url": url, "content": "", "success": False}
    except Exception as e:
        logger.error(f"Async error scraping {url[:60]}: {str(e)}")
        return {"url": url, "content": "", "success": False}


async def scrape_urls_parallel(
    session: aiohttp.ClientSession,
    urls: List[str],
    scraped_urls: Set[str],
    semaphore: asyncio.Semaphore,
    max_age: int = 0,
    max_retries: int = MAX_RETRIES
) -> List[Dict[str, str]]:
    """
    Scrape multiple URLs in parallel with semaphore-limited concurrency
    
    Args:
        session: aiohttp ClientSession
        urls: List of URLs to scrape
        scraped_urls: Set tracking already-scraped URLs (modified in place)
        semaphore: Asyncio semaphore to limit concurrent connections
        max_age: Cache age in milliseconds
        max_retries: Max retry attempts per URL (default: 1)
    
    Returns:
        List of successfully scraped results with content
    """
    async def scrape_with_semaphore(url: str) -> Optional[Dict[str, str]]:
        """Wrapper to enforce semaphore limit on each scrape"""
        async with semaphore:
            # Try scraping with optional retry
            for attempt in range(max_retries + 1):
                result = await scrape_with_firecrawl_async(
                    session, url, scraped_urls, max_age=max_age
                )
                
                # If scraped_urls already had this URL, result is None (skip)
                if result is None:
                    return None
                
                # Success - return result
                if result.get("success") and result.get("content"):
                    return result
                
                # Failed - retry if we have attempts left
                if attempt < max_retries:
                    logger.debug(f"Retry {attempt + 1}/{max_retries} for {url[:50]}...")
                    # Remove from tracked set to allow retry
                    scraped_urls.discard(url)
                    await asyncio.sleep(0.5)  # Brief delay before retry
            
            return result  # Return last attempt result (even if failed)
    
    # Launch all scrapes in parallel (semaphore limits actual concurrency)
    logger.info(f"Parallel scraping {len(urls)} URLs (max {MAX_CONCURRENT_SCRAPES} concurrent)...")
    
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results with content
    successful = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Parallel scrape exception: {result}")
            continue
        if result and result.get("success") and result.get("content"):
            successful.append(result)
    
    logger.info(f"Parallel scraping complete: {len(successful)}/{len(urls)} successful")
    return successful


def scrape_aggregator_headlines(aggregator: str = "tldr") -> List[Dict[str, str]]:
    """
    Extract 1-2 latest headlines from aggregator sites for topic discovery
    
    Args:
        aggregator: One of "tldr", "bensbites", or "smol"
    
    Returns:
        List of headline dicts with {title, url, section}
    """
    headlines = []
    
    try:
        if aggregator == "tldr":
            url = "https://tldr.tech/"
            # Use 1 hour cache for aggregators (they update frequently)
            content = scrape_with_firecrawl(url, max_age=3600000, timeout=20)
            
            if content:
                # Extract first 2 significant headlines from markdown
                # TLDR uses various markdown patterns: ##, ###, or just bold text
                lines = content.split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Look for headline patterns in markdown
                    # Pattern 1: ## Headline or ### Headline
                    if (line_stripped.startswith('##') or line_stripped.startswith('###')) and len(line_stripped) > 20:
                        title = line_stripped.replace('###', '').replace('##', '').strip()
                        if len(headlines) < 2 and title and not any(skip in title.lower() for skip in ['subscribe', 'join', 'tldr', 'newsletter', 'advertisement']):
                            headlines.append({
                                'title': title,
                                'url': url,
                                'section': 'tech',
                                'source': 'TLDR'
                            })
                    
                    # Pattern 2: Lines with links and substantial text (likely headlines)
                    elif line_stripped.startswith('[') and '](' in line_stripped and len(line_stripped) > 40:
                        # Extract text from [Title](url) format
                        if ']' in line_stripped:
                            title = line_stripped.split(']')[0].replace('[', '').strip()
                            if len(headlines) < 2 and title and len(title) > 20 and not any(skip in title.lower() for skip in ['subscribe', 'join', 'tldr', 'newsletter']):
                                headlines.append({
                                    'title': title,
                                    'url': url,
                                    'section': 'tech',
                                    'source': 'TLDR'
                                })
                    
                    if len(headlines) >= 2:
                        break
        
        elif aggregator == "bensbites":
            url = "https://www.bensbites.com/"
            content = scrape_with_firecrawl(url, max_age=3600000, timeout=20)
            
            if content:
                lines = content.split('\n')
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Pattern 1: Markdown headers (##, ###)
                    if (line_stripped.startswith('##') or line_stripped.startswith('###')) and len(line_stripped) > 20:
                        title = line_stripped.replace('###', '').replace('##', '').strip()
                        if len(headlines) < 2 and title and not any(skip in title.lower() for skip in ['ben', 'subscribe', 'newsletter', 'bites']):
                            headlines.append({
                                'title': title,
                                'url': url,
                                'section': 'tech',
                                'source': 'BensBites'
                            })
                    
                    # Pattern 2: Lines with bold text and links
                    elif line_stripped.startswith('**') and '**' in line_stripped[2:] and len(line_stripped) > 30:
                        # Extract bold text: **Title**
                        title = line_stripped.split('**')[1].strip() if '**' in line_stripped[2:] else ''
                        if len(headlines) < 2 and title and len(title) > 15 and not any(skip in title.lower() for skip in ['ben', 'subscribe', 'newsletter']):
                            headlines.append({
                                'title': title,
                                'url': url,
                                'section': 'tech',
                                'source': 'BensBites'
                            })
                    
                    # Pattern 3: Links with substantial text
                    elif line_stripped.startswith('[') and '](' in line_stripped and len(line_stripped) > 30:
                        if ']' in line_stripped:
                            title = line_stripped.split(']')[0].replace('[', '').strip()
                            if len(headlines) < 2 and title and len(title) > 15:
                                headlines.append({
                                    'title': title,
                                    'url': url,
                                    'section': 'tech',
                                    'source': 'BensBites'
                                })
                    
                    if len(headlines) >= 2:
                        break
        
        elif aggregator == "smol":
            url = "https://news.smol.ai/issues/"
            content = scrape_with_firecrawl(url, max_age=3600000, timeout=20)
            
            if content:
                lines = content.split('\n')
                # Look for recent date headlines (e.g., "Dec 08 not much happened today")
                # and extract the detailed summaries that follow
                in_recent_section = False
                for i, line in enumerate(lines):
                    # Check if we're in a recent date section (December 2025)
                    if '2025' in line or 'December' in line:
                        in_recent_section = True
                    
                    # Look for headlines with date patterns or "Show details"
                    if in_recent_section and len(headlines) < 2:
                        # Pattern: "Dec 08 not much happened today Show details"
                        if line.startswith(('Dec', '* Dec')) or 'Show details' in line:
                            # The actual content is usually in the next lines after model names
                            title_line = line.replace('Show details', '').replace('* ', '').strip()
                            
                            # Look ahead for the actual content summary
                            if i + 1 < len(lines):
                                # Skip the model tags line and get the summary
                                for j in range(i + 1, min(i + 5, len(lines))):
                                    summary_line = lines[j].strip()
                                    # Get the first substantial paragraph
                                    if len(summary_line) > 50 and not summary_line.startswith(('*', '#', 'Show')):
                                        # Extract first sentence as title
                                        first_sentence = summary_line.split('.')[0] + '.'
                                        if len(first_sentence) > 20:
                                            headlines.append({
                                                'title': first_sentence[:150],  # Limit length
                                                'url': url,
                                                'section': 'tech',
                                                'source': 'AINews (smol.ai)'
                                            })
                                            break
                    
                    if len(headlines) >= 2:
                        break
        
        logger.info(f"Scraped {len(headlines)} headlines from {aggregator}")
        return headlines
        
    except Exception as e:
        logger.error(f"Error scraping {aggregator}: {str(e)}")
        return []


def extract_keywords_from_headlines(headlines: List[Dict[str, str]], provider: str = "litellm") -> List[str]:
    """
    Use LLM to extract key topics/entities from aggregator headlines
    
    Args:
        headlines: List of headline dicts
        provider: LLM provider to use
    
    Returns:
        List of extracted keywords/topics
    """
    if not headlines:
        return []
    
    try:
        client = get_client(provider=provider, model_tier="fast")
        
        headlines_text = "\n".join([f"- {h['title']}" for h in headlines])
        
        prompt = f"""Extract 2-3 specific keywords or topics from these headlines that would make good search queries for finding recent news articles.

Headlines:
{headlines_text}

Return ONLY the keywords/topics, one per line. Focus on specific technologies, companies, events, or trends.
Examples: "Claude AI update", "Nvidia GPU announcement", "Federal Reserve interest rates"

Keywords:"""
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        # Parse keywords from response
        keywords = [k.strip('- ').strip() for k in response.split('\n') if k.strip()]
        keywords = [k for k in keywords if len(k) > 3][:3]  # Limit to 3 best keywords
        
        logger.info(f"Extracted keywords: {keywords}")
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []


def search_fresh_sources_firecrawl(query: str, limit: int = 5, timeout: int = 120) -> List[Dict[str, str]]:
    """
    Search for fresh news sources using Firecrawl v2 search API
    
    Args:
        query: Search query
        limit: Maximum number of results
        timeout: Request timeout in seconds (default: 120)
    
    Returns:
        List of search results with {title, url, description, date}
    """
    try:
        # Firecrawl v2 API: server_url already includes /v2
        search_url = f"{firecrawl_server_url}/search"
        
        # Firecrawl v2 search API format with explicit date filtering
        today_date = datetime.now().strftime('%B %d, %Y')  # e.g., "December 9, 2025"
        
        payload = {
            "query": f"{query} {today_date} latest breaking news today",
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "maxAge": 0  # CRITICAL: Always fetch fresh data, no cache
            }
        }
        
        logger.info(f"Searching Firecrawl v2 for: {query} (timeout: {timeout}s)")
        logger.debug(f"Search URL: {search_url}")
        logger.debug(f"Payload: {payload}")
        
        response = requests.post(search_url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Firecrawl response keys: {result.keys()}")
            
            # Parse search results - Firecrawl v2 format: {success: true, data: {web: [...]}}
            results = []
            
            # Handle different Firecrawl response formats
            search_results = []
            if "data" in result:
                if isinstance(result["data"], dict) and "web" in result["data"]:
                    # Format: {data: {web: [...]}}
                    search_results = result["data"]["web"]
                elif isinstance(result["data"], list):
                    # Format: {data: [...]}
                    search_results = result["data"]
            
            logger.info(f"Firecrawl returned {len(search_results)} search results")
            
            # Now scrape each URL to get full content (with retry logic)
            for item in search_results[:limit]:
                url = item.get('url', '')
                if not url:
                    continue
                
                logger.info(f"Scraping search result: {url[:60]}...")
                
                # Try scraping with retry
                markdown_content = None
                for attempt in range(2):  # 2 attempts
                    try:
                        markdown_content = scrape_with_firecrawl(url, max_age=0, timeout=30)
                        if markdown_content and len(markdown_content) > 100:
                            break
                    except Exception as scrape_error:
                        if attempt == 0:
                            logger.warning(f"Scrape attempt {attempt + 1} failed for {url}: {scrape_error}, retrying...")
                        else:
                            logger.warning(f"Failed to scrape {url} after {attempt + 1} attempts: {scrape_error}")
                
                if markdown_content and len(markdown_content) > 100:
                    # Validate freshness - check if content mentions today's date
                    today_markers = [
                        datetime.now().strftime('%B %d'),  # "December 9"
                        datetime.now().strftime('%b %d'),   # "Dec 9"
                        datetime.now().strftime('%d %B'),   # "9 December"
                        'today', 'breaking', 'just now', 'hours ago', 'latest'
                    ]
                    
                    # Also check for recent dates (yesterday, 1-2 days ago)
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%B %d')
                    today_markers.append(yesterday)
                    
                    # Check if content is likely fresh (contains today's markers)
                    content_preview = markdown_content[:500].lower()
                    is_fresh = any(marker.lower() in content_preview for marker in today_markers)
                    
                    # Accept more sources: first 2 sources always, or if fresh markers found
                    # This balances quality with diversity - trust Firecrawl's search but verify top results
                    if is_fresh or len(results) < 2:  # Accept first 2 sources per query
                        results.append({
                            'title': item.get('title', 'Untitled'),
                            'url': url,
                            'description': item.get('description', '')[:300],
                            'markdown': markdown_content[:1500],
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': extract_source_name(url),
                            'freshness_validated': is_fresh
                        })
                    else:
                        logger.debug(f"Skipping source without fresh markers: {url}")
                else:
                    # Add URL info even without content
                    logger.info(f"Adding {url} without content (scraping failed)")
                    results.append({
                        'title': item.get('title', 'Untitled'),
                        'url': url,
                        'description': item.get('description', '')[:300],
                        'markdown': '',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': extract_source_name(url)
                    })
            
            logger.info(f"Successfully processed {len(results)} sources from Firecrawl for: {query}")
            return results
        else:
            error_msg = f"Firecrawl search failed: status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            logger.error(error_msg)
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Error searching Firecrawl: {str(e)}")
        raise  # Re-raise to avoid fallback


# DuckDuckGo fallback removed per user request - Firecrawl only


async def search_fresh_sources_firecrawl_async(
    session: aiohttp.ClientSession,
    query: str,
    scraped_urls: Set[str],
    semaphore: asyncio.Semaphore,
    limit: int = 5,
    timeout: int = SEARCH_TIMEOUT
) -> List[Dict[str, str]]:
    """
    Async version: Search for fresh news sources using Firecrawl v2 search API
    with parallel URL scraping for discovered results.
    
    Args:
        session: aiohttp ClientSession
        query: Search query
        scraped_urls: Set tracking already-scraped URLs (prevents redundant work)
        semaphore: Semaphore for limiting concurrent scrapes
        limit: Maximum number of results
        timeout: Search API timeout in seconds (default: 90s)
    
    Returns:
        List of search results with {title, url, description, date, markdown, source}
    """
    try:
        search_url = f"{firecrawl_server_url}/search"
        today_date = datetime.now().strftime('%B %d, %Y')
        
        payload = {
            "query": f"{query} {today_date} latest breaking news today",
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "maxAge": 0  # Always fetch fresh data
            }
        }
        
        logger.info(f"Async searching Firecrawl for: {query} (timeout: {timeout}s)")
        
        client_timeout = aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)
        
        async with session.post(search_url, json=payload, timeout=client_timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Firecrawl async search failed: status {response.status} - {error_text[:200]}")
                return []
            
            result = await response.json()
            
            # Parse search results from Firecrawl response
            search_results = []
            if "data" in result:
                if isinstance(result["data"], dict) and "web" in result["data"]:
                    search_results = result["data"]["web"]
                elif isinstance(result["data"], list):
                    search_results = result["data"]
            
            logger.info(f"Firecrawl async returned {len(search_results)} search results")
            
            if not search_results:
                return []
            
            # Extract URLs to scrape (filter out already-scraped ones)
            urls_to_scrape = []
            url_to_item = {}
            for item in search_results[:limit]:
                url = item.get('url', '')
                if url and url not in scraped_urls:
                    urls_to_scrape.append(url)
                    url_to_item[url] = item
            
            logger.info(f"Scraping {len(urls_to_scrape)} new URLs in parallel...")
            
            # Parallel scrape all URLs
            scrape_results = await scrape_urls_parallel(
                session, urls_to_scrape, scraped_urls, semaphore, max_age=0
            )
            
            # Build results list with scraped content
            results = []
            scraped_content = {r["url"]: r["content"] for r in scrape_results}
            
            for item in search_results[:limit]:
                url = item.get('url', '')
                if not url:
                    continue
                
                markdown_content = scraped_content.get(url, '')
                
                # Validate freshness markers if we have content
                is_fresh = False
                if markdown_content and len(markdown_content) > 100:
                    today_markers = [
                        datetime.now().strftime('%B %d'),
                        datetime.now().strftime('%b %d'),
                        datetime.now().strftime('%d %B'),
                        'today', 'breaking', 'just now', 'hours ago', 'latest'
                    ]
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%B %d')
                    today_markers.append(yesterday)
                    
                    content_preview = markdown_content[:500].lower()
                    is_fresh = any(marker.lower() in content_preview for marker in today_markers)
                
                # Accept source if: has fresh markers, or is in first 2 results, or has content
                if is_fresh or len(results) < 2 or (markdown_content and len(markdown_content) > 100):
                    results.append({
                        'title': item.get('title', 'Untitled'),
                        'url': url,
                        'description': item.get('description', '')[:300],
                        'markdown': markdown_content[:1500] if markdown_content else '',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': extract_source_name(url),
                        'freshness_validated': is_fresh
                    })
            
            logger.info(f"Async search complete: {len(results)} sources from '{query}'")
            return results
            
    except asyncio.TimeoutError:
        logger.error(f"Async search timeout ({timeout}s) for query: {query}")
        return []
    except Exception as e:
        logger.error(f"Async search error for '{query}': {str(e)}")
        return []


def extract_source_name(url: str) -> str:
    """Extract clean source name from URL"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        # Remove www. and common TLDs
        domain = domain.replace('www.', '').split('.')[0]
        return domain.title()
    except:
        return "Unknown"


async def gather_daily_news_async(
    category: str,
    max_sources: int = 5,
    aggregator_limit: int = 1,
    freshness_hours: int = 24,
    provider: str = "litellm"
) -> str:
    """
    Async orchestrator: Gather fresh daily news for a category using parallel scraping.
    
    This is the performance-optimized version that uses:
    - Async HTTP requests with aiohttp
    - Parallel URL scraping with semaphore-limited concurrency
    - URL tracking to prevent redundant scraping
    - Aggressive timeouts (15s) for fast failure
    
    Strategy:
    1. Scrape 1-2 headlines from aggregators (topic discovery) - sync for simplicity
    2. Extract keywords from those headlines
    3. Search for fresh original sources with parallel scraping
    4. Synthesize into comprehensive news summary
    
    Args:
        category: "technology", "financial", or "india"
        max_sources: Maximum total sources to gather
        aggregator_limit: Max headlines to take from aggregators (1-2)
        freshness_hours: How recent news should be (24 = last day)
        provider: LLM provider for synthesis
    
    Returns:
        Comprehensive news summary as markdown string
    """
    import random
    import time
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"GATHERING DAILY NEWS (ASYNC): {category.upper()}")
    logger.info("="*80)
    
    all_sources = []
    keywords = []
    
    # Step 1: Get trending topics from aggregators (sync - not performance critical)
    # Aggregator scraping is kept sync as it's a single URL and caching helps
    if category in ["technology", "financial"]:
        logger.info("Step 1: Scraping aggregator for trending topics...")
        
        aggregators = ["tldr", "smol", "bensbites"] if category == "technology" else ["tldr"]
        aggregator = random.choice(aggregators)
        
        if aggregator:
            logger.info(f"Selected aggregator: {aggregator}")
            headlines = scrape_aggregator_headlines(aggregator)
            
            if headlines:
                all_sources.extend(headlines[:aggregator_limit])
                
                logger.info("Step 2: Extracting keywords from headlines...")
                keywords = extract_keywords_from_headlines(headlines, provider)
            else:
                logger.warning("No headlines from aggregator, using category-based search")
    
    # Step 3: Async search and parallel scraping
    logger.info("Step 3: Searching for fresh sources with PARALLEL scraping...")
    
    # Build search queries
    today_str = datetime.now().strftime('%B %d %Y')
    search_queries = []
    
    if category in CATEGORY_QUERIES:
        base_queries = CATEGORY_QUERIES[category]
        search_queries.extend([f"{query} {today_str}" for query in base_queries])
        logger.info(f"Using {len(base_queries)} diverse queries for {category}")
    else:
        search_queries = [f"{category} news {today_str}"]
    
    if keywords:
        search_queries[:0] = [f"{kw} {today_str}" for kw in keywords[:1]]
        logger.info("Added keyword-based query from aggregator trending topics")
    
    # Track all scraped URLs across queries to prevent redundant work
    scraped_urls: Set[str] = set()
    temp_sources: List[Dict] = []
    sources_per_query = 3
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_SCRAPES, limit_per_host=3)
    timeout = aiohttp.ClientTimeout(total=SEARCH_TIMEOUT, connect=CONNECT_TIMEOUT)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for query_idx, query in enumerate(search_queries):
            logger.info(f"Query {query_idx + 1}/{len(search_queries)}: {query}")
            
            try:
                results = await search_fresh_sources_firecrawl_async(
                    session=session,
                    query=query,
                    scraped_urls=scraped_urls,
                    semaphore=semaphore,
                    limit=sources_per_query
                )
            except Exception as e:
                logger.error(f"Async search failed for '{query}': {e}")
                continue
            
            # Add results with content to temp collection
            for result in results:
                markdown = result.get('markdown', '')
                if markdown and len(markdown) > 100:
                    temp_sources.append({
                        'title': result['title'],
                        'url': result['url'],
                        'description': result.get('description', ''),
                        'content': markdown,
                        'date': result.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'source': result['source'],
                        'is_aggregator': False
                    })
    
    # Deduplicate and finalize
    logger.info(f"Collected {len(temp_sources)} sources from {len(search_queries)} queries")
    unique_sources = deduplicate_sources(temp_sources)
    
    remaining_slots = max_sources - len(all_sources)
    all_sources.extend(unique_sources[:remaining_slots])
    
    elapsed = time.time() - start_time
    logger.info(f"Gathered {len(all_sources)} total sources in {elapsed:.1f}s")
    logger.info(f"Scraped URLs tracked: {len(scraped_urls)} (prevented redundant scraping)")
    
    if not all_sources:
        logger.error("No sources found! Returning error message.")
        return f"Unable to gather news for {category}. Please check your internet connection or try again later."
    
    # Step 4: Synthesize news summary using LLM (sync - LLM calls are already optimized)
    logger.info("Step 4: Synthesizing news summary with LLM...")
    
    return synthesize_news_report(all_sources, category, provider)


def gather_daily_news(
    category: str,
    max_sources: int = 5,
    aggregator_limit: int = 1,
    freshness_hours: int = 24,
    provider: str = "litellm"
) -> str:
    """
    Sync wrapper: Gather fresh daily news for a category.
    
    This is a synchronous wrapper around gather_daily_news_async() for backward
    compatibility. It uses asyncio.run() to execute the async version.
    
    Performance optimizations in the async version:
    - Parallel URL scraping with aiohttp (up to 5 concurrent)
    - Aggressive 15s timeouts (down from 30s)
    - URL tracking prevents redundant scraping
    - Max 1 retry per URL (down from 2)
    
    Args:
        category: "technology", "financial", or "india"
        max_sources: Maximum total sources to gather
        aggregator_limit: Max headlines to take from aggregators (1-2)
        freshness_hours: How recent news should be (24 = last day)
        provider: LLM provider for synthesis
    
    Returns:
        Comprehensive news summary as markdown string
    """
    # Use asyncio.run() to execute the async version
    # This handles event loop creation and cleanup automatically
    return asyncio.run(
        gather_daily_news_async(
            category=category,
            max_sources=max_sources,
            aggregator_limit=aggregator_limit,
            freshness_hours=freshness_hours,
            provider=provider
        )
    )


def deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    """
    Remove duplicate sources by URL and near-duplicate by title similarity
    
    Args:
        sources: List of source dictionaries
    
    Returns:
        Deduplicated list of sources
    """
    seen_urls = set()
    seen_titles = []
    unique_sources = []
    
    for source in sources:
        url = source.get('url', '')
        title = source.get('title', '')
        
        # Skip exact URL duplicates
        if url in seen_urls:
            logger.debug(f"Skipping duplicate URL: {url}")
            continue
            
        # Skip very similar titles (simple check)
        is_duplicate = False
        for seen_title in seen_titles:
            # If 80%+ of words overlap, consider duplicate
            title_words = set(title.lower().split())
            seen_words = set(seen_title.lower().split())
            
            if not title_words or not seen_words:
                continue
                
            overlap = len(title_words & seen_words)
            max_len = max(len(title_words), len(seen_words))
            
            if max_len > 0 and overlap / max_len > 0.8:
                logger.debug(f"Skipping similar title: {title}")
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_urls.add(url)
            seen_titles.append(title)
            unique_sources.append(source)
    
    logger.info(f"Deduplication: {len(sources)} sources -> {len(unique_sources)} unique sources")
    return unique_sources


def analyze_source_relevance(sources: List[Dict], category: str, provider: str = "litellm") -> List[Dict]:
    """
    Use fast model to quickly filter and rank sources by relevance
    
    Args:
        sources: List of source dicts
        category: News category
        provider: LLM provider
    
    Returns:
        Filtered and ranked sources
    """
    try:
        fast_client = get_client(provider=provider, model_tier="fast")
        
        # Create source summary for analysis
        sources_text = ""
        for i, source in enumerate(sources[:10], 1):  # Analyze up to 10 sources
            sources_text += f"{i}. {source['title']}\n"
            desc = source.get('description', '')[:200]
            if desc:
                sources_text += f"   {desc}\n"
        
        prompt = f"""Analyze these news sources for {category} news and rate their relevance (1-10 scale).
Return ONLY a comma-separated list of numbers corresponding to each source's relevance score.

Sources:
{sources_text}

Relevance scores (comma-separated, e.g., "9,7,8,6,9"):"""
        
        response = fast_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        
        # Parse scores
        scores = [int(s.strip()) for s in response.split(',') if s.strip().isdigit()]
        
        # Add scores to sources and sort
        for i, score in enumerate(scores[:len(sources)]):
            sources[i]['relevance_score'] = score
        
        # Sort by relevance
        sources.sort(key=lambda x: x.get('relevance_score', 5), reverse=True)
        
        logger.info(f"Ranked {len(sources)} sources by relevance (fast model)")
        return sources
        
    except Exception as e:
        logger.warning(f"Source ranking failed: {e}, using original order")
        return sources


def synthesize_news_report(sources: List[Dict], category: str, provider: str = "litellm") -> str:
    """
    Use multi-model approach to synthesize news sources into comprehensive report
    
    Strategy:
    1. Fast model: Quick source relevance filtering
    2. Smart model: Initial news synthesis with deep reasoning
    3. Strategic model: Final editorial polish and enhancement
    
    Args:
        sources: List of source dicts with content
        category: News category
        provider: LLM provider
    
    Returns:
        Comprehensive news summary as markdown
    """
    try:
        # Step 1: Use fast model to rank sources by relevance
        logger.info("Step 1: Ranking sources with fast model...")
        sources = analyze_source_relevance(sources, category, provider)
        
        # Step 2: Use smart model for initial synthesis
        logger.info("Step 2: Synthesizing news with smart model...")
        smart_client = get_client(provider=provider, model_tier="smart")
        
        # Prepare context
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        context = f"Current date and time: {current_time}\n\n"
        context += f"Category: {category.title()}\n\n"
        context += "=== NEWS SOURCES (Ranked by Relevance) ===\n\n"
        
        for i, source in enumerate(sources, 1):
            context += f"Source {i}: {source.get('source', 'Unknown')} - {source['title']}\n"
            context += f"URL: {source['url']}\n"
            if source.get('date'):
                context += f"Date: {source['date']}\n"
            if source.get('relevance_score'):
                context += f"Relevance: {source['relevance_score']}/10\n"
            
            # Add content
            content = source.get('content', source.get('markdown', source.get('description', '')))
            if content:
                context += f"Content:\n{content[:1500]}\n"  # Limit per source
            
            context += "\n" + "="*80 + "\n\n"
        
        # Synthesis prompt for smart model with explicit date requirement and diversity enforcement
        synthesis_prompt = f"""You are a professional news analyst with deep reasoning capabilities. Analyze these sources and create a comprehensive news summary for {category} news.

Your summary should:
1. START with an explicit date reference (e.g., "On [Date]" or "[Date] saw" or "As of [Date]")
2. Provide a clear overview of the most important developments
3. Include specific facts, figures, and key details from the sources
4. Analyze connections and implications between different news items
5. Maintain journalistic objectivity
6. Be well-structured with clear paragraphs
7. Focus on the most newsworthy and impactful information
8. Be comprehensive but concise (aim for 400-600 words)
9. Include temporal markers throughout (e.g., "today", "this week", specific dates)

CRITICAL DATE REQUIREMENTS:
1. The opening sentence MUST include TODAY's date: {datetime.now().strftime('%B %d, %Y')} (NOT any other date!)
2. If sources mention older dates (e.g., December 7), you MUST reframe them as "As of [today's date], reports indicate..."
3. Use present tense and today's perspective throughout

DIVERSITY REQUIREMENTS (VERY IMPORTANT):
1. Cover MULTIPLE distinct topics/stories (minimum 3-4 different stories)
2. Limit to 1-2 sources per specific topic/story (avoid over-focusing on one story)
3. If one story dominates the sources, summarize it BRIEFLY (1-2 paragraphs max) and move on
4. Prioritize BREADTH over depth - mention various developments across the category
5. Group related sources together but keep each topic concise
6. Ensure balanced coverage - don't let any single story consume more than 30% of the summary

Think deeply about the significance and interconnections of these news items while maintaining today's date context and topic diversity.

{context}

Write a professional news summary with explicit date references and diverse topic coverage:"""
        
        initial_draft = smart_client.chat_completion(
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        
        # Step 3: Use strategic model for final polish
        logger.info("Step 3: Enhancing report with strategic model (gpt-5.1)...")
        strategic_client = get_client(provider=provider, model_tier="strategic")
        
        editorial_prompt = f"""You are an expert editor reviewing a news summary. Enhance this draft to make it publication-ready.

Category: {category.title()}
Current Date: {current_time.split()[0]}

Draft:
{initial_draft}

Your tasks:
1. VERIFY the opening includes explicit date reference (e.g., "On 9 December" or "December 9 saw")
2. If missing, ADD an explicit date reference to the opening sentence
3. VERIFY diverse topic coverage (3-4+ distinct stories, not dominated by one topic)
4. If one topic is over-represented, condense it and ensure other stories get adequate coverage
3. Improve clarity and flow
4. Ensure accuracy and proper context
5. Add impactful opening and closing statements
6. Maintain professional tone
7. Enhance readability without changing core facts
8. Keep it concise (400-600 words)
9. Ensure temporal markers are present throughout

CRITICAL: The final version MUST have a clear date reference in the opening.

Return the polished, publication-ready version with explicit dates:"""
        
        final_report = strategic_client.chat_completion(
            messages=[{"role": "user", "content": editorial_prompt}],
            temperature=0.2,
            max_tokens=3000
        )
        
        # Add source citations
        final_report += "\n\n---\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            relevance = f" (Relevance: {source['relevance_score']}/10)" if source.get('relevance_score') else ""
            final_report += f"{i}. [{source['source']}] {source['title']}{relevance}\n"
            final_report += f"   {source['url']}\n"
        
        logger.info("Multi-model news synthesis complete! (fast→smart→strategic)")
        return final_report
        
    except Exception as e:
        logger.error(f"Error synthesizing news: {str(e)}")
        
        # Fallback: Return basic summary
        fallback = f"## {category.title()} News Update\n\n"
        for i, source in enumerate(sources[:5], 1):
            fallback += f"{i}. **{source['title']}**\n"
            fallback += f"   Source: {source.get('source', 'Unknown')}\n"
            if source.get('description'):
                fallback += f"   {source['description']}\n"
            fallback += f"   URL: {source['url']}\n\n"
        
        return fallback


if __name__ == "__main__":
    # Test the news researcher
    logging.basicConfig(level=logging.INFO)
    
    print("\nTesting Technology News Gathering...")
    tech_news = gather_daily_news(
        category="technology",
        max_sources=5,
        aggregator_limit=1,
        freshness_hours=24,
        provider="litellm"
    )
    
    print("\n" + "="*80)
    print("TECHNOLOGY NEWS REPORT")
    print("="*80)
    print(tech_news)

