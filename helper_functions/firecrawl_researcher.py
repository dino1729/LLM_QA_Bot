"""
Custom Research Feature using Firecrawl and LLM
Simple, reliable research without external API dependencies
"""
import requests
import logging
from datetime import datetime
from config import config
from helper_functions.llm_client import get_client
from helper_functions.debug_logger import log_debug_data

# Configuration
firecrawl_server_url = config.firecrawl_server_url


def scrape_with_firecrawl(url, timeout=30):
    """
    Scrape a URL using Firecrawl server
    
    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds
    
    Returns:
        Scraped text content or None if failed
    """
    try:
        scrape_url = f"{firecrawl_server_url}/scrape"
        
        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "includeTags": ["article", "main", "content", "p", "h1", "h2", "h3"],
            "excludeTags": ["nav", "footer", "header", "aside", "script", "style"],
            "waitFor": 1000
        }
        
        response = requests.post(scrape_url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            # Log successful scrape
            log_debug_data("firecrawl_scrape", {
                "url": url,
                "status": "success",
                "response": result
            })
            
            if "data" in result and "markdown" in result["data"]:
                return result["data"]["markdown"]
            elif "data" in result and "html" in result["data"]:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(result["data"]["html"], 'html.parser')
                return soup.get_text()
        else:
            print(f"Firecrawl scrape failed for {url}: status {response.status_code}")
            # Log failed scrape
            log_debug_data("firecrawl_scrape_error", {
                "url": url,
                "status": "error",
                "status_code": response.status_code,
                "response_text": response.text
            })
            return None
            
    except Exception as e:
        print(f"Error scraping {url} with Firecrawl: {str(e)}")
        return None




def get_search_urls_fallback(query, count=5):
    """
    Get search result URLs using DuckDuckGo HTML scraping as fallback
    
    Args:
        query: Search query
        count: Number of results
    
    Returns:
        List of URLs
    """
    try:
        # Use DuckDuckGo HTML search (no API key needed)
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract URLs from DuckDuckGo results
            urls = []
            for result in soup.find_all('a', class_='result__url', limit=count):
                href = result.get('href')
                if href and href.startswith('http'):
                    urls.append(href)
            
            return urls if urls else generate_topic_urls(query)
        else:
            return generate_topic_urls(query)
            
    except Exception as e:
        print(f"Error with DuckDuckGo search: {str(e)}")
        return generate_topic_urls(query)


def generate_topic_urls(query):
    """
    Generate likely URLs for common topics when search fails
    
    Args:
        query: Search query
    
    Returns:
        List of URLs
    """
    # Common authoritative sources for different topics
    topic_sources = {
        "news": [
            f"https://www.reuters.com/search/news?query={requests.utils.quote(query)}",
            f"https://apnews.com/search?q={requests.utils.quote(query)}",
        ],
        "technology": [
            f"https://techcrunch.com/?s={requests.utils.quote(query)}",
            f"https://arstechnica.com/?s={requests.utils.quote(query)}",
        ],
        "science": [
            f"https://www.sciencedaily.com/search/?keyword={requests.utils.quote(query)}",
        ],
        "general": [
            f"https://en.wikipedia.org/wiki/{requests.utils.quote(query.replace(' ', '_'))}",
        ]
    }
    
    # Try to determine topic
    query_lower = query.lower()
    if any(word in query_lower for word in ["news", "latest", "update", "current"]):
        return topic_sources["news"][:2]
    elif any(word in query_lower for word in ["tech", "ai", "software", "computer"]):
        return topic_sources["technology"][:2]
    elif any(word in query_lower for word in ["science", "research", "study"]):
        return topic_sources["science"][:1]
    else:
        return topic_sources["general"][:1]


def conduct_research_firecrawl(query, provider="litellm", model_name=None, max_sources=5):
    """
    Conduct research using Firecrawl and LLM synthesis
    
    Args:
        query: Research query
        provider: LLM provider ("litellm" or "ollama")
        model_name: Specific model name (optional)
        max_sources: Maximum number of sources to scrape
    
    Returns:
        Research report string
    """
    try:
        print(f"🔍 Starting research for: {query}")
        
        # Step 1: Get search URLs using fallback search
        urls = get_search_urls_fallback(query, count=max_sources)
        
        if not urls:
            return "Unable to find search results. Please check your internet connection or search query."
        
        print(f"📋 Found {len(urls)} URLs to research")
        
        # Step 2: Scrape content from URLs using Firecrawl
        scraped_contents = []
        for i, url in enumerate(urls, 1):
            print(f"📄 Scraping source {i}/{len(urls)}: {url[:60]}...")
            content = scrape_with_firecrawl(url, timeout=20)
            if content and len(content.strip()) > 100:  # Minimum content threshold
                scraped_contents.append({
                    "url": url,
                    "content": content[:8000]  # Limit per source to manage token count
                })
            
            # Stop if we have enough good sources
            if len(scraped_contents) >= max_sources:
                break
        
        if not scraped_contents:
            return "Unable to scrape content from search results. The sources may be behind paywalls or blocking automated access."
        
        print(f"✓ Successfully scraped {len(scraped_contents)} sources")
        
        # Step 3: Prepare context for LLM
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        context = f"Current date and time: {current_time}\n\n"
        context += f"Research Query: {query}\n\n"
        context += "=== SOURCES ===\n\n"
        
        for i, source in enumerate(scraped_contents, 1):
            context += f"Source {i}: {source['url']}\n"
            context += f"{source['content']}\n"
            context += "\n" + "="*80 + "\n\n"
        
        # Step 4: Use LLM to synthesize research report
        print("🤖 Synthesizing research report using LLM...")
        
        research_prompt = f"""You are a research assistant. Based on the sources provided below, create a comprehensive research report about: "{query}"

Your report should:
1. Provide a clear, well-structured overview of the topic
2. Include key facts and findings from the sources
3. Be factual and cite information from the sources when relevant
4. Be comprehensive but concise
5. Use markdown formatting for better readability

{context}

Now, write a comprehensive research report:"""

        # Get LLM client based on provider
        if model_name:
            # Direct API call for dynamic models
            from openai import OpenAI
            if provider == "litellm":
                client = OpenAI(
                    base_url=config.litellm_base_url,
                    api_key=config.litellm_api_key
                )
            else:  # ollama
                client = OpenAI(
                    base_url=config.ollama_base_url,
                    api_key="ollama"
                )
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": research_prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Handle reasoning models (o1, o3, gpt-oss-120b, etc.) that return content in reasoning_content
            message = response.choices[0].message
            report = message.content
            
            # Check if this is a reasoning model response
            if not report and hasattr(message, 'reasoning_content') and message.reasoning_content:
                report = message.reasoning_content
            
            # Ensure we have some content
            if not report:
                report = "Unable to generate research report. The model returned an empty response."
        else:
            # Use configured client
            llm_client = get_client(provider=provider, model_tier="smart")
            report = llm_client.chat_completion(
                messages=[{"role": "user", "content": research_prompt}],
                temperature=0.3,
                max_tokens=4000
            )
        
        # Add source citations at the end
        report += "\n\n---\n\n**Sources:**\n"
        for i, source in enumerate(scraped_contents, 1):
            report += f"{i}. {source['url']}\n"
        
        print("✓ Research complete!")
        return report
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Research failed: {str(e)}. Please try again or check your configuration."


if __name__ == "__main__":
    # Test the researcher
    query = "Latest developments in artificial intelligence"
    print(f"Testing Firecrawl Researcher with query: {query}\n")
    
    report = conduct_research_firecrawl(query, provider="litellm", model_name="gpt-oss-120b", max_sources=3)
    
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80)
    print(report)

