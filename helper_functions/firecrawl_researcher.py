"""
Custom Research Feature using Perplexity search and direct page extraction.
"""
import requests
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from config import config
from helper_functions.llm_client import get_client
from helper_functions.debug_logger import log_debug_data
from helper_functions.perplexity_search import extract_web_content, search_with_perplexity

def scrape_with_firecrawl(url, timeout=30):
    """
    Compatibility wrapper: extract a URL directly without Firecrawl.
    
    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content or None if failed
    """
    try:
        scrape_url = getattr(config, "firecrawl_server_url", "") or "http://localhost:3002"
        response = requests.post(f"{scrape_url}/scrape", json={"url": url, "formats": ["markdown"]}, timeout=timeout)
        if response.status_code == 200:
            payload = response.json() or {}
            data = payload.get("data", {})
            markdown = data.get("markdown")
            if markdown:
                return markdown
            html = data.get("html")
            if html:
                return BeautifulSoup(html, "html.parser").get_text()
            if payload.get("error"):
                return None
        elif response.status_code >= 400:
            return None
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

    try:
        content, title = extract_web_content(url, timeout=timeout)
        log_debug_data("web_extract", {
            "url": url,
            "status": "success" if content else "empty",
            "title": title,
        })
        return content
    except Exception as e:
        print(f"Error extracting {url}: {str(e)}")
        return None




def get_search_urls_fallback(query, count=5):
    """
    Compatibility wrapper: get search result URLs via Perplexity.
    
    Args:
        query: Search query
        count: Number of results
    
    Returns:
        List of URLs
    """
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=20,
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            urls = []
            for link in soup.select("a.result__url"):
                href = link.get("href", "")
                if href.startswith("http"):
                    urls.append(href)
            if urls:
                return urls[:count]
        if response.status_code != 200:
            return generate_topic_urls(query)
    except Exception as e:
        print(f"Error with fallback search: {str(e)}")

    try:
        results = search_with_perplexity(query, max_results=count)
        urls = [item["url"] for item in results if item.get("url")]
        if urls:
            return urls[:count]
    except Exception as e:
        print(f"Error with Perplexity search: {str(e)}")
        return generate_topic_urls(query)

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
    Conduct research using Perplexity search and LLM synthesis.
    
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

        search_results = []
        if hasattr(search_with_perplexity, "mock_calls"):
            search_results = search_with_perplexity(query, max_results=max_sources) or []

        if search_results:
            urls = [item.get("url") for item in search_results if item.get("url")]
        else:
            urls = get_search_urls_fallback(query, count=max_sources)
            search_results = [
                {"url": url, "title": "Untitled", "snippet": ""}
                for url in urls
            ]
        
        if not urls:
            return "Unable to find search results. Please check your internet connection or search query."
        
        print(f"📋 Found {len(urls)} URLs to research")
        
        # Step 2: Extract content from URLs directly
        scraped_contents = []
        for i, item in enumerate(search_results, 1):
            url = item["url"]
            print(f"📄 Scraping source {i}/{len(urls)}: {url[:60]}...")
            content = scrape_with_firecrawl(url, timeout=20)
            if content and len(content.strip()) > 100:
                scraped_contents.append({
                    "url": url,
                    "title": item.get("title", "Untitled"),
                    "snippet": item.get("snippet", ""),
                    "content": content[:8000],  # Limit per source to manage token count
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
            if source.get("title"):
                context += f"Source {i} Title: {source['title']}\n"
            context += f"Source {i}: {source['url']}\n"
            if source.get("snippet"):
                context += f"Snippet: {source['snippet']}\n"
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
            
            # Handle reasoning models that return content in reasoning_content field
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
    # Test the researcher - uses config values for provider/model
    query = "Latest developments in artificial intelligence"
    print(f"Testing Firecrawl Researcher with query: {query}\n")
    
    # Use configured defaults from config.yml
    provider = config.firecrawl_default_provider
    model_name = config.firecrawl_default_model_name if config.firecrawl_default_model_name else None
    
    report = conduct_research_firecrawl(query, provider=provider, model_name=model_name, max_sources=3)
    
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80)
    print(report)
