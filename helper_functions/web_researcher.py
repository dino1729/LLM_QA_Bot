"""
Web research helpers backed by LiteLLM Perplexity search.

Discovery uses the configured LiteLLM Perplexity search endpoint. Page content
is extracted directly with newspaper3k/BeautifulSoup via perplexity_search.py.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests

from config import config
from helper_functions.debug_logger import log_debug_data
from helper_functions.llm_client import get_client
from helper_functions.perplexity_search import extract_web_content, search_with_perplexity

logger = logging.getLogger(__name__)


def extract_page_content(url: str, timeout: int = 30) -> Optional[str]:
    """
    Extract readable text from a URL with direct HTTP/article parsing.

    Returns extracted text or None when the page cannot be read.
    """
    try:
        content, title = extract_web_content(url, timeout=timeout)
        log_debug_data("web_extract", {
            "url": url,
            "status": "success" if content else "empty",
            "title": title,
        })
        return content
    except Exception as e:
        logger.warning("Error extracting %s: %s", url, e)
        return None


def generate_topic_urls(query: str) -> List[str]:
    """
    Generate likely authoritative URLs when configured search returns no results.
    """
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
        ],
    }

    query_lower = query.lower()
    if any(word in query_lower for word in ["news", "latest", "update", "current"]):
        return topic_sources["news"][:2]
    if any(word in query_lower for word in ["tech", "ai", "software", "computer"]):
        return topic_sources["technology"][:2]
    if any(word in query_lower for word in ["science", "research", "study"]):
        return topic_sources["science"][:1]
    return topic_sources["general"][:1]


def search_web_sources(query: str, max_sources: int = 5, timeout: int = 60) -> List[Dict[str, str]]:
    """
    Search for sources through LiteLLM's Perplexity search endpoint.

    If the search endpoint is unavailable, return a small deterministic fallback
    set of likely source URLs so callers can still attempt direct extraction.
    """
    results = search_with_perplexity(query, max_results=max_sources, timeout=timeout)
    if results:
        return results[:max_sources]

    return [
        {"url": url, "title": "Untitled", "snippet": "", "date": "", "source": ""}
        for url in generate_topic_urls(query)[:max_sources]
    ]


def conduct_web_research(query: str, provider: str = "litellm", model_name: str | None = None, max_sources: int = 5) -> str:
    """
    Conduct web research using LiteLLM Perplexity search and direct extraction.

    Args:
        query: Research query
        provider: LLM provider ("litellm" or "ollama")
        model_name: Specific model name for synthesis, if any
        max_sources: Maximum number of sources to read

    Returns:
        Research report string with source URLs appended.
    """
    try:
        print(f"Searching web for: {query}")
        search_results = search_web_sources(query, max_sources=max_sources)

        if not search_results:
            return "Unable to find search results. Please check your internet connection or search query."

        print(f"Found {len(search_results)} URLs to research")

        scraped_contents = []
        for i, item in enumerate(search_results, 1):
            url = item.get("url", "")
            if not url:
                continue
            print(f"Reading source {i}/{len(search_results)}: {url[:60]}...")
            content = extract_page_content(url, timeout=20)
            if content and len(content.strip()) > 100:
                scraped_contents.append({
                    "url": url,
                    "title": item.get("title", "Untitled"),
                    "snippet": item.get("snippet", ""),
                    "content": content[:8000],
                })

            if len(scraped_contents) >= max_sources:
                break

        if not scraped_contents:
            return "Unable to extract readable content from search results. The sources may be behind paywalls or blocking automated access."

        print(f"Successfully read {len(scraped_contents)} sources")

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            context += "\n" + "=" * 80 + "\n\n"

        print("Synthesizing research report using LLM...")

        research_prompt = f"""You are a research assistant. Based on the sources provided below, create a comprehensive research report about: "{query}"

Your report should:
1. Provide a clear, well-structured overview of the topic
2. Include key facts and findings from the sources
3. Be factual and cite information from the sources when relevant
4. Be comprehensive but concise
5. Use markdown formatting for better readability

{context}

Now, write a comprehensive research report:"""

        if model_name:
            from openai import OpenAI

            if provider == "litellm":
                client = OpenAI(
                    base_url=config.litellm_base_url,
                    api_key=config.litellm_api_key,
                )
            else:
                client = OpenAI(
                    base_url=config.ollama_base_url,
                    api_key="ollama",
                )

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": research_prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

            message = response.choices[0].message
            report = message.content
            if not report and hasattr(message, "reasoning_content") and message.reasoning_content:
                report = message.reasoning_content
            if not report:
                report = "Unable to generate research report. The model returned an empty response."
        else:
            llm_client = get_client(provider=provider, model_tier="smart")
            report = llm_client.chat_completion(
                messages=[{"role": "user", "content": research_prompt}],
                temperature=0.3,
                max_tokens=4000,
            )

        report += "\n\n---\n\n**Sources:**\n"
        for i, source in enumerate(scraped_contents, 1):
            report += f"{i}. {source['url']}\n"

        print("Research complete")
        return report

    except Exception as e:
        logger.exception("Error during web research")
        return f"Research failed: {str(e)}. Please check your Perplexity search endpoint and LLM configuration."


if __name__ == "__main__":
    sample_query = "Latest developments in artificial intelligence"
    print(f"Testing web researcher with query: {sample_query}\n")
    provider_name = config.web_research_default_provider
    model = config.web_research_default_model_name or None
    print(conduct_web_research(sample_query, provider=provider_name, model_name=model, max_sources=3))
