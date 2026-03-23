"""
Internet-connected chatbot with Perplexity-backed web research.
Note: Agent functionality temporarily disabled for llama-index 0.14.8 compatibility
"""
import os
import requests
from datetime import datetime
from config import config
from helper_functions.chat_generation import generate_chat
from helper_functions.llm_client import get_client
from bs4 import BeautifulSoup
from helper_functions.firecrawl_researcher import conduct_research_firecrawl
from helper_functions.debug_logger import log_debug_data
from helper_functions.perplexity_search import extract_web_content

try:
    from newspaper import Article
except ImportError:
    Article = None

try:
    from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
except ImportError:
    SimpleDirectoryReader = None
    SummaryIndex = None
    VectorStoreIndex = None

# Configuration
openweather_api_key = config.openweather_api_key
retriever = config.retriever
bing_api_key = getattr(config, "bing_api_key", "")

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name
keywords = config.keywords

WEB_SEARCH_FOLDER = config.WEB_SEARCH_FOLDER
if not os.path.exists(WEB_SEARCH_FOLDER):
    os.makedirs(WEB_SEARCH_FOLDER)
BING_FOLDER = WEB_SEARCH_FOLDER

system_prompt = config.system_prompt


def saveextractedtext_to_file(text, filename):
    """Save extracted text to a file in the web search folder"""
    file_path = os.path.join(BING_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)
    return f"Text saved to {file_path}"


def clearallfiles_websearch():
    """Clear all files in the web search folder"""
    for root, dirs, files in os.walk(BING_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def clearallfiles_bing():
    """Backward-compatible alias for clearing extracted web-search files."""
    clearallfiles_websearch()


def get_weather_data(query):
    """Get weather data using OpenWeatherMap"""
    # Temporarily disabled - OpenAIAgent incompatible with llama-index-core 0.14.8
    # weather_tool = OpenWeatherMapToolSpec(key=openweather_api_key)
    # agent = OpenAIAgent.from_tools(
    #     weather_tool.to_tool_list(),
    #     llm=Settings.llm,
    #     verbose=False,
    # )
    # return str(agent.chat(query))
    return "Weather data functionality is temporarily disabled while upgrading to llama-index 0.14.8. Please use Firecrawl or GPT Researcher for weather information."


def scrape_with_firecrawl(url):
    """
    Compatibility wrapper: extract text directly from a URL.

    Args:
        url: The URL to scrape

    Returns:
        Scraped text content or None if failed
    """
    try:
        content, title = extract_web_content(url, timeout=30)
        if content:
            log_debug_data("chat_web_extract", {
                "url": url,
                "status": "success",
                "title": title,
            })
            return content

    except Exception as e:
        print(f"Error extracting content: {str(e)}")

    try:
        scrape_url = getattr(config, "firecrawl_server_url", "") or "http://localhost:3002/v1/scrape"
        response = requests.post(scrape_url, json={"url": url}, timeout=30)
        if response.status_code != 200:
            return None

        payload = response.json() or {}
        data = payload.get("data", {})
        markdown = data.get("markdown")
        if markdown:
            return markdown

        html = data.get("html")
        if html:
            return BeautifulSoup(html, "html.parser").get_text()
    except Exception:
        return None

    return None




def text_extractor(url, debug=False):
    """
    Extract text from URL using multiple methods:
    1. Direct URL extraction when web retrieval is enabled
    2. Newspaper3k
    3. BeautifulSoup (fallback)
    """
    # Try direct extraction first if web retrieval is enabled
    if retriever in ("firecrawl", "perplexity"):
        firecrawl_text = scrape_with_firecrawl(url)
        if firecrawl_text:
            return firecrawl_text

    # Fallback to newspaper3k
    if url:
        if Article is not None:
            try:
                article = Article(url)
                article.download()
                article.parse()
                if article.text:
                    return article.text
            except Exception as e:
                if debug:
                    print(f"Failed to download and parse article from URL using newspaper package: {url}. Error: {str(e)}")
        elif debug:
            print("newspaper3k not installed, falling back to BeautifulSoup extraction.")

        try:
            req = requests.get(url)
            html_text = getattr(req, "text", "")
            if isinstance(html_text, str) and html_text:
                html = html_text
            else:
                html = getattr(req, "content", b"")
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        except Exception as e:
            if debug:
                print(f"Failed to download article using beautifulsoup method from URL: {url}. Error: {str(e)}")
            return None
    return None


def search_with_firecrawl(query, limit=5):
    """Backward-compatible Bing search + scrape flow used by older tests and callers."""
    if not bing_api_key:
        return None

    try:
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": bing_api_key},
            params={"q": query, "count": limit},
            timeout=20,
        )
        if response.status_code != 200:
            return None

        payload = response.json() or {}
        results = payload.get("webPages", {}).get("value", [])
        scraped_chunks = []
        for item in results[:limit]:
            url = item.get("url")
            if not url:
                continue
            content = scrape_with_firecrawl(url)
            if content and len(content.split()) >= 50:
                scraped_chunks.append(f"Source: {url}\n{content}")

        if not scraped_chunks:
            return None
        return f"Search query: {query}\n\n" + "\n\n".join(scraped_chunks)
    except Exception:
        return None


def summarize(folder_path):
    """Backward-compatible summary helper for a folder of extracted documents."""
    if SimpleDirectoryReader is None or SummaryIndex is None:
        return None

    try:
        documents = SimpleDirectoryReader(folder_path).load_data()
        if not documents:
            return None
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return query_engine.query("Summarize the documents.")
    except Exception:
        return None


def simple_query(folder_path, query):
    """Backward-compatible vector query helper for a folder of extracted documents."""
    if SimpleDirectoryReader is None or VectorStoreIndex is None:
        return None

    try:
        documents = SimpleDirectoryReader(folder_path).load_data()
        if not documents:
            return None
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return query_engine.query(query)
    except Exception:
        return None


def get_bing_agent(query):
    """Use Bing search agent (deprecated - replaced with Perplexity research)"""
    # Bing search functionality has been replaced with Perplexity-backed research
    # Temporarily disabled - OpenAIAgent incompatible with llama-index-core 0.14.8
    # bing_tool = BingSearchToolSpec(api_key=bing_api_key)
    # agent = OpenAIAgent.from_tools(
    #     bing_tool.to_tool_list(),
    #     llm=Settings.llm,
    #     verbose=False,
    # )
    # return str(agent.chat(query))
    return "Bing search agent functionality is temporarily disabled while upgrading to llama-index 0.14.8. Please use Perplexity-backed research for web search."




def get_bing_news_results(query, num=5, provider="litellm", model_name=None):
    """Get news results using Perplexity-backed researcher."""
    try:
        return firecrawl_researcher(query + " latest news", provider or "litellm", model_name)
    except Exception as e:
        return f"News results not available: {e}"




def get_web_results(query, num=10, provider="litellm", model_name=None):
    """
    Get web results using Perplexity-backed researcher

    Args:
        query: Search query
        num: Number of results (used for max_sources in researcher)
        provider: LLM provider to use
        model_name: Specific model name to use

    Returns:
        Processed answer or research report fallback.
    """
    clearallfiles_bing()

    search_results = search_with_firecrawl(query, limit=num)
    if search_results:
        saveextractedtext_to_file(search_results, "web_search_results.txt")
        response = simple_query(BING_FOLDER, query)
        if hasattr(response, "response"):
            return response.response
        if response:
            return str(response)

    return firecrawl_researcher(query, provider or "litellm", model_name)


def parse_dynamic_model_name(model_name):
    """
    Parse dynamic model name to extract provider and actual model
    
    Args:
        model_name: Model name (e.g., "LITELLM:model-name" or "OLLAMA:model:size" or "LITELLM_SMART")
    
    Returns:
        Tuple of (provider, actual_model_name)
    """
    if model_name.startswith("LITELLM:"):
        provider = "litellm"
        actual_model = model_name.split("LITELLM:", 1)[1]
    elif model_name.startswith("OLLAMA:"):
        provider = "ollama"
        actual_model = model_name.split("OLLAMA:", 1)[1]
    elif model_name.startswith("LITELLM_"):
        provider = "litellm"
        actual_model = model_name.split("LITELLM_", 1)[1].lower()
    elif model_name.startswith("OLLAMA_"):
        provider = "ollama"
        actual_model = model_name.split("OLLAMA_", 1)[1].lower()
    else:
        provider = None
        actual_model = model_name
    
    return provider, actual_model


def internet_connected_chatbot(query, history, model_name, max_tokens, temperature, fast_response=True):
    """
    Internet-connected chatbot with web search capabilities

    Args:
        query: User query
        history: Chat history
        model_name: Model to use (can be dynamic "PROVIDER:model" or predefined)
        max_tokens: Max tokens for response
        temperature: Sampling temperature
        fast_response: Use fast search vs deep research

    Returns:
        Assistant's response
    """
    assistant_reply = "Sorry, I couldn't generate a response. Please try again."

    try:
        # Build conversation from history
        conversation = system_prompt.copy()
        for entry in history:
            if isinstance(entry, dict):
                conversation.append({
                    "role": entry.get("role", "user"),
                    "content": entry.get("content", ""),
                })
            else:
                human, assistant = entry
                conversation.append({"role": "user", "content": human})
                conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": query})
        query_lower = query.lower()

        try:
            # Check if query requires web search
            requires_web_search = (
                any(keyword in query_lower for keyword in keywords)
                or "search" in query_lower
                or "news" in query_lower
            )
            if requires_web_search:
                # Parse model name for GPT Researcher
                provider, actual_model = parse_dynamic_model_name(model_name)
                context_text = None
                
                # News queries
                if "news" in query_lower and "search" not in query_lower:
                    if fast_response:
                        context_text = get_bing_news_results(query, provider=provider, model_name=actual_model)
                    else:
                        context_text = firecrawl_researcher(query, provider or "litellm", actual_model)
                # Weather queries
                elif "weather" in query_lower:
                    context_text = get_weather_data(query)
                # General web search
                else:
                    if fast_response:
                        context_text = get_web_results(query, provider=provider, model_name=actual_model)
                    else:
                        context_text = firecrawl_researcher(query, provider or "litellm", actual_model)

                if context_text:
                    conversation.append({
                        "role": "assistant",
                        "content": f"Web context:\n{context_text}",
                    })
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
            else:
                # Generate response using selected model
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)

        except Exception as e:
            print(f"Model error: {str(e)}")
            print("Resetting conversation...")
            conversation = system_prompt.copy()

    except Exception as e:
        print(f"Error occurred while generating response: {str(e)}")
        conversation = system_prompt.copy()

    return assistant_reply


def firecrawl_researcher(query, provider="litellm", model_name=None):
    """
    Conducts research on a given query using Perplexity search and LLM

    Args:
        query: The research query
        provider: Either "litellm" or "ollama"
        model_name: Optional specific model name

    Returns:
        Research report or error message
    """
    try:
        report = conduct_research_firecrawl(query, provider=provider, model_name=model_name, max_sources=5)
        return report if report else "Research failed to complete. Please check your configuration."
    except Exception as e:
        print(f"Error in Perplexity-backed researcher: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error conducting research: {str(e)}. Please check your Perplexity search endpoint and LLM configuration."
