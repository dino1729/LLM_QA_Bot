"""
Internet-Connected Chatbot with Firecrawl Researcher
Supports web scraping via Firecrawl and custom research using LLM synthesis
Note: Agent functionality temporarily disabled for llama-index 0.14.8 compatibility
"""
import os
import requests
from datetime import datetime
from config import config
from helper_functions.chat_generation import generate_chat
from helper_functions.llm_client import get_client
# from llama_index.agent.openai import OpenAIAgent  # Temporarily disabled - incompatible with llama-index-core 0.14.8
# from llama_index.tools.weather import OpenWeatherMapToolSpec  # Temporarily disabled
# from llama_index.tools.bing_search import BingSearchToolSpec  # Temporarily disabled
from newspaper import Article
from bs4 import BeautifulSoup
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, PromptHelper, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.indices import SummaryIndex
from llama_index.core import Settings
from helper_functions.firecrawl_researcher import conduct_research_firecrawl
from helper_functions.debug_logger import log_debug_data

# Configuration
openweather_api_key = config.openweather_api_key
firecrawl_server_url = config.firecrawl_server_url
retriever = config.retriever

sum_template = config.sum_template
ques_template = config.ques_template
summary_template = PromptTemplate(sum_template)
qa_template = PromptTemplate(ques_template)

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name
num_output = config.num_output
max_chunk_overlap_ratio = config.max_chunk_overlap_ratio
max_input_size = config.max_input_size
context_window = config.context_window
keywords = config.keywords

# Initialize default LLM client (will be overridden in functions based on user selection)
default_client = get_client(provider="litellm", model_tier="smart")

# Configure LlamaIndex Settings with default client
Settings.llm = default_client.get_llamaindex_llm()
Settings.embed_model = default_client.get_llamaindex_embedding()
text_splitter = SentenceSplitter()
Settings.text_splitter = text_splitter
Settings.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

WEB_SEARCH_FOLDER = config.WEB_SEARCH_FOLDER
if not os.path.exists(WEB_SEARCH_FOLDER):
    os.makedirs(WEB_SEARCH_FOLDER)

system_prompt = config.system_prompt


def saveextractedtext_to_file(text, filename):
    """Save extracted text to a file in the web search folder"""
    file_path = os.path.join(WEB_SEARCH_FOLDER, filename)
    with open(file_path, 'w') as file:
        file.write(text)
    return f"Text saved to {file_path}"


def clearallfiles_websearch():
    """Clear all files in the web search folder"""
    for root, dirs, files in os.walk(WEB_SEARCH_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


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
    Scrape a URL using Firecrawl server

    Args:
        url: The URL to scrape

    Returns:
        Scraped text content or None if failed
    """
    try:
        # Firecrawl v2 API endpoint
        scrape_url = f"{firecrawl_server_url}/scrape"

        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "includeTags": ["article", "main", "content", "p", "h1", "h2", "h3"],
            "excludeTags": ["nav", "footer", "header", "aside"],
            "waitFor": 1000
        }

        response = requests.post(scrape_url, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # Log successful scrape
            log_debug_data("chat_firecrawl_scrape", {
                "url": url,
                "status": "success",
                "response": result
            })
            
            # Extract markdown or HTML content
            if "data" in result and "markdown" in result["data"]:
                return result["data"]["markdown"]
            elif "data" in result and "html" in result["data"]:
                soup = BeautifulSoup(result["data"]["html"], 'html.parser')
                return soup.get_text()
        else:
            print(f"Firecrawl scrape failed with status {response.status_code}")
            # Log failed scrape
            log_debug_data("chat_firecrawl_scrape_error", {
                "url": url,
                "status": "error",
                "status_code": response.status_code,
                "response_text": response.text
            })
            return None

    except Exception as e:
        print(f"Error scraping with Firecrawl: {str(e)}")
        return None




def text_extractor(url, debug=False):
    """
    Extract text from URL using multiple methods:
    1. Firecrawl (if configured)
    2. Newspaper3k
    3. BeautifulSoup (fallback)
    """
    # Try Firecrawl first if configured
    if retriever == "firecrawl":
        firecrawl_text = scrape_with_firecrawl(url)
        if firecrawl_text:
            return firecrawl_text

    # Fallback to newspaper3k
    if url:
        article = Article(url)
        try:
            article.download()
            article.parse()
            if len(article.text.split()) < 75:
                raise Exception("Article is too short. Probably the article is behind a paywall.")
            return article.text
        except Exception as e:
            if debug:
                print(f"Failed to download and parse article from URL using newspaper package: {url}. Error: {str(e)}")
            # Try BeautifulSoup method
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                return soup.get_text()
            except Exception as e:
                if debug:
                    print(f"Failed to download article using beautifulsoup method from URL: {url}. Error: {str(e)}")
                return None
    return None


def get_bing_agent(query):
    """Use Bing search agent (deprecated - replaced with Firecrawl)"""
    # Bing search functionality has been replaced with Firecrawl Researcher
    # Temporarily disabled - OpenAIAgent incompatible with llama-index-core 0.14.8
    # bing_tool = BingSearchToolSpec(api_key=bing_api_key)
    # agent = OpenAIAgent.from_tools(
    #     bing_tool.to_tool_list(),
    #     llm=Settings.llm,
    #     verbose=False,
    # )
    # return str(agent.chat(query))
    return "Bing search agent functionality is temporarily disabled while upgrading to llama-index 0.14.8. Please use Firecrawl or GPT Researcher for web search."




def get_bing_news_results(query, num=5, provider="litellm", model_name=None):
    """Get news results using Firecrawl Researcher"""
    # Use Firecrawl Researcher for all news queries
    return firecrawl_researcher(query + " latest news", provider, model_name)




def get_web_results(query, num=10, provider="litellm", model_name=None):
    """
    Get web results using Firecrawl Researcher

    Args:
        query: Search query
        num: Number of results (used for max_sources in researcher)
        provider: LLM provider to use
        model_name: Specific model name to use

    Returns:
        Research report from Firecrawl Researcher
    """
    # Use Firecrawl Researcher for all web queries
    return firecrawl_researcher(query, provider, model_name)


def parse_dynamic_model_name(model_name):
    """
    Parse dynamic model name to extract provider and actual model
    
    Args:
        model_name: Model name (e.g., "LITELLM:gpt-4" or "OLLAMA:llama3.2:3b" or "LITELLM_SMART")
    
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
        actual_model = None  # Use default configured model
    elif model_name.startswith("OLLAMA_"):
        provider = "ollama"
        actual_model = None  # Use default configured model
    else:
        # Fallback to litellm for other models (GEMINI, COHERE, GROQ)
        provider = "litellm"
        actual_model = None
    
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
        fast_response: Use fast search (Firecrawl/Bing) vs deep research

    Returns:
        Assistant's response
    """
    assistant_reply = "Sorry, I couldn't generate a response. Please try again."

    try:
        # Build conversation from history
        conversation = system_prompt.copy()
        for human, assistant in history:
            conversation.append({"role": "user", "content": human})
            conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": query})

        try:
            # Check if query requires web search
            if any(keyword in query.lower() for keyword in keywords):
                # Parse model name for GPT Researcher
                provider, actual_model = parse_dynamic_model_name(model_name)
                
                # News queries
                if "news" in query.lower():
                    if fast_response:
                        assistant_reply = get_bing_news_results(query, provider=provider, model_name=actual_model)
                    else:
                        assistant_reply = firecrawl_researcher(query, provider, actual_model)
                # Weather queries
                elif "weather" in query.lower():
                    assistant_reply = get_weather_data(query)
                # General web search
                else:
                    if fast_response:
                        assistant_reply = get_web_results(query, provider=provider, model_name=actual_model)
                    else:
                        assistant_reply = firecrawl_researcher(query, provider, actual_model)
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
    Conducts research on a given query using Firecrawl and LLM

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
        print(f"Error in Firecrawl Researcher: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error conducting research: {str(e)}. Please check your Firecrawl server and LLM configuration."
