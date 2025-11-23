from gpt_researcher import GPTResearcher
import dotenv
import asyncio
import logging
import os
from config import config

# Configure logging to show only errors
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

def setup_researcher_env(provider="litellm", model_name=None):
    """
    Setup environment variables for GPT Researcher based on provider
    
    Args:
        provider: Either "litellm" or "ollama"
        model_name: Optional specific model name (e.g., "gpt-oss-120b" or "deepseek-r1:14b")
    """
    if provider == "litellm":
        os.environ["OPENAI_API_KEY"] = config.litellm_api_key
        os.environ["OPENAI_API_BASE"] = config.litellm_base_url
        # Use the smart model as default for research, or the provided model
        if model_name:
            os.environ["SMART_LLM_MODEL"] = model_name
            os.environ["FAST_LLM_MODEL"] = model_name
        else:
            smart_model = config.litellm_smart_llm.split(":", 1)[-1] if ":" in config.litellm_smart_llm else config.litellm_smart_llm
            fast_model = config.litellm_fast_llm.split(":", 1)[-1] if ":" in config.litellm_fast_llm else config.litellm_fast_llm
            os.environ["SMART_LLM_MODEL"] = smart_model
            os.environ["FAST_LLM_MODEL"] = fast_model
        # Set embedding model
        embedding_model = config.litellm_embedding.split(":", 1)[-1] if ":" in config.litellm_embedding else config.litellm_embedding
        os.environ["EMBEDDING_MODEL"] = embedding_model
    elif provider == "ollama":
        # For Ollama, we use the ollama base URL
        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_API_BASE"] = config.ollama_base_url
        # Use the smart model as default for research, or the provided model
        if model_name:
            os.environ["SMART_LLM_MODEL"] = model_name
            os.environ["FAST_LLM_MODEL"] = model_name
        else:
            smart_model = config.ollama_smart_llm.split(":", 1)[-1] if ":" in config.ollama_smart_llm else config.ollama_smart_llm
            fast_model = config.ollama_fast_llm.split(":", 1)[-1] if ":" in config.ollama_fast_llm else config.ollama_fast_llm
            os.environ["SMART_LLM_MODEL"] = smart_model
            os.environ["FAST_LLM_MODEL"] = fast_model
        # Set embedding model
        embedding_model = config.ollama_embedding.split(":", 1)[-1] if ":" in config.ollama_embedding else config.ollama_embedding
        os.environ["EMBEDDING_MODEL"] = embedding_model
    
    # Set retriever configuration for GPT Researcher
    # Check if Tavily API key is configured in environment or config
    tavily_key = os.environ.get("TAVILY_API_KEY") or getattr(config, "tavily_api_key", None)
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
        os.environ["RETRIEVER"] = "tavily"
    else:
        # Use DuckDuckGo as fallback if no Tavily key
        os.environ["RETRIEVER"] = "duckduckgo"
        # Set a dummy Tavily key to prevent errors
        os.environ["TAVILY_API_KEY"] = "tvly-placeholder"

async def get_report(query: str, report_type: str, provider="litellm", model_name=None):
    """
    Get research report using GPT Researcher
    
    Args:
        query: Research query
        report_type: Type of report (e.g., "research_report")
        provider: Either "litellm" or "ollama"
        model_name: Optional specific model name
    
    Returns:
        Tuple of (report, context, costs, images, sources)
    """
    try:
        # Setup environment for GPT Researcher
        setup_researcher_env(provider, model_name)
        
        researcher = GPTResearcher(query, report_type)
        research_result = await researcher.conduct_research()
        report = await researcher.write_report()
        
        # Get additional information
        research_context = researcher.get_research_context()
        research_costs = researcher.get_costs()
        research_images = researcher.get_research_images()
        research_sources = researcher.get_research_sources()
        
        return report, research_context, research_costs, research_images, research_sources
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

if __name__ == "__main__":
    query = "Latest news headlines from Financial markets. Answer only with headlines and brief descriptions."
    report_type = "research_report"
    provider = "litellm"  # or "ollama"
    model_name = None  # Use default configured model

    try:
        report, context, costs, images, sources = asyncio.run(get_report(query, report_type, provider, model_name))
        
        if report:
            print("Report:")
            print(report)
        else:
            logger.error("Research failed to complete.")
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")