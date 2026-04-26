"""
Internet-connected chatbot backed by LiteLLM Perplexity search.
"""
from config import config
from helper_functions.chat_generation import generate_chat
from helper_functions.web_researcher import conduct_web_research, extract_page_content, search_web_sources

try:
    from newspaper import Article
except ImportError:
    Article = None

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name
keywords = config.keywords
system_prompt = config.system_prompt

QUICK_SEARCH_TIMEOUT_SECONDS = 12

NO_SEARCH_EXACT_QUERIES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "who are you",
}

NO_SEARCH_STARTERS = (
    "write ",
    "draft ",
    "rewrite ",
    "rephrase ",
    "translate ",
    "brainstorm ",
    "create ",
    "generate ",
)

QUESTION_STARTERS = (
    "who ",
    "what ",
    "when ",
    "where ",
    "why ",
    "how ",
    "which ",
    "is ",
    "are ",
    "can ",
    "does ",
    "do ",
    "should ",
    "tell me about ",
    "compare ",
    "find ",
    "look up ",
)

LEGACY_MODEL_TOKENS = {
    "LITELLM",
    "LITELLM_FAST",
    "LITELLM_SMART",
    "LITELLM_STRATEGIC",
    "OLLAMA",
    "OLLAMA_FAST",
    "OLLAMA_SMART",
    "OLLAMA_STRATEGIC",
    "GEMINI",
    "GEMINI_THINKING",
    "GROQ",
    "GROQ_LLAMA",
    "GROQ_MIXTRAL",
}


def get_weather_data(query: str) -> str:
    """Weather support is currently disabled in this chat surface."""
    return "Weather data functionality is currently disabled. Please use web research or a weather provider integration for weather information."


def text_extractor(url: str, debug: bool = False):
    """
    Extract text from a URL using direct page extraction with a newspaper3k fallback.
    """
    content = extract_page_content(url)
    if content:
        return content

    if Article is None:
        if debug:
            print("newspaper3k not installed.")
        return None

    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text or None
    except Exception as e:
        if debug:
            print(f"Failed to extract article from {url}: {e}")
        return None


def get_news_results(query: str, num: int = 5, provider: str = "litellm", model_name: str | None = None) -> str:
    """Get news-oriented quick search context using LiteLLM Perplexity search."""
    try:
        sources = get_quick_web_sources(f"{query} latest news", max_results=num)
        return format_quick_web_context(query, sources)
    except Exception as e:
        return f"News results not available: {e}"


def get_web_results(query: str, num: int = 5, provider: str = "litellm", model_name: str | None = None) -> str:
    """Get quick search context using LiteLLM Perplexity search."""
    sources = get_quick_web_sources(query, max_results=num)
    return format_quick_web_context(query, sources)


def query_requires_web_search(query: str) -> bool:
    query_lower = query.lower().strip()
    if not query_lower:
        return False

    if query_lower in NO_SEARCH_EXACT_QUERIES:
        return False

    if any(keyword in query_lower for keyword in keywords):
        return True

    if "search" in query_lower or "news" in query_lower:
        return True

    if _is_simple_math_query(query_lower):
        return False

    if query_lower.startswith(NO_SEARCH_STARTERS):
        return False

    return query_lower.startswith(QUESTION_STARTERS) or query_lower.endswith("?")


def _is_simple_math_query(query_lower: str) -> bool:
    candidate = query_lower
    for prefix in ("what is ", "what's ", "calculate ", "solve "):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break

    compact = candidate.replace(" ", "")
    return bool(compact) and all(char in "0123456789+-*/().=?" for char in compact)


def build_search_query(query: str) -> str:
    query_lower = query.lower()
    if "news" in query_lower and "latest" not in query_lower:
        return f"{query} latest news"
    return query


def get_quick_web_sources(query: str, max_results: int = 3):
    """Return a small set of search results without reading full pages."""
    return search_web_sources(query, max_sources=max_results, timeout=QUICK_SEARCH_TIMEOUT_SECONDS)[:max_results]


def format_quick_web_context(query: str, sources) -> str:
    """Build compact answer context from search-result metadata and snippets."""
    if not sources:
        return "No search results were found."

    lines = [
        f"Search query: {query}",
        "Use these quick search results. Treat snippets as partial context and mention uncertainty when needed.",
        "",
    ]
    for index, source in enumerate(sources, 1):
        title = source.get("title") or "Untitled"
        url = source.get("url") or ""
        snippet = source.get("snippet") or ""
        date = source.get("date") or ""
        source_name = source.get("source") or ""
        lines.append(f"Source {index}: {title}")
        if source_name:
            lines.append(f"Publisher: {source_name}")
        if date:
            lines.append(f"Date: {date}")
        if url:
            lines.append(f"URL: {url}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def parse_dynamic_model_name(model_name: str):
    """
    Parse dynamic model name to extract provider and actual model.
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


def normalize_internet_model_name(model_name: str | None) -> str:
    """Normalize internet-chat model names into tokens generate_chat accepts."""
    if not model_name:
        provider = (getattr(config, "default_internet_chat_provider", "litellm") or "litellm").upper()
        tier = (getattr(config, "default_internet_chat_tier", "smart") or "smart").upper()
        return f"{provider}_{tier}"

    model_name = model_name.strip()
    upper_model_name = model_name.upper()

    if upper_model_name.startswith("LITELLM:"):
        return f"LITELLM:{model_name.split(':', 1)[1]}"
    if upper_model_name.startswith("OLLAMA:"):
        return f"OLLAMA:{model_name.split(':', 1)[1]}"
    if upper_model_name in LEGACY_MODEL_TOKENS:
        return upper_model_name

    provider = (getattr(config, "default_internet_chat_provider", "litellm") or "litellm").upper()
    if provider == "OLLAMA":
        return f"OLLAMA:{model_name}"
    return f"LITELLM:{model_name}"


def _build_conversation(history, query: str):
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
    return conversation


def _append_web_prompt(conversation, query: str, context_text: str | None) -> None:
    if context_text:
        conversation.append({
            "role": "user",
            "content": (
                "Answer the user's question using the web context below. "
                "If the context is incomplete or failed, say that plainly and "
                "avoid claiming you have no internet access.\n\n"
                f"User question: {query}\n\n"
                f"Web context:\n{context_text}"
            ),
        })
    else:
        conversation.append({"role": "user", "content": query})


def answer_with_context(query: str, history, model_name: str, max_tokens: int, temperature: float, context_text: str | None):
    model_name = normalize_internet_model_name(model_name)
    conversation = _build_conversation(history, query)
    _append_web_prompt(conversation, query, context_text)
    return generate_chat(model_name, conversation, temperature, max_tokens)


def internet_connected_chatbot(query, history, model_name, max_tokens, temperature, fast_response=True):
    """
    Internet-connected chatbot with web search capabilities.
    """
    assistant_reply = "Sorry, I couldn't generate a response. Please try again."
    model_name = normalize_internet_model_name(model_name)

    try:
        conversation = _build_conversation(history, query)
        query_lower = query.lower()

        try:
            if query_requires_web_search(query):
                provider, actual_model = parse_dynamic_model_name(model_name)
                context_text = None

                if "weather" in query_lower:
                    context_text = get_weather_data(query)
                elif "news" in query_lower and "search" not in query_lower and fast_response:
                    context_text = get_news_results(query, num=3, provider=provider, model_name=actual_model)
                else:
                    context_text = get_web_results(query, num=3, provider=provider, model_name=actual_model)

                _append_web_prompt(conversation, query, context_text)
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)
            else:
                conversation.append({"role": "user", "content": query})
                assistant_reply = generate_chat(model_name, conversation, temperature, max_tokens)

        except Exception as e:
            print(f"Model error: {str(e)}")

    except Exception as e:
        print(f"Error occurred while generating response: {str(e)}")

    return assistant_reply


def research_web(query: str, provider: str = "litellm", model_name: str | None = None, max_sources: int = 5) -> str:
    """
    Conduct web research using LiteLLM Perplexity search and LLM synthesis.
    """
    try:
        report = conduct_web_research(query, provider=provider, model_name=model_name, max_sources=max_sources)
        return report if report else "Research failed to complete. Please check your configuration."
    except Exception as e:
        print(f"Error in web researcher: {str(e)}")
        return f"Error conducting research: {str(e)}. Please check your Perplexity search endpoint and LLM configuration."
