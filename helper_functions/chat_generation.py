"""
Chat Generation Module
Supports multiple LLM providers: LiteLLM, Ollama, Gemini aliases via LiteLLM, and Groq
"""
import logging

from openai import OpenAI
from config import config
from helper_functions.llm_client import get_client

try:
    from groq import Groq
except ImportError:
    Groq = None

# Configured aliases for Gemini-compatible routes on LiteLLM
gemini_model_name = config.gemini_model_name or config.litellm_fast_llm
gemini_thinkingmodel_name = config.gemini_thinkingmodel_name or config.litellm_smart_llm
groq_api_key = config.groq_api_key
groq_model_name = config.groq_model_name
groq_llama_model_name = config.groq_llama_model_name
groq_qwen_model_name = config.groq_qwen_model_name

logger = logging.getLogger(__name__)


def _extract_message_content(message):
    """Extract text from OpenAI-compatible chat messages across providers."""
    content = getattr(message, "content", None)

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    text_parts.append(str(text))
            elif part:
                text_parts.append(str(part))
        content = "\n".join(text_parts)

    if not content and hasattr(message, "reasoning_content") and message.reasoning_content:
        content = message.reasoning_content

    if content and "</think>" in content:
        content = content.split("</think>", 1)[-1].lstrip()
    elif content and "<think>" in content and "</think>" not in content:
        logger.warning("Truncated <think> block detected (no closing tag), response may be incomplete")

    return content.strip() if isinstance(content, str) else ""


def _generate_litellm_alias_chat(model_name, conversation, temperature, max_tokens, model_tier):
    """Route compatibility aliases through LiteLLM with an explicit model name."""
    client = get_client(provider="litellm", model_tier=model_tier, model_name=model_name)
    return client.chat_completion(conversation, temperature, max_tokens)


def generate_chat(model_name, conversation, temperature, max_tokens):
    """
    Generate chat completion using the specified model

    Args:
        model_name: Name of the model provider/type (can be predefined or dynamic "PROVIDER:model_name")
        conversation: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    
    # Check if this is a dynamic model name with provider prefix (e.g., "LITELLM:deepseek-v3.1" or "OLLAMA:granite4:3b")
    if model_name.startswith("LITELLM:"):
        # Extract the actual model name after the prefix
        actual_model = model_name.split("LITELLM:", 1)[1]
        # Use LiteLLM with the specific model
        client = OpenAI(
            base_url=config.litellm_base_url,
            api_key=config.litellm_api_key
        )
        response = client.chat.completions.create(
            model=actual_model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return _extract_message_content(response.choices[0].message)
    
    elif model_name.startswith("OLLAMA:"):
        # Extract the actual model name after the prefix
        actual_model = model_name.split("OLLAMA:", 1)[1]
        # Use Ollama with the specific model
        client = OpenAI(
            base_url=config.ollama_base_url,
            api_key="ollama"  # Ollama doesn't need a real API key
        )
        response = client.chat.completions.create(
            model=actual_model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return _extract_message_content(response.choices[0].message)

    # LiteLLM models (predefined tiers)
    elif model_name == "LITELLM_FAST":
        client = get_client(provider="litellm", model_tier="fast")
        return client.chat_completion(conversation, temperature, max_tokens)

    elif model_name == "LITELLM_SMART":
        client = get_client(provider="litellm", model_tier="smart")
        return client.chat_completion(conversation, temperature, max_tokens)

    elif model_name == "LITELLM_STRATEGIC":
        client = get_client(provider="litellm", model_tier="strategic")
        return client.chat_completion(conversation, temperature, max_tokens)

    # Ollama models (predefined tiers)
    elif model_name == "OLLAMA_FAST":
        client = get_client(provider="ollama", model_tier="fast")
        return client.chat_completion(conversation, temperature, max_tokens)

    elif model_name == "OLLAMA_SMART":
        client = get_client(provider="ollama", model_tier="smart")
        return client.chat_completion(conversation, temperature, max_tokens)

    elif model_name == "OLLAMA_STRATEGIC":
        client = get_client(provider="ollama", model_tier="strategic")
        return client.chat_completion(conversation, temperature, max_tokens)

    # Gemini compatibility aliases now route through LiteLLM
    elif model_name == "GEMINI":
        return _generate_litellm_alias_chat(
            gemini_model_name,
            conversation,
            temperature,
            max_tokens,
            model_tier="fast",
        )

    elif model_name == "GEMINI_THINKING":
        return _generate_litellm_alias_chat(
            gemini_thinkingmodel_name,
            conversation,
            temperature,
            max_tokens,
            model_tier="smart",
        )

    # Groq
    elif model_name == "GROQ":
        if Groq is None:
            raise ImportError("groq package not installed")
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_model_name,
            messages=conversation,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.9
        )
        return _extract_message_content(response.choices[0].message)

    elif model_name == "GROQ_LLAMA":
        if Groq is None:
            raise ImportError("groq package not installed")
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_llama_model_name,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return _extract_message_content(response.choices[0].message)

    elif model_name == "GROQ_MIXTRAL":
        if Groq is None:
            raise ImportError("groq package not installed")
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_qwen_model_name,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return _extract_message_content(response.choices[0].message)

    else:
        return f"Invalid model name: {model_name}. Please choose from: LITELLM_FAST, LITELLM_SMART, LITELLM_STRATEGIC, OLLAMA_FAST, OLLAMA_SMART, OLLAMA_STRATEGIC, GEMINI, GEMINI_THINKING, GROQ, GROQ_LLAMA, GROQ_MIXTRAL"
