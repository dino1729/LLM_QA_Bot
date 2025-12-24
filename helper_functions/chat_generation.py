"""
Chat Generation Module
Supports multiple LLM providers: LiteLLM, Ollama, Gemini, and Groq
"""
from google import genai
from google.genai import types
from groq import Groq
from config import config
from helper_functions.llm_client import get_client

# API Keys
google_api_key = config.google_api_key
gemini_model_name = config.gemini_model_name
gemini_thinkingmodel_name = config.gemini_thinkingmodel_name
groq_api_key = config.groq_api_key
groq_model_name = config.groq_model_name
groq_llama_model_name = config.groq_llama_model_name
groq_mixtral_model_name = config.groq_mixtral_model_name


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
        from openai import OpenAI
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
        return response.choices[0].message.content
    
    elif model_name.startswith("OLLAMA:"):
        # Extract the actual model name after the prefix
        actual_model = model_name.split("OLLAMA:", 1)[1]
        # Use Ollama with the specific model
        from openai import OpenAI
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
        return response.choices[0].message.content

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

    # Gemini models
    elif model_name == "GEMINI":
        client = genai.Client(api_key=google_api_key)
        response = client.models.generate_content(
            model=gemini_model_name,
            contents=str(conversation).replace("'", '"'),
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.9,
                top_k=1,
            )
        )
        return response.text

    elif model_name == "GEMINI_THINKING":
        client = genai.Client(api_key=google_api_key)
        response = client.models.generate_content(
            model=gemini_thinkingmodel_name,
            contents=str(conversation).replace("'", '"'),
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.9,
                top_k=1,
            )
        )
        return response.text

    # Groq
    elif model_name == "GROQ":
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_model_name,
            messages=conversation,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content

    elif model_name == "GROQ_LLAMA":
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
        return response.choices[0].message.content

    elif model_name == "GROQ_MIXTRAL":
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model=groq_mixtral_model_name,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.6,
            presence_penalty=0.1
        )
        return response.choices[0].message.content

    else:
        return f"Invalid model name: {model_name}. Please choose from: LITELLM_FAST, LITELLM_SMART, LITELLM_STRATEGIC, OLLAMA_FAST, OLLAMA_SMART, OLLAMA_STRATEGIC, GEMINI, GEMINI_THINKING, GROQ, GROQ_LLAMA, GROQ_MIXTRAL"