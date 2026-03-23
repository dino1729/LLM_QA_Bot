"""
Unified LLM Client for LiteLLM and Ollama
Provides a consistent interface for interacting with different LLM providers
"""
import logging
from openai import OpenAI
from typing import List, Optional
from config import config

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Unified client for LiteLLM and Ollama that uses OpenAI-compatible API
    """

    def __init__(self, provider="litellm", model_tier="smart", model_name=None):
        """
        Initialize the unified LLM client

        Args:
            provider: Either "litellm" or "ollama"
            model_tier: Either "fast", "smart", or "strategic" (ignored if model_name is provided)
            model_name: Optional specific model name to use (overrides model_tier)
        """
        self.provider = provider
        self.model_tier = model_tier

        if provider == "litellm":
            self.base_url = config.litellm_base_url
            self.api_key = config.litellm_api_key
            self.embedding_model = config.litellm_embedding

            # Use explicit model name if provided, otherwise use tier
            if model_name:
                self.model = model_name
            elif model_tier == "fast":
                self.model = config.litellm_fast_llm
            elif model_tier == "smart":
                self.model = config.litellm_smart_llm
            elif model_tier == "strategic":
                self.model = config.litellm_strategic_llm
            else:
                # Use configured default tier instead of hardcoded "smart"
                default_tier = config.default_llm_tier
                if default_tier == "fast":
                    self.model = config.litellm_fast_llm
                elif default_tier == "strategic":
                    self.model = config.litellm_strategic_llm
                else:
                    self.model = config.litellm_smart_llm

        elif provider == "ollama":
            self.base_url = config.ollama_base_url
            self.api_key = "ollama"  # Ollama doesn't need a real API key
            self.embedding_model = config.ollama_embedding

            # Use explicit model name if provided, otherwise use tier
            if model_name:
                self.model = model_name
            elif model_tier == "fast":
                self.model = config.ollama_fast_llm
            elif model_tier == "smart":
                self.model = config.ollama_smart_llm
            elif model_tier == "strategic":
                self.model = config.ollama_strategic_llm
            else:
                # Use configured default tier instead of hardcoded "smart"
                default_tier = config.default_llm_tier
                if default_tier == "fast":
                    self.model = config.ollama_fast_llm
                elif default_tier == "strategic":
                    self.model = config.ollama_strategic_llm
                else:
                    self.model = config.ollama_smart_llm
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Create OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        # Log the model being used for debugging
        logger.info(f"LLM Client initialized: provider={provider}, tier={model_tier}, model={self.model}")

    @staticmethod
    def _strip_prefix(name: str) -> str:
        """Strip provider prefix (openai:, ollama:) from model names."""
        if ":" in name:
            return name.split(":", 1)[1]
        return name

    def chat_completion(self, messages, temperature=0.7, max_tokens=1024, **kwargs):
        """
        Generate a chat completion

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            The generated text response
        """
        model_name = self._strip_prefix(self.model)

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Handle reasoning models that return content in reasoning_content field
        message = response.choices[0].message
        content = message.content

        # If content is empty, try reasoning_content field
        if not content and hasattr(message, 'reasoning_content') and message.reasoning_content:
            content = message.reasoning_content

        # Strip <think>...</think> reasoning blocks (nemotron, deepseek, etc.)
        if content and "</think>" in content:
            content = content.split("</think>", 1)[-1].lstrip()
        elif content and "<think>" in content and "</think>" not in content:
            logger.warning("Truncated <think> block detected (no closing tag), response may be incomplete")

        return content if content else ""

    def stream_chat_completion(self, messages, temperature=0.7, max_tokens=1024, **kwargs):
        """
        Stream a chat completion, yielding text chunks.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they arrive
        """
        model_name = self._strip_prefix(self.model)

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_embedding(self, text, input_type="passage"):
        """
        Generate embeddings for text

        Args:
            text: Text to generate embeddings for
            input_type: Type of input - "query" or "passage" (for asymmetric models)

        Returns:
            Embedding vector
        """
        model_name = self._strip_prefix(self.embedding_model)
        is_asymmetric = "nv-embedqa" in model_name or "nv-embed" in model_name

        extra_body = {"input_type": input_type} if is_asymmetric else {}
        response = self.client.embeddings.create(
            model=model_name,
            input=text,
            extra_body=extra_body
        )
        return response.data[0].embedding


def get_client(provider="litellm", model_tier="smart", model_name=None):
    """
    Factory function to get a unified LLM client

    Args:
        provider: Either "litellm" or "ollama"
        model_tier: Either "fast", "smart", or "strategic" (ignored if model_name is provided)
        model_name: Optional specific model name to use (overrides model_tier)

    Returns:
        UnifiedLLMClient instance
    """
    return UnifiedLLMClient(provider=provider, model_tier=model_tier, model_name=model_name)


def list_available_models(provider="litellm"):
    """
    List all available models from LiteLLM or Ollama

    Args:
        provider: Either "litellm" or "ollama"

    Returns:
        List of available model names, or empty list if failed
    """
    try:
        if provider == "litellm":
            base_url = config.litellm_base_url
            api_key = config.litellm_api_key
        elif provider == "ollama":
            base_url = config.ollama_base_url
            api_key = "ollama"
        else:
            return []

        client = OpenAI(base_url=base_url, api_key=api_key)

        # Call /v1/models endpoint
        models = client.models.list()

        # Extract model names
        model_names = [model.id for model in models.data]

        return sorted(model_names)

    except Exception as e:
        print(f"Error fetching models from {provider}: {str(e)}")
        return []
