"""
Unified LLM Client for LiteLLM and Ollama
Provides a consistent interface for interacting with different LLM providers
"""
from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ChatResponseGen
from typing import List, Optional, Any, Sequence
from config import config


class CustomOpenAIEmbedding(BaseEmbedding):
    """
    Custom OpenAI-compatible embedding class that bypasses LlamaIndex's model validation.
    Works with LiteLLM and Ollama by directly calling the OpenAI API.
    Supports asymmetric embedding models (like NVIDIA NIM) that require input_type parameter.
    """
    
    def __init__(self, model_name: str, api_key: str, api_base: str, embed_batch_size: int = 10):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size
        )
        # Store client creation parameters instead of client itself to avoid Pydantic validation
        self._api_key = api_key
        self._api_base = api_base
        self._model_name = model_name
        # Check if this is an asymmetric model that needs input_type parameter
        self._is_asymmetric = "nv-embedqa" in model_name or "nv-embed" in model_name
    
    def _get_client(self):
        """Get or create OpenAI client."""
        return OpenAI(api_key=self._api_key, base_url=self._api_base)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        client = self._get_client()
        extra_body = {"input_type": "query"} if self._is_asymmetric else {}
        response = client.embeddings.create(
            input=[query],
            model=self._model_name,
            extra_body=extra_body
        )
        return response.data[0].embedding
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding (for document/passage)."""
        client = self._get_client()
        extra_body = {"input_type": "passage"} if self._is_asymmetric else {}
        response = client.embeddings.create(
            input=[text],
            model=self._model_name,
            extra_body=extra_body
        )
        return response.data[0].embedding
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings (for documents/passages)."""
        client = self._get_client()
        extra_body = {"input_type": "passage"} if self._is_asymmetric else {}
        response = client.embeddings.create(
            input=texts,
            model=self._model_name,
            extra_body=extra_body
        )
        return [data.embedding for data in response.data]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding."""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get text embedding."""
        return self._get_text_embedding(text)


class CustomOpenAILLM(CustomLLM):
    """
    Custom LLM class that bypasses LlamaIndex's OpenAI model validation.
    Works with any OpenAI-compatible API (LiteLLM, Ollama, etc.)
    """
    
    def __init__(self, model_name: str, api_key: str, api_base: str, 
                 temperature: float = 0.7, max_tokens: int = 1024):
        super().__init__()
        self._model_name = model_name
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._api_base)
        return self._client
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=128000,  # Large context window
            num_output=self._max_tokens,
            model_name=self._model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        
        # For reasoning models, the actual response may be in reasoning_content field
        message = response.choices[0].message
        content = message.content
        
        # Check if this is a reasoning model response
        if not content and hasattr(message, 'reasoning_content') and message.reasoning_content:
            content = message.reasoning_content
        
        return CompletionResponse(text=content if content else "")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream complete a prompt."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            stream=True,
        )
        
        def gen():
            text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    text += delta
                    yield CompletionResponse(text=text, delta=delta)
        
        return gen()
    
    @llm_completion_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the LLM."""
        client = self._get_client()
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        
        # For reasoning models, the actual response may be in reasoning_content field
        message = response.choices[0].message
        content = message.content
        
        # Check if this is a reasoning model response
        if not content and hasattr(message, 'reasoning_content') and message.reasoning_content:
            content = message.reasoning_content
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=content if content else ""
            )
        )
    
    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream chat with the LLM."""
        client = self._get_client()
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            stream=True,
        )
        
        def gen():
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    content += delta
                    yield ChatResponse(
                        message=ChatMessage(role="assistant", content=content),
                        delta=delta
                    )
        
        return gen()


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
                self.model = config.litellm_smart_llm  # default to smart

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
                self.model = config.ollama_smart_llm  # default to smart
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Create OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Log the model being used for debugging
        print(f"  🤖 LLM Client initialized: provider={provider}, tier={model_tier}, model={self.model}")

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
        # Strip the provider prefix (openai: or ollama:) from the model name
        # LiteLLM/Ollama proxies don't need the prefix in the API call
        model_name = self.model
        if ":" in model_name:
            model_name = model_name.split(":", 1)[1]
        
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
        
        return content if content else ""

    def get_embedding(self, text, input_type="passage"):
        """
        Generate embeddings for text

        Args:
            text: Text to generate embeddings for
            input_type: Type of input - "query" or "passage" (for asymmetric models)

        Returns:
            Embedding vector
        """
        # Strip the provider prefix (openai: or ollama:) from the embedding model name
        model_name = self.embedding_model.split(":", 1)[-1] if ":" in self.embedding_model else self.embedding_model
        is_asymmetric = "nv-embedqa" in model_name or "nv-embed" in model_name
        
        extra_body = {"input_type": input_type} if is_asymmetric else {}
        response = self.client.embeddings.create(
            model=model_name,
            input=text,
            extra_body=extra_body
        )
        return response.data[0].embedding

    def get_llamaindex_llm(self):
        """
        Get a LlamaIndex-compatible LLM instance that works with any model on LiteLLM/Ollama

        Returns:
            Custom LLM instance that bypasses model validation
        """
        # Strip the provider prefix (openai: or ollama:) from the model name
        # LiteLLM proxy doesn't need the prefix in the API call
        model_name = self.model
        if ":" in model_name:
            model_name = model_name.split(":", 1)[1]
        
        # Use our CustomOpenAILLM class that completely bypasses validation
        # This allows any model available on LiteLLM/Ollama to work
        return CustomOpenAILLM(
            model_name=model_name,
            api_key=self.api_key,
            api_base=self.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    def get_llamaindex_embedding(self):
        """
        Get a LlamaIndex-compatible embedding model instance

        Returns:
            Custom OpenAI Embedding instance that bypasses model validation
        """
        # Strip the provider prefix (openai: or ollama:) from the embedding model name
        model_name = self.embedding_model
        if ":" in model_name:
            model_name = model_name.split(":", 1)[1]
        
        # Use our custom embedding class that bypasses LlamaIndex's model validation
        # This allows any model available on LiteLLM/Ollama to work
        return CustomOpenAIEmbedding(
            model_name=model_name,
            api_key=self.api_key,
            api_base=self.base_url,
            embed_batch_size=10
        )

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
