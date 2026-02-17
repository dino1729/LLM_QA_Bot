"""
Unit tests for helper_functions/llm_client.py
Tests unified LLM client for LiteLLM and Ollama

Note: These tests intentionally use specific provider names ("litellm", "ollama")
and tier names ("fast", "smart", "strategic") to test the provider/tier routing
logic. This is correct behavior for unit tests - we're testing that the
UnifiedLLMClient class correctly handles each provider and tier.

For integration tests that use actual config values, use fixtures from conftest.py.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestUnifiedLLMClient:
    """Test UnifiedLLMClient class"""

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_initialization_litellm(self, mock_openai, mock_config):
        """Test initialization with LiteLLM provider"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_fast_llm = "test-fast-model"
        mock_config.litellm_smart_llm = "test-smart-model"
        mock_config.litellm_strategic_llm = "test-strategic-model"

        mock_openai.return_value = Mock()

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        assert client.provider == "litellm"
        assert client.model_tier == "smart"
        assert client.model == "test-smart-model"
        assert client.base_url == "http://litellm:4000"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_initialization_ollama(self, mock_openai, mock_config):
        """Test initialization with Ollama provider"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_embedding = "test-ollama-embed"
        mock_config.ollama_fast_llm = "test-ollama-fast"
        mock_config.ollama_smart_llm = "test-ollama-smart"
        mock_config.ollama_strategic_llm = "test-ollama-strategic"

        mock_openai.return_value = Mock()

        client = UnifiedLLMClient(provider="ollama", model_tier="fast")

        assert client.provider == "ollama"
        assert client.model_tier == "fast"
        assert client.model == "test-ollama-fast"
        assert client.api_key == "ollama"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_invalid_provider(self, mock_openai, mock_config):
        """Test initialization with invalid provider"""
        from helper_functions.llm_client import UnifiedLLMClient

        with pytest.raises(ValueError, match="Unsupported provider"):
            UnifiedLLMClient(provider="invalid", model_tier="smart")

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_model_tier_selection(self, mock_openai, mock_config):
        """Test model tier selection (fast, smart, strategic)"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_fast_llm = "test-fast-model"
        mock_config.litellm_smart_llm = "test-smart-model"
        mock_config.litellm_strategic_llm = "test-strategic-model"

        mock_openai.return_value = Mock()

        # Test fast
        client_fast = UnifiedLLMClient(provider="litellm", model_tier="fast")
        assert client_fast.model == "test-fast-model"

        # Test smart
        client_smart = UnifiedLLMClient(provider="litellm", model_tier="smart")
        assert client_smart.model == "test-smart-model"

        # Test strategic
        client_strategic = UnifiedLLMClient(provider="litellm", model_tier="strategic")
        assert client_strategic.model == "test-strategic-model"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_chat_completion(self, mock_openai, mock_config):
        """Test chat completion"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_completion(messages)

        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_chat_completion_reasoning_model(self, mock_openai, mock_config):
        """Test chat completion with reasoning model (content in reasoning_content)"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock(content=None, reasoning_content="Reasoning response")
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_completion(messages)

        assert result == "Reasoning response"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_chat_completion_empty_response(self, mock_openai, mock_config):
        """Test chat completion returns empty string when no content"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock(content=None, reasoning_content=None)
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "Hello"}]
        result = client.chat_completion(messages)

        assert result == ""

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_stream_chat_completion(self, mock_openai, mock_config):
        """Test streaming chat completion yields text chunks"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(client.stream_chat_completion(messages))

        assert chunks == ["Hello", " world", "!"]
        # Verify stream=True was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['stream'] is True

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_stream_chat_completion_skips_empty_chunks(self, mock_openai, mock_config):
        """Test streaming chat completion skips chunks with no content"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),
            Mock(choices=[Mock(delta=Mock(content=""))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(client.stream_chat_completion(messages))

        assert chunks == ["Hello", " world"]

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_embedding(self, mock_openai, mock_config):
        """Test get embedding"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        result = client.get_embedding("test text")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_embedding_asymmetric(self, mock_openai, mock_config):
        """Test get embedding with asymmetric model"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "nvidia/nv-embedqa-e5-v5"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        client.get_embedding("test", input_type="query")

        # Verify input_type was passed
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs['extra_body']['input_type'] == 'query'

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_embedding_asymmetric_nv_embed(self, mock_openai, mock_config):
        """Test asymmetric model detection for nv-embed models"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "nvidia/nv-embed-v2"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        client.get_embedding("test passage text", input_type="passage")

        # Verify input_type=passage was passed for nv-embed model
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs['extra_body']['input_type'] == 'passage'

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_embedding_non_asymmetric_no_extra_body(self, mock_openai, mock_config):
        """Test that non-asymmetric models do not send extra_body with input_type"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "text-embedding-ada-002"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        client.get_embedding("test text")

        # Verify extra_body is empty (no input_type) for non-asymmetric model
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs['extra_body'] == {}

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_provider_prefix_stripping(self, mock_openai, mock_config):
        """Test that provider prefixes are stripped from model names"""
        from helper_functions.llm_client import UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "openai:test-embed-model"
        mock_config.litellm_smart_llm = "openai:test-smart-model"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        messages = [{"role": "user", "content": "test"}]
        client.chat_completion(messages)

        # Verify prefix was stripped
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "test-smart-model"


class TestGetClient:
    """Test get_client factory function"""

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_client_litellm(self, mock_openai, mock_config):
        """Test get_client with LiteLLM"""
        from helper_functions.llm_client import get_client, UnifiedLLMClient

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_openai.return_value = Mock()

        client = get_client(provider="litellm", model_tier="smart")

        assert isinstance(client, UnifiedLLMClient)
        assert client.provider == "litellm"
        assert client.model_tier == "smart"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_client_ollama(self, mock_openai, mock_config):
        """Test get_client with Ollama"""
        from helper_functions.llm_client import get_client, UnifiedLLMClient

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_embedding = "test-ollama-embed"
        mock_config.ollama_smart_llm = "test-ollama-model"

        mock_openai.return_value = Mock()

        client = get_client(provider="ollama", model_tier="smart")

        assert isinstance(client, UnifiedLLMClient)
        assert client.provider == "ollama"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_client_different_tiers(self, mock_openai, mock_config):
        """Test get_client with different tiers"""
        from helper_functions.llm_client import get_client

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_fast_llm = "test-fast-model"
        mock_config.litellm_smart_llm = "test-smart-model"
        mock_config.litellm_strategic_llm = "test-strategic-model"

        mock_openai.return_value = Mock()

        # Test different tiers
        for tier in ["fast", "smart", "strategic"]:
            client = get_client(provider="litellm", model_tier=tier)
            assert client.model_tier == tier


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
