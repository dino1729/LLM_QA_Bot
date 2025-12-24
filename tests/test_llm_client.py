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


class TestCustomOpenAIEmbedding:
    """Test CustomOpenAIEmbedding class - uses test placeholder model names"""

    def test_initialization(self):
        """Test CustomOpenAIEmbedding initialization"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        embedding = CustomOpenAIEmbedding(
            model_name="test-embed-model",
            api_key="test-key",
            api_base="http://localhost:4000",
            embed_batch_size=10
        )

        assert embedding._model_name == "test-embed-model"
        assert embedding._api_key == "test-key"
        assert embedding._api_base == "http://localhost:4000"
        assert embedding._is_asymmetric == False

    def test_asymmetric_model_detection(self):
        """Test asymmetric model detection (NVIDIA NIM models)"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        # Test NVIDIA NIM model - asymmetric detection is path-based
        embedding = CustomOpenAIEmbedding(
            model_name="nvidia/test-asymmetric-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        assert embedding._is_asymmetric == True

    @patch('helper_functions.llm_client.OpenAI')
    def test_get_query_embedding(self, mock_openai):
        """Test get query embedding"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="test-embed-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = embedding._get_query_embedding("test query")

        assert len(result) == 1536
        assert result[0] == 0.1
        mock_client.embeddings.create.assert_called_once()

    @patch('helper_functions.llm_client.OpenAI')
    def test_get_text_embedding(self, mock_openai):
        """Test get text embedding"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.2] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="test-embed-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = embedding._get_text_embedding("test text")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()

    @patch('helper_functions.llm_client.OpenAI')
    def test_get_text_embeddings_batch(self, mock_openai):
        """Test batch text embeddings"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
            Mock(embedding=[0.3] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="test-embed-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = embedding._get_text_embeddings(["text1", "text2", "text3"])

        assert len(result) == 3
        assert all(len(emb) == 1536 for emb in result)

    @patch('helper_functions.llm_client.OpenAI')
    def test_asymmetric_input_type_query(self, mock_openai):
        """Test asymmetric model with query input_type"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="nvidia/nv-embedqa-e5-v5",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        embedding._get_query_embedding("test")

        # Verify input_type=query was passed
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs['extra_body']['input_type'] == 'query'

    @patch('helper_functions.llm_client.OpenAI')
    def test_asymmetric_input_type_passage(self, mock_openai):
        """Test asymmetric model with passage input_type"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="nvidia/nv-embed-v2",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        embedding._get_text_embedding("test")

        # Verify input_type=passage was passed
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs['extra_body']['input_type'] == 'passage'

    @pytest.mark.asyncio
    @patch('helper_functions.llm_client.OpenAI')
    async def test_async_get_query_embedding(self, mock_openai):
        """Test async get query embedding"""
        from helper_functions.llm_client import CustomOpenAIEmbedding

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedding = CustomOpenAIEmbedding(
            model_name="test-embed-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = await embedding._aget_query_embedding("test")

        assert len(result) == 1536


class TestCustomOpenAILLM:
    """Test CustomOpenAILLM class"""

    def test_initialization(self):
        """Test CustomOpenAILLM initialization"""
        from helper_functions.llm_client import CustomOpenAILLM

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000",
            temperature=0.7,
            max_tokens=2000
        )

        assert llm._model_name == "test-smart-model"
        assert llm._api_key == "test-key"
        assert llm._api_base == "http://localhost:4000"
        assert llm._temperature == 0.7
        assert llm._max_tokens == 2000

    def test_metadata_property(self):
        """Test metadata property"""
        from helper_functions.llm_client import CustomOpenAILLM

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000",
            max_tokens=2000
        )

        metadata = llm.metadata

        assert metadata.model_name == "test-smart-model"
        assert metadata.num_output == 2000
        assert metadata.context_window == 128000

    @patch('helper_functions.llm_client.OpenAI')
    def test_complete(self, mock_openai):
        """Test complete method"""
        from helper_functions.llm_client import CustomOpenAILLM

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock(content="Test response", reasoning_content=None)
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = llm.complete("Test prompt")

        assert result.text == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('helper_functions.llm_client.OpenAI')
    def test_complete_reasoning_model(self, mock_openai):
        """Test complete with reasoning model (o1, o3)"""
        from helper_functions.llm_client import CustomOpenAILLM

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock(content=None, reasoning_content="Reasoning response")
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = CustomOpenAILLM(
            model_name="o1-preview",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = llm.complete("Test prompt")

        # Should extract reasoning_content
        assert result.text == "Reasoning response"

    @patch('helper_functions.llm_client.OpenAI')
    def test_stream_complete(self, mock_openai):
        """Test stream complete method"""
        from helper_functions.llm_client import CustomOpenAILLM

        mock_client = Mock()
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))])
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        result = llm.stream_complete("Test prompt")
        chunks = list(result)

        assert len(chunks) == 3
        assert chunks[-1].text == "Hello world!"

    @patch('helper_functions.llm_client.OpenAI')
    def test_chat(self, mock_openai):
        """Test chat method"""
        from helper_functions.llm_client import CustomOpenAILLM
        from llama_index.core.base.llms.types import ChatMessage

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock(content="Chat response", reasoning_content=None)
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        messages = [
            ChatMessage(role="user", content="Hello")
        ]
        result = llm.chat(messages)

        assert result.message.content == "Chat response"
        assert result.message.role == "assistant"

    @patch('helper_functions.llm_client.OpenAI')
    def test_stream_chat(self, mock_openai):
        """Test stream chat method"""
        from helper_functions.llm_client import CustomOpenAILLM
        from llama_index.core.base.llms.types import ChatMessage

        mock_client = Mock()
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" there"))])
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai.return_value = mock_client

        llm = CustomOpenAILLM(
            model_name="test-smart-model",
            api_key="test-key",
            api_base="http://localhost:4000"
        )

        messages = [ChatMessage(role="user", content="Hi")]
        result = llm.stream_chat(messages)
        chunks = list(result)

        assert len(chunks) == 2
        assert chunks[-1].message.content == "Hello there"


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
    def test_get_llamaindex_llm(self, mock_openai, mock_config):
        """Test get LlamaIndex LLM"""
        from helper_functions.llm_client import UnifiedLLMClient, CustomOpenAILLM

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"
        mock_config.temperature = 0.7
        mock_config.max_tokens = 2000

        mock_openai.return_value = Mock()

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        llm = client.get_llamaindex_llm()

        assert isinstance(llm, CustomOpenAILLM)
        assert llm._model_name == "test-smart-model"

    @patch('helper_functions.llm_client.config')
    @patch('helper_functions.llm_client.OpenAI')
    def test_get_llamaindex_embedding(self, mock_openai, mock_config):
        """Test get LlamaIndex embedding"""
        from helper_functions.llm_client import UnifiedLLMClient, CustomOpenAIEmbedding

        mock_config.litellm_base_url = "http://litellm:4000"
        mock_config.litellm_api_key = "test-key"
        mock_config.litellm_embedding = "test-embed-model"
        mock_config.litellm_smart_llm = "test-smart-model"

        mock_openai.return_value = Mock()

        client = UnifiedLLMClient(provider="litellm", model_tier="smart")

        embedding = client.get_llamaindex_embedding()

        assert isinstance(embedding, CustomOpenAIEmbedding)
        assert embedding._model_name == "test-embed-model"

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
