"""
Unit tests for helper_functions/chat_stream.py
Tests for preparing chat streaming with SimpleVectorStore and UnifiedLLMClient
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call


@pytest.fixture(autouse=True)
def reset_settings():
    """Override conftest reset_settings to remove LlamaIndex dependency for this module."""
    yield


class TestPrepareChatStream:
    """Tests for prepare_chat_stream() function"""

    @pytest.fixture
    def mock_vector_folder(self, tmp_path):
        """Create a temporary vector folder with mock files"""
        vector_folder = tmp_path / "vector"
        vector_folder.mkdir()
        # Create some mock files to simulate existing index
        (vector_folder / "index.json").write_text("{}")
        (vector_folder / "embeddings.json").write_text("{}")
        return str(vector_folder)

    @pytest.fixture
    def mock_qa_template(self):
        """Create a plain string QA template"""
        return "Context: {context_str}\nQuestion: {query_str}\nAnswer:"

    @pytest.fixture
    def mock_parse_model_func(self):
        """Create a mock model name parser - uses test placeholder model names"""
        def parser(model_name):
            if model_name.startswith("LITELLM"):
                return ("litellm", "smart", "test-litellm-model")
            elif model_name.startswith("OLLAMA"):
                return ("ollama", "smart", "test-ollama-model")
            else:
                return ("litellm", "smart", model_name)
        return parser

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_litellm_provider(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test prepare_chat_stream with LiteLLM provider"""
        from helper_functions.chat_stream import prepare_chat_stream

        # Setup mocks
        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(["chunk1", "chunk2"])
        mock_get_client.return_value = mock_client

        mock_result1 = Mock()
        mock_result1.text = "Document chunk 1"
        mock_result2 = Mock()
        mock_result2.text = "Document chunk 2"

        mock_store = Mock()
        mock_store.search.return_value = [mock_result1, mock_result2]
        mock_vector_store_cls.return_value = mock_store

        # Execute
        response = prepare_chat_stream(
            question="What is AI?",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Assertions
        mock_get_client.assert_called_once_with(
            provider="litellm", model_tier="smart", model_name="test-litellm-model"
        )
        mock_vector_store_cls.assert_called_once_with(
            persist_dir=mock_vector_folder, embed_fn=mock_client.get_embedding
        )
        mock_store.search.assert_called_once_with("What is AI?", top_k=10)
        assert hasattr(response, 'response_gen')

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_ollama_provider(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test prepare_chat_stream with Ollama provider"""
        from helper_functions.chat_stream import prepare_chat_stream

        # Setup mocks
        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(["chunk1", "chunk2"])
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.text = "Some relevant content"
        mock_store = Mock()
        mock_store.search.return_value = [mock_result]
        mock_vector_store_cls.return_value = mock_store

        # Execute
        response = prepare_chat_stream(
            question="Explain machine learning",
            model_name="OLLAMA_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Assertions
        mock_get_client.assert_called_once_with(
            provider="ollama", model_tier="smart", model_name="test-ollama-model"
        )
        assert hasattr(response, 'response_gen')

    def test_prepare_chat_stream_missing_index(
        self, mock_qa_template, mock_parse_model_func, tmp_path
    ):
        """Test prepare_chat_stream raises error when index doesn't exist (empty folder)"""
        from helper_functions.chat_stream import prepare_chat_stream

        empty_folder = str(tmp_path / "empty_vector")
        os.makedirs(empty_folder, exist_ok=True)

        with patch('helper_functions.chat_stream.get_client'):
            with pytest.raises(Exception) as excinfo:
                prepare_chat_stream(
                    question="Test question",
                    model_name="LITELLM_SMART",
                    vector_folder=empty_folder,
                    qa_template=mock_qa_template,
                    parse_model_name_func=mock_parse_model_func
                )

        assert "Index not found" in str(excinfo.value) or "not found" in str(excinfo.value).lower()

    def test_prepare_chat_stream_nonexistent_folder(
        self, mock_qa_template, mock_parse_model_func, tmp_path
    ):
        """Test prepare_chat_stream raises error when folder doesn't exist"""
        from helper_functions.chat_stream import prepare_chat_stream

        nonexistent_folder = str(tmp_path / "nonexistent")

        with patch('helper_functions.chat_stream.get_client'):
            with pytest.raises(Exception):
                prepare_chat_stream(
                    question="Test question",
                    model_name="LITELLM_SMART",
                    vector_folder=nonexistent_folder,
                    qa_template=mock_qa_template,
                    parse_model_name_func=mock_parse_model_func
                )

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_search_top_k(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that vector store search is called with top_k=10"""
        from helper_functions.chat_stream import prepare_chat_stream

        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(["chunk"])
        mock_get_client.return_value = mock_client

        mock_store = Mock()
        mock_store.search.return_value = []
        mock_vector_store_cls.return_value = mock_store

        prepare_chat_stream(
            question="Test",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Check search was called with top_k=10
        mock_store.search.assert_called_once_with("Test", top_k=10)

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_streaming_behavior(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that streaming is enabled and response_gen yields chunks"""
        from helper_functions.chat_stream import prepare_chat_stream

        expected_chunks = ["Hello", " world", "!"]

        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(expected_chunks)
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.text = "Some context"
        mock_store = Mock()
        mock_store.search.return_value = [mock_result]
        mock_vector_store_cls.return_value = mock_store

        response = prepare_chat_stream(
            question="Test",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # stream_chat_completion should have been called
        mock_client.stream_chat_completion.assert_called_once()

        # Verify response_gen yields chunks
        chunks = list(response.response_gen)
        assert chunks == expected_chunks

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_prompt_formatting(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that the prompt is correctly formatted with context and question"""
        from helper_functions.chat_stream import prepare_chat_stream

        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(["answer"])
        mock_get_client.return_value = mock_client

        mock_result1 = Mock()
        mock_result1.text = "First chunk"
        mock_result2 = Mock()
        mock_result2.text = "Second chunk"
        mock_store = Mock()
        mock_store.search.return_value = [mock_result1, mock_result2]
        mock_vector_store_cls.return_value = mock_store

        prepare_chat_stream(
            question="What is AI?",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Verify the prompt was formatted correctly
        call_kwargs = mock_client.stream_chat_completion.call_args
        messages = call_kwargs[1].get('messages') or call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]['messages']
        prompt_content = messages[0]["content"]

        expected_context = "First chunk\n\nSecond chunk"
        assert expected_context in prompt_content
        assert "What is AI?" in prompt_content

    @patch('helper_functions.chat_stream.SimpleVectorStore')
    @patch('helper_functions.chat_stream.get_client')
    def test_prepare_chat_stream_gemini_provider(
        self, mock_get_client, mock_vector_store_cls,
        mock_vector_folder, mock_qa_template
    ):
        """Test prepare_chat_stream with a non-litellm/ollama provider (gemini)"""
        from helper_functions.chat_stream import prepare_chat_stream

        def gemini_parser(model_name):
            return ("gemini", "smart", "test-gemini-model")

        mock_client = Mock()
        mock_client.get_embedding = Mock()
        mock_client.stream_chat_completion.return_value = iter(["response"])
        mock_get_client.return_value = mock_client

        mock_store = Mock()
        mock_store.search.return_value = []
        mock_vector_store_cls.return_value = mock_store

        response = prepare_chat_stream(
            question="Test",
            model_name="GEMINI",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=gemini_parser
        )

        # get_client should be called for all providers now
        mock_get_client.assert_called_once_with(
            provider="gemini", model_tier="smart", model_name="test-gemini-model"
        )
        assert hasattr(response, 'response_gen')
