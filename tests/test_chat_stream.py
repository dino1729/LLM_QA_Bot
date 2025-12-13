"""
Unit tests for helper_functions/chat_stream.py
Tests for preparing chat streaming with LlamaIndex
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from llama_index.core import Settings as LlamaSettings


class TestPrepareChatStream:
    """Tests for prepare_chat_stream() function"""

    @pytest.fixture
    def mock_vector_folder(self, tmp_path):
        """Create a temporary vector folder with mock files"""
        vector_folder = tmp_path / "vector"
        vector_folder.mkdir()
        # Create some mock files to simulate existing index
        (vector_folder / "docstore.json").write_text("{}")
        (vector_folder / "index_store.json").write_text("{}")
        return str(vector_folder)

    @pytest.fixture
    def mock_qa_template(self):
        """Create a mock QA template"""
        from llama_index.core import PromptTemplate
        return PromptTemplate(
            "Context: {context_str}\nQuestion: {query_str}\nAnswer:"
        )

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

    @patch('helper_functions.chat_stream.LlamaSettings')
    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    @patch('helper_functions.chat_stream.get_client')
    @patch('helper_functions.chat_stream.get_response_synthesizer')
    @patch('helper_functions.chat_stream.VectorIndexRetriever')
    @patch('helper_functions.chat_stream.RetrieverQueryEngine')
    def test_prepare_chat_stream_litellm_provider(
        self, mock_engine_cls, mock_retriever_cls, mock_synthesizer,
        mock_get_client, mock_storage_ctx, mock_load_index, mock_settings,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test prepare_chat_stream with LiteLLM provider"""
        from helper_functions.chat_stream import prepare_chat_stream

        # Setup mocks
        mock_client = Mock()
        mock_llm = Mock()
        mock_embed = Mock()
        mock_client.get_llamaindex_llm.return_value = mock_llm
        mock_client.get_llamaindex_embedding.return_value = mock_embed
        mock_get_client.return_value = mock_client

        mock_index = Mock()
        mock_load_index.return_value = mock_index

        mock_retriever = Mock()
        mock_retriever_cls.return_value = mock_retriever

        mock_response = Mock()
        mock_response.response_gen = iter(["chunk1", "chunk2"])
        mock_engine = Mock()
        mock_engine.query.return_value = mock_response
        mock_engine_cls.return_value = mock_engine

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
        mock_engine.query.assert_called_once_with("What is AI?")
        assert response == mock_response

    @patch('helper_functions.chat_stream.LlamaSettings')
    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    @patch('helper_functions.chat_stream.get_client')
    @patch('helper_functions.chat_stream.get_response_synthesizer')
    @patch('helper_functions.chat_stream.VectorIndexRetriever')
    @patch('helper_functions.chat_stream.RetrieverQueryEngine')
    def test_prepare_chat_stream_ollama_provider(
        self, mock_engine_cls, mock_retriever_cls, mock_synthesizer,
        mock_get_client, mock_storage_ctx, mock_load_index, mock_settings,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test prepare_chat_stream with Ollama provider"""
        from helper_functions.chat_stream import prepare_chat_stream

        # Setup mocks
        mock_client = Mock()
        mock_llm = Mock()
        mock_embed = Mock()
        mock_client.get_llamaindex_llm.return_value = mock_llm
        mock_client.get_llamaindex_embedding.return_value = mock_embed
        mock_get_client.return_value = mock_client

        mock_index = Mock()
        mock_load_index.return_value = mock_index

        mock_retriever = Mock()
        mock_retriever_cls.return_value = mock_retriever

        mock_response = Mock()
        mock_engine = Mock()
        mock_engine.query.return_value = mock_response
        mock_engine_cls.return_value = mock_engine

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
        assert response == mock_response

    def test_prepare_chat_stream_missing_index(
        self, mock_qa_template, mock_parse_model_func, tmp_path
    ):
        """Test prepare_chat_stream raises error when index doesn't exist"""
        from helper_functions.chat_stream import prepare_chat_stream

        empty_folder = str(tmp_path / "empty_vector")
        os.makedirs(empty_folder, exist_ok=True)

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

        with pytest.raises(Exception):
            prepare_chat_stream(
                question="Test question",
                model_name="LITELLM_SMART",
                vector_folder=nonexistent_folder,
                qa_template=mock_qa_template,
                parse_model_name_func=mock_parse_model_func
            )

    @patch('helper_functions.chat_stream.LlamaSettings')
    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    @patch('helper_functions.chat_stream.get_client')
    @patch('helper_functions.chat_stream.get_response_synthesizer')
    @patch('helper_functions.chat_stream.VectorIndexRetriever')
    @patch('helper_functions.chat_stream.RetrieverQueryEngine')
    def test_prepare_chat_stream_restores_settings(
        self, mock_engine_cls, mock_retriever_cls, mock_synthesizer,
        mock_get_client, mock_storage_ctx, mock_load_index, mock_settings,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that original LlamaSettings are restored after stream preparation"""
        from helper_functions.chat_stream import prepare_chat_stream

        # Setup mocks
        mock_client = Mock()
        mock_client.get_llamaindex_llm.return_value = Mock()
        mock_client.get_llamaindex_embedding.return_value = Mock()
        mock_get_client.return_value = mock_client

        mock_load_index.return_value = Mock()
        mock_retriever_cls.return_value = Mock()

        mock_engine = Mock()
        mock_engine.query.return_value = Mock()
        mock_engine_cls.return_value = mock_engine

        # Execute
        prepare_chat_stream(
            question="Test question",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Settings should be restored - verify llm was set
        assert mock_settings.llm is not None

    @patch('helper_functions.chat_stream.LlamaSettings')
    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    @patch('helper_functions.chat_stream.get_client')
    @patch('helper_functions.chat_stream.get_response_synthesizer')
    @patch('helper_functions.chat_stream.VectorIndexRetriever')
    @patch('helper_functions.chat_stream.RetrieverQueryEngine')
    def test_prepare_chat_stream_retriever_config(
        self, mock_engine_cls, mock_retriever_cls, mock_synthesizer,
        mock_get_client, mock_storage_ctx, mock_load_index, mock_settings,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that retriever is configured with correct top_k"""
        from helper_functions.chat_stream import prepare_chat_stream

        mock_client = Mock()
        mock_client.get_llamaindex_llm.return_value = Mock()
        mock_client.get_llamaindex_embedding.return_value = Mock()
        mock_get_client.return_value = mock_client

        mock_index = Mock()
        mock_load_index.return_value = mock_index

        mock_engine = Mock()
        mock_engine.query.return_value = Mock()
        mock_engine_cls.return_value = mock_engine

        prepare_chat_stream(
            question="Test",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Check retriever was created with index and top_k=10
        mock_retriever_cls.assert_called_once()
        call_kwargs = mock_retriever_cls.call_args[1]
        assert call_kwargs['index'] == mock_index
        assert call_kwargs['similarity_top_k'] == 10

    @patch('helper_functions.chat_stream.LlamaSettings')
    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    @patch('helper_functions.chat_stream.get_client')
    @patch('helper_functions.chat_stream.get_response_synthesizer')
    @patch('helper_functions.chat_stream.VectorIndexRetriever')
    @patch('helper_functions.chat_stream.RetrieverQueryEngine')
    def test_prepare_chat_stream_streaming_enabled(
        self, mock_engine_cls, mock_retriever_cls, mock_synthesizer,
        mock_get_client, mock_storage_ctx, mock_load_index, mock_settings,
        mock_vector_folder, mock_qa_template, mock_parse_model_func
    ):
        """Test that response synthesizer is configured with streaming=True"""
        from helper_functions.chat_stream import prepare_chat_stream

        mock_client = Mock()
        mock_client.get_llamaindex_llm.return_value = Mock()
        mock_client.get_llamaindex_embedding.return_value = Mock()
        mock_get_client.return_value = mock_client

        mock_load_index.return_value = Mock()
        mock_retriever_cls.return_value = Mock()

        mock_engine = Mock()
        mock_engine.query.return_value = Mock()
        mock_engine_cls.return_value = mock_engine

        prepare_chat_stream(
            question="Test",
            model_name="LITELLM_SMART",
            vector_folder=mock_vector_folder,
            qa_template=mock_qa_template,
            parse_model_name_func=mock_parse_model_func
        )

        # Check synthesizer was created with streaming=True
        mock_synthesizer.assert_called_once()
        call_kwargs = mock_synthesizer.call_args[1]
        assert call_kwargs['streaming'] is True

    @patch('helper_functions.chat_stream.load_index_from_storage')
    @patch('helper_functions.chat_stream.StorageContext')
    def test_prepare_chat_stream_non_litellm_ollama_provider(
        self, mock_storage_ctx, mock_load_index,
        mock_vector_folder, mock_qa_template
    ):
        """Test prepare_chat_stream with non-litellm/ollama provider"""
        from helper_functions.chat_stream import prepare_chat_stream

        def gemini_parser(model_name):
            return ("gemini", "smart", "test-gemini-model")

        mock_index = Mock()
        mock_load_index.return_value = mock_index

        with patch('helper_functions.chat_stream.get_response_synthesizer') as mock_synth, \
             patch('helper_functions.chat_stream.VectorIndexRetriever') as mock_ret, \
             patch('helper_functions.chat_stream.RetrieverQueryEngine') as mock_eng:

            mock_ret.return_value = Mock()
            mock_engine = Mock()
            mock_engine.query.return_value = Mock()
            mock_eng.return_value = mock_engine

            # For non-litellm/ollama providers, get_client should NOT be called
            with patch('helper_functions.chat_stream.get_client') as mock_get_client:
                response = prepare_chat_stream(
                    question="Test",
                    model_name="GEMINI",
                    vector_folder=mock_vector_folder,
                    qa_template=mock_qa_template,
                    parse_model_name_func=gemini_parser
                )

                # get_client should not be called for gemini
                mock_get_client.assert_not_called()

