"""
Unit tests for helper_functions/memory_palace_local.py
Tests for local memory palace storage and retrieval
"""
import pytest
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from llama_index.core import Document, Settings
from helper_functions.memory_palace_local import (
    save_memory,
    search_memories,
    reset_memory_palace,
    get_memory_palace_index_dir,
    _sanitize_filename,
    load_or_create_index,
    prepare_memory_stream
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM

# Mock embedding model to avoid API calls
class MockEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _get_text_embedding(self, text):
        return [0.1] * 1536
    
    def _get_query_embedding(self, query):
        return [0.1] * 1536
        
    async def _aget_text_embedding(self, text):
        return [0.1] * 1536
        
    async def _aget_query_embedding(self, query):
        return [0.1] * 1536

@pytest.fixture
def mock_settings(monkeypatch):
    mock_embed = MockEmbedding()
    monkeypatch.setattr(Settings, "embed_model", mock_embed)
    monkeypatch.setattr(Settings, "llm", MockLLM())
    return Settings

@pytest.fixture
def temp_memory_root(tmp_path, monkeypatch):
    # Override MEMORY_PALACE_ROOT in the module
    root = tmp_path / "memory_palace"
    monkeypatch.setattr("helper_functions.memory_palace_local.MEMORY_PALACE_ROOT", str(root))
    monkeypatch.setattr("config.config.MEMORY_PALACE_FOLDER", str(root))
    return root


class TestSanitizeFilename:
    """Tests for _sanitize_filename() function"""
    
    def test_sanitize_simple_string(self):
        """Test sanitizing a simple alphanumeric string"""
        assert _sanitize_filename("test123") == "test123"
    
    def test_sanitize_string_with_dashes(self):
        """Test sanitizing string with dashes"""
        assert _sanitize_filename("test-embed-model") == "test-embed-model"
    
    def test_sanitize_string_with_underscores(self):
        """Test sanitizing string with underscores"""
        assert _sanitize_filename("test_embedding") == "test_embedding"
    
    def test_sanitize_string_with_slashes(self):
        """Test sanitizing string with slashes"""
        assert _sanitize_filename("provider/embedding") == "provider_embedding"
    
    def test_sanitize_string_with_colons(self):
        """Test sanitizing string with colons"""
        assert _sanitize_filename("model:v1") == "model_v1"
    
    def test_sanitize_string_with_spaces(self):
        """Test sanitizing string with spaces"""
        assert _sanitize_filename("test embedding") == "test_embedding"
    
    def test_sanitize_string_with_special_chars(self):
        """Test sanitizing string with various special characters"""
        assert _sanitize_filename("model@v1.0!#$%") == "model_v1_0____"
    
    def test_sanitize_empty_string(self):
        """Test sanitizing empty string"""
        assert _sanitize_filename("") == ""


class TestGetMemoryPalaceIndexDir:
    """Tests for get_memory_palace_index_dir() function - uses test placeholder model names"""
    
    def test_get_memory_palace_index_dir(self, temp_memory_root):
        """Test basic directory path generation"""
        provider = "litellm"
        model = "test-embed-model"
        expected = str(temp_memory_root / "litellm__test-embed-model")
        assert get_memory_palace_index_dir(provider, model) == expected
    
    def test_get_memory_palace_index_dir_ollama(self, temp_memory_root):
        """Test directory path for Ollama provider"""
        provider = "ollama"
        model = "test-ollama-embed"
        expected = str(temp_memory_root / "ollama__test-ollama-embed")
        assert get_memory_palace_index_dir(provider, model) == expected
    
    def test_get_memory_palace_index_dir_with_special_chars(self, temp_memory_root):
        """Test directory path with special characters in model name"""
        provider = "litellm"
        model = "provider/test-embed:v1"
        expected = str(temp_memory_root / "litellm__provider_test-embed_v1")
        assert get_memory_palace_index_dir(provider, model) == expected


class TestLoadOrCreateIndex:
    """Tests for load_or_create_index() function"""
    
    def test_load_or_create_index_new_dir(self, tmp_path, mock_settings):
        """Test creating index in new directory"""
        persist_dir = str(tmp_path / "new_index")
        
        index = load_or_create_index(persist_dir)
        
        assert index is not None
        assert os.path.exists(persist_dir)
        assert os.path.exists(os.path.join(persist_dir, "docstore.json"))
    
    def test_load_or_create_index_existing_dir(self, tmp_path, mock_settings):
        """Test loading index from existing directory"""
        persist_dir = str(tmp_path / "existing_index")
        
        # Create index first
        index1 = load_or_create_index(persist_dir)
        
        # Load it again
        index2 = load_or_create_index(persist_dir)
        
        assert index2 is not None
    
    def test_load_or_create_index_empty_dir(self, tmp_path, mock_settings):
        """Test creating index when directory exists but is empty"""
        persist_dir = str(tmp_path / "empty_dir")
        os.makedirs(persist_dir)
        
        index = load_or_create_index(persist_dir)
        
        assert index is not None
        assert os.path.exists(os.path.join(persist_dir, "docstore.json"))


class TestSaveMemory:
    """Tests for save_memory() function - uses test placeholder model names"""
    
    def test_save_memory(self, temp_memory_root, mock_settings):
        """Test basic memory saving"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            status = save_memory(
                title="Test Title",
                content="Test Content",
                source_type="test",
                source_ref="test_ref",
                model_name="LITELLM:test-model"
            )
            
            assert status == "Saved to Memory Palace"
            
            index_dir = temp_memory_root / "litellm__test-embed-model"
            assert index_dir.exists()
            assert (index_dir / "docstore.json").exists()
    
    def test_save_memory_ollama_provider(self, temp_memory_root, mock_settings):
        """Test saving memory with Ollama provider"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.ollama_embedding = "test-ollama-embed"
            
            status = save_memory(
                title="Ollama Memory",
                content="Test content for Ollama",
                source_type="note",
                source_ref="note.txt",
                model_name="OLLAMA:test-ollama-model"
            )
            
            assert status == "Saved to Memory Palace"
            
            index_dir = temp_memory_root / "ollama__test-ollama-embed"
            assert index_dir.exists()
    
    def test_save_memory_multiple_entries(self, temp_memory_root, mock_settings):
        """Test saving multiple memories"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            # Save multiple memories
            for i in range(3):
                status = save_memory(
                    title=f"Memory {i}",
                    content=f"Content for memory {i}",
                    source_type="test",
                    source_ref=f"ref_{i}",
                    model_name="LITELLM:test-model"
                )
                assert status == "Saved to Memory Palace"


class TestSearchMemories:
    """Tests for search_memories() function - uses test placeholder model names"""
    
    def test_search_memories(self, temp_memory_root, mock_settings):
        """Test basic memory search"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            # First save something
            save_memory(
                title="Test Memory",
                content="Important information about AI",
                source_type="note",
                source_ref="note.txt",
                model_name="LITELLM:test-model"
            )
            
            # Then search
            results = search_memories("AI", "LITELLM:test-model")
            
            assert len(results) > 0
            assert "Important information" in results[0]["content"]
            assert results[0]["metadata"]["source_title"] == "Test Memory"
    
    def test_search_memories_empty_palace(self, temp_memory_root, mock_settings):
        """Test searching empty memory palace"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            results = search_memories("AI", "LITELLM:test-model")
            
            assert results == []
    
    def test_search_memories_custom_top_k(self, temp_memory_root, mock_settings):
        """Test search with custom top_k parameter"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            # Save multiple memories
            for i in range(5):
                save_memory(
                    title=f"Memory {i}",
                    content=f"AI related content {i}",
                    source_type="test",
                    source_ref=f"ref_{i}",
                    model_name="LITELLM:test-model"
                )
            
            # Search with top_k=2
            results = search_memories("AI", "LITELLM:test-model", top_k=2)
            
            # Should return at most 2 results
            assert len(results) <= 2
    
    def test_search_memories_ollama_provider(self, temp_memory_root, mock_settings):
        """Test searching with Ollama provider"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.ollama_embedding = "test-ollama-embed"
            
            save_memory(
                title="Ollama Memory",
                content="Content about Ollama models",
                source_type="test",
                source_ref="ref",
                model_name="OLLAMA:test-ollama-model"
            )
            
            results = search_memories("Ollama", "OLLAMA:test-ollama-model")
            
            assert len(results) > 0


class TestPrepareMemoryStream:
    """Tests for prepare_memory_stream() function"""
    
    @patch('helper_functions.memory_palace_local.get_response_synthesizer')
    @patch('helper_functions.memory_palace_local.VectorIndexRetriever')
    @patch('helper_functions.memory_palace_local.RetrieverQueryEngine')
    @patch('helper_functions.memory_palace_local.load_index_from_storage')
    @patch('helper_functions.memory_palace_local.StorageContext')
    def test_prepare_memory_stream_success(
        self, mock_storage_ctx, mock_load_index, mock_engine_cls,
        mock_retriever_cls, mock_synthesizer, temp_memory_root, mock_settings
    ):
        """Test successful memory stream preparation"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            # Create the index directory
            index_dir = temp_memory_root / "litellm__test-embed-model"
            index_dir.mkdir(parents=True)
            (index_dir / "docstore.json").write_text("{}")
            
            mock_index = Mock()
            mock_load_index.return_value = mock_index
            
            mock_response = Mock()
            mock_engine = Mock()
            mock_engine.query.return_value = mock_response
            mock_engine_cls.return_value = mock_engine
            
            response = prepare_memory_stream(
                message="What do I know about AI?",
                history=[],
                model_name="LITELLM:test-model",
                top_k=5
            )
            
            mock_engine.query.assert_called_once_with("What do I know about AI?")
            assert response == mock_response
    
    def test_prepare_memory_stream_empty_palace(self, temp_memory_root, mock_settings):
        """Test prepare_memory_stream raises error when palace is empty"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            with pytest.raises(ValueError) as excinfo:
                prepare_memory_stream(
                    message="Test question",
                    history=[],
                    model_name="LITELLM:test-model"
                )
            
            assert "empty" in str(excinfo.value).lower()


class TestResetMemoryPalace:
    """Tests for reset_memory_palace() function - uses test placeholder model names"""
    
    def test_reset_memory_palace(self, temp_memory_root):
        """Test basic memory palace reset"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            index_dir = temp_memory_root / "litellm__test-embed-model"
            index_dir.mkdir(parents=True)
            
            assert index_dir.exists()
            
            status = reset_memory_palace("LITELLM:test-model")
            
            assert "reset" in status
            assert not index_dir.exists()
    
    def test_reset_memory_palace_nonexistent(self, temp_memory_root):
        """Test resetting non-existent memory palace"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            
            status = reset_memory_palace("LITELLM:test-model")
            
            assert "empty" in status.lower()
    
    def test_reset_memory_palace_ollama(self, temp_memory_root):
        """Test resetting Ollama memory palace"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.ollama_embedding = "test-ollama-embed"
            
            index_dir = temp_memory_root / "ollama__test-ollama-embed"
            index_dir.mkdir(parents=True)
            
            status = reset_memory_palace("OLLAMA:test-ollama-model")
            
            assert "reset" in status.lower() or "ollama" in status.lower()
            assert not index_dir.exists()

