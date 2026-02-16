"""
Unit tests for helper_functions/memory_palace_local.py
Tests for local memory palace storage and retrieval (SimpleVectorStore-based)
"""
import pytest
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from helper_functions.vector_store import SimpleVectorStore, Document
from helper_functions.memory_palace_local import (
    save_memory,
    search_memories,
    reset_memory_palace,
    get_memory_palace_index_dir,
    _sanitize_filename,
    prepare_memory_stream
)


@pytest.fixture
def temp_memory_root(tmp_path, monkeypatch):
    # Override MEMORY_PALACE_ROOT in the module
    root = tmp_path / "memory_palace"
    monkeypatch.setattr("helper_functions.memory_palace_local.MEMORY_PALACE_ROOT", str(root))
    monkeypatch.setattr("config.config.MEMORY_PALACE_FOLDER", str(root))
    return root


@pytest.fixture
def mock_get_client():
    """Mock get_client to return a mock client with get_embedding."""
    with patch("helper_functions.memory_palace_local.get_client") as mock_gc:
        mock_client = Mock()
        mock_client.get_embedding.return_value = [0.1] * 1536
        mock_client.stream_chat_completion.return_value = iter(["chunk1", "chunk2"])
        mock_gc.return_value = mock_client
        yield mock_gc


@pytest.fixture
def mock_vector_store():
    """Mock SimpleVectorStore to avoid actual file I/O and embedding calls."""
    with patch("helper_functions.memory_palace_local.SimpleVectorStore") as mock_cls:
        mock_store = Mock()
        mock_store.search.return_value = []
        mock_cls.return_value = mock_store
        yield mock_cls


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


class TestSaveMemory:
    """Tests for save_memory() function - uses test placeholder model names"""

    def test_save_memory(self, temp_memory_root, mock_get_client, mock_vector_store):
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
            mock_vector_store.assert_called_once()
            mock_vector_store.return_value.insert.assert_called_once()

    def test_save_memory_ollama_provider(self, temp_memory_root, mock_get_client, mock_vector_store):
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
            mock_vector_store.return_value.insert.assert_called_once()

    def test_save_memory_multiple_entries(self, temp_memory_root, mock_get_client, mock_vector_store):
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

    def test_search_memories(self, temp_memory_root, mock_get_client, mock_vector_store):
        """Test basic memory search"""
        from helper_functions.vector_store import RetrievalResult

        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"

            # Create the persist dir so it passes the existence check
            index_dir = temp_memory_root / "litellm__test-embed-model"
            index_dir.mkdir(parents=True)
            (index_dir / "vector_store.json").write_text("{}")

            mock_result = Mock()
            mock_result.text = "Title: Test Memory\n\nKey Takeaways:\nImportant information about AI"
            mock_result.score = 0.9
            mock_result.metadata = {"source_title": "Test Memory"}
            mock_vector_store.return_value.search.return_value = [mock_result]

            results = search_memories("AI", "LITELLM:test-model")

            assert len(results) > 0
            assert "Important information" in results[0]["content"]
            assert results[0]["metadata"]["source_title"] == "Test Memory"

    def test_search_memories_empty_palace(self, temp_memory_root):
        """Test searching empty memory palace"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"

            results = search_memories("AI", "LITELLM:test-model")

            assert results == []

    def test_search_memories_custom_top_k(self, temp_memory_root, mock_get_client, mock_vector_store):
        """Test search with custom top_k parameter"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"

            # Create the persist dir so it passes the existence check
            index_dir = temp_memory_root / "litellm__test-embed-model"
            index_dir.mkdir(parents=True)
            (index_dir / "vector_store.json").write_text("{}")

            mock_results = []
            for i in range(2):
                r = Mock()
                r.text = f"AI related content {i}"
                r.score = 0.9 - i * 0.1
                r.metadata = {"source_title": f"Memory {i}"}
                mock_results.append(r)
            mock_vector_store.return_value.search.return_value = mock_results

            results = search_memories("AI", "LITELLM:test-model", top_k=2)

            # Should return at most 2 results
            assert len(results) <= 2
            # Verify top_k was passed to store.search
            mock_vector_store.return_value.search.assert_called_once_with("AI", top_k=2)

    def test_search_memories_ollama_provider(self, temp_memory_root, mock_get_client, mock_vector_store):
        """Test searching with Ollama provider"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.ollama_embedding = "test-ollama-embed"

            index_dir = temp_memory_root / "ollama__test-ollama-embed"
            index_dir.mkdir(parents=True)
            (index_dir / "vector_store.json").write_text("{}")

            mock_result = Mock()
            mock_result.text = "Content about Ollama models"
            mock_result.score = 0.8
            mock_result.metadata = {}
            mock_vector_store.return_value.search.return_value = [mock_result]

            results = search_memories("Ollama", "OLLAMA:test-ollama-model")

            assert len(results) > 0


class TestPrepareMemoryStream:
    """Tests for prepare_memory_stream() function"""

    def test_prepare_memory_stream_success(self, temp_memory_root, mock_get_client, mock_vector_store):
        """Test successful memory stream preparation"""
        with patch("helper_functions.memory_palace_local.config") as mock_config:
            mock_config.litellm_embedding = "test-embed-model"
            mock_config.temperature = 0.7
            mock_config.max_tokens = 1000

            # Create the index directory
            index_dir = temp_memory_root / "litellm__test-embed-model"
            index_dir.mkdir(parents=True)
            (index_dir / "vector_store.json").write_text("{}")

            mock_result = Mock()
            mock_result.text = "Some memory context"
            mock_vector_store.return_value.search.return_value = [mock_result]

            response = prepare_memory_stream(
                message="What do I know about AI?",
                history=[],
                model_name="LITELLM:test-model",
                top_k=5
            )

            # Should return a StreamingResult with response_gen
            assert hasattr(response, 'response_gen')
            # The mock client's stream_chat_completion should have been called
            mock_client = mock_get_client.return_value
            mock_client.stream_chat_completion.assert_called_once()

    def test_prepare_memory_stream_empty_palace(self, temp_memory_root):
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
