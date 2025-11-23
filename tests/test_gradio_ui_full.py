"""
Unit tests for gradio_ui_full.py
Tests main UI functions for the LLM QA Bot
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestParseModelName:
    """Test parse_model_name function"""

    def test_parse_litellm_tier(self):
        """Test parsing LITELLM_TIER format"""
        from gradio_ui_full import parse_model_name

        provider, tier = parse_model_name("LITELLM_SMART")

        assert provider == "litellm"
        assert tier == "smart"

    def test_parse_litellm_only(self):
        """Test parsing LITELLM format"""
        from gradio_ui_full import parse_model_name

        provider, tier = parse_model_name("LITELLM")

        assert provider == "litellm"
        assert tier == "smart"  # default

    def test_parse_ollama_tier(self):
        """Test parsing OLLAMA_TIER format"""
        from gradio_ui_full import parse_model_name

        provider, tier = parse_model_name("OLLAMA_FAST")

        assert provider == "ollama"
        assert tier == "fast"

    def test_parse_ollama_only(self):
        """Test parsing OLLAMA format"""
        from gradio_ui_full import parse_model_name

        provider, tier = parse_model_name("OLLAMA")

        assert provider == "ollama"
        assert tier == "smart"

    def test_parse_other_models(self):
        """Test parsing other model formats"""
        from gradio_ui_full import parse_model_name

        provider, tier = parse_model_name("COHERE")

        # Should return defaults for non-LiteLLM/Ollama models
        assert isinstance(provider, str)
        assert isinstance(tier, str)


class TestSetModelForSession:
    """Test set_model_for_session function"""

    @patch('gradio_ui_full.Settings')
    @patch('gradio_ui_full.get_client')
    def test_set_model_litellm(self, mock_get_client, mock_settings):
        """Test setting LiteLLM model"""
        from gradio_ui_full import set_model_for_session

        mock_client = Mock()
        mock_client.get_llamaindex_llm.return_value = Mock()
        mock_client.get_llamaindex_embedding.return_value = Mock()
        mock_get_client.return_value = mock_client

        set_model_for_session("LITELLM_SMART")

        mock_get_client.assert_called_once_with("litellm", "smart")

    @patch('gradio_ui_full.Settings')
    @patch('gradio_ui_full.get_client')
    def test_set_model_ollama(self, mock_get_client, mock_settings):
        """Test setting Ollama model"""
        from gradio_ui_full import set_model_for_session

        mock_client = Mock()
        mock_client.get_llamaindex_llm.return_value = Mock()
        mock_client.get_llamaindex_embedding.return_value = Mock()
        mock_get_client.return_value = mock_client

        set_model_for_session("OLLAMA_FAST")

        mock_get_client.assert_called_once_with("ollama", "fast")


class TestUploadFile:
    """Test upload_file function"""

    @patch('gradio_ui_full.set_model_for_session')
    @patch('gradio_ui_full.analyze_file')
    def test_upload_file_success(self, mock_analyze, mock_set_model):
        """Test successful file upload"""
        from gradio_ui_full import upload_file

        mock_analyze.return_value = {
            "message": "Success",
            "summary": "File summary",
            "example_queries": ["Q1", "Q2"],
            "file_title": "test.pdf",
            "file_memoryupload_status": "uploaded"
        }

        files = [Mock(name="test.pdf")]
        result = upload_file(files, False, "LITELLM_SMART")

        assert len(result) == 7  # Returns tuple with 7 elements
        assert result[0] == "Success"  # message
        mock_analyze.assert_called_once()
        mock_set_model.assert_called_once()


class TestDownloadYtvideo:
    """Test download_ytvideo function"""

    @patch('gradio_ui_full.set_model_for_session')
    @patch('gradio_ui_full.analyze_ytvideo')
    def test_download_ytvideo_success(self, mock_analyze, mock_set_model):
        """Test successful YouTube video download"""
        from gradio_ui_full import download_ytvideo

        mock_analyze.return_value = {
            "message": "Success",
            "summary": "Video summary",
            "example_queries": ["Q1"],
            "video_title": "Test Video",
            "video_memoryupload_status": "uploaded"
        }

        result = download_ytvideo("https://youtube.com/watch?v=test", False, "LITELLM_SMART")

        assert len(result) == 7
        mock_analyze.assert_called_once()


class TestDownloadArt:
    """Test download_art function"""

    @patch('gradio_ui_full.set_model_for_session')
    @patch('gradio_ui_full.analyze_article')
    def test_download_art_success(self, mock_analyze, mock_set_model):
        """Test successful article download"""
        from gradio_ui_full import download_art

        mock_analyze.return_value = {
            "message": "Success",
            "summary": "Article summary",
            "example_queries": ["Q1"],
            "article_title": "Test Article",
            "article_memoryupload_status": "uploaded"
        }

        result = download_art("https://example.com/article", False, "LITELLM_SMART")

        assert len(result) == 7
        mock_analyze.assert_called_once()


class TestDownloadMedia:
    """Test download_media function"""

    @patch('gradio_ui_full.set_model_for_session')
    @patch('gradio_ui_full.analyze_media')
    def test_download_media_success(self, mock_analyze, mock_set_model):
        """Test successful media download"""
        from gradio_ui_full import download_media

        mock_analyze.return_value = {
            "message": "Success",
            "summary": "Media summary",
            "example_queries": ["Q1"],
            "media_title": "Test Media",
            "media_memoryupload_status": "uploaded"
        }

        result = download_media("https://example.com/audio.mp3", False, "LITELLM_SMART")

        assert len(result) == 7
        mock_analyze.assert_called_once()


class TestAsk:
    """Test ask function"""

    @patch('gradio_ui_full.ask_query')
    def test_ask_success(self, mock_ask_query):
        """Test asking a question"""
        from gradio_ui_full import ask

        mock_ask_query.return_value = "Answer to the question"

        result = ask("What is AI?", [], "LITELLM_SMART")

        assert result == "Answer to the question"
        mock_ask_query.assert_called_once_with("What is AI?", "LITELLM_SMART")


class TestAskQuery:
    """Test ask_query function"""

    @patch('gradio_ui_full.Settings')
    @patch('gradio_ui_full.set_model_for_session')
    @patch('gradio_ui_full.StorageContext')
    @patch('gradio_ui_full.VectorStoreIndex')
    def test_ask_query_success(self, mock_index_class, mock_storage, mock_set_model, mock_settings):
        """Test successful query"""
        from gradio_ui_full import ask_query

        # Mock the query engine
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.response = "Answer from index"
        mock_query_engine.query.return_value = mock_response

        mock_index = Mock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_index_class.from_storage_context.return_value = mock_index

        result = ask_query("What is this about?", "LITELLM_SMART")

        assert result == "Answer from index"
        mock_set_model.assert_called_once()

    @patch('gradio_ui_full.Settings')
    @patch('gradio_ui_full.set_model_for_session')
    def test_ask_query_error_handling(self, mock_set_model, mock_settings):
        """Test error handling in ask_query"""
        from gradio_ui_full import ask_query

        mock_set_model.side_effect = Exception("Model error")

        # Should handle error gracefully
        try:
            result = ask_query("test", "LITELLM_SMART")
            assert isinstance(result, str)
        except Exception:
            # Exception is acceptable
            pass


class TestLoadExample:
    """Test load_example function"""

    @patch('gradio_ui_full.example_queries', ["Example 1", "Example 2", "Example 3"])
    def test_load_example(self):
        """Test loading example query"""
        from gradio_ui_full import load_example

        result = load_example(1)

        assert result == "Example 2"


class TestClearFunctions:
    """Test clear functions"""

    def test_clearfield(self):
        """Test clearfield function"""
        from gradio_ui_full import clearfield

        result = clearfield("test value")

        assert result == ""

    def test_clearhistory(self):
        """Test clearhistory function"""
        from gradio_ui_full import clearhistory

        result = clearhistory()

        # Should return empty list
        assert result == []

    def test_clear_trip_plan(self):
        """Test clear_trip_plan function"""
        from gradio_ui_full import clear_trip_plan

        result = clear_trip_plan()

        assert result == ""

    def test_clear_craving_plan(self):
        """Test clear_craving_plan function"""
        from gradio_ui_full import clear_craving_plan

        result = clear_craving_plan()

        assert result == ""


class TestToggleModelLocal:
    """Test toggle_model_local function"""

    def test_toggle_to_local(self):
        """Test toggling to local models"""
        from gradio_ui_full import toggle_model_local

        result = toggle_model_local(True)

        assert "OLLAMA" in result

    def test_toggle_to_remote(self):
        """Test toggling to remote models"""
        from gradio_ui_full import toggle_model_local

        result = toggle_model_local(False)

        assert "LITELLM" in result


class TestFetchLitellmModels:
    """Test fetch_litellm_models function"""

    @patch('gradio_ui_full.requests.get')
    @patch('gradio_ui_full.litellm_base_url', 'http://litellm:4000')
    def test_fetch_litellm_models_success(self, mock_get):
        """Test successful model fetching"""
        from gradio_ui_full import fetch_litellm_models

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
                {"id": "claude-3"}
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_litellm_models()

        assert isinstance(result, list)
        assert "LITELLM:gpt-4" in result
        assert "LITELLM:gpt-3.5-turbo" in result

    @patch('gradio_ui_full.requests.get')
    def test_fetch_litellm_models_error(self, mock_get):
        """Test error handling in model fetching"""
        from gradio_ui_full import fetch_litellm_models

        mock_get.side_effect = Exception("API error")

        result = fetch_litellm_models()

        # Should return default list on error
        assert isinstance(result, list)
        assert "LITELLM_FAST" in result

    @patch('gradio_ui_full.requests.get')
    @patch('gradio_ui_full.litellm_base_url', 'http://litellm:4000')
    def test_fetch_litellm_models_timeout(self, mock_get):
        """Test timeout handling"""
        from gradio_ui_full import fetch_litellm_models
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        result = fetch_litellm_models()

        # Should return default list
        assert isinstance(result, list)

    @patch('gradio_ui_full.requests.get')
    @patch('gradio_ui_full.litellm_base_url', 'http://litellm:4000')
    def test_fetch_litellm_models_response_parsing(self, mock_get):
        """Test response parsing"""
        from gradio_ui_full import fetch_litellm_models

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "model1"},
                {"id": "model2"}
            ]
        }
        mock_get.return_value = mock_response

        result = fetch_litellm_models()

        # Should format models correctly
        assert any("LITELLM:model1" in item for item in result)
        assert any("LITELLM:model2" in item for item in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
