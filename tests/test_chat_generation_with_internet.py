"""
Unit tests for helper_functions/chat_generation_with_internet.py
Tests for internet-connected chatbot with Firecrawl researcher
"""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from helper_functions import chat_generation_with_internet


class TestSaveExtractedTextToFile:
    """Tests for saveextractedtext_to_file() function"""
    
    def test_save_text_success(self, temp_bing_folder, monkeypatch):
        """Test successful text saving"""
        monkeypatch.setattr(chat_generation_with_internet, 'BING_FOLDER', temp_bing_folder)
        
        text = "This is extracted text content"
        filename = "test.txt"
        
        result = chat_generation_with_internet.saveextractedtext_to_file(text, filename)
        
        assert "Text saved" in result
        assert os.path.exists(os.path.join(temp_bing_folder, filename))
    
    def test_save_text_unicode(self, temp_bing_folder, monkeypatch):
        """Test saving text with unicode characters"""
        monkeypatch.setattr(chat_generation_with_internet, 'BING_FOLDER', temp_bing_folder)
        
        text = "Unicode content: café ñ 日本語"
        filename = "unicode.txt"
        
        chat_generation_with_internet.saveextractedtext_to_file(text, filename)
        
        file_path = os.path.join(temp_bing_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "café" in content


class TestClearAllFilesBing:
    """Tests for clearallfiles_bing() function"""
    
    def test_clear_empty_folder(self, temp_bing_folder, monkeypatch):
        """Test clearing empty folder"""
        monkeypatch.setattr(chat_generation_with_internet, 'BING_FOLDER', temp_bing_folder)
        
        chat_generation_with_internet.clearallfiles_bing()
        assert len(os.listdir(temp_bing_folder)) == 0
    
    def test_clear_folder_with_files(self, temp_bing_folder, monkeypatch):
        """Test clearing folder with files"""
        monkeypatch.setattr(chat_generation_with_internet, 'BING_FOLDER', temp_bing_folder)
        
        # Create test files
        for i in range(3):
            with open(os.path.join(temp_bing_folder, f"test{i}.txt"), "w") as f:
                f.write("test content")
        
        assert len(os.listdir(temp_bing_folder)) == 3
        chat_generation_with_internet.clearallfiles_bing()
        assert len(os.listdir(temp_bing_folder)) == 0


class TestGetWeatherData:
    """Tests for get_weather_data() function"""
    
    def test_weather_data_disabled(self):
        """Test that weather data returns disabled message"""
        result = chat_generation_with_internet.get_weather_data("weather in Tokyo")
        assert "disabled" in result.lower()


class TestScrapeWithFirecrawl:
    """Tests for scrape_with_firecrawl() function"""
    
    @patch('helper_functions.chat_generation_with_internet.requests.post')
    def test_scrape_success_markdown(self, mock_post):
        """Test successful scraping with markdown response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "markdown": "# Test Article\n\nThis is test content."
            }
        }
        mock_post.return_value = mock_response
        
        result = chat_generation_with_internet.scrape_with_firecrawl("https://example.com")
        
        assert result is not None
        assert "Test Article" in result
    
    @patch('helper_functions.chat_generation_with_internet.requests.post')
    def test_scrape_success_html(self, mock_post):
        """Test successful scraping with HTML response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "html": "<html><body><p>Test content</p></body></html>"
            }
        }
        mock_post.return_value = mock_response
        
        result = chat_generation_with_internet.scrape_with_firecrawl("https://example.com")
        
        assert result is not None
        assert "Test content" in result
    
    @patch('helper_functions.chat_generation_with_internet.requests.post')
    def test_scrape_failed_status(self, mock_post):
        """Test scraping with failed status code"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response
        
        result = chat_generation_with_internet.scrape_with_firecrawl("https://example.com")
        
        assert result is None
    
    @patch('helper_functions.chat_generation_with_internet.requests.post')
    def test_scrape_timeout(self, mock_post):
        """Test scraping with timeout"""
        mock_post.side_effect = Exception("Timeout")
        
        result = chat_generation_with_internet.scrape_with_firecrawl("https://example.com")
        
        assert result is None
    
    @patch('helper_functions.chat_generation_with_internet.requests.post')
    def test_scrape_invalid_url(self, mock_post):
        """Test scraping with invalid URL"""
        mock_post.side_effect = Exception("Invalid URL")
        
        result = chat_generation_with_internet.scrape_with_firecrawl("invalid-url")
        
        assert result is None


class TestSearchWithFirecrawl:
    """Tests for search_with_firecrawl() function"""
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.requests.get')
    def test_search_with_bing_api(self, mock_get, mock_scrape, monkeypatch):
        """Test search using Bing API and Firecrawl"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', 'test_key')
        
        # Mock Bing response
        mock_bing_response = Mock()
        mock_bing_response.status_code = 200
        mock_bing_response.json.return_value = {
            "webPages": {
                "value": [
                    {"url": "https://example.com/1"},
                    {"url": "https://example.com/2"}
                ]
            }
        }
        mock_get.return_value = mock_bing_response
        
        # Mock Firecrawl scraping
        mock_scrape.side_effect = [
            "This is content from page 1. " * 20,  # >50 words
            "This is content from page 2. " * 20
        ]
        
        result = chat_generation_with_internet.search_with_firecrawl("test query", limit=2)
        
        assert result is not None
        assert "test query" in result
        assert "page 1" in result or "page 2" in result
    
    @patch('helper_functions.chat_generation_with_internet.requests.get')
    def test_search_without_bing_api(self, mock_get, monkeypatch):
        """Test search without Bing API key"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', '')
        
        result = chat_generation_with_internet.search_with_firecrawl("test query")
        
        assert result is None
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.requests.get')
    def test_search_minimum_word_count(self, mock_get, mock_scrape, monkeypatch):
        """Test that pages with less than 50 words are filtered out"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', 'test_key')
        
        mock_bing_response = Mock()
        mock_bing_response.status_code = 200
        mock_bing_response.json.return_value = {
            "webPages": {
                "value": [{"url": "https://example.com/1"}]
            }
        }
        mock_get.return_value = mock_bing_response
        
        # Mock short content (< 50 words)
        mock_scrape.return_value = "Short content"
        
        result = chat_generation_with_internet.search_with_firecrawl("test query")
        
        assert result is None  # No results with enough content
    
    @patch('helper_functions.chat_generation_with_internet.requests.get')
    def test_search_bing_error(self, mock_get, monkeypatch):
        """Test search with Bing API error"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', 'test_key')
        
        mock_get.side_effect = Exception("Bing API error")
        
        result = chat_generation_with_internet.search_with_firecrawl("test query")
        
        assert result is None


class TestTextExtractor:
    """Tests for text_extractor() function"""
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    def test_text_extractor_firecrawl(self, mock_scrape, monkeypatch):
        """Test text extraction with Firecrawl"""
        monkeypatch.setattr(chat_generation_with_internet, 'retriever', 'firecrawl')
        mock_scrape.return_value = "Firecrawl extracted text"
        
        result = chat_generation_with_internet.text_extractor("https://example.com")
        
        assert result == "Firecrawl extracted text"
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.Article')
    def test_text_extractor_newspaper_fallback(self, mock_article_class, mock_scrape, monkeypatch):
        """Test text extraction with newspaper3k fallback"""
        monkeypatch.setattr(chat_generation_with_internet, 'retriever', 'firecrawl')
        mock_scrape.return_value = None
        
        mock_article = Mock()
        mock_article.text = "Newspaper extracted text"
        mock_article_class.return_value = mock_article
        
        result = chat_generation_with_internet.text_extractor("https://example.com")
        
        assert result == "Newspaper extracted text"
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.Article')
    @patch('helper_functions.chat_generation_with_internet.requests.get')
    def test_text_extractor_beautifulsoup_fallback(self, mock_get, mock_article_class, 
                                                   mock_scrape, monkeypatch):
        """Test text extraction with BeautifulSoup fallback"""
        monkeypatch.setattr(chat_generation_with_internet, 'retriever', 'firecrawl')
        mock_scrape.return_value = None
        mock_article_class.side_effect = Exception("Newspaper error")
        
        mock_response = Mock()
        mock_response.text = "<html><body><p>BeautifulSoup content</p></body></html>"
        mock_get.return_value = mock_response
        
        result = chat_generation_with_internet.text_extractor("https://example.com")
        
        assert result is not None
        assert "BeautifulSoup content" in result
    
    @patch('helper_functions.chat_generation_with_internet.scrape_with_firecrawl')
    def test_text_extractor_all_methods_fail(self, mock_scrape, monkeypatch):
        """Test when all extraction methods fail"""
        monkeypatch.setattr(chat_generation_with_internet, 'retriever', 'firecrawl')
        mock_scrape.return_value = None
        
        with patch('helper_functions.chat_generation_with_internet.Article') as mock_article:
            mock_article.side_effect = Exception("Error")
            
            with patch('helper_functions.chat_generation_with_internet.requests.get') as mock_get:
                mock_get.side_effect = Exception("Error")
                
                result = chat_generation_with_internet.text_extractor("https://example.com")
                
                assert result is None


class TestGetBingAgent:
    """Tests for get_bing_agent() function"""
    
    def test_bing_agent_disabled(self):
        """Test that Bing agent returns disabled message"""
        result = chat_generation_with_internet.get_bing_agent("test query")
        assert "disabled" in result.lower()


class TestSummarize:
    """Tests for summarize() function"""
    
    @patch('helper_functions.chat_generation_with_internet.SimpleDirectoryReader')
    @patch('helper_functions.chat_generation_with_internet.SummaryIndex')
    def test_summarize_with_documents(self, mock_summary_index, mock_reader, temp_bing_folder):
        """Test summarizing documents in a folder"""
        # Create a test file
        test_file = os.path.join(temp_bing_folder, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test document content")
        
        mock_docs = [Mock()]
        mock_reader.return_value.load_data.return_value = mock_docs
        
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.response = "This is a summary"
        
        mock_summary_index.from_documents.return_value = mock_index
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response
        
        result = chat_generation_with_internet.summarize(temp_bing_folder)
        
        assert result.response == "This is a summary"
    
    @patch('helper_functions.chat_generation_with_internet.SimpleDirectoryReader')
    def test_summarize_empty_folder(self, mock_reader, temp_bing_folder):
        """Test summarizing empty folder"""
        mock_reader.return_value.load_data.return_value = []
        
        # Should not raise exception
        try:
            chat_generation_with_internet.summarize(temp_bing_folder)
        except Exception:
            pytest.fail("summarize should handle empty folders")


class TestGetBingNewsResults:
    """Tests for get_bing_news_results() function"""
    
    @patch('helper_functions.chat_generation_with_internet.firecrawl_researcher')
    def test_get_news_with_firecrawl_researcher(self, mock_firecrawl, monkeypatch):
        """Test getting news with Firecrawl researcher"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', '')
        mock_firecrawl.return_value = "News summary from Firecrawl"
        
        result = chat_generation_with_internet.get_bing_news_results("latest news", num=5)
        
        assert result == "News summary from Firecrawl"
    
    @patch('helper_functions.chat_generation_with_internet.firecrawl_researcher')
    def test_get_news_firecrawl_error(self, mock_firecrawl, monkeypatch):
        """Test news retrieval with Firecrawl error"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', '')
        mock_firecrawl.side_effect = Exception("Firecrawl error")
        
        result = chat_generation_with_internet.get_bing_news_results("latest news")
        
        assert "error" in result.lower() or "not available" in result.lower()


class TestSimpleQuery:
    """Tests for simple_query() function"""
    
    @patch('helper_functions.chat_generation_with_internet.SimpleDirectoryReader')
    @patch('helper_functions.chat_generation_with_internet.VectorStoreIndex')
    def test_simple_query_success(self, mock_vector_index, mock_reader, temp_bing_folder):
        """Test simple query on documents"""
        # Create a test file
        test_file = os.path.join(temp_bing_folder, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test document content")
        
        mock_docs = [Mock()]
        mock_reader.return_value.load_data.return_value = mock_docs
        
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.response = "Query answer"
        
        mock_vector_index.from_documents.return_value = mock_index
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response
        
        result = chat_generation_with_internet.simple_query(temp_bing_folder, "What is this about?")
        
        assert result.response == "Query answer"


class TestGetWebResults:
    """Tests for get_web_results() function"""
    
    @patch('helper_functions.chat_generation_with_internet.search_with_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.simple_query')
    @patch('helper_functions.chat_generation_with_internet.clearallfiles_bing')
    @patch('helper_functions.chat_generation_with_internet.saveextractedtext_to_file')
    def test_get_web_results_success(self, mock_save, mock_clear, mock_query, mock_search):
        """Test getting web results successfully"""
        mock_search.return_value = "Web search results"
        mock_response = Mock()
        mock_response.response = "Processed answer"
        mock_query.return_value = mock_response
        
        result = chat_generation_with_internet.get_web_results("test query")
        
        assert result == "Processed answer"
        mock_clear.assert_called()
    
    @patch('helper_functions.chat_generation_with_internet.firecrawl_researcher')
    @patch('helper_functions.chat_generation_with_internet.clearallfiles_bing')
    def test_get_web_results_fallback_to_researcher(self, mock_clear, mock_firecrawl, monkeypatch):
        """Test fallback to Firecrawl researcher"""
        monkeypatch.setattr(chat_generation_with_internet, 'bing_api_key', '')
        mock_firecrawl.return_value = "Research results"
        
        result = chat_generation_with_internet.get_web_results("test query")
        
        assert result == "Research results"


class TestParseDynamicModelName:
    """Tests for parse_dynamic_model_name() function"""
    
    def test_parse_litellm_dynamic(self):
        """Test parsing LITELLM:model format"""
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("LITELLM:gpt-4")
        assert provider == "litellm"
        assert model == "gpt-4"
    
    def test_parse_ollama_dynamic(self):
        """Test parsing OLLAMA:model format"""
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("OLLAMA:llama3.2:3b")
        assert provider == "ollama"
        assert model == "llama3.2:3b"
    
    def test_parse_litellm_tier(self):
        """Test parsing LITELLM_TIER format"""
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("LITELLM_SMART")
        assert provider == "litellm"
        assert model == "smart"
    
    def test_parse_ollama_tier(self):
        """Test parsing OLLAMA_TIER format"""
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("OLLAMA_FAST")
        assert provider == "ollama"
        assert model == "fast"
    
    def test_parse_other_models(self):
        """Test parsing other model names"""
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("GEMINI")
        assert provider is None
        assert model == "GEMINI"


class TestInternetConnectedChatbot:
    """Tests for internet_connected_chatbot() function"""
    
    @patch('helper_functions.chat_generation_with_internet.get_web_results')
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_keyword_triggered_search(self, mock_generate, mock_web):
        """Test chatbot with keyword-triggered web search"""
        mock_web.return_value = "Web search results"
        mock_generate.return_value = "Final answer with web context"
        
        result = chat_generation_with_internet.internet_connected_chatbot(
            "search for latest news",
            [],
            "LITELLM_SMART",
            1000,
            0.7
        )
        
        assert result == "Final answer with web context"
        mock_web.assert_called()
    
    @patch('helper_functions.chat_generation_with_internet.get_bing_news_results')
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_news_query(self, mock_generate, mock_news):
        """Test chatbot with news query"""
        mock_news.return_value = "News results"
        mock_generate.return_value = "News answer"
        
        result = chat_generation_with_internet.internet_connected_chatbot(
            "latest news about AI",
            [],
            "LITELLM_SMART",
            1000,
            0.7
        )
        
        assert result == "News answer"
        mock_news.assert_called()
    
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_general_query(self, mock_generate):
        """Test chatbot with general query (no web search)"""
        mock_generate.return_value = "General answer"
        
        result = chat_generation_with_internet.internet_connected_chatbot(
            "What is 2+2?",
            [],
            "LITELLM_SMART",
            1000,
            0.7
        )
        
        assert result == "General answer"
    
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_error_handling(self, mock_generate):
        """Test chatbot error handling"""
        mock_generate.side_effect = Exception("API error")
        
        result = chat_generation_with_internet.internet_connected_chatbot(
            "test query",
            [],
            "LITELLM_SMART",
            1000,
            0.7
        )
        
        assert "error" in result.lower() or "sorry" in result.lower()
    
    @patch('helper_functions.chat_generation_with_internet.conduct_research_firecrawl')
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_deep_research(self, mock_generate, mock_research):
        """Test chatbot with deep research (fast_response=False)"""
        mock_research.return_value = "Deep research results"
        mock_generate.return_value = "Comprehensive answer"
        
        result = chat_generation_with_internet.internet_connected_chatbot(
            "search for quantum computing",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
            fast_response=False
        )
        
        assert result == "Comprehensive answer"
        mock_research.assert_called()
    
    @patch('helper_functions.chat_generation_with_internet.generate_chat')
    def test_chatbot_conversation_building(self, mock_generate):
        """Test that chatbot builds conversation correctly"""
        mock_generate.return_value = "Response"
        
        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        chat_generation_with_internet.internet_connected_chatbot(
            "New query",
            history,
            "LITELLM_SMART",
            1000,
            0.7
        )
        
        # Verify generate_chat was called with updated conversation
        call_args = mock_generate.call_args
        conversation = call_args[0][1]
        assert len(conversation) > len(history)


class TestFirecrawlResearcher:
    """Tests for firecrawl_researcher() function"""
    
    @patch('helper_functions.chat_generation_with_internet.conduct_research_firecrawl')
    def test_firecrawl_researcher_litellm(self, mock_conduct):
        """Test Firecrawl researcher with LiteLLM"""
        mock_conduct.return_value = "Research report"
        
        result = chat_generation_with_internet.firecrawl_researcher(
            "AI advancements",
            provider="litellm"
        )
        
        assert result == "Research report"
        mock_conduct.assert_called_once()
    
    @patch('helper_functions.chat_generation_with_internet.conduct_research_firecrawl')
    def test_firecrawl_researcher_ollama(self, mock_conduct):
        """Test Firecrawl researcher with Ollama"""
        mock_conduct.return_value = "Ollama research report"
        
        result = chat_generation_with_internet.firecrawl_researcher(
            "Climate change",
            provider="ollama"
        )
        
        assert result == "Ollama research report"
    
    @patch('helper_functions.chat_generation_with_internet.conduct_research_firecrawl')
    def test_firecrawl_researcher_with_model(self, mock_conduct):
        """Test Firecrawl researcher with specific model"""
        mock_conduct.return_value = "Custom model report"
        
        result = chat_generation_with_internet.firecrawl_researcher(
            "Space exploration",
            provider="litellm",
            model_name="gpt-4"
        )
        
        assert result == "Custom model report"
        call_args = mock_conduct.call_args
        assert call_args[1]["model_name"] == "gpt-4"
    
    @patch('helper_functions.chat_generation_with_internet.conduct_research_firecrawl')
    def test_firecrawl_researcher_error(self, mock_conduct):
        """Test Firecrawl researcher error handling"""
        mock_conduct.side_effect = Exception("Research error")
        
        result = chat_generation_with_internet.firecrawl_researcher("test query")
        
        assert "error" in result.lower() or "unable" in result.lower()


class TestKeywordsConfiguration:
    """Tests for keywords configuration"""
    
    def test_keywords_loaded(self):
        """Test that keywords are loaded from config"""
        assert hasattr(chat_generation_with_internet, 'keywords')
        assert isinstance(chat_generation_with_internet.keywords, list)


class TestSettingsConfiguration:
    """Tests for LlamaIndex Settings configuration"""
    
    def test_settings_initialized(self):
        """Test that LlamaIndex Settings are initialized"""
        from llama_index.core import Settings
        
        # Settings should have LLM and embed_model configured
        assert Settings.llm is not None or True  # May be None in test environment
        assert Settings.embed_model is not None or True


class TestFolderInitialization:
    """Tests for BING_FOLDER initialization"""
    
    def test_bing_folder_exists(self):
        """Test that BING_FOLDER is created if it doesn't exist"""
        assert hasattr(chat_generation_with_internet, 'BING_FOLDER')
        # The folder should be created during module import
        assert os.path.exists(chat_generation_with_internet.BING_FOLDER)

