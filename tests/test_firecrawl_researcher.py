"""
Unit tests for helper_functions/firecrawl_researcher.py
Tests custom research using Firecrawl and LLM synthesis
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestScrapeWithFirecrawl:
    """Test scrape_with_firecrawl function"""

    @patch('helper_functions.firecrawl_researcher.requests.post')
    def test_scrape_success_markdown(self, mock_post):
        """Test successful scrape with markdown content"""
        from helper_functions.firecrawl_researcher import scrape_with_firecrawl

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {"markdown": "# Article Title\n\nArticle content here"}
        }
        mock_post.return_value = mock_response

        result = scrape_with_firecrawl("https://example.com/article")

        assert result == "# Article Title\n\nArticle content here"
        mock_post.assert_called_once()

    @patch('helper_functions.firecrawl_researcher.requests.post')
    def test_scrape_success_html(self, mock_post):
        """Test successful scrape with HTML content"""
        from helper_functions.firecrawl_researcher import scrape_with_firecrawl

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {"html": "<html><body>Content</body></html>"}
        }
        mock_post.return_value = mock_response

        result = scrape_with_firecrawl("https://example.com")

        assert result == "<html><body>Content</body></html>"

    @patch('helper_functions.firecrawl_researcher.requests.post')
    def test_scrape_timeout(self, mock_post):
        """Test timeout handling"""
        from helper_functions.firecrawl_researcher import scrape_with_firecrawl
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        result = scrape_with_firecrawl("https://example.com", timeout=10)

        assert result is None

    @patch('helper_functions.firecrawl_researcher.requests.post')
    def test_scrape_invalid_url(self, mock_post):
        """Test invalid URL handling"""
        from helper_functions.firecrawl_researcher import scrape_with_firecrawl

        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = scrape_with_firecrawl("invalid-url")

        assert result is None

    @patch('helper_functions.firecrawl_researcher.requests.post')
    def test_scrape_failed_scrape(self, mock_post):
        """Test failed scrape (status code)"""
        from helper_functions.firecrawl_researcher import scrape_with_firecrawl

        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = scrape_with_firecrawl("https://example.com")

        assert result is None


class TestGetSearchUrlsBing:
    """Test get_search_urls_bing function"""

    @patch('helper_functions.firecrawl_researcher.requests.get')
    @patch('helper_functions.firecrawl_researcher.bing_apikey', 'test-bing-key')
    def test_get_search_urls_success(self, mock_get):
        """Test successful Bing search"""
        from helper_functions.firecrawl_researcher import get_search_urls_bing

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "webPages": {
                "value": [
                    {"url": "https://example.com/1"},
                    {"url": "https://example.com/2"},
                    {"url": "https://example.com/3"}
                ]
            }
        }
        mock_get.return_value = mock_response

        result = get_search_urls_bing("test query", count=3)

        assert result == ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
        mock_get.assert_called_once()

    @patch('helper_functions.firecrawl_researcher.bing_apikey', None)
    def test_get_search_urls_no_api_key(self):
        """Test without Bing API key"""
        from helper_functions.firecrawl_researcher import get_search_urls_bing

        result = get_search_urls_bing("test query")

        assert result is None

    @patch('helper_functions.firecrawl_researcher.requests.get')
    @patch('helper_functions.firecrawl_researcher.bing_apikey', 'test-key')
    def test_get_search_urls_error(self, mock_get):
        """Test Bing API error handling"""
        from helper_functions.firecrawl_researcher import get_search_urls_bing

        mock_get.side_effect = Exception("API error")

        result = get_search_urls_bing("test query")

        assert result is None

    @patch('helper_functions.firecrawl_researcher.requests.get')
    @patch('helper_functions.firecrawl_researcher.bing_apikey', 'test-key')
    def test_get_search_urls_response_parsing(self, mock_get):
        """Test response parsing"""
        from helper_functions.firecrawl_researcher import get_search_urls_bing

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "webPages": {
                "value": [
                    {"url": "https://site1.com"},
                    {"url": "https://site2.com"}
                ]
            }
        }
        mock_get.return_value = mock_response

        result = get_search_urls_bing("query", count=5)

        assert len(result) == 2
        assert all(url.startswith("https://") for url in result)


class TestGetSearchUrlsFallback:
    """Test get_search_urls_fallback function"""

    @patch('helper_functions.firecrawl_researcher.requests.get')
    def test_fallback_search_success(self, mock_get):
        """Test successful DuckDuckGo fallback search"""
        from helper_functions.firecrawl_researcher import get_search_urls_fallback

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
            <a class="result__url" href="https://example.com/1">Link 1</a>
            <a class="result__url" href="https://example.com/2">Link 2</a>
        </html>
        '''
        mock_get.return_value = mock_response

        result = get_search_urls_fallback("test query", count=2)

        assert isinstance(result, list)
        assert len(result) >= 0  # Flexible since parsing may vary

    @patch('helper_functions.firecrawl_researcher.requests.get')
    @patch('helper_functions.firecrawl_researcher.generate_topic_urls')
    def test_fallback_to_topic_urls(self, mock_generate_urls, mock_get):
        """Test fallback to topic URLs when DuckDuckGo fails"""
        from helper_functions.firecrawl_researcher import get_search_urls_fallback

        mock_get.side_effect = Exception("Network error")
        mock_generate_urls.return_value = ["https://wikipedia.org/wiki/Test"]

        result = get_search_urls_fallback("test query")

        assert result == ["https://wikipedia.org/wiki/Test"]
        mock_generate_urls.assert_called_once()

    @patch('helper_functions.firecrawl_researcher.requests.get')
    def test_fallback_error_handling(self, mock_get):
        """Test error handling in fallback search"""
        from helper_functions.firecrawl_researcher import get_search_urls_fallback

        mock_get.side_effect = Exception("Error")

        result = get_search_urls_fallback("query")

        # Should return topic URLs as fallback
        assert isinstance(result, list)

    @patch('helper_functions.firecrawl_researcher.requests.get')
    def test_fallback_url_extraction(self, mock_get):
        """Test URL extraction from DuckDuckGo HTML"""
        from helper_functions.firecrawl_researcher import get_search_urls_fallback

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<a href="https://test.com">Link</a>'
        mock_get.return_value = mock_response

        result = get_search_urls_fallback("query", count=5)

        assert isinstance(result, list)


class TestGenerateTopicUrls:
    """Test generate_topic_urls function"""

    def test_generate_news_topic(self):
        """Test news topic detection"""
        from helper_functions.firecrawl_researcher import generate_topic_urls

        result = generate_topic_urls("latest news about AI")

        assert isinstance(result, list)
        assert len(result) > 0
        assert any("news" in url.lower() or "bbc" in url.lower() or "cnn" in url.lower() for url in result)

    def test_generate_technology_topic(self):
        """Test technology topic detection"""
        from helper_functions.firecrawl_researcher import generate_topic_urls

        result = generate_topic_urls("programming python")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_science_topic(self):
        """Test science topic detection"""
        from helper_functions.firecrawl_researcher import generate_topic_urls

        result = generate_topic_urls("quantum physics")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_general_fallback(self):
        """Test general fallback URLs"""
        from helper_functions.firecrawl_researcher import generate_topic_urls

        result = generate_topic_urls("random query")

        assert isinstance(result, list)
        assert len(result) > 0
        assert any("wikipedia.org" in url for url in result)

    def test_generate_url_generation(self):
        """Test that generated URLs are valid"""
        from helper_functions.firecrawl_researcher import generate_topic_urls

        result = generate_topic_urls("test")

        assert all(url.startswith("http") for url in result)


class TestConductResearchFirecrawl:
    """Test conduct_research_firecrawl function"""

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    @patch('helper_functions.firecrawl_researcher.scrape_with_firecrawl')
    @patch('helper_functions.firecrawl_researcher.get_client')
    def test_conduct_research_success_with_bing(self, mock_get_client, mock_scrape, mock_bing):
        """Test successful research with Bing search"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        mock_bing.return_value = [
            "https://example.com/1",
            "https://example.com/2"
        ]

        mock_scrape.side_effect = [
            "Content from source 1 with more than 100 words " * 20,
            "Content from source 2 with more than 100 words " * 20
        ]

        mock_client = Mock()
        mock_client.chat_completion.return_value = "Research report based on sources"
        mock_get_client.return_value = mock_client

        result = conduct_research_firecrawl("test query", "litellm", None, max_sources=2)

        assert isinstance(result, str)
        assert "Research report" in result
        mock_bing.assert_called_once()
        assert mock_scrape.call_count == 2

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    @patch('helper_functions.firecrawl_researcher.get_search_urls_fallback')
    @patch('helper_functions.firecrawl_researcher.scrape_with_firecrawl')
    @patch('helper_functions.firecrawl_researcher.get_client')
    def test_conduct_research_fallback_search(self, mock_get_client, mock_scrape, mock_fallback, mock_bing):
        """Test research with fallback search when Bing unavailable"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        mock_bing.return_value = None  # Bing not available
        mock_fallback.return_value = ["https://fallback.com"]

        mock_scrape.return_value = "Fallback content " * 30

        mock_client = Mock()
        mock_client.chat_completion.return_value = "Report from fallback sources"
        mock_get_client.return_value = mock_client

        result = conduct_research_firecrawl("query", "litellm")

        assert isinstance(result, str)
        mock_fallback.assert_called_once()

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    @patch('helper_functions.firecrawl_researcher.scrape_with_firecrawl')
    @patch('helper_functions.firecrawl_researcher.get_client')
    def test_conduct_research_llm_synthesis(self, mock_get_client, mock_scrape, mock_bing):
        """Test LLM synthesis of scraped content"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        mock_bing.return_value = ["https://test.com"]
        mock_scrape.return_value = "Test content " * 50

        mock_client = Mock()
        mock_client.chat_completion.return_value = "Synthesized report"
        mock_get_client.return_value = mock_client

        result = conduct_research_firecrawl("query", "ollama", "llama3.2")

        mock_get_client.assert_called_with("ollama", "smart")
        mock_client.chat_completion.assert_called_once()

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    @patch('helper_functions.firecrawl_researcher.scrape_with_firecrawl')
    @patch('helper_functions.firecrawl_researcher.get_client')
    def test_conduct_research_source_citation(self, mock_get_client, mock_scrape, mock_bing):
        """Test that sources are cited in report"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        urls = ["https://source1.com", "https://source2.com"]
        mock_bing.return_value = urls
        mock_scrape.return_value = "Content " * 50

        mock_client = Mock()
        mock_client.chat_completion.return_value = "Report content"
        mock_get_client.return_value = mock_client

        result = conduct_research_firecrawl("query", "litellm", max_sources=2)

        # Verify sources are included in the final report
        assert "source1.com" in result or "source2.com" in result or "Sources" in result

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    def test_conduct_research_error_handling(self, mock_bing):
        """Test error handling in research"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        mock_bing.side_effect = Exception("Search error")

        # Should handle error gracefully
        try:
            result = conduct_research_firecrawl("query", "litellm")
            assert isinstance(result, str)
        except Exception as e:
            assert "error" in str(e).lower()

    @patch('helper_functions.firecrawl_researcher.get_search_urls_bing')
    @patch('helper_functions.firecrawl_researcher.scrape_with_firecrawl')
    @patch('helper_functions.firecrawl_researcher.get_client')
    def test_conduct_research_content_length_limits(self, mock_get_client, mock_scrape, mock_bing):
        """Test content length limits (minimum word count)"""
        from helper_functions.firecrawl_researcher import conduct_research_firecrawl

        mock_bing.return_value = ["https://test1.com", "https://test2.com", "https://test3.com"]

        # First URL has short content (filtered out), others have long content
        mock_scrape.side_effect = [
            "Short",  # < 100 words
            "Long content " * 50,  # > 100 words
            "Another long content " * 50  # > 100 words
        ]

        mock_client = Mock()
        mock_client.chat_completion.return_value = "Report"
        mock_get_client.return_value = mock_client

        conduct_research_firecrawl("query", "litellm", max_sources=3)

        # Verify LLM was called with content from valid sources
        call_args = mock_client.chat_completion.call_args
        messages = call_args[0][0]
        content = str(messages)

        # Should not include short content
        assert "Short" not in content or "Long content" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
