from unittest.mock import Mock, patch

from helper_functions import web_researcher


class TestExtractPageContent:
    @patch("helper_functions.web_researcher.extract_web_content")
    def test_extract_page_content_success(self, mock_extract):
        mock_extract.return_value = ("Article body", "Article title")

        result = web_researcher.extract_page_content("https://example.com")

        assert result == "Article body"
        mock_extract.assert_called_once()

    @patch("helper_functions.web_researcher.extract_web_content")
    def test_extract_page_content_failure(self, mock_extract):
        mock_extract.side_effect = Exception("boom")

        result = web_researcher.extract_page_content("https://example.com")

        assert result is None


class TestSearchWebSources:
    @patch("helper_functions.web_researcher.search_with_perplexity")
    def test_search_web_sources_uses_perplexity(self, mock_search):
        mock_search.return_value = [
            {"title": "AI News", "url": "https://example.com/ai", "snippet": "summary"}
        ]

        results = web_researcher.search_web_sources("latest AI news", max_sources=3)

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/ai"
        mock_search.assert_called_once()

    @patch("helper_functions.web_researcher.search_with_perplexity")
    def test_search_web_sources_falls_back_to_topic_urls(self, mock_search):
        mock_search.return_value = []

        results = web_researcher.search_web_sources("latest AI news", max_sources=3)

        assert results
        assert all("url" in result for result in results)


class TestConductWebResearch:
    @patch("helper_functions.web_researcher.get_client")
    @patch("helper_functions.web_researcher.extract_page_content")
    @patch("helper_functions.web_researcher.search_with_perplexity")
    def test_conduct_web_research_uses_perplexity_results(
        self,
        mock_search,
        mock_extract,
        mock_get_client,
    ):
        mock_search.return_value = [
            {
                "title": "AI News",
                "url": "https://example.com/ai",
                "snippet": "Short summary of the article.",
            }
        ]
        mock_extract.return_value = (
            "This is a long extracted article body with enough detail to pass the "
            "minimum threshold for the research synthesis stage."
        )
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Research Report"
        mock_get_client.return_value = mock_client

        result = web_researcher.conduct_web_research("latest AI news")

        assert "Research Report" in result
        assert "https://example.com/ai" in result
        mock_search.assert_called_once()

    @patch("helper_functions.web_researcher.search_with_perplexity")
    @patch("helper_functions.web_researcher.extract_page_content")
    def test_conduct_web_research_handles_empty_content(self, mock_extract, mock_search):
        mock_search.return_value = [
            {"title": "Blocked", "url": "https://example.com/blocked", "snippet": ""}
        ]
        mock_extract.return_value = None

        result = web_researcher.conduct_web_research("blocked source")

        assert "Unable to extract readable content" in result
