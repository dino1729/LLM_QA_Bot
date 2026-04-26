from unittest.mock import Mock, patch


class TestPerplexitySearch:
    @patch("helper_functions.perplexity_search.requests.post")
    def test_search_with_perplexity_success(self, mock_post):
        from helper_functions.perplexity_search import search_with_perplexity

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "AI News",
                    "url": "https://example.com/ai",
                    "snippet": "2026-03-17 AI systems continue to evolve rapidly.",
                }
            ]
        }
        mock_post.return_value = mock_response

        results = search_with_perplexity("latest AI news", max_results=3)

        assert len(results) == 1
        assert results[0]["title"] == "AI News"
        assert results[0]["url"] == "https://example.com/ai"
        assert results[0]["date"] == "2026-03-17"

    @patch("helper_functions.perplexity_search.requests.post")
    def test_search_with_perplexity_failure(self, mock_post):
        from helper_functions.perplexity_search import search_with_perplexity

        mock_post.side_effect = Exception("boom")

        results = search_with_perplexity("latest AI news", max_results=3)

        assert results == []


class TestResearchWrappers:
    @patch("helper_functions.web_researcher.get_client")
    @patch("helper_functions.web_researcher.extract_page_content")
    @patch("helper_functions.web_researcher.search_with_perplexity")
    def test_conduct_web_research_uses_perplexity_results(
        self,
        mock_search,
        mock_extract,
        mock_get_client,
    ):
        from helper_functions.web_researcher import conduct_web_research

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

        result = conduct_web_research("latest AI news")

        assert "Research Report" in result
        assert "https://example.com/ai" in result
        mock_search.assert_called_once()

    @patch("helper_functions.news_researcher.extract_page_content")
    @patch("helper_functions.news_researcher.search_with_perplexity")
    def test_search_fresh_sources_uses_perplexity(
        self,
        mock_search,
        mock_extract,
    ):
        from helper_functions.news_researcher import search_fresh_sources

        mock_search.return_value = [
            {
                "title": "AI News",
                "url": "https://example.com/ai",
                "snippet": "2026-03-17 latest AI system launch.",
                "date": "2026-03-17",
            }
        ]
        mock_extract.return_value = "Detailed article body for the AI news item."

        results = search_fresh_sources("latest AI news", limit=1)

        assert len(results) == 1
        assert results[0]["title"] == "AI News"
        assert results[0]["url"] == "https://example.com/ai"
        assert results[0]["description"].startswith("2026-03-17")
        mock_search.assert_called_once()
