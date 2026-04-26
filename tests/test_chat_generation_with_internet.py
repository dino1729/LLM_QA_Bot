"""
Unit tests for the internet-connected chatbot.
"""
from unittest.mock import patch

from helper_functions import chat_generation_with_internet


class TestGetWeatherData:
    def test_weather_data_disabled(self):
        result = chat_generation_with_internet.get_weather_data("weather in Tokyo")
        assert "disabled" in result.lower()


class TestTextExtractor:
    @patch("helper_functions.chat_generation_with_internet.extract_page_content")
    def test_text_extractor_direct(self, mock_extract):
        mock_extract.return_value = "Direct extracted text"

        result = chat_generation_with_internet.text_extractor("https://example.com")

        assert result == "Direct extracted text"


class TestResearchHelpers:
    @patch("helper_functions.chat_generation_with_internet.search_web_sources")
    def test_get_quick_web_sources_uses_short_timeout(self, mock_search):
        mock_search.return_value = [
            {"title": "One", "url": "https://example.com/1"},
            {"title": "Two", "url": "https://example.com/2"},
            {"title": "Three", "url": "https://example.com/3"},
        ]

        result = chat_generation_with_internet.get_quick_web_sources("AI news", max_results=2)

        assert len(result) == 2
        mock_search.assert_called_once_with("AI news", max_sources=2, timeout=12)

    @patch("helper_functions.chat_generation_with_internet.get_quick_web_sources")
    def test_get_news_results(self, mock_sources):
        mock_sources.return_value = [
            {"title": "News", "url": "https://example.com/news", "snippet": "Summary"}
        ]

        result = chat_generation_with_internet.get_news_results("latest news", num=3)

        assert "Search query: latest news" in result
        assert "https://example.com/news" in result
        assert mock_sources.call_args.kwargs["max_results"] == 3

    @patch("helper_functions.chat_generation_with_internet.get_quick_web_sources")
    def test_get_web_results(self, mock_sources):
        mock_sources.return_value = [
            {"title": "Result", "url": "https://example.com/result", "snippet": "Summary"}
        ]

        result = chat_generation_with_internet.get_web_results("test query", num=4)

        assert "Search query: test query" in result
        assert "https://example.com/result" in result
        assert mock_sources.call_args.kwargs["max_results"] == 4


class TestSearchDecision:
    def test_current_query_uses_web_search(self):
        assert chat_generation_with_internet.query_requires_web_search("latest AI news")

    def test_normal_information_question_uses_web_search(self):
        assert chat_generation_with_internet.query_requires_web_search("who is the current president of France?")

    def test_simple_math_skips_web_search(self):
        assert not chat_generation_with_internet.query_requires_web_search("What is 2 + 2?")

    def test_creative_prompt_skips_web_search(self):
        assert not chat_generation_with_internet.query_requires_web_search("write a short poem about rain")


class TestParseDynamicModelName:
    def test_normalize_raw_litellm_model_name(self):
        result = chat_generation_with_internet.normalize_internet_model_name("gemini-3-pro-preview")

        assert result == "LITELLM:gemini-3-pro-preview"

    def test_normalize_dynamic_model_name_case(self):
        result = chat_generation_with_internet.normalize_internet_model_name("litellm:test-model")

        assert result == "LITELLM:test-model"

    def test_normalize_legacy_model_token(self):
        result = chat_generation_with_internet.normalize_internet_model_name("litellm_smart")

        assert result == "LITELLM_SMART"

    def test_parse_litellm_dynamic(self):
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("LITELLM:test-model")
        assert provider == "litellm"
        assert model == "test-model"

    def test_parse_ollama_dynamic(self):
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("OLLAMA:test-ollama:3b")
        assert provider == "ollama"
        assert model == "test-ollama:3b"

    def test_parse_litellm_tier(self):
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("LITELLM_SMART")
        assert provider == "litellm"
        assert model == "smart"

    def test_parse_ollama_tier(self):
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("OLLAMA_FAST")
        assert provider == "ollama"
        assert model == "fast"

    def test_parse_other_models(self):
        provider, model = chat_generation_with_internet.parse_dynamic_model_name("GEMINI")
        assert provider is None
        assert model == "GEMINI"


class TestInternetConnectedChatbot:
    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_answer_with_context_normalizes_raw_model_name(self, mock_generate):
        mock_generate.return_value = "Answer"

        result = chat_generation_with_internet.answer_with_context(
            "latest AI news",
            [],
            "gemini-3-pro-preview",
            1000,
            0.7,
            "Source context",
        )

        assert result == "Answer"
        assert mock_generate.call_args[0][0] == "LITELLM:gemini-3-pro-preview"

    @patch("helper_functions.chat_generation_with_internet.get_web_results")
    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_keyword_triggered_search(self, mock_generate, mock_web):
        mock_web.return_value = "Web search results"
        mock_generate.return_value = "Final answer with web context"

        result = chat_generation_with_internet.internet_connected_chatbot(
            "search for latest news",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
        )

        assert result == "Final answer with web context"
        mock_web.assert_called()

    @patch("helper_functions.chat_generation_with_internet.get_news_results")
    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_news_query(self, mock_generate, mock_news):
        mock_news.return_value = "News results"
        mock_generate.return_value = "News answer"

        result = chat_generation_with_internet.internet_connected_chatbot(
            "latest news about AI",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
        )

        assert result == "News answer"
        mock_news.assert_called()

    @patch("helper_functions.chat_generation_with_internet.get_news_results")
    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_places_web_context_in_final_user_message(self, mock_generate, mock_news):
        mock_news.return_value = "Fresh source facts"
        mock_generate.return_value = "Answer using web context"

        result = chat_generation_with_internet.internet_connected_chatbot(
            "latest news about AI",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
        )

        conversation = mock_generate.call_args[0][1]
        final_message = conversation[-1]
        assert result == "Answer using web context"
        assert final_message["role"] == "user"
        assert "User question: latest news about AI" in final_message["content"]
        assert "Web context:\nFresh source facts" in final_message["content"]

    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_general_query(self, mock_generate):
        mock_generate.return_value = "General answer"

        result = chat_generation_with_internet.internet_connected_chatbot(
            "What is 2+2?",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
        )

        assert result == "General answer"

    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_error_handling(self, mock_generate):
        mock_generate.side_effect = Exception("API error")

        result = chat_generation_with_internet.internet_connected_chatbot(
            "test query",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
        )

        assert "error" in result.lower() or "sorry" in result.lower()

    @patch("helper_functions.chat_generation_with_internet.get_web_results")
    @patch("helper_functions.chat_generation_with_internet.generate_chat")
    def test_chatbot_deep_research(self, mock_generate, mock_web):
        mock_web.return_value = "Deep research results"
        mock_generate.return_value = "Comprehensive answer"

        result = chat_generation_with_internet.internet_connected_chatbot(
            "search for quantum computing",
            [],
            "LITELLM_SMART",
            1000,
            0.7,
            fast_response=False,
        )

        assert result == "Comprehensive answer"
        mock_web.assert_called()


class TestResearchWeb:
    @patch("helper_functions.chat_generation_with_internet.conduct_web_research")
    def test_research_web(self, mock_conduct):
        mock_conduct.return_value = "Research report"

        result = chat_generation_with_internet.research_web("AI advancements", provider="litellm")

        assert result == "Research report"
        mock_conduct.assert_called_once()
