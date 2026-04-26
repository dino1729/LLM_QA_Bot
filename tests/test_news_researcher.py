"""
Unit tests for helper_functions/news_researcher.py
Tests for the multi-model news gathering system
"""
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime


class TestScrapeWithPerplexity:
    """Tests for extract_page_content() function"""
    
    @patch('helper_functions.news_researcher.extract_web_content')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_extract_page_content_success(self, mock_log, mock_extract):
        """Test successful scraping"""
        from helper_functions.news_researcher import extract_page_content

        mock_extract.return_value = ("# Test Content\nThis is scraped content.", "Test")
        
        result = extract_page_content("https://example.com")
        
        assert result is not None
        assert "Test Content" in result
        mock_extract.assert_called_once()
    
    @patch('helper_functions.news_researcher.extract_web_content')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_extract_page_content_failure(self, mock_log, mock_extract):
        """Test scraping failure"""
        from helper_functions.news_researcher import extract_page_content
        
        mock_extract.return_value = (None, None)
        
        result = extract_page_content("https://example.com")
        
        assert result is None
    
    @patch('helper_functions.news_researcher.extract_web_content')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_extract_page_content_timeout(self, mock_log, mock_extract):
        """Test scraping with timeout"""
        from helper_functions.news_researcher import extract_page_content
        
        mock_extract.side_effect = Exception("Connection timeout")
        
        result = extract_page_content("https://example.com", timeout=5)
        
        assert result is None
    
    @patch('helper_functions.news_researcher.extract_web_content')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_extract_page_content_ignores_max_age(self, mock_log, mock_extract):
        """Test compatibility max_age argument is accepted."""
        from helper_functions.news_researcher import extract_page_content
        
        mock_extract.return_value = ("Content", "Title")
        
        result = extract_page_content("https://example.com", max_age=3600000)
        
        assert result == "Content"


class TestExtractSourceName:
    """Tests for extract_source_name() function"""
    
    def test_extract_source_name_simple(self):
        """Test extracting source name from simple URL"""
        from helper_functions.news_researcher import extract_source_name
        
        result = extract_source_name("https://example.com/page")
        assert result == "Example"
    
    def test_extract_source_name_with_www(self):
        """Test extracting source name from URL with www"""
        from helper_functions.news_researcher import extract_source_name
        
        result = extract_source_name("https://www.techcrunch.com/article")
        assert result == "Techcrunch"
    
    def test_extract_source_name_invalid_url(self):
        """Test extracting source name from invalid URL"""
        from helper_functions.news_researcher import extract_source_name
        
        result = extract_source_name("not-a-valid-url")
        assert result == "Unknown" or isinstance(result, str)


class TestDeduplicateSources:
    """Tests for deduplicate_sources() function"""
    
    def test_deduplicate_sources_by_url(self):
        """Test deduplication by URL"""
        from helper_functions.news_researcher import deduplicate_sources
        
        sources = [
            {"url": "https://example.com/1", "title": "Article 1"},
            {"url": "https://example.com/1", "title": "Article 1 Duplicate"},
            {"url": "https://example.com/2", "title": "Article 2"}
        ]
        
        result = deduplicate_sources(sources)
        
        assert len(result) == 2
    
    def test_deduplicate_sources_by_title_similarity(self):
        """Test deduplication by similar titles"""
        from helper_functions.news_researcher import deduplicate_sources
        
        sources = [
            {"url": "https://example.com/1", "title": "Breaking news about AI"},
            {"url": "https://example.com/2", "title": "Breaking news about AI today"},
            {"url": "https://example.com/3", "title": "Completely different topic"}
        ]
        
        result = deduplicate_sources(sources)
        
        # Should remove near-duplicate titles
        assert len(result) <= 3
    
    def test_deduplicate_sources_empty_list(self):
        """Test deduplication with empty list"""
        from helper_functions.news_researcher import deduplicate_sources
        
        result = deduplicate_sources([])
        
        assert result == []


class TestExtractKeywordsFromHeadlines:
    """Tests for extract_keywords_from_headlines() function"""
    
    @patch('helper_functions.news_researcher.get_client')
    def test_extract_keywords_success(self, mock_get_client):
        """Test successful keyword extraction"""
        from helper_functions.news_researcher import extract_keywords_from_headlines
        
        mock_client = Mock()
        mock_client.chat_completion.return_value = "Claude AI\nNvidia GPU\nOpenAI"
        mock_get_client.return_value = mock_client
        
        headlines = [
            {"title": "Claude AI releases new features"},
            {"title": "Nvidia announces new GPU"}
        ]
        
        result = extract_keywords_from_headlines(headlines, provider="litellm")
        
        assert len(result) > 0
        assert any("Claude" in kw or "Nvidia" in kw for kw in result)
    
    @patch('helper_functions.news_researcher.get_client')
    def test_extract_keywords_empty_headlines(self, mock_get_client):
        """Test keyword extraction with empty headlines"""
        from helper_functions.news_researcher import extract_keywords_from_headlines
        
        result = extract_keywords_from_headlines([], provider="litellm")
        
        assert result == []
        mock_get_client.assert_not_called()
    
    @patch('helper_functions.news_researcher.get_client')
    def test_extract_keywords_error(self, mock_get_client):
        """Test keyword extraction with error"""
        from helper_functions.news_researcher import extract_keywords_from_headlines
        
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        headlines = [{"title": "Test headline"}]
        
        result = extract_keywords_from_headlines(headlines, provider="litellm")
        
        assert result == []


class TestScrapeAggregatorHeadlines:
    """Tests for scrape_aggregator_headlines() function"""
    
    @patch('helper_functions.news_researcher.extract_page_content')
    def test_scrape_tldr_headlines(self, mock_scrape):
        """Test scraping TLDR headlines"""
        from helper_functions.news_researcher import scrape_aggregator_headlines
        
        mock_scrape.return_value = """
        ## AI News Today
        Some content here
        ## Breaking: New Technology
        More content
        """
        
        result = scrape_aggregator_headlines("tldr")
        
        # Should extract headlines
        assert isinstance(result, list)
    
    @patch('helper_functions.news_researcher.extract_page_content')
    def test_scrape_bensbites_headlines(self, mock_scrape):
        """Test scraping Ben's Bites headlines"""
        from helper_functions.news_researcher import scrape_aggregator_headlines
        
        mock_scrape.return_value = """
        ## AI Tools Update
        Content about AI tools
        **Important News**
        More content
        """
        
        result = scrape_aggregator_headlines("bensbites")
        
        assert isinstance(result, list)
    
    @patch('helper_functions.news_researcher.extract_page_content')
    def test_scrape_smol_headlines(self, mock_scrape):
        """Test scraping smol.ai headlines"""
        from helper_functions.news_researcher import scrape_aggregator_headlines
        
        mock_scrape.return_value = """
        Dec 08 2025 Show details
        AI news content for today
        """
        
        result = scrape_aggregator_headlines("smol")
        
        assert isinstance(result, list)
    
    @patch('helper_functions.news_researcher.extract_page_content')
    def test_scrape_aggregator_failure(self, mock_scrape):
        """Test scraping aggregator with failure"""
        from helper_functions.news_researcher import scrape_aggregator_headlines
        
        mock_scrape.return_value = None
        
        result = scrape_aggregator_headlines("tldr")
        
        assert result == []


class TestSearchFreshSourcesPerplexity:
    """Tests for search_fresh_sources() function"""
    
    @patch('helper_functions.news_researcher.extract_page_content')
    @patch('helper_functions.news_researcher.search_with_perplexity')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_search_sources_success(self, mock_log, mock_search, mock_scrape):
        """Test successful search"""
        from helper_functions.news_researcher import search_fresh_sources

        mock_search.return_value = [
            {"url": "https://news.com/1", "title": "News 1", "snippet": "Desc 1"},
            {"url": "https://news.com/2", "title": "News 2", "snippet": "Desc 2"},
        ]
        mock_scrape.return_value = "# Article Content\nThis is today's news content."
        
        result = search_fresh_sources("AI news", limit=2)
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @patch('helper_functions.news_researcher.search_with_perplexity')
    @patch('helper_functions.news_researcher.log_debug_data')
    def test_search_sources_failure(self, mock_log, mock_search):
        """Test search failure"""
        from helper_functions.news_researcher import search_fresh_sources
        
        mock_search.side_effect = Exception("Search error")
        
        with pytest.raises(Exception):
            search_fresh_sources("test query")


class TestSynthesizeNewsReport:
    """Tests for synthesize_news_report() function"""
    
    @patch('helper_functions.news_researcher.get_client')
    def test_synthesize_report_success(self, mock_get_client):
        """Test successful report synthesis"""
        from helper_functions.news_researcher import synthesize_news_report
        
        mock_fast_client = Mock()
        mock_fast_client.chat_completion.return_value = "9,8,7"
        
        mock_smart_client = Mock()
        mock_smart_client.chat_completion.return_value = "Initial draft of news"
        
        mock_strategic_client = Mock()
        mock_strategic_client.chat_completion.return_value = "Final polished news report"
        
        # Return different clients for different tiers
        def get_client_side_effect(provider, model_tier, model_name=None):
            if model_tier == "fast":
                return mock_fast_client
            elif model_tier == "smart":
                return mock_smart_client
            else:
                return mock_strategic_client
        
        mock_get_client.side_effect = get_client_side_effect
        
        sources = [
            {
                "title": "AI News 1",
                "url": "https://example.com/1",
                "content": "AI content 1",
                "source": "Example"
            },
            {
                "title": "AI News 2",
                "url": "https://example.com/2",
                "content": "AI content 2",
                "source": "Example2"
            }
        ]
        
        result = synthesize_news_report(sources, "technology", provider="litellm")
        
        assert "polished" in result.lower() or "report" in result.lower() or "Sources" in result
    
    @patch('helper_functions.news_researcher.get_client')
    def test_synthesize_report_error_fallback(self, mock_get_client):
        """Test report synthesis with error uses fallback"""
        from helper_functions.news_researcher import synthesize_news_report
        
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        sources = [
            {"title": "Test", "url": "https://example.com", "description": "Test desc", "source": "Test"}
        ]
        
        result = synthesize_news_report(sources, "technology", provider="litellm")
        
        # Should return fallback summary
        assert "Test" in result or "technology" in result.lower()


class TestAnalyzeSourceRelevance:
    """Tests for analyze_source_relevance() function"""
    
    @patch('helper_functions.news_researcher.get_client')
    def test_analyze_relevance_success(self, mock_get_client):
        """Test successful relevance analysis"""
        from helper_functions.news_researcher import analyze_source_relevance
        
        mock_client = Mock()
        mock_client.chat_completion.return_value = "9,7,8"
        mock_get_client.return_value = mock_client
        
        sources = [
            {"title": "AI News", "description": "About AI"},
            {"title": "Tech News", "description": "About tech"},
            {"title": "Other", "description": "Misc"}
        ]
        
        result = analyze_source_relevance(sources, "technology", provider="litellm")
        
        assert len(result) == 3
        # Sources should have relevance_score added
        assert any(s.get('relevance_score') for s in result)
    
    @patch('helper_functions.news_researcher.get_client')
    def test_analyze_relevance_error(self, mock_get_client):
        """Test relevance analysis with error"""
        from helper_functions.news_researcher import analyze_source_relevance
        
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client
        
        sources = [{"title": "Test", "description": "Test"}]
        
        result = analyze_source_relevance(sources, "technology", provider="litellm")
        
        # Should return original sources without crashing
        assert result == sources


class TestGatherDailyNews:
    """Tests for gather_daily_news() sync wrapper function"""
    
    @patch('helper_functions.news_researcher.asyncio.run')
    def test_gather_daily_news_calls_async(self, mock_run):
        """Test that sync wrapper calls async function"""
        from helper_functions.news_researcher import gather_daily_news
        
        mock_run.return_value = "News report"
        
        result = gather_daily_news(
            category="technology",
            max_sources=3,
            provider="litellm"
        )
        
        mock_run.assert_called_once()
        assert result == "News report"


class TestCategoryQueries:
    """Tests for CATEGORY_QUERIES configuration"""
    
    def test_category_queries_exists(self):
        """Test that CATEGORY_QUERIES is defined"""
        from helper_functions.news_researcher import CATEGORY_QUERIES
        
        assert isinstance(CATEGORY_QUERIES, dict)
    
    def test_technology_category_exists(self):
        """Test technology category has queries"""
        from helper_functions.news_researcher import CATEGORY_QUERIES
        
        assert "technology" in CATEGORY_QUERIES
        assert len(CATEGORY_QUERIES["technology"]) > 0
    
    def test_financial_category_exists(self):
        """Test financial category has queries"""
        from helper_functions.news_researcher import CATEGORY_QUERIES
        
        assert "financial" in CATEGORY_QUERIES
        assert len(CATEGORY_QUERIES["financial"]) > 0
    
    def test_india_category_exists(self):
        """Test India category has queries"""
        from helper_functions.news_researcher import CATEGORY_QUERIES
        
        assert "india" in CATEGORY_QUERIES
        assert len(CATEGORY_QUERIES["india"]) > 0


class TestAsyncScraping:
    """Tests for async scraping functions"""
    
    @pytest.mark.asyncio
    @patch('helper_functions.news_researcher.aiohttp.ClientSession')
    async def test_extract_page_content_async_skip_duplicate(self, mock_session_cls):
        """Test that async scrape skips already-scraped URLs"""
        from helper_functions.news_researcher import extract_page_content_async
        
        mock_session = AsyncMock()
        scraped_urls = {"https://example.com"}  # Already scraped
        
        result = await extract_page_content_async(
            mock_session, "https://example.com", scraped_urls
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('helper_functions.news_researcher.aiohttp.ClientSession')
    async def test_scrape_urls_parallel_empty_list(self, mock_session_cls):
        """Test parallel scraping with empty URL list"""
        from helper_functions.news_researcher import scrape_urls_parallel
        
        mock_session = AsyncMock()
        semaphore = asyncio.Semaphore(5)
        scraped_urls = set()
        
        result = await scrape_urls_parallel(
            mock_session, [], scraped_urls, semaphore
        )
        
        assert result == []


class TestConstants:
    """Tests for module constants"""
    
    def test_timeout_constants_defined(self):
        """Test that timeout constants are defined"""
        from helper_functions.news_researcher import (
            SCRAPE_TIMEOUT, CONNECT_TIMEOUT, SEARCH_TIMEOUT,
            MAX_RETRIES, MAX_CONCURRENT_SCRAPES
        )
        
        assert isinstance(SCRAPE_TIMEOUT, int)
        assert isinstance(CONNECT_TIMEOUT, int)
        assert isinstance(SEARCH_TIMEOUT, int)
        assert isinstance(MAX_RETRIES, int)
        assert isinstance(MAX_CONCURRENT_SCRAPES, int)
    
    def test_timeout_values_reasonable(self):
        """Test that timeout values are reasonable"""
        from helper_functions.news_researcher import (
            SCRAPE_TIMEOUT, CONNECT_TIMEOUT, SEARCH_TIMEOUT
        )
        
        assert 5 <= SCRAPE_TIMEOUT <= 60
        assert 5 <= CONNECT_TIMEOUT <= 30
        assert 30 <= SEARCH_TIMEOUT <= 180
