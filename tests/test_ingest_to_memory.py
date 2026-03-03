"""
Tests for the unified content ingestion script.

Tests cover:
- Input type detection (YouTube, URL, file)
- YouTube video ID extraction
- Content extraction from YouTube, articles, and files
- Takeaway extraction and preamble filtering
- Upload to EDITH, Local Memory Palace, and Knowledge Archive
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from scripts.ingest_to_memory import (
    InputType,
    ExtractedContent,
    detect_input_type,
    _extract_video_id,
    _get_video_title,
    extract_youtube,
    extract_article,
    extract_file,
    extract_takeaways,
    upload_to_edith,
    upload_to_local_memory,
    upload_to_knowledge_archive,
)


# --- Fixtures ---

@pytest.fixture
def sample_content():
    """Create a sample ExtractedContent for testing."""
    return ExtractedContent(
        text="This is a long enough sample text. " * 50,
        title="Test Content Title",
        source_type="article",
        source_ref="https://example.com/article",
        word_count=350,
    )


@pytest.fixture
def sample_takeaways():
    """Return sample takeaways for upload tests."""
    return [
        "First insight about learning from mistakes and iterating quickly.",
        "Second insight about the importance of surrounding yourself with smart people.",
        "Third insight about eliminating debt to preserve long-term optionality.",
    ]


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for file extraction tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document with enough content to extract takeaways. " * 20)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_empty_file():
    """Create a temporary empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_md_file():
    """Create a temporary markdown file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nSome markdown content with useful information. " * 15)
        path = f.name
    yield path
    os.unlink(path)


# --- Input Type Detection ---

class TestDetectInputType:
    """Tests for detect_input_type()."""

    def test_youtube_full_url(self):
        assert detect_input_type("https://www.youtube.com/watch?v=abc123") == InputType.YOUTUBE

    def test_youtube_short_url(self):
        assert detect_input_type("https://youtu.be/abc123") == InputType.YOUTUBE

    def test_youtube_with_params(self):
        assert detect_input_type("https://youtube.com/watch?v=abc&t=120") == InputType.YOUTUBE

    def test_https_url(self):
        assert detect_input_type("https://paulgraham.com/ds.html") == InputType.URL

    def test_http_url(self):
        assert detect_input_type("http://example.com/article") == InputType.URL

    def test_local_file(self, temp_text_file):
        assert detect_input_type(temp_text_file) == InputType.FILE

    def test_url_without_scheme(self):
        assert detect_input_type("paulgraham.com/ds.html") == InputType.URL

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Cannot determine input type"):
            detect_input_type("not_a_valid_input")


# --- YouTube Video ID Extraction ---

class TestExtractVideoId:
    """Tests for _extract_video_id()."""

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/0-LAT4HjWPo") == "0-LAT4HjWPo"

    def test_full_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_params(self):
        assert _extract_video_id("https://youtube.com/watch?v=abc123&t=60") == "abc123"

    def test_short_url_with_params(self):
        assert _extract_video_id("https://youtu.be/abc123?si=tracking") == "abc123"

    def test_bare_id_passthrough(self):
        assert _extract_video_id("abc123") == "abc123"


# --- Video Title ---

class TestGetVideoTitle:
    """Tests for _get_video_title()."""

    def test_fallback_when_ytdlp_unavailable(self):
        """Title falls back gracefully when yt-dlp is not installed."""
        # yt-dlp is not installed in test env, so this tests the real fallback path
        title = _get_video_title("https://youtu.be/abc", "abc")
        assert title == "YouTube video abc"


# --- YouTube Extraction ---

class TestExtractYoutube:
    """Tests for extract_youtube()."""

    def test_successful_extraction(self):
        mock_api = Mock()
        snippet1 = Mock(text="Hello world")
        snippet2 = Mock(text="this is a test")
        mock_api.fetch.return_value = [snippet1, snippet2]

        mock_yt_module = Mock()
        mock_yt_module.YouTubeTranscriptApi.return_value = mock_api

        with patch("scripts.ingest_to_memory._get_video_title", return_value="Test Video Title"), \
             patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            result = extract_youtube("https://youtu.be/abc123")

        assert result.title == "Test Video Title"
        assert result.text == "Hello world this is a test"
        assert result.source_type == "video"
        assert result.source_ref == "https://youtu.be/abc123"
        assert result.word_count == 6

    def test_transcript_api_failure(self):
        mock_api = Mock()
        mock_api.fetch.side_effect = Exception("No transcript available")

        mock_yt_module = Mock()
        mock_yt_module.YouTubeTranscriptApi.return_value = mock_api

        with patch("scripts.ingest_to_memory._get_video_title", return_value="Test Video"), \
             patch.dict("sys.modules", {"youtube_transcript_api": mock_yt_module}):
            with pytest.raises(Exception, match="No transcript available"):
                extract_youtube("https://youtu.be/abc123")


# --- Article Extraction ---

class TestExtractArticle:
    """Tests for extract_article()."""

    def test_firecrawl_success(self):
        from helper_functions.knowledge_archive_scraper import ScrapedContent
        mock_scraped = ScrapedContent(
            content="Article body text. " * 50,
            title="Great Article",
            author=None,
            publish_date=None,
            word_count=150,
        )

        with patch("helper_functions.knowledge_archive_scraper.scrape_article", return_value=mock_scraped):
            result = extract_article("https://example.com/article")

        assert result.title == "Great Article"
        assert result.source_type == "article"
        assert result.word_count == 150

    def test_firecrawl_too_short_falls_through(self):
        """Firecrawl result under 75 words triggers fallback."""
        from helper_functions.knowledge_archive_scraper import ScrapedContent
        mock_scraped = ScrapedContent(
            content="Too short.",
            title="Short",
            author=None,
            publish_date=None,
            word_count=2,
        )

        mock_resp = Mock()
        mock_resp.text = "<html><head><title>Fallback Title</title></head><body><p>Body text. " + "word " * 100 + "</p></body></html>"

        # newspaper may not be installed; inject a fake module so patch() can resolve it
        mock_newspaper = MagicMock()
        mock_newspaper.Article.side_effect = Exception("no newspaper")

        with patch("helper_functions.knowledge_archive_scraper.scrape_article", return_value=mock_scraped), \
             patch.dict("sys.modules", {"newspaper": mock_newspaper}), \
             patch("requests.get", return_value=mock_resp):
            result = extract_article("https://example.com/short")

        assert result.title == "Fallback Title"
        assert result.source_type == "article"

    def test_all_methods_fail(self):
        """RuntimeError when all extraction methods fail."""
        # newspaper may not be installed; inject a fake module so patch() can resolve it
        mock_newspaper = MagicMock()
        mock_newspaper.Article.side_effect = Exception("fail")

        with patch("helper_functions.knowledge_archive_scraper.scrape_article", side_effect=Exception("fail")), \
             patch.dict("sys.modules", {"newspaper": mock_newspaper}), \
             patch("requests.get", side_effect=Exception("fail")):
            with pytest.raises(RuntimeError, match="All article extraction methods failed"):
                extract_article("https://example.com/broken")


# --- File Extraction ---

class TestExtractFile:
    """Tests for extract_file()."""

    def test_txt_file(self, temp_text_file):
        result = extract_file(temp_text_file)
        assert result.source_type == "file"
        assert result.word_count > 0
        assert "test document" in result.text.lower()

    def test_md_file(self, temp_md_file):
        result = extract_file(temp_md_file)
        assert result.source_type == "file"
        assert "markdown content" in result.text.lower()

    def test_title_from_filename(self, temp_text_file):
        result = extract_file(temp_text_file)
        # Title should be derived from filename stem
        assert result.title  # Not empty
        assert isinstance(result.title, str)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_file("/nonexistent/path/file.txt")

    def test_empty_file(self, temp_empty_file):
        with pytest.raises(ValueError, match="No text content extracted"):
            extract_file(temp_empty_file)

    def test_pdf_extraction(self):
        """PDF extraction uses pypdf."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content from page one. " * 20
        mock_reader = Mock()
        mock_reader.pages = [mock_page]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            path = f.name

        try:
            with patch("pypdf.PdfReader", return_value=mock_reader):
                result = extract_file(path)
            assert result.word_count > 0
            assert "PDF content" in result.text
        finally:
            os.unlink(path)


# --- Takeaway Extraction ---

class TestExtractTakeaways:
    """Tests for extract_takeaways() including preamble filtering."""

    @patch("scripts.ingest_to_memory.get_client")
    def test_parses_numbered_list(self, mock_get_client, sample_content):
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1. **First insight.** Learning from mistakes is essential for growth.\n"
            "2. **Second insight.** Surround yourself with smart people who challenge you.\n"
            "3. **Third insight.** Eliminate debt to maintain freedom of choice.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 3, "fast")

        assert len(result) == 3
        assert "Learning from mistakes" in result[0]
        assert "Surround yourself" in result[1]
        assert "Eliminate debt" in result[2]

    @patch("scripts.ingest_to_memory.get_client")
    def test_parses_markdown_wrapped_numbered_headings(self, mock_get_client, sample_content):
        """Handles responses where numbering is wrapped in markdown bold."""
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "Here are key takeaways:\n"
            "**1. First insight title**\n"
            "The first insight body is detailed enough to pass filtering.\n"
            "**2. Second insight title**\n"
            "The second insight body is also substantial and actionable.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 2, "fast")

        assert len(result) == 2
        assert "First insight title" in result[0]
        assert "second insight body" in result[1].lower()

    @patch("scripts.ingest_to_memory.get_client")
    def test_parses_parenthesized_numbering(self, mock_get_client, sample_content):
        """Handles alternate list formats like `1)`."""
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1) First point with enough supporting detail to be useful in practice.\n"
            "2) Second point with enough supporting detail to be useful in practice.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 2, "fast")

        assert len(result) == 2
        assert "First point" in result[0]
        assert "Second point" in result[1]

    @patch("scripts.ingest_to_memory.get_client")
    def test_strips_markdown_bold(self, mock_get_client, sample_content):
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1. **Bold title.** The actual insight content goes here with enough words.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 1, "fast")

        assert len(result) == 1
        assert "**" not in result[0]
        assert "Bold title." in result[0]

    @patch("scripts.ingest_to_memory.get_client")
    def test_filters_preamble_lines(self, mock_get_client, sample_content):
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1. Here are the key takeaways from the transcript:\n"
            "2. **Real insight.** Debt destroys freedom and forces short-term optimization at the expense of growth.\n"
            "3. **Another insight.** Younger people serve as early warning systems for future trends.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 2, "fast")

        assert len(result) == 2
        # Preamble "Here are the key takeaways..." should be filtered out
        assert not any("here are" in t.lower() for t in result)

    @patch("scripts.ingest_to_memory.get_client")
    def test_filters_based_on_preamble(self, mock_get_client, sample_content):
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1. Based on the video, these are the main points from the speaker:\n"
            "2. **Insight.** Real content that should be kept because it has enough substance.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 1, "fast")

        assert len(result) == 1
        assert "Real content" in result[0]

    @patch("scripts.ingest_to_memory.get_client")
    def test_multiline_takeaways(self, mock_get_client, sample_content):
        """Takeaways split across multiple lines are joined correctly."""
        mock_client = Mock()
        mock_client.chat_completion.return_value = (
            "1. **First insight.** This is the start of a long takeaway\n"
            "   that continues on the next line with more details and evidence.\n"
            "2. **Second insight.** Another standalone point about career decisions.\n"
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 2, "fast")

        assert len(result) == 2
        assert "continues on the next line" in result[0]

    @patch("scripts.ingest_to_memory.get_client")
    def test_truncates_long_content(self, mock_get_client):
        """Content longer than 15000 chars is truncated."""
        long_content = ExtractedContent(
            text="word " * 20000,
            title="Long Doc",
            source_type="file",
            source_ref="/tmp/long.txt",
            word_count=20000,
        )
        mock_client = Mock()
        mock_client.chat_completion.return_value = "1. **Insight.** A takeaway that meets the minimum length requirement for filtering.\n"
        mock_get_client.return_value = mock_client

        extract_takeaways(long_content, 1, "fast")

        # Verify the prompt content was truncated
        call_args = mock_client.chat_completion.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert len(prompt_content) < 20000 * 5  # Rough check it was truncated

    @patch("scripts.ingest_to_memory.get_client")
    def test_respects_num_limit(self, mock_get_client, sample_content):
        """Returns at most num_takeaways items even if LLM returns more."""
        mock_client = Mock()
        mock_client.chat_completion.return_value = "\n".join(
            [f"{i}. Takeaway number {i} with sufficient length to pass the forty character filter." for i in range(1, 10)]
        )
        mock_get_client.return_value = mock_client

        result = extract_takeaways(sample_content, 3, "fast")

        assert len(result) <= 3


# --- Upload to EDITH ---

class TestUploadToEdith:
    """Tests for upload_to_edith()."""

    @patch("scripts.ingest_to_memory.MemoryPalaceDB")
    def test_skip_distill_stores_raw(self, mock_db_class, sample_content, sample_takeaways):
        mock_db = Mock()
        mock_db.add_lesson.return_value = "lesson-id-123"
        mock_db_class.return_value = mock_db

        count = upload_to_edith(sample_takeaways, sample_content, skip_distill=True)

        assert count == 3
        assert mock_db.add_lesson.call_count == 3

        # Verify lesson metadata
        first_call = mock_db.add_lesson.call_args_list[0]
        lesson = first_call[0][0]
        assert lesson.metadata.source == "manual"
        assert lesson.metadata.distilled_by_model == "raw"
        assert sample_content.source_type in lesson.metadata.tags

    @patch("scripts.ingest_to_memory.distill_lesson")
    @patch("scripts.ingest_to_memory.MemoryPalaceDB")
    def test_with_distillation(self, mock_db_class, mock_distill, sample_content, sample_takeaways):
        mock_db = Mock()
        mock_db.add_lesson.return_value = "lesson-id-456"
        mock_db.get_few_shot_examples.return_value = ["Example 1", "Example 2"]
        mock_db_class.return_value = mock_db

        mock_distill.return_value = Mock(
            distilled_text="Distilled insight here.",
            suggested_category="psychology",
            suggested_tags=["learning"],
        )

        count = upload_to_edith(sample_takeaways, sample_content, skip_distill=False)

        assert count == 3
        assert mock_distill.call_count == 3
        assert mock_db.get_few_shot_examples.call_count == 3

    @patch("scripts.ingest_to_memory.MemoryPalaceDB")
    def test_handles_individual_failures(self, mock_db_class, sample_content, sample_takeaways):
        """Failures on individual lessons don't stop the batch."""
        mock_db = Mock()
        mock_db.add_lesson.side_effect = [
            "id-1",
            Exception("embedding failed"),
            "id-3",
        ]
        mock_db_class.return_value = mock_db

        count = upload_to_edith(sample_takeaways, sample_content, skip_distill=True)

        assert count == 2  # 2 of 3 succeeded

    @patch("scripts.ingest_to_memory.MemoryPalaceDB")
    def test_skip_distill_skips_source_referential_takeaway(self, mock_db_class, sample_content):
        """Skip-distill mode rejects obvious source-referential preamble entries."""
        mock_db = Mock()
        mock_db.add_lesson.return_value = "lesson-id"
        mock_db_class.return_value = mock_db
        takeaways = [
            "Based on the video, these are the main points from the speaker:",
            "Focused feedback loops reveal weak assumptions before they get expensive.",
        ]

        count = upload_to_edith(takeaways, sample_content, skip_distill=True)

        assert count == 1
        assert mock_db.add_lesson.call_count == 1
        stored = mock_db.add_lesson.call_args[0][0]
        assert "speaker" not in stored.distilled_text.lower()

    @patch("scripts.ingest_to_memory.distill_lesson")
    @patch("scripts.ingest_to_memory.MemoryPalaceDB")
    def test_distillation_skips_non_objective_output(self, mock_db_class, mock_distill, sample_content):
        """Distilled entries that still contain source framing are skipped."""
        mock_db = Mock()
        mock_db.add_lesson.return_value = "lesson-id"
        mock_db.get_few_shot_examples.return_value = ["Example 1"]
        mock_db_class.return_value = mock_db

        mock_distill.side_effect = [
            Mock(
                distilled_text="The author observes that excess resources lead to waste.",
                suggested_category="psychology",
                suggested_tags=["behavior"],
            ),
            Mock(
                distilled_text="Excess resources often lead to waste when constraints are weak.",
                suggested_category="psychology",
                suggested_tags=["behavior"],
            ),
        ]

        count = upload_to_edith(
            ["Takeaway 1", "Takeaway 2"],
            sample_content,
            skip_distill=False,
        )

        assert count == 1
        assert mock_db.add_lesson.call_count == 1


# --- Upload to Local Memory ---

class TestUploadToLocalMemory:
    """Tests for upload_to_local_memory()."""

    @patch("scripts.ingest_to_memory.save_memory")
    def test_combines_takeaways(self, mock_save, sample_content, sample_takeaways):
        mock_save.return_value = "Saved to Memory Palace"

        result = upload_to_local_memory(sample_content, sample_takeaways)

        assert result == "Saved to Memory Palace"
        mock_save.assert_called_once()

        call_kwargs = mock_save.call_args[1]
        assert call_kwargs["title"] == "Test Content Title"
        assert call_kwargs["source_type"] == "article"
        assert "1." in call_kwargs["content"]
        assert "2." in call_kwargs["content"]
        assert "3." in call_kwargs["content"]


# --- Upload to Knowledge Archive ---

class TestUploadToKnowledgeArchive:
    """Tests for upload_to_knowledge_archive()."""

    @patch("helper_functions.knowledge_archive_db.KnowledgeArchiveDB")
    @patch("helper_functions.knowledge_archive_processor.process_article")
    def test_successful_archive(self, mock_process, mock_db_class, sample_content):
        mock_db = Mock()
        mock_db.url_exists.return_value = False
        mock_db_class.return_value = mock_db

        mock_entry = Mock()
        mock_entry.metadata.title = "Test Article"
        mock_entry.metadata.tags = ["tech", "ai"]
        mock_process.return_value = mock_entry

        upload_to_knowledge_archive(sample_content)

        mock_db.url_exists.assert_called_once_with(sample_content.source_ref)
        mock_db.add_entry.assert_called_once_with(mock_entry)

    @patch("helper_functions.knowledge_archive_db.KnowledgeArchiveDB")
    def test_skips_duplicate(self, mock_db_class, sample_content):
        mock_db = Mock()
        mock_db.url_exists.return_value = True
        mock_db_class.return_value = mock_db

        upload_to_knowledge_archive(sample_content)

        mock_db.add_entry.assert_not_called()


# --- ExtractedContent dataclass ---

class TestExtractedContent:
    """Tests for the ExtractedContent dataclass."""

    def test_creation(self):
        content = ExtractedContent(
            text="Hello world",
            title="Test",
            source_type="file",
            source_ref="/tmp/test.txt",
            word_count=2,
        )
        assert content.text == "Hello world"
        assert content.word_count == 2

    def test_source_types(self):
        for stype in ("video", "article", "file"):
            content = ExtractedContent(
                text="text", title="t", source_type=stype,
                source_ref="ref", word_count=1,
            )
            assert content.source_type == stype
