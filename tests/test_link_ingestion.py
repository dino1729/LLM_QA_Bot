"""
Tests for shared link ingestion helpers.
"""
from unittest.mock import Mock, patch

from helper_functions.link_ingestion import (
    ExtractedContent,
    LinkPreview,
    prepare_link_preview,
    save_link_preview,
    upload_to_edith,
)
from helper_functions.memory_palace_db import LessonDistillationResult


def sample_content():
    return ExtractedContent(
        text="This is a long enough sample text. " * 50,
        title="Test Content Title",
        source_type="article",
        source_ref="https://example.com/article",
        word_count=350,
    )


@patch("helper_functions.link_ingestion.extract_takeaways")
@patch("helper_functions.link_ingestion.extract_article")
@patch("helper_functions.link_ingestion.KnowledgeArchiveDB")
def test_prepare_link_preview_uses_existing_archive_entry(
    mock_db_class,
    mock_extract_article,
    mock_extract_takeaways,
):
    mock_entry = Mock()
    mock_entry.summary = "Archived summary"
    mock_entry.takeaways = (
        "1. First archived takeaway with enough detail to be useful.\n"
        "2. Second archived takeaway with enough detail to be useful.\n"
        "3. Third archived takeaway with enough detail to be useful."
    )
    mock_entry.metadata.title = "Archived Title"
    mock_entry.metadata.word_count = 1200

    mock_db = Mock()
    mock_db.get_entry_by_url.return_value = mock_entry
    mock_db_class.return_value = mock_db

    preview = prepare_link_preview(
        "https://example.com/article",
        num_takeaways=3,
        tier="fast",
    )

    assert preview.already_archived is True
    assert preview.content.title == "Archived Title"
    assert len(preview.takeaways) == 3
    mock_extract_article.assert_not_called()
    mock_extract_takeaways.assert_not_called()


@patch("helper_functions.link_ingestion.upload_to_knowledge_archive")
@patch("helper_functions.link_ingestion.upload_to_local_memory")
@patch("helper_functions.link_ingestion.upload_to_edith")
def test_save_link_preview_skips_archive_when_already_archived(
    mock_upload_to_edith,
    mock_upload_to_local,
    mock_upload_to_archive,
):
    mock_upload_to_edith.return_value = 3
    mock_upload_to_local.return_value = "Saved to Memory Palace"

    preview = LinkPreview(
        content=sample_content(),
        takeaways=[
            "First insight about learning from mistakes and iterating quickly.",
            "Second insight about the importance of surrounding yourself with smart people.",
            "Third insight about eliminating debt to preserve long-term optionality.",
        ],
        already_archived=True,
    )

    result = save_link_preview(
        preview,
        save_archive=True,
        edith_delay_seconds=0.0,
    )

    assert result.edith_count == 3
    assert result.local_memory_status == "Saved to Memory Palace"
    assert result.archive_status == "Already in Knowledge Archive (skipped)"
    mock_upload_to_archive.assert_not_called()


@patch("helper_functions.link_ingestion.distill_lesson")
@patch("helper_functions.link_ingestion.MemoryPalaceDB")
def test_upload_to_edith_skips_placeholder_distillation(mock_db_class, mock_distill):
    """A placeholder distillation ('...') must never be stored.

    Regression guard for the 2026-06-09 bulk ingest that wrote 61 lessons with
    distilled_text='...' to the store, producing an empty newsletter section.
    """
    mock_db = Mock()
    mock_db.get_few_shot_examples.return_value = []
    mock_db_class.return_value = mock_db

    # Simulate the small model emitting a placeholder distillation.
    mock_distill.return_value = LessonDistillationResult(
        distilled_text="...",
        suggested_category="observations",
        suggested_tags=[],
    )

    stored = upload_to_edith(
        ["Mastery is forged in the dark hours through deliberate practice."],
        sample_content(),
        per_item_delay_seconds=0.0,
    )

    assert stored == 0
    mock_db.add_lesson.assert_not_called()
