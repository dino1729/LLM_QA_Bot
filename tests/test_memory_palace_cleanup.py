"""
Tests for Memory Palace lesson audit and cleanup workflow.
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from helper_functions.memory_palace_db import (
    Lesson,
    LessonCategory,
    LessonMetadata,
    MemoryPalaceDB,
    is_objective_lesson_text,
)
from scripts.audit_memory_palace_lessons import audit_lessons


@pytest.fixture
def db_with_mock_embeddings(tmp_path):
    """MemoryPalaceDB with mocked embeddings for offline tests."""
    index_folder = tmp_path / "lessons_index"
    with patch("helper_functions.memory_palace_db.get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.get_embedding.return_value = [0.1] * 1536
        mock_client.chat_completion.return_value = json.dumps(
            {"distilled_text": "Objective rewritten lesson for validation."}
        )
        mock_get_client.return_value = mock_client
        db = MemoryPalaceDB(index_folder=str(index_folder))
        yield db


def _add_lessons(db: MemoryPalaceDB) -> tuple[str, str]:
    bad = Lesson(
        distilled_text="The author predicts that domain-specific chatbots will emerge as a game-changing technology.",
        metadata=LessonMetadata(
            category=LessonCategory.TECHNOLOGY,
            source="migration:Little Observations.pdf",
            original_input="raw input",
            distilled_by_model="migration",
        ),
    )
    good = Lesson(
        distilled_text="Domain-specific chatbots can become game-changing tools in constrained workflows.",
        metadata=LessonMetadata(
            category=LessonCategory.TECHNOLOGY,
            source="manual",
            original_input="raw input",
            distilled_by_model="test-model",
        ),
    )
    db.add_lesson(bad)
    db.add_lesson(good)
    return bad.id, good.id


def _read_journal_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def test_audit_dry_run_flags_without_mutation(db_with_mock_embeddings, tmp_path):
    db = db_with_mock_embeddings
    bad_id, _ = _add_lessons(db)
    journal = tmp_path / "audit_dry_run.jsonl"

    stats = audit_lessons(
        db,
        apply=False,
        include_forgotten=False,
        source_filter=None,
        max_rewrites=50,
        batch_size=10,
        sleep_ms=0,
        provider=None,
        model_tier=None,
        model_name="test-model",
        journal_path=journal,
        resume_journal=None,
    )

    assert stats["total_scanned"] == 2
    assert stats["flagged"] == 1
    assert stats["rewritten"] == 0
    assert stats["unchanged"] == 1

    rows = _read_journal_rows(journal)
    statuses = {row["status"] for row in rows}
    assert "flagged" in statuses
    assert "unchanged" in statuses

    bad = db.get_lesson_by_id(bad_id)
    assert bad is not None
    assert bad.distilled_text.startswith("The author predicts")


def test_audit_apply_rewrites_flagged_entries(db_with_mock_embeddings, tmp_path):
    db = db_with_mock_embeddings
    bad_id, _ = _add_lessons(db)
    journal = tmp_path / "audit_apply.jsonl"

    stats = audit_lessons(
        db,
        apply=True,
        include_forgotten=False,
        source_filter=None,
        max_rewrites=50,
        batch_size=1,
        sleep_ms=0,
        provider=None,
        model_tier=None,
        model_name="cleanup-model",
        journal_path=journal,
        resume_journal=None,
    )

    assert stats["flagged"] == 1
    assert stats["rewritten"] == 1
    assert stats["errors"] == 0

    rewritten = db.get_lesson_by_id(bad_id)
    assert rewritten is not None
    assert "author" not in rewritten.distilled_text.lower()
    assert rewritten.metadata.distilled_by_model == "cleanup-model"
    assert "auto-cleanup" in rewritten.metadata.tags
    ok, reasons = is_objective_lesson_text(rewritten.distilled_text)
    assert ok is True, reasons

    rows = _read_journal_rows(journal)
    rewritten_rows = [row for row in rows if row["status"] == "rewritten"]
    assert len(rewritten_rows) == 1
    assert rewritten_rows[0]["lesson_id"] == bad_id
    assert rewritten_rows[0].get("after_text")


def test_audit_resume_journal_skips_processed_entries(db_with_mock_embeddings, tmp_path):
    db = db_with_mock_embeddings
    _add_lessons(db)
    first_journal = tmp_path / "audit_first.jsonl"

    first_stats = audit_lessons(
        db,
        apply=True,
        include_forgotten=False,
        source_filter=None,
        max_rewrites=50,
        batch_size=10,
        sleep_ms=0,
        provider=None,
        model_tier=None,
        model_name="cleanup-model",
        journal_path=first_journal,
        resume_journal=None,
    )
    assert first_stats["rewritten"] == 1

    second_journal = tmp_path / "audit_second.jsonl"
    second_stats = audit_lessons(
        db,
        apply=True,
        include_forgotten=False,
        source_filter=None,
        max_rewrites=50,
        batch_size=10,
        sleep_ms=0,
        provider=None,
        model_tier=None,
        model_name="cleanup-model",
        journal_path=second_journal,
        resume_journal=first_journal,
    )

    assert second_stats["rewritten"] == 0
    assert second_stats["resumed_skips"] >= 1
