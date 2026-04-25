"""Tests for wiki.telegram_commands and EDITH integration hooks."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from wiki.telegram_commands import (
    _search_wiki_pages,
    _run_lint,
    _compute_stats,
    trigger_wiki_ingest,
    trigger_wiki_auto_file,
)
from wiki.page_writer import PageWriter


CONCEPT_PAGE = """---
title: Compound Interest
type: concept
category: economics
tags: [finance, patience, long-term]
sources: 3
people: [Warren Buffett]
last_updated: '2026-04-14'
confidence: medium
wiki_version: 1
---

# Compound Interest

## Core Idea
The exponential growth of returns reinvested over time.

## Key Insights
- [[Warren Buffett]]: Time beats timing

## Connections

### Related
- [[Long-Term Thinking]]

## Sources
- [[src-a]]
"""

PERSON_PAGE = """---
title: Warren Buffett
type: person
category: leadership
tags: [investing, patience]
sources: 2
people: []
last_updated: '2026-04-14'
confidence: medium
wiki_version: 1
---

# Warren Buffett

## Overview
American investor known for value investing.

## Key Ideas
- [[Compound Interest]] as the eighth wonder
- [[Patience]] in investing

## Sources
- [[src-b]]
"""


class TestSearchWikiPages:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries"]:
            (vault / d).mkdir(parents=True)
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(CONCEPT_PAGE)
        (vault / "wiki" / "entities" / "warren-buffett.md").write_text(PERSON_PAGE)
        return vault

    @patch("wiki.telegram_commands._get_vault_root")
    def test_search_by_title(self, mock_root, vault):
        mock_root.return_value = vault
        results = _search_wiki_pages("compound interest")
        assert len(results) >= 1
        assert results[0][0] == "Compound Interest"

    @patch("wiki.telegram_commands._get_vault_root")
    def test_search_by_tag(self, mock_root, vault):
        mock_root.return_value = vault
        results = _search_wiki_pages("patience")
        assert len(results) >= 1
        # Both pages have "patience" tag
        titles = {r[0] for r in results}
        assert "Compound Interest" in titles or "Warren Buffett" in titles

    @patch("wiki.telegram_commands._get_vault_root")
    def test_search_no_results(self, mock_root, vault):
        mock_root.return_value = vault
        results = _search_wiki_pages("quantum mechanics")
        assert len(results) == 0

    @patch("wiki.telegram_commands._get_vault_root")
    def test_search_disabled_wiki(self, mock_root):
        mock_root.return_value = None
        results = _search_wiki_pages("anything")
        assert results == []


class TestRunLint:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries", "raw/refs/legacy-schema"]:
            (vault / d).mkdir(parents=True)
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(CONCEPT_PAGE)
        # Write empty entity registry
        (vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json").write_text("{}")
        return vault

    @patch("wiki.telegram_commands._get_vault_root")
    def test_lint_returns_report(self, mock_root, vault):
        mock_root.return_value = vault
        report = _run_lint()
        assert "Lint Report" in report
        assert "Pages checked:" in report

    @patch("wiki.telegram_commands._get_vault_root")
    def test_lint_disabled(self, mock_root):
        mock_root.return_value = None
        result = _run_lint()
        assert "not enabled" in result


class TestComputeStats:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries", "outputs/queries", "raw/articles"]:
            (vault / d).mkdir(parents=True)
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(CONCEPT_PAGE)
        (vault / "wiki" / "entities" / "warren-buffett.md").write_text(PERSON_PAGE)
        return vault

    @patch("wiki.telegram_commands._get_vault_root")
    def test_stats_returns_formatted(self, mock_root, vault):
        mock_root.return_value = vault
        stats = _compute_stats()
        assert "Wiki Stats" in stats
        assert "Concepts: 1" in stats
        assert "Entities: 1" in stats


class TestTriggerWikiIngest:
    @patch("wiki.get_wiki_builder")
    def test_ingest_calls_builder(self, mock_get_builder):
        mock_builder = MagicMock()
        mock_builder.process_ingest.return_value = MagicMock(
            success=True, pages_created=["test"], pages_updated=[], errors=[]
        )
        mock_get_builder.return_value = mock_builder

        trigger_wiki_ingest({"text": "test", "title": "Test"}, "ingest_link")

        mock_builder.process_ingest.assert_called_once()

    @patch("wiki.get_wiki_builder")
    def test_ingest_handles_disabled(self, mock_get_builder):
        mock_get_builder.return_value = None
        # Should not raise
        trigger_wiki_ingest({"text": "test"}, "ingest_link")

    @patch("wiki.get_wiki_builder")
    def test_ingest_handles_errors_silently(self, mock_get_builder):
        mock_builder = MagicMock()
        mock_builder.process_ingest.side_effect = RuntimeError("test error")
        mock_get_builder.return_value = mock_builder

        # Should not raise
        trigger_wiki_ingest({"text": "test"}, "ingest_link")


class TestTriggerWikiAutoFile:
    @patch("wiki.get_wiki_builder")
    def test_auto_file_calls_process_answer(self, mock_get_builder):
        mock_builder = MagicMock()
        mock_builder.process_answer.return_value = MagicMock(
            success=True, pages_created=["test-analysis"]
        )
        mock_get_builder.return_value = mock_builder

        trigger_wiki_auto_file(
            "What is compound interest?",
            "Compound interest is...",
            ["source1", "source2", "source3"],
        )

        mock_builder.process_answer.assert_called_once_with(
            "What is compound interest?",
            "Compound interest is...",
            ["source1", "source2", "source3"],
        )

    @patch("wiki.get_wiki_builder")
    def test_auto_file_handles_disabled(self, mock_get_builder):
        mock_get_builder.return_value = None
        # Should not raise
        trigger_wiki_auto_file("q", "a", ["s1", "s2", "s3"])
