"""Tests for wiki.indexer - WikiIndexer index and log operations."""
import pytest
from pathlib import Path

from wiki.indexer import WikiIndexer, IndexEntry, LogEntry
from wiki.page_writer import PageWriter


SAMPLE_PAGE_A = """---
title: Compound Interest
type: concept
category: economics
tags: [finance]
sources: 3
people: [Warren Buffett]
last_updated: '2026-04-14'
confidence: medium
wiki_version: 1
---

# Compound Interest

## Core Idea
The exponential growth of returns over time.

## Key Insights
- [[Warren Buffett]]: Patience is key

## Connections

### Related
- [[Long-Term Thinking]]

## Sources
- [[src-a]]
- [[src-b]]
- [[src-c]]
"""

SAMPLE_PAGE_B = """---
title: Long-Term Thinking
type: concept
category: strategy
tags: [patience]
sources: 1
people: []
last_updated: '2026-04-14'
confidence: low
wiki_version: 1
---

# Long-Term Thinking

## Core Idea
Planning with a long time horizon beats short-term optimization.

## Key Insights
- Connected to [[Compound Interest]]

## Sources
- [[src-x]]
"""


class TestWikiIndexer:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries", "log", "outputs/queries"]:
            (vault / d).mkdir(parents=True)
        return vault

    @pytest.fixture
    def populated_vault(self, vault):
        """Vault with two sample concept pages."""
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(SAMPLE_PAGE_A)
        (vault / "wiki" / "concepts" / "long-term-thinking.md").write_text(SAMPLE_PAGE_B)
        return vault

    @pytest.fixture
    def indexer(self, vault):
        return WikiIndexer(vault)

    def test_update_index_empty_vault(self, indexer, vault):
        indexer.update_index()
        assert (vault / "wiki" / "index.md").exists()
        content = (vault / "wiki" / "index.md").read_text()
        assert "*(none yet)*" in content

    def test_update_index_with_pages(self, vault):
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(SAMPLE_PAGE_A)
        (vault / "wiki" / "concepts" / "long-term-thinking.md").write_text(SAMPLE_PAGE_B)
        indexer = WikiIndexer(vault)

        indexer.update_index()

        content = (vault / "wiki" / "index.md").read_text()
        assert "[[concepts/compound-interest|Compound Interest]]" in content
        assert "Compound Interest" in content
        assert "Long-Term Thinking" in content

    def test_append_log(self, indexer, vault):
        entry = LogEntry(
            timestamp="2026-04-14T10:00:00Z",
            operation="ingest_link",
            pages_affected="compound-interest.md",
            details="test",
        )
        indexer.append_log(entry)

        content = (vault / "log" / "20260414.md").read_text()
        assert "ingest_link" in content
        assert "compound-interest" in content

    def test_append_log_multiple(self, indexer, vault):
        for i in range(3):
            indexer.append_log(LogEntry(
                timestamp=f"2026-04-14T10:0{i}:00Z",
                operation=f"op_{i}",
                pages_affected=f"page_{i}",
                details="",
            ))

        content = (vault / "log" / "20260414.md").read_text()
        assert "op_0" in content
        assert "op_1" in content
        assert "op_2" in content

    def test_make_log_entry(self, indexer):
        entry = indexer.make_log_entry(
            operation="ingest_link",
            pages=["compound-interest", "patience"],
            details="test source",
        )
        assert entry.operation == "ingest_link"
        assert "compound-interest" in entry.pages_affected

    def test_make_log_entry_truncates(self, indexer):
        pages = [f"page-{i}" for i in range(10)]
        entry = indexer.make_log_entry("test", pages)
        assert "+5 more" in entry.pages_affected

    def test_update_connections(self, populated_vault):
        indexer = WikiIndexer(populated_vault)
        indexer.update_connections()

        content = (populated_vault / "wiki" / "concepts" / "knowledge-connections.md").read_text()
        assert "Knowledge Connections" in content

    def test_compute_stats(self, populated_vault):
        indexer = WikiIndexer(populated_vault)
        stats = indexer._compute_stats()
        assert stats["concepts"] == 2
        assert stats["total_pages"] == 2
        assert stats["total_links"] > 0

    def test_build_link_graph(self, populated_vault):
        indexer = WikiIndexer(populated_vault)
        graph = indexer._build_link_graph()
        assert "compound-interest" in graph
        # compound-interest links to Long-Term Thinking and Warren Buffett
        assert len(graph["compound-interest"]) >= 1
