"""Tests for wiki.page_writer - PageWriter and page operations."""
import pytest
from pathlib import Path

from wiki.page_writer import PageMetadata, PageWriter, _atomic_write


SAMPLE_PAGE = """---
title: Compound Interest
type: concept
category: wealth-and-investing
tags: [finance, patience]
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
- [[Warren Buffett]]: Time in the market beats timing the market
- Compounding works in knowledge too, not just finance

## Connections

### Related
- [[Long-Term Thinking]] - patience enables compounding

### Tensions
- [[Active Trading]] - argues for market timing

### Applied In
- Dollar cost averaging

## Sources
- [[src-sahil-bloom-2026-03]]
- [[src-fs-brain-food-2026-02]]
- [[src-nzs-2026-01]]
"""


class TestPageMetadata:
    def test_frozen_dataclass(self):
        meta = PageMetadata(
            title="Test", entity_type="concept", category="test"
        )
        with pytest.raises(AttributeError):
            meta.title = "Changed"  # type: ignore


class TestPageWriter:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "wiki" / "concepts").mkdir(parents=True)
        return vault

    @pytest.fixture
    def writer(self, vault):
        return PageWriter(vault)

    def test_create_page(self, writer, vault):
        metadata = PageMetadata(
            title="Test Concept",
            entity_type="concept",
            category="strategy",
            tags=("strategy", "test"),
            sources=1,
            last_updated="2026-04-14",
            confidence="low",
        )
        page_path = vault / "wiki" / "concepts" / "test-concept.md"
        writer.create_page(page_path, metadata, "# Test Concept\n\n## Core Idea\nA test.")

        assert page_path.exists()
        content = page_path.read_text()
        assert "title: Test Concept" in content
        assert "type: concept" in content
        assert "## Core Idea" in content

    def test_create_page_from_full_markdown(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "full.md"
        writer.create_page_from_full_markdown(page_path, SAMPLE_PAGE)

        assert page_path.exists()
        content = page_path.read_text()
        assert "Compound Interest" in content

    def test_read_page_metadata(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "test.md"
        page_path.write_text(SAMPLE_PAGE)

        metadata = writer.read_page_metadata(page_path)
        assert metadata is not None
        assert metadata.title == "Compound Interest"
        assert metadata.entity_type == "concept"
        assert metadata.sources == 3
        assert metadata.confidence == "medium"
        assert "Warren Buffett" in metadata.people

    def test_read_page_body(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "test.md"
        page_path.write_text(SAMPLE_PAGE)

        body = writer.read_page_body(page_path)
        assert "## Core Idea" in body
        assert "---" not in body  # Frontmatter stripped

    def test_get_one_line_summary(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "test.md"
        page_path.write_text(SAMPLE_PAGE)

        summary = writer.get_one_line_summary(page_path)
        assert "exponential growth" in summary.lower()
        assert len(summary) <= 200

    def test_read_nonexistent_page(self, writer, vault):
        assert writer.read_page_metadata(vault / "nope.md") is None
        assert writer.read_page_body(vault / "nope.md") == ""

    def test_update_page_with_xml_diff(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "test.md"
        page_path.write_text(SAMPLE_PAGE)

        xml_diff = """
        <wiki_update>
          <update_frontmatter>
            <sources>INCREMENT by 1</sources>
            <last_updated>2026-04-15</last_updated>
          </update_frontmatter>
          <section name="Key Insights">
            <add_item>New insight about [[Patience]] from latest article</add_item>
          </section>
          <section name="Sources">
            <add_item>[[src-new-source-2026-04]]</add_item>
          </section>
        </wiki_update>
        """

        result = writer.update_page(page_path, xml_diff)
        assert result is True

        updated = page_path.read_text()
        assert "sources: 4" in updated  # Incremented from 3
        assert "New insight about [[Patience]]" in updated
        assert "[[src-new-source-2026-04]]" in updated

    def test_update_page_no_xml_block(self, writer, vault):
        page_path = vault / "wiki" / "concepts" / "test.md"
        page_path.write_text(SAMPLE_PAGE)

        result = writer.update_page(page_path, "no xml here")
        assert result is False  # No changes

    def test_update_page_nonexistent(self, writer, vault):
        result = writer.update_page(vault / "nope.md", "<wiki_update></wiki_update>")
        assert result is False

    def test_render_frontmatter(self, writer):
        metadata = PageMetadata(
            title="Test",
            entity_type="concept",
            category="strategy",
            tags=("a", "b"),
            sources=2,
            last_updated="2026-04-14",
            confidence="high",
        )
        fm = writer.render_frontmatter(metadata)
        assert fm.startswith("---\n")
        assert fm.endswith("---\n")
        assert "title: Test" in fm


class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        path = tmp_path / "test.md"
        _atomic_write(path, "hello world")
        assert path.read_text() == "hello world"

    def test_no_tmp_file_remains(self, tmp_path):
        path = tmp_path / "test.md"
        _atomic_write(path, "hello")
        assert not (tmp_path / "test.tmp").exists()
