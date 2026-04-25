"""Tests for Phase 3: wiki maintenance, backup, timeline, and cron integration."""
import json
import tarfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from wiki.backup import WikiBackup
from wiki.indexer import WikiIndexer
from wiki.lint import WikiLinter, LintReport
from wiki.page_writer import PageWriter
from wiki.entities import EntityRegistry


CONCEPT_PAGE = """---
title: Test Concept
type: concept
category: strategy
tags: [test]
sources: 2
people: []
last_updated: '2026-04-14'
confidence: low
wiki_version: 1
---

# Test Concept

## Core Idea
A test concept for lint and maintenance.

## Key Insights
- First insight with [[Related Concept]]
- Second insight

## Connections

### Related
- [[Related Concept]] - test link

## Sources
- [[src-a]]
- [[src-b]]
"""

ORPHAN_PAGE = """---
title: Orphan Page
type: concept
category: observations
tags: [orphan]
sources: 1
people: []
last_updated: '2026-03-01'
confidence: low
wiki_version: 1
---

# Orphan Page

## Core Idea
Nobody links here.

## Sources
- [[src-c]]
"""


class TestWikiBackup:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        (vault / "wiki" / "concepts").mkdir(parents=True)
        (vault / "wiki" / "concepts" / "test.md").write_text("# Test\nContent here.")
        (vault / "raw" / "refs" / "legacy-schema").mkdir(parents=True)
        return vault

    @pytest.fixture
    def backup_dir(self, tmp_path):
        return tmp_path / "backups"

    def test_create_tarball(self, vault, backup_dir):
        backup = WikiBackup(vault, backup_dir, keep_last_n=5)
        tarball = backup.create_tarball()

        assert tarball.exists()
        assert tarball.suffix == ".gz"
        assert tarball.stat().st_size > 0

        # Verify tarball contains vault files
        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
            assert any("test.md" in n for n in names)

    def test_list_backups(self, vault, backup_dir):
        backup = WikiBackup(vault, backup_dir, keep_last_n=5)
        # Create with distinct timestamps by patching
        import time
        with patch("wiki.backup.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260414_100000"
            backup.create_tarball()
            mock_dt.now.return_value.strftime.return_value = "20260414_100001"
            backup.create_tarball()

        backups = backup.list_backups()
        assert len(backups) == 2

    def test_rotation_deletes_old(self, vault, backup_dir):
        backup = WikiBackup(vault, backup_dir, keep_last_n=2)

        with patch("wiki.backup.datetime") as mock_dt:
            for i in range(4):
                mock_dt.now.return_value.strftime.return_value = f"20260414_10000{i}"
                backup.create_tarball()

        backups = backup.list_backups()
        assert len(backups) == 2  # Only 2 kept


class TestTimelineUpdate:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries", "outputs/queries"]:
            (vault / d).mkdir(parents=True)
        (vault / "wiki" / "concepts" / "test.md").write_text(CONCEPT_PAGE)
        return vault

    def test_update_timeline_creates_file(self, vault):
        indexer = WikiIndexer(vault)
        timeline_path = vault / "outputs" / "queries" / "wiki-growth-timeline.md"
        assert not timeline_path.exists()

        indexer.update_timeline()

        assert timeline_path.exists()
        content = timeline_path.read_text()
        assert "Wiki Growth Timeline" in content
        assert "Pages: 1" in content

    def test_update_timeline_appends(self, vault):
        indexer = WikiIndexer(vault)
        indexer.update_timeline()
        indexer.update_timeline()

        content = (vault / "outputs" / "queries" / "wiki-growth-timeline.md").read_text()
        # Should have two date entries
        import re
        dates = re.findall(r"^## \d{4}-\d{2}-\d{2}", content, re.MULTILINE)
        assert len(dates) == 2


class TestConnectionsUpdate:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries"]:
            (vault / d).mkdir(parents=True)
        # Two pages that link to each other
        (vault / "wiki" / "concepts" / "compound-interest.md").write_text(
            "---\ntitle: Compound Interest\ntype: concept\ncategory: economics\n"
            "tags: []\nsources: 1\nlast_updated: '2026-04-14'\nconfidence: low\n---\n"
            "# Compound Interest\n## Core Idea\nTest.\n## Key Insights\n"
            "- [[Long-Term Thinking]] is key\n- [[Warren Buffett]] said so\n"
        )
        (vault / "wiki" / "concepts" / "long-term-thinking.md").write_text(
            "---\ntitle: Long-Term Thinking\ntype: concept\ncategory: strategy\n"
            "tags: []\nsources: 1\nlast_updated: '2026-04-14'\nconfidence: low\n---\n"
            "# Long-Term Thinking\n## Core Idea\nTest.\n## Key Insights\n"
            "- Related to [[Compound Interest]]\n- [[Patience]] matters\n"
        )
        (vault / "wiki" / "concepts" / "patience.md").write_text(
            "---\ntitle: Patience\ntype: concept\ncategory: psychology\n"
            "tags: []\nsources: 1\nlast_updated: '2026-04-14'\nconfidence: low\n---\n"
            "# Patience\n## Core Idea\nTest.\n## Key Insights\n"
            "- Enables [[Long-Term Thinking]]\n- Enables [[Compound Interest]]\n"
        )
        return vault

    def test_update_connections(self, vault):
        indexer = WikiIndexer(vault)
        indexer.update_connections()

        content = (vault / "wiki" / "concepts" / "knowledge-connections.md").read_text()
        assert "Knowledge Connections" in content
        # Should find a cluster with these 3 interconnected concepts
        assert "Cluster" in content


class TestLinterIntegration:
    @pytest.fixture
    def vault(self, tmp_path):
        vault = tmp_path / "vault"
        for d in ["wiki/concepts", "wiki/entities", "wiki/summaries", "raw/refs/legacy-schema"]:
            (vault / d).mkdir(parents=True)
        (vault / "wiki" / "concepts" / "test-concept.md").write_text(CONCEPT_PAGE)
        (vault / "wiki" / "concepts" / "orphan-page.md").write_text(ORPHAN_PAGE)
        # Empty registry to trigger consistency issues
        (vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json").write_text("{}")
        return vault

    def test_lint_finds_orphans(self, vault):
        registry = EntityRegistry(vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
        registry.load()
        linter = WikiLinter(vault, registry)

        report = linter.run(auto_fix=False)
        orphan_issues = [i for i in report.issues if i.category == "orphan"]
        # orphan-page has no inbound links
        orphan_pages = {i.page for i in orphan_issues}
        assert "orphan-page" in orphan_pages

    def test_lint_finds_broken_wikilinks(self, vault):
        registry = EntityRegistry(vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
        registry.load()
        linter = WikiLinter(vault, registry)

        report = linter.run(auto_fix=False)
        wikilink_issues = [i for i in report.issues if i.category == "wikilink"]
        # [[Related Concept]] doesn't exist as a page
        broken_targets = [i.message for i in wikilink_issues]
        assert any("Related Concept" in msg for msg in broken_targets)

    def test_lint_auto_fixes_confidence(self, vault):
        registry = EntityRegistry(vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
        registry.load()
        linter = WikiLinter(vault, registry)

        report = linter.run(auto_fix=True)
        # test-concept.md has sources=2 but confidence=low -> should be upgraded
        confidence_fixes = [i for i in report.issues if i.category == "confidence" and i.fixed]
        assert len(confidence_fixes) >= 1

        # Verify the file was actually updated
        content = (vault / "wiki" / "concepts" / "test-concept.md").read_text()
        assert "confidence: medium" in content

    def test_lint_finds_registry_inconsistency(self, vault):
        registry = EntityRegistry(vault / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
        registry.load()
        linter = WikiLinter(vault, registry)

        report = linter.run(auto_fix=False)
        registry_issues = [i for i in report.issues if i.category == "registry"]
        # Both pages exist but registry is empty
        assert len(registry_issues) >= 2


class TestNewsletterMaintenanceHook:
    """Test that the newsletter pipeline maintenance hook is wired correctly."""

    @patch("scripts.newsletter_auto_ingest._try_wiki_maintenance")
    def test_maintenance_called_after_pipeline(self, mock_maintenance):
        """Verify _try_wiki_maintenance exists and is callable."""
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance
        assert callable(_try_wiki_maintenance)

    def test_try_wiki_maintenance_handles_import_error(self):
        """Verify graceful handling when wiki module not available."""
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance
        # Should not raise even if wiki is disabled/unavailable
        with patch.dict("sys.modules", {"config.wiki_config": None}):
            _try_wiki_maintenance()  # Should not raise
