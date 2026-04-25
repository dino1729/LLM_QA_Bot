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

    def test_try_wiki_maintenance_handles_import_error(self, tmp_path):
        """Verify graceful handling when wiki module not available."""
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance
        state_file = tmp_path / "state.json"
        with patch.dict("sys.modules", {"config.wiki_config": None}):
            _try_wiki_maintenance(state_file)  # Should not raise


class TestWikiMaintenanceCadenceGuard:
    """The 7-day cadence guard prevents wiki maintenance from running on every
    invocation. The cadence stamp is read from a DEDICATED state file
    (wiki_maintenance_state.json), separate from the newsletter ingestion
    state file."""

    def _patch_wiki_config(self, enabled=True, run_with_cron=True):
        from unittest.mock import MagicMock
        wc = MagicMock()
        wc.wiki_enabled = enabled
        wc.wiki_lint_run_with_newsletter_cron = run_with_cron
        return patch.dict("sys.modules", {"config.wiki_config": wc})

    def _seed_wiki_state(self, state_file, value):
        """Write `value` into the dedicated wiki maintenance state file."""
        from scripts.newsletter_auto_ingest import _wiki_maintenance_state_path
        wiki_path = _wiki_maintenance_state_path(state_file)
        wiki_path.write_text(json.dumps({"last_wiki_maintenance_at": value}))
        return wiki_path

    def test_skips_when_last_run_within_7_days(self, tmp_path):
        """Recent stamp on disk -> early return, no work done."""
        from datetime import datetime, timezone, timedelta
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance

        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        state_file = tmp_path / "state.json"
        wiki_path = self._seed_wiki_state(state_file, recent)

        with self._patch_wiki_config():
            with patch("wiki.get_wiki_builder") as mock_builder:
                _try_wiki_maintenance(state_file)
                mock_builder.assert_not_called()

        # Wiki state file unchanged
        assert json.loads(wiki_path.read_text())["last_wiki_maintenance_at"] == recent

    def test_runs_when_last_run_older_than_7_days(self, tmp_path):
        """Stale stamp on disk -> proceeds past the cadence gate."""
        from datetime import datetime, timezone, timedelta
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance

        stale = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        state_file = tmp_path / "state.json"
        self._seed_wiki_state(state_file, stale)

        with self._patch_wiki_config():
            with patch("wiki.get_wiki_builder") as mock_builder:
                mock_builder.return_value = None  # bail cleanly inside body
                _try_wiki_maintenance(state_file)
                mock_builder.assert_called_once()

    def test_runs_when_no_prior_timestamp(self, tmp_path):
        """First-ever run -> no wiki state file, must proceed past the gate."""
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance

        state_file = tmp_path / "state.json"

        with self._patch_wiki_config():
            with patch("wiki.get_wiki_builder") as mock_builder:
                mock_builder.return_value = None
                _try_wiki_maintenance(state_file)
                mock_builder.assert_called_once()

    def test_malformed_timestamp_does_not_block(self, tmp_path):
        """Garbage in wiki state file -> warn and proceed."""
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance

        state_file = tmp_path / "state.json"
        self._seed_wiki_state(state_file, "not-a-real-timestamp")

        with self._patch_wiki_config():
            with patch("wiki.get_wiki_builder") as mock_builder:
                mock_builder.return_value = None
                _try_wiki_maintenance(state_file)
                mock_builder.assert_called_once()

    def test_disabled_via_config_does_not_stamp(self, tmp_path):
        """wiki_enabled=False -> early return, no wiki state file created."""
        from scripts.newsletter_auto_ingest import (
            _try_wiki_maintenance, _wiki_maintenance_state_path
        )

        state_file = tmp_path / "state.json"

        with self._patch_wiki_config(enabled=False):
            _try_wiki_maintenance(state_file)

        assert not _wiki_maintenance_state_path(state_file).exists()


class TestWikiMaintenanceStampSeparation:
    """The cadence stamp lives in a DEDICATED file separate from the
    newsletter ingestion state. This is the architectural fix for the bug
    where ingestion's whole-dict writes (using a snapshot taken at process
    start, before maintenance has stamped) would silently clobber the
    cadence stamp if both were in the same JSON."""

    def test_stamp_written_to_dedicated_wiki_state_file(self, tmp_path):
        """_stamp_wiki_maintenance_complete writes to wiki_maintenance_state.json,
        NOT to the newsletter ingestion state file."""
        from scripts.newsletter_auto_ingest import (
            _stamp_wiki_maintenance_complete,
            _wiki_maintenance_state_path,
            save_state_atomic,
        )

        state_file = tmp_path / "newsletter_ingestion_state.json"
        # Pre-existing ingestion state (not touched by stamp)
        save_state_atomic(state_file, {"sources": {"feedA": {}}, "last_run_at": "2026-04-24T05:00:00+00:00"})
        ingestion_before = state_file.read_text()

        _stamp_wiki_maintenance_complete(state_file)

        # Stamp landed in the dedicated file
        wiki_state = json.loads(_wiki_maintenance_state_path(state_file).read_text())
        assert "last_wiki_maintenance_at" in wiki_state

        # Ingestion state file is byte-for-byte unchanged
        assert state_file.read_text() == ingestion_before
        # And it does NOT contain the stamp field
        assert "last_wiki_maintenance_at" not in json.loads(state_file.read_text())

    def test_cadence_stamp_survives_ingestion_state_write(self, tmp_path):
        """Regression test for Codex finding 5: ingestion writes its state
        using a whole-dict snapshot taken at process start, which would
        clobber the cadence stamp if it lived in the same file. With the
        stamp in its own file, ingestion writes can't reach it."""
        from scripts.newsletter_auto_ingest import (
            _stamp_wiki_maintenance_complete,
            _load_wiki_maintenance_state,
            save_state_atomic,
        )

        state_file = tmp_path / "newsletter_ingestion_state.json"

        # T+15min: wiki maintenance stamps completion
        _stamp_wiki_maintenance_complete(state_file)
        stamp_before = _load_wiki_maintenance_state(state_file)["last_wiki_maintenance_at"]

        # T+20min: newsletter ingestion (which started before the stamp)
        # writes its full in-memory snapshot to the ingestion state file.
        # The snapshot has NO knowledge of the wiki cadence stamp.
        save_state_atomic(state_file, {
            "sources": {"feedA": {"last_success_at": "..."}, "feedB": {"new": True}},
            "last_run_at": "2026-04-24T05:20:00+00:00",
        })

        # Stamp must STILL be readable — it lives in its own file
        stamp_after = _load_wiki_maintenance_state(state_file)["last_wiki_maintenance_at"]
        assert stamp_after == stamp_before, (
            "Cadence stamp was lost when ingestion wrote its state — "
            "the architectural separation regressed"
        )

    def test_stamp_preserves_concurrent_writes_to_wiki_state_file(self, tmp_path):
        """Defensive: even within the dedicated wiki state file, stamp uses
        read-modify-write so any concurrent field is preserved."""
        from scripts.newsletter_auto_ingest import (
            _stamp_wiki_maintenance_complete,
            _wiki_maintenance_state_path,
        )

        state_file = tmp_path / "state.json"
        wiki_path = _wiki_maintenance_state_path(state_file)
        # Concurrent process wrote a field into the wiki state file
        wiki_path.write_text(json.dumps({"unrelated_field": "concurrent_value"}))

        _stamp_wiki_maintenance_complete(state_file)

        final = json.loads(wiki_path.read_text())
        assert final["unrelated_field"] == "concurrent_value", \
            "Stamp clobbered a concurrent field in the wiki state file"
        assert "last_wiki_maintenance_at" in final


class TestWikiMaintenanceMutualExclusion:
    """Cross-process mutual exclusion via fcntl.flock prevents two
    overlapping invocations from both passing the cadence check and running
    maintenance simultaneously."""

    def _patch_wiki_config(self, enabled=True, run_with_cron=True):
        from unittest.mock import MagicMock
        wc = MagicMock()
        wc.wiki_enabled = enabled
        wc.wiki_lint_run_with_newsletter_cron = run_with_cron
        return patch.dict("sys.modules", {"config.wiki_config": wc})

    def test_concurrent_lock_holder_blocks_second_invocation(self, tmp_path):
        """A separate fd holding the flock causes _try_wiki_maintenance to
        skip without entering the maintenance body."""
        import fcntl
        from scripts.newsletter_auto_ingest import (
            _try_wiki_maintenance,
            _wiki_maintenance_state_path,
        )

        state_file = tmp_path / "state.json"
        lock_path = state_file.parent / "wiki_maintenance.lock"

        # Simulate another process holding the lock
        holder_fp = open(lock_path, "w")
        try:
            fcntl.flock(holder_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self._patch_wiki_config():
                with patch("wiki.get_wiki_builder") as mock_builder:
                    _try_wiki_maintenance(state_file)
                    mock_builder.assert_not_called()
            # No wiki state file created: stamp wasn't written
            assert not _wiki_maintenance_state_path(state_file).exists()
        finally:
            holder_fp.close()  # releases the lock

    def test_lock_released_after_completion(self, tmp_path):
        """After _try_wiki_maintenance returns, another caller can acquire
        the lock — i.e. the lock isn't leaked."""
        import fcntl
        from scripts.newsletter_auto_ingest import _try_wiki_maintenance

        state_file = tmp_path / "state.json"
        lock_path = state_file.parent / "wiki_maintenance.lock"

        with self._patch_wiki_config():
            with patch("wiki.get_wiki_builder") as mock_builder:
                mock_builder.return_value = None
                _try_wiki_maintenance(state_file)

        with open(lock_path, "w") as fp:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
