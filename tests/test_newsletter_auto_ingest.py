"""
Tests for scripts/newsletter_auto_ingest.py.
"""
from pathlib import Path
from unittest.mock import Mock

from scripts.newsletter_auto_ingest import (
    FeedEntry,
    fetch_listing_entries,
    filter_feed_entries,
    ingest_with_retry,
    load_state,
    mark_seen,
    parse_feed_xml,
    run_pipeline,
    save_state_atomic,
    select_candidates,
)


def test_parse_feed_xml_rss():
    xml = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example Feed</title>
    <item>
      <title>Issue 1</title>
      <link>https://example.com/newsletter/issue-1</link>
      <guid>id-1</guid>
      <pubDate>Sun, 01 Mar 2026 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""
    entries = parse_feed_xml(xml)
    assert len(entries) == 1
    assert entries[0].title == "Issue 1"
    assert entries[0].link == "https://example.com/newsletter/issue-1"
    assert entries[0].entry_id
    assert entries[0].published_ts > 0


def test_parse_feed_xml_atom():
    xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Atom</title>
  <entry>
    <id>entry-1</id>
    <title>Atom Issue</title>
    <link rel="alternate" href="https://example.com/newsletter/atom-issue"/>
    <updated>2026-03-01T12:00:00Z</updated>
  </entry>
</feed>"""
    entries = parse_feed_xml(xml)
    assert len(entries) == 1
    assert entries[0].title == "Atom Issue"
    assert entries[0].link == "https://example.com/newsletter/atom-issue"
    assert entries[0].published_ts > 0


def test_filter_feed_entries_include_exclude():
    entries = [
        FeedEntry(entry_id="1", link="https://x.com/brain-food/a", title="A"),
        FeedEntry(entry_id="2", link="https://x.com/podcast/b", title="B"),
        FeedEntry(entry_id="3", link="https://x.com/brain-food/c", title="C"),
    ]
    filtered = filter_feed_entries(
        entries,
        include_patterns=[r"/brain-food/"],
        exclude_patterns=[r"/brain-food/c"],
    )
    assert [e.entry_id for e in filtered] == ["1"]


def test_select_candidates_backfill_and_skip():
    entries = [
        FeedEntry(entry_id="1", link="https://x/1", title="1"),
        FeedEntry(entry_id="2", link="https://x/2", title="2"),
        FeedEntry(entry_id="3", link="https://x/3", title="3"),
        FeedEntry(entry_id="4", link="https://x/4", title="4"),
    ]
    source_state = {"seen_ids": ["4"], "backfill_completed": False}
    candidates, skipped, unseen_count = select_candidates(
        entries=entries,
        source_state=source_state,
        backfill_count=2,
        per_source_limit=20,
    )
    assert [c.entry_id for c in candidates] == ["1", "2"]
    assert skipped == ["3"]
    assert unseen_count == 3


def test_select_candidates_incremental():
    entries = [
        FeedEntry(entry_id="1", link="https://x/1", title="1"),
        FeedEntry(entry_id="2", link="https://x/2", title="2"),
        FeedEntry(entry_id="3", link="https://x/3", title="3"),
    ]
    source_state = {"seen_ids": ["1"], "backfill_completed": True}
    candidates, skipped, unseen_count = select_candidates(
        entries=entries,
        source_state=source_state,
        backfill_count=10,
        per_source_limit=1,
    )
    assert [c.entry_id for c in candidates] == ["2"]
    assert skipped == []
    assert unseen_count == 2


def test_state_roundtrip_and_mark_seen(tmp_path):
    state_path = tmp_path / "state.json"
    state = {"sources": {"demo": {"seen_ids": ["a"], "backfill_completed": False}}}
    save_state_atomic(state_path, state)
    loaded = load_state(state_path)
    assert loaded["sources"]["demo"]["seen_ids"] == ["a"]

    source_state = loaded["sources"]["demo"]
    mark_seen(source_state, "b", max_seen_ids=2)
    mark_seen(source_state, "c", max_seen_ids=2)
    assert source_state["seen_ids"] == ["c", "b"]


def test_ingest_with_retry_succeeds_after_retry(monkeypatch):
    call_count = {"n": 0}

    def fake_ingest(url, model_tier, dry_run=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient")
        return {"url": url, "status": "success"}

    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.ingest_newsletter_url",
        fake_ingest,
    )

    sleep_calls = []
    result = ingest_with_retry(
        "https://x.com/newsletter/1",
        model_tier="smart",
        max_retries=2,
        base_retry_delay_seconds=0.01,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert result["success"] is True
    assert call_count["n"] == 2
    assert sleep_calls


def test_fetch_listing_entries_discovers_matches(monkeypatch):
    html = """
<html>
  <body>
    <a href="/brain-food/march-1-2026/">Brain Food Mar 1</a>
    <a href="https://fs.blog/brain-food/february-22-2026/">Brain Food Feb 22</a>
    <a href="/podcast/some-episode/">Podcast</a>
    <a href="/brain-food/march-1-2026/">Duplicate</a>
  </body>
</html>
"""
    response = Mock()
    response.text = html
    response.raise_for_status = Mock()
    monkeypatch.setattr("scripts.newsletter_auto_ingest.requests.get", Mock(return_value=response))

    entries = fetch_listing_entries(
        "https://fs.blog/brain-food/",
        r"/brain-food/[a-z]+-\d{1,2}-\d{4}/?$",
        timeout=10,
    )
    assert len(entries) == 2
    assert entries[0].link == "https://fs.blog/brain-food/march-1-2026/"
    assert entries[1].link == "https://fs.blog/brain-food/february-22-2026/"


def test_run_pipeline_source_isolation(monkeypatch, tmp_path):
    # Configure pipeline paths and runtime limits.
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_log_dir", str(tmp_path / "logs"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_lock_file", str(tmp_path / "lock" / "run.lock"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_model_tier", "smart")
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_max_retries", 1)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_base_retry_delay_seconds", 0.01)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_run_timeout_minutes", 10)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_per_source_limit", 20)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_feed_timeout_seconds", 5)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_backfill_count", 10)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_max_seen_ids", 2000)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_telegram_alerts", False)
    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.config.newsletter_ingestion_sources",
        [
            {
                "name": "good_source",
                "enabled": True,
                "feed_url": "https://feed.good/rss",
                "include_url_patterns": [r"/newsletter/"],
                "exclude_url_patterns": [],
                "backfill_count": 10,
            },
            {
                "name": "bad_source",
                "enabled": True,
                "feed_url": "https://feed.bad/rss",
                "include_url_patterns": [r"/newsletter/"],
                "exclude_url_patterns": [],
                "backfill_count": 10,
            },
        ],
    )

    good_entries = [FeedEntry(entry_id="g1", link="https://good.com/newsletter/1", title="good")]
    bad_entries = [FeedEntry(entry_id="b1", link="https://bad.com/newsletter/1", title="bad")]

    def fake_fetch(feed_url, timeout=20):
        if "feed.good" in feed_url:
            return good_entries
        return bad_entries

    monkeypatch.setattr("scripts.newsletter_auto_ingest.fetch_feed_entries", fake_fetch)

    def fake_ingest_with_retry(url, model_tier, max_retries, base_retry_delay_seconds, dry_run=False, sleep_fn=None):
        if "bad.com" in url:
            return {"success": False, "error": "boom", "attempts": max_retries + 1}
        return {"success": True, "result": {"url": url, "status": "success"}}

    monkeypatch.setattr("scripts.newsletter_auto_ingest.ingest_with_retry", fake_ingest_with_retry)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.send_telegram_message", Mock(return_value=True))

    summary = run_pipeline()
    assert summary["totals"]["sources_processed"] == 2
    assert summary["totals"]["ingested"] == 1
    assert summary["totals"]["failed"] == 1

    # Running again should be idempotent for the successful source because state recorded seen IDs.
    summary_2 = run_pipeline()
    assert summary_2["totals"]["ingested"] == 0

    state_file = Path(tmp_path / "state" / "newsletter_ingestion_state.json")
    loaded_state = load_state(state_file)
    assert "good_source" in loaded_state["sources"]
    assert loaded_state["sources"]["good_source"]["seen_ids"]


def test_run_pipeline_uses_listing_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_log_dir", str(tmp_path / "logs"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_lock_file", str(tmp_path / "lock" / "run.lock"))
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_model_tier", "smart")
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_max_retries", 1)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_base_retry_delay_seconds", 0.01)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_run_timeout_minutes", 10)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_per_source_limit", 20)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_feed_timeout_seconds", 5)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_backfill_count", 10)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_max_seen_ids", 2000)
    monkeypatch.setattr("scripts.newsletter_auto_ingest.config.newsletter_ingestion_telegram_alerts", False)
    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.config.newsletter_ingestion_sources",
        [
            {
                "name": "brain_food",
                "enabled": True,
                "feed_url": "https://feed.example/rss",
                "include_url_patterns": [r"/brain-food/"],
                "exclude_url_patterns": [],
                "backfill_count": 10,
                "listing_page_url": "https://fs.blog/brain-food/",
                "listing_link_pattern": r"/brain-food/[a-z]+-\d{1,2}-\d{4}/?$",
            }
        ],
    )

    # Feed discovery returns non-matching entries.
    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.fetch_feed_entries",
        lambda feed_url, timeout=20: [
            FeedEntry(entry_id="pod1", link="https://fs.blog/knowledge-project-podcast/x", title="pod"),
        ],
    )
    # Listing fallback returns matching entries.
    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.fetch_listing_entries",
        lambda listing_page_url, link_pattern, timeout=20: [
            FeedEntry(entry_id="bf1", link="https://fs.blog/brain-food/march-1-2026/", title="bf"),
        ],
    )
    monkeypatch.setattr(
        "scripts.newsletter_auto_ingest.ingest_with_retry",
        lambda **kwargs: {"success": True, "result": {"status": "success"}},
    )

    summary = run_pipeline()
    assert summary["totals"]["ingested"] == 1
    assert summary["sources"][0]["used_listing_fallback"] is True
    assert summary["sources"][0]["eligible_entries"] == 1
