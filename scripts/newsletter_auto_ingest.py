#!/usr/bin/env python3
"""
Reliable newsletter auto-ingestion pipeline.

This script polls configured RSS/Atom feeds, filters matching newsletter URLs,
and ingests unseen issues into:
1) EDITH Lessons (MemoryPalaceDB)
2) Local Memory Palace vector store
3) Knowledge Archive

Design goals:
- Idempotent ingestion with persisted state
- Backfill latest N issues on first run per source
- Retry with exponential backoff
- Source-level isolation (one source failing does not block others)
- Optional Telegram run summaries
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

# Add project root for imports when executed as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from helper_functions.link_ingestion import (  # noqa: E402
    extract_article,
    extract_takeaways,
    upload_to_edith,
    upload_to_knowledge_archive,
    upload_to_local_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    entry_id: str
    link: str
    title: str
    published: Optional[str] = None
    published_ts: float = 0.0


@dataclass
class SourceSpec:
    name: str
    feed_url: str
    include_url_patterns: List[str]
    exclude_url_patterns: List[str]
    enabled: bool = True
    backfill_count: int = 10
    listing_page_url: Optional[str] = None
    listing_link_pattern: Optional[str] = None


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_child_text(elem: ElementTree.Element, child_name: str) -> Optional[str]:
    for child in list(elem):
        if _tag_name(child.tag) == child_name:
            if child.text:
                return child.text.strip()
            return None
    return None


def _entry_sort_ts(raw_date: Optional[str]) -> float:
    if not raw_date:
        return 0.0

    dt = None

    try:
        dt = parsedate_to_datetime(raw_date)
    except Exception:
        dt = None

    if dt is None:
        try:
            normalized = raw_date.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
        except Exception:
            dt = None

    if dt is None:
        return 0.0

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


def _build_entry_id(guid: Optional[str], link: str, title: str) -> str:
    raw = f"{guid or ''}|{link}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_feed_xml(xml_text: str) -> List[FeedEntry]:
    """
    Parse RSS/Atom XML into normalized feed entries.
    """
    root = ElementTree.fromstring(xml_text)
    root_name = _tag_name(root.tag).lower()
    entries: List[FeedEntry] = []

    if root_name == "rss":
        channel = None
        for child in list(root):
            if _tag_name(child.tag) == "channel":
                channel = child
                break

        if channel is None:
            return entries

        for item in list(channel):
            if _tag_name(item.tag) != "item":
                continue

            title = _first_child_text(item, "title") or "Untitled"
            link = _first_child_text(item, "link") or ""
            guid = _first_child_text(item, "guid")
            published = _first_child_text(item, "pubDate")

            if not link:
                continue

            entries.append(
                FeedEntry(
                    entry_id=_build_entry_id(guid, link, title),
                    link=link.strip(),
                    title=title.strip(),
                    published=published,
                    published_ts=_entry_sort_ts(published),
                )
            )

        return entries

    if root_name == "feed":
        for entry in list(root):
            if _tag_name(entry.tag) != "entry":
                continue

            title = _first_child_text(entry, "title") or "Untitled"
            guid = _first_child_text(entry, "id")
            published = _first_child_text(entry, "published") or _first_child_text(entry, "updated")

            link = ""
            for child in list(entry):
                if _tag_name(child.tag) != "link":
                    continue
                rel = (child.attrib.get("rel") or "").strip()
                href = (child.attrib.get("href") or "").strip()
                if not href:
                    continue
                if rel in ("", "alternate"):
                    link = href
                    break
                if not link:
                    link = href

            if not link:
                continue

            entries.append(
                FeedEntry(
                    entry_id=_build_entry_id(guid, link, title),
                    link=link.strip(),
                    title=title.strip(),
                    published=published,
                    published_ts=_entry_sort_ts(published),
                )
            )

    return entries


def fetch_feed_entries(feed_url: str, timeout: int = 20) -> List[FeedEntry]:
    response = requests.get(
        feed_url,
        timeout=timeout,
        headers={"User-Agent": "LLM_QA_Bot Newsletter Ingestor/1.0"},
    )
    response.raise_for_status()
    return parse_feed_xml(response.text)


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1].replace("-", " ").strip()
    if not slug:
        return "Untitled"
    return slug.title()


def fetch_listing_entries(
    listing_page_url: str,
    link_pattern: str,
    timeout: int = 20,
) -> List[FeedEntry]:
    """
    Discover entries from a listing page by matching anchor href values.
    """
    response = requests.get(
        listing_page_url,
        timeout=timeout,
        headers={"User-Agent": "LLM_QA_Bot Newsletter Ingestor/1.0"},
    )
    response.raise_for_status()

    pattern = re.compile(link_pattern)
    soup = BeautifulSoup(response.text, "html.parser")

    entries: List[FeedEntry] = []
    seen_urls = set()
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        absolute_url = urljoin(listing_page_url, href)
        if absolute_url in seen_urls:
            continue
        if not pattern.search(absolute_url):
            continue

        seen_urls.add(absolute_url)
        title = anchor.get_text(" ", strip=True) or _title_from_url(absolute_url)
        entries.append(
            FeedEntry(
                entry_id=_build_entry_id(None, absolute_url, title),
                link=absolute_url,
                title=title,
            )
        )

    return entries


def _matches_any_regex(text: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return True
        except re.error as exc:
            logger.warning("Invalid regex pattern '%s': %s", pattern, exc)
    return False


def url_allowed(url: str, include_patterns: Sequence[str], exclude_patterns: Sequence[str]) -> bool:
    if include_patterns and not _matches_any_regex(url, include_patterns):
        return False
    if exclude_patterns and _matches_any_regex(url, exclude_patterns):
        return False
    return True


def filter_feed_entries(
    entries: Sequence[FeedEntry],
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> List[FeedEntry]:
    return [e for e in entries if url_allowed(e.link, include_patterns, exclude_patterns)]


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {"sources": {}, "created_at": datetime.now(timezone.utc).isoformat()}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("sources", {})
            return data
    except Exception as exc:
        logger.warning("Failed to load state file '%s': %s", state_path, exc)

    return {"sources": {}, "created_at": datetime.now(timezone.utc).isoformat()}


def save_state_atomic(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    temp_path.replace(state_path)


def get_source_state(state: Dict, source_name: str) -> Dict:
    sources = state.setdefault("sources", {})
    source_state = sources.setdefault(
        source_name,
        {
            "seen_ids": [],
            "backfill_completed": False,
            "last_success_at": None,
            "consecutive_failures": 0,
        },
    )
    source_state.setdefault("seen_ids", [])
    source_state.setdefault("backfill_completed", False)
    source_state.setdefault("last_success_at", None)
    source_state.setdefault("consecutive_failures", 0)
    return source_state


def mark_seen(source_state: Dict, entry_id: str, max_seen_ids: int) -> None:
    seen_ids = source_state.setdefault("seen_ids", [])
    if entry_id in seen_ids:
        seen_ids.remove(entry_id)
    seen_ids.insert(0, entry_id)
    del seen_ids[max_seen_ids:]


def select_candidates(
    entries: Sequence[FeedEntry],
    source_state: Dict,
    backfill_count: int,
    per_source_limit: int,
) -> Tuple[List[FeedEntry], List[str], int]:
    seen = set(source_state.get("seen_ids", []))
    unseen = [entry for entry in entries if entry.entry_id not in seen]
    unseen_count = len(unseen)

    if source_state.get("backfill_completed", False):
        return unseen[:per_source_limit], [], unseen_count

    initial_cap = min(backfill_count, per_source_limit)
    selected = unseen[:initial_cap]
    skipped_after_backfill = [entry.entry_id for entry in unseen[initial_cap:]]
    return selected, skipped_after_backfill, unseen_count


def determine_takeaway_count(word_count: int) -> int:
    return 3 if word_count < 2000 else 8


def ingest_newsletter_url(url: str, model_tier: str, dry_run: bool = False) -> Dict:
    if dry_run:
        return {
            "url": url,
            "status": "dry_run",
            "edith_lessons_saved": 0,
            "local_memory_status": "dry_run",
            "takeaway_count": 0,
        }

    content = extract_article(url)
    takeaway_count = determine_takeaway_count(content.word_count)
    takeaways = extract_takeaways(content, takeaway_count, model_tier)

    if not takeaways:
        raise RuntimeError(f"No takeaways extracted for URL: {url}")

    edith_saved = upload_to_edith(takeaways, content, skip_distill=False)
    local_status = upload_to_local_memory(content, takeaways)
    upload_to_knowledge_archive(content)

    return {
        "url": url,
        "status": "success",
        "edith_lessons_saved": edith_saved,
        "local_memory_status": local_status,
        "takeaway_count": len(takeaways),
    }


def ingest_with_retry(
    url: str,
    model_tier: str,
    max_retries: int,
    base_retry_delay_seconds: float,
    dry_run: bool = False,
    sleep_fn=time.sleep,
) -> Dict:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = ingest_newsletter_url(url, model_tier=model_tier, dry_run=dry_run)
            result["attempts"] = attempt + 1
            return {"success": True, "result": result}
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            delay = base_retry_delay_seconds * (2 ** attempt)
            logger.warning(
                "Ingestion failed for %s (attempt %d/%d): %s. Retrying in %.1fs",
                url,
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            sleep_fn(delay)

    return {
        "success": False,
        "error": last_error or "unknown_error",
        "attempts": max_retries + 1,
    }


def load_source_specs(default_backfill_count: int) -> List[SourceSpec]:
    raw_sources = getattr(config, "newsletter_ingestion_sources", []) or []
    specs: List[SourceSpec] = []

    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name") or "").strip()
        feed_url = str(raw.get("feed_url") or "").strip()
        if not name or not feed_url:
            continue

        include_patterns = raw.get("include_url_patterns") or []
        exclude_patterns = raw.get("exclude_url_patterns") or []
        include_patterns = [str(p) for p in include_patterns if str(p).strip()]
        exclude_patterns = [str(p) for p in exclude_patterns if str(p).strip()]

        specs.append(
            SourceSpec(
                name=name,
                feed_url=feed_url,
                include_url_patterns=include_patterns,
                exclude_url_patterns=exclude_patterns,
                enabled=bool(raw.get("enabled", True)),
                backfill_count=int(raw.get("backfill_count", default_backfill_count)),
                listing_page_url=(str(raw.get("listing_page_url", "")).strip() or None),
                listing_link_pattern=(str(raw.get("listing_link_pattern", "")).strip() or None),
            )
        )

    return specs


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_logs(log_dir: Path, summary: Dict) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"run_{ts}.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    jsonl_path = log_dir / "runs.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")

    return summary_path


def send_telegram_message(message: str) -> bool:
    token = getattr(config, "telegram_bot_token", None)
    chat_id = getattr(config, "memory_palace_telegram_user_id", None)
    if not token or not chat_id:
        logger.info("Telegram token/chat id missing, skipping alert")
        return False

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Failed to send Telegram alert: %s", exc)
        return False


def format_telegram_summary(summary: Dict) -> str:
    totals = summary["totals"]
    lines = [
        "Newsletter Ingestion Run",
        f"Started: {summary['started_at']}",
        f"Finished: {summary['finished_at']}",
        (
            f"Sources processed: {totals['sources_processed']} | "
            f"Candidates: {totals['candidates']} | "
            f"Ingested: {totals['ingested']} | "
            f"Failed: {totals['failed']}"
        ),
    ]

    for source in summary["sources"]:
        fallback_suffix = " (listing fallback)" if source.get("used_listing_fallback") else ""
        lines.append(
            (
                f"- {source['name']}{fallback_suffix}: feed={source['feed_entries']} "
                f"eligible={source['eligible_entries']} candidates={source['candidates']} "
                f"ingested={source['ingested']} failed={source['failed']}"
            )
        )

    if summary.get("timed_out"):
        lines.append("Run timed out before completion.")

    failures: List[str] = []
    for source in summary["sources"]:
        for error in source.get("errors", [])[:2]:
            failures.append(f"{source['name']}: {error}")

    if failures:
        lines.append("Failures:")
        lines.extend(f"- {item}" for item in failures[:6])

    message = "\n".join(lines)
    if len(message) > 3900:
        message = message[:3897] + "..."
    return message


try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


@contextmanager
def single_run_lock(lock_path: Path):
    if fcntl is None:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another newsletter ingestion run is active") from exc

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def run_pipeline(
    source_name_filter: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> Dict:
    state_dir = Path(getattr(config, "newsletter_ingestion_state_dir", "./memory_palace/newsletter_ingestion"))
    log_dir = Path(getattr(config, "newsletter_ingestion_log_dir", "./memory_palace/newsletter_ingestion/logs"))
    lock_file = Path(
        getattr(
            config,
            "newsletter_ingestion_lock_file",
            "./memory_palace/newsletter_ingestion/newsletter_ingestion.lock",
        )
    )
    state_file = state_dir / "newsletter_ingestion_state.json"

    model_tier = getattr(config, "newsletter_ingestion_model_tier", "smart")
    max_retries = _coerce_positive_int(getattr(config, "newsletter_ingestion_max_retries", 3), 3)
    retry_delay = float(getattr(config, "newsletter_ingestion_base_retry_delay_seconds", 30))
    run_timeout_minutes = _coerce_positive_int(
        getattr(config, "newsletter_ingestion_run_timeout_minutes", 90),
        90,
    )
    per_source_limit = _coerce_positive_int(
        getattr(config, "newsletter_ingestion_per_source_limit", 20),
        20,
    )
    feed_timeout_seconds = _coerce_positive_int(
        getattr(config, "newsletter_ingestion_feed_timeout_seconds", 20),
        20,
    )
    default_backfill_count = _coerce_positive_int(
        getattr(config, "newsletter_ingestion_backfill_count", 10),
        10,
    )
    max_seen_ids = _coerce_positive_int(
        getattr(config, "newsletter_ingestion_max_seen_ids", 2000),
        2000,
    )

    sources = [source for source in load_source_specs(default_backfill_count) if source.enabled]
    if source_name_filter:
        requested = set(source_name_filter)
        sources = [source for source in sources if source.name in requested]

    started = time.time()
    timeout_seconds = run_timeout_minutes * 60
    timed_out = False

    summary = {
        "started_at": _utc_now_iso(),
        "finished_at": None,
        "dry_run": dry_run,
        "sources": [],
        "timed_out": False,
        "totals": {
            "sources_processed": 0,
            "feed_entries": 0,
            "eligible_entries": 0,
            "candidates": 0,
            "ingested": 0,
            "failed": 0,
        },
    }

    with single_run_lock(lock_file):
        state = load_state(state_file)
        state_dir.mkdir(parents=True, exist_ok=True)

        for source in sources:
            if time.time() - started > timeout_seconds:
                timed_out = True
                break

            source_summary = {
                "name": source.name,
                "feed_url": source.feed_url,
                "feed_entries": 0,
                "listing_entries": 0,
                "eligible_entries": 0,
                "unseen_entries": 0,
                "candidates": 0,
                "ingested": 0,
                "failed": 0,
                "errors": [],
                "used_listing_fallback": False,
            }
            summary["sources"].append(source_summary)

            source_state = get_source_state(state, source.name)
            entries: List[FeedEntry] = []
            filtered: List[FeedEntry] = []
            feed_error: Optional[Exception] = None

            try:
                entries = fetch_feed_entries(source.feed_url, timeout=feed_timeout_seconds)
                entries = sorted(entries, key=lambda e: e.published_ts, reverse=True)
                filtered = filter_feed_entries(
                    entries,
                    include_patterns=source.include_url_patterns,
                    exclude_patterns=source.exclude_url_patterns,
                )
            except Exception as exc:
                feed_error = exc

            listing_entries: List[FeedEntry] = []
            if source.listing_page_url and source.listing_link_pattern and (feed_error or not filtered):
                try:
                    listing_entries = fetch_listing_entries(
                        source.listing_page_url,
                        source.listing_link_pattern,
                        timeout=feed_timeout_seconds,
                    )
                    fallback_filtered = filter_feed_entries(
                        listing_entries,
                        include_patterns=source.include_url_patterns,
                        exclude_patterns=source.exclude_url_patterns,
                    )
                    if fallback_filtered:
                        filtered = fallback_filtered
                        source_summary["used_listing_fallback"] = True
                except Exception as listing_exc:
                    if feed_error is not None:
                        source_summary["errors"].append(
                            f"Feed fetch/parse failed: {feed_error}; listing fallback failed: {listing_exc}"
                        )
                        source_state["consecutive_failures"] = int(source_state.get("consecutive_failures", 0)) + 1
                        save_state_atomic(state_file, state)
                        continue
                    source_summary["errors"].append(f"Listing fallback failed: {listing_exc}")

            if feed_error is not None and not source_summary["used_listing_fallback"]:
                source_summary["errors"].append(f"Feed fetch/parse failed: {feed_error}")
                source_state["consecutive_failures"] = int(source_state.get("consecutive_failures", 0)) + 1
                save_state_atomic(state_file, state)
                continue

            candidates, skipped_after_backfill, unseen_count = select_candidates(
                entries=filtered,
                source_state=source_state,
                backfill_count=source.backfill_count,
                per_source_limit=per_source_limit,
            )

            source_summary["feed_entries"] = len(entries)
            source_summary["listing_entries"] = len(listing_entries)
            source_summary["eligible_entries"] = len(filtered)
            source_summary["unseen_entries"] = unseen_count
            source_summary["candidates"] = len(candidates)

            all_candidates_succeeded = True
            for candidate in candidates:
                if time.time() - started > timeout_seconds:
                    timed_out = True
                    break

                result = ingest_with_retry(
                    url=candidate.link,
                    model_tier=model_tier,
                    max_retries=max_retries,
                    base_retry_delay_seconds=retry_delay,
                    dry_run=dry_run,
                )

                if result["success"]:
                    source_summary["ingested"] += 1
                    mark_seen(source_state, candidate.entry_id, max_seen_ids=max_seen_ids)
                    source_state["last_success_at"] = _utc_now_iso()
                    source_state["consecutive_failures"] = 0
                    save_state_atomic(state_file, state)
                else:
                    all_candidates_succeeded = False
                    source_summary["failed"] += 1
                    source_summary["errors"].append(
                        f"{candidate.link} (attempts={result['attempts']}): {result['error']}"
                    )
                    source_state["consecutive_failures"] = int(
                        source_state.get("consecutive_failures", 0)
                    ) + 1
                    save_state_atomic(state_file, state)

            # On successful initial backfill, mark remaining older entries as seen.
            if not source_state.get("backfill_completed", False):
                if not candidates or all_candidates_succeeded:
                    for skipped_id in skipped_after_backfill:
                        mark_seen(source_state, skipped_id, max_seen_ids=max_seen_ids)
                    source_state["backfill_completed"] = True
                    save_state_atomic(state_file, state)

            if timed_out:
                break

        summary["timed_out"] = timed_out
        summary["finished_at"] = _utc_now_iso()

        summary["totals"]["sources_processed"] = len(summary["sources"])
        summary["totals"]["feed_entries"] = sum(s["feed_entries"] for s in summary["sources"])
        summary["totals"]["eligible_entries"] = sum(s["eligible_entries"] for s in summary["sources"])
        summary["totals"]["candidates"] = sum(s["candidates"] for s in summary["sources"])
        summary["totals"]["ingested"] = sum(s["ingested"] for s in summary["sources"])
        summary["totals"]["failed"] = sum(s["failed"] for s in summary["sources"])

        state["last_run_at"] = summary["finished_at"]
        save_state_atomic(state_file, state)

    summary_path = write_run_logs(log_dir, summary)
    logger.info("Run summary saved to %s", summary_path)

    alerts_enabled = bool(getattr(config, "newsletter_ingestion_telegram_alerts", True))
    if alerts_enabled:
        send_telegram_message(format_telegram_summary(summary))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-ingest newsletter issues from configured feeds into Memory Palace",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (default behavior).",
    )
    parser.add_argument(
        "--source",
        help="Comma-separated source names to ingest (defaults to all enabled sources).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to databases. Useful for feed/filter validation.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not getattr(config, "newsletter_ingestion_enabled", False) and not args.dry_run:
        logger.warning(
            "newsletter_ingestion.enabled=false in config.yml; running anyway because command was invoked directly"
        )

    source_filter = None
    if args.source:
        source_filter = [s.strip() for s in args.source.split(",") if s.strip()]

    try:
        summary = run_pipeline(
            source_name_filter=source_filter,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return 0
    except Exception as exc:
        logger.exception("Newsletter ingestion failed: %s", exc)
        return 1

    failed = summary["totals"]["failed"]
    if summary.get("timed_out") or failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
