#!/usr/bin/env python3
"""
Batch Article Archiver - Robust overnight batch processing for Knowledge Archive.

Features:
- Retry logic with exponential backoff for network failures
- Resume capability from previous runs
- Graceful shutdown handling (Ctrl+C saves state)
- Connection health checks
- Detailed progress tracking

Usage:
    # Overnight run (1 article per minute, with tracking)
    python scripts/batch_archive_articles.py urls.txt --track progress.txt -v

    # Resume from previous run
    python scripts/batch_archive_articles.py urls.txt --track progress.txt --resume -v

    # Custom delay (default: 60s for rate limit safety)
    python scripts/batch_archive_articles.py urls.txt --delay 30 --track progress.txt -v
"""
import argparse
import json
import logging
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions.knowledge_archive_db import KnowledgeArchiveDB
from helper_functions.knowledge_archive_scraper import scrape_article
from helper_functions.knowledge_archive_processor import process_article
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(_signum, _frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.warning("Shutdown requested - finishing current article and saving state...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RobustProgressTracker:
    """
    Robust progress tracking with state persistence and resume capability.

    Saves state to both a human-readable tracking file and a JSON state file
    for crash recovery.
    """

    def __init__(self, filepath: str, urls: List[str], resume: bool = False):
        self.filepath = filepath
        self.state_filepath = filepath + ".state.json"
        self.urls = urls
        self.url_set = set(urls)

        # Progress tracking
        self.done_success: List[Dict] = []
        self.done_duplicate: List[str] = []
        self.done_short: List[Dict] = []
        self.failed_scrape: List[Dict] = []  # Now includes error details and retry count
        self.failed_process: List[Dict] = []
        self.pending: List[str] = list(urls)

        # Runtime stats
        self.started_at = datetime.now()
        self.last_success_at: Optional[datetime] = None
        self.total_retries = 0

        # Resume from previous state if requested
        if resume and Path(self.state_filepath).exists():
            self._load_state()

        self._write()

    def _load_state(self):
        """Load state from previous run."""
        try:
            with open(self.state_filepath, 'r') as f:
                state = json.load(f)

            self.done_success = state.get("done_success", [])
            self.done_duplicate = state.get("done_duplicate", [])
            self.done_short = state.get("done_short", [])
            self.failed_scrape = state.get("failed_scrape", [])
            self.failed_process = state.get("failed_process", [])

            # Rebuild pending list from what's not done
            done_urls = set()
            done_urls.update(item["url"] for item in self.done_success)
            done_urls.update(self.done_duplicate)
            done_urls.update(item["url"] for item in self.done_short)
            done_urls.update(item["url"] for item in self.failed_scrape)
            done_urls.update(item["url"] for item in self.failed_process)

            self.pending = [url for url in self.urls if url not in done_urls]

            logger.info(f"Resumed from previous state: {len(done_urls)} done, {len(self.pending)} pending")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}. Starting fresh.")

    def _save_state(self):
        """Save state to JSON file for crash recovery."""
        state = {
            "saved_at": datetime.now().isoformat(),
            "done_success": self.done_success,
            "done_duplicate": self.done_duplicate,
            "done_short": self.done_short,
            "failed_scrape": self.failed_scrape,
            "failed_process": self.failed_process,
            "total_retries": self.total_retries,
        }

        # Write atomically (write to temp, then rename)
        temp_path = self.state_filepath + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
            Path(temp_path).rename(self.state_filepath)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def get_pending_urls(self) -> List[str]:
        """Get list of URLs still pending."""
        return list(self.pending)

    def mark_success(self, url: str, title: str, tags: List[str]):
        """Mark URL as successfully processed."""
        if url in self.pending:
            self.pending.remove(url)
        self.done_success.append({
            "url": url,
            "title": title,
            "tags": tags,
            "completed_at": datetime.now().isoformat()
        })
        self.last_success_at = datetime.now()
        self._write()

    def mark_duplicate(self, url: str):
        """Mark URL as skipped (duplicate)."""
        if url in self.pending:
            self.pending.remove(url)
        self.done_duplicate.append(url)
        self._write()

    def mark_short(self, url: str, word_count: int):
        """Mark URL as skipped (too short)."""
        if url in self.pending:
            self.pending.remove(url)
        self.done_short.append({"url": url, "word_count": word_count})
        self._write()

    def mark_failed_scrape(self, url: str, error: str, retries: int = 0):
        """Mark URL as failed to scrape after all retries."""
        if url in self.pending:
            self.pending.remove(url)
        self.failed_scrape.append({
            "url": url,
            "error": error,
            "retries": retries,
            "failed_at": datetime.now().isoformat()
        })
        self._write()

    def mark_failed_process(self, url: str, error: str, retries: int = 0):
        """Mark URL as failed during LLM processing."""
        if url in self.pending:
            self.pending.remove(url)
        self.failed_process.append({
            "url": url,
            "error": error,
            "retries": retries,
            "failed_at": datetime.now().isoformat()
        })
        self._write()

    def increment_retries(self):
        """Track total retry count."""
        self.total_retries += 1

    def _write(self):
        """Write current state to tracking file."""
        # Save JSON state first (for crash recovery)
        self._save_state()

        # Now write human-readable tracking file
        total = len(self.urls)
        done_count = len(self.done_success) + len(self.done_duplicate) + len(self.done_short)
        failed_count = len(self.failed_scrape) + len(self.failed_process)
        pending_count = len(self.pending)

        # Calculate runtime stats
        runtime = datetime.now() - self.started_at
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        runtime_str = f"{hours}h {minutes}m {seconds}s"

        # Calculate rate
        processed = done_count + failed_count
        rate_per_hour = (processed / runtime.total_seconds() * 3600) if runtime.total_seconds() > 0 else 0

        # Estimate remaining time
        if rate_per_hour > 0 and pending_count > 0:
            remaining_hours = pending_count / rate_per_hour
            eta_str = f"{remaining_hours:.1f}h"
        else:
            eta_str = "calculating..."

        lines = []
        lines.append(f"# Knowledge Archive - Batch Progress Tracker")
        lines.append(f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Started: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Runtime: {runtime_str}")
        lines.append(f"#")
        lines.append(f"# Progress: {done_count + failed_count}/{total} ({pending_count} pending)")
        lines.append(f"#   Success: {len(self.done_success)}")
        lines.append(f"#   Duplicate: {len(self.done_duplicate)}")
        lines.append(f"#   Too Short: {len(self.done_short)}")
        lines.append(f"#   Failed: {failed_count}")
        lines.append(f"#")
        lines.append(f"# Rate: {rate_per_hour:.1f}/hour | ETA: {eta_str}")
        lines.append(f"# Total retries: {self.total_retries}")
        lines.append("")

        # Success section
        lines.append("=" * 60)
        lines.append(f"SUCCESS ({len(self.done_success)})")
        lines.append("=" * 60)
        for item in self.done_success:
            lines.append(f"{item['url']}")
            lines.append(f"  -> {item['title']}")
            lines.append(f"     Tags: {', '.join(item['tags'])}")
        lines.append("")

        # Duplicate section
        lines.append("=" * 60)
        lines.append(f"SKIPPED - DUPLICATE ({len(self.done_duplicate)})")
        lines.append("=" * 60)
        for url in self.done_duplicate:
            lines.append(url)
        lines.append("")

        # Too short section
        lines.append("=" * 60)
        lines.append(f"SKIPPED - TOO SHORT ({len(self.done_short)})")
        lines.append("=" * 60)
        for item in self.done_short:
            lines.append(f"{item['url']} ({item['word_count']} words)")
        lines.append("")

        # Failed scrape section
        lines.append("=" * 60)
        lines.append(f"FAILED - SCRAPE ({len(self.failed_scrape)})")
        lines.append("=" * 60)
        for item in self.failed_scrape:
            lines.append(f"{item['url']}")
            lines.append(f"  -> Error: {item['error']} (retries: {item.get('retries', 0)})")
        lines.append("")

        # Failed process section
        lines.append("=" * 60)
        lines.append(f"FAILED - PROCESS ({len(self.failed_process)})")
        lines.append("=" * 60)
        for item in self.failed_process:
            lines.append(f"{item['url']}")
            lines.append(f"  -> Error: {item['error']} (retries: {item.get('retries', 0)})")
        lines.append("")

        # Pending section
        lines.append("=" * 60)
        lines.append(f"PENDING ({len(self.pending)})")
        lines.append("=" * 60)
        for url in self.pending:
            lines.append(url)

        # Write atomically
        temp_path = self.filepath + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                f.write('\n'.join(lines))
            Path(temp_path).rename(self.filepath)
        except Exception as e:
            logger.error(f"Failed to write tracking file: {e}")


def check_internet_connection(timeout: int = 5) -> bool:
    """Check if internet connection is available."""
    test_hosts = [
        ("8.8.8.8", 53),      # Google DNS
        ("1.1.1.1", 53),      # Cloudflare DNS
        ("208.67.222.222", 53) # OpenDNS
    ]

    for host, port in test_hosts:
        try:
            socket.setdefaulttimeout(timeout)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.error, socket.timeout):
            continue

    return False


def wait_for_connection(max_wait: int = 300, check_interval: int = 10) -> bool:
    """
    Wait for internet connection to be restored.

    Args:
        max_wait: Maximum seconds to wait
        check_interval: Seconds between checks

    Returns:
        True if connection restored, False if timeout
    """
    logger.warning("Internet connection lost. Waiting for reconnection...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if shutdown_requested:
            return False

        if check_internet_connection():
            logger.info("Internet connection restored!")
            return True

        elapsed = int(time.time() - start_time)
        logger.info(f"Still waiting for connection... ({elapsed}s / {max_wait}s)")
        time.sleep(check_interval)

    logger.error(f"Connection not restored after {max_wait}s")
    return False


def scrape_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 30.0,
    tracker: Optional[RobustProgressTracker] = None
) -> Optional[Any]:
    """
    Scrape article with exponential backoff retry logic.

    Args:
        url: URL to scrape
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries (exponential backoff)
        tracker: Progress tracker for retry counting

    Returns:
        ScrapedContent or None if all retries failed
    """
    last_error = "Unknown error"

    for attempt in range(max_retries + 1):
        if shutdown_requested:
            return None

        try:
            # Check connection before attempting
            if not check_internet_connection():
                if not wait_for_connection():
                    return None

            result = scrape_article(url)
            return result

        except requests.exceptions.Timeout as e:
            last_error = f"Timeout: {e}"
            logger.warning(f"Scrape timeout for {url} (attempt {attempt + 1}/{max_retries + 1})")

        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"Connection error for {url} (attempt {attempt + 1}/{max_retries + 1})")

            # Wait for connection to be restored
            if not wait_for_connection():
                return None

        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {e}"
            logger.warning(f"Request error for {url}: {e} (attempt {attempt + 1}/{max_retries + 1})")

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.warning(f"Unexpected error scraping {url}: {e} (attempt {attempt + 1}/{max_retries + 1})")

        # Retry with exponential backoff
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retrying in {delay:.0f}s...")
            if tracker:
                tracker.increment_retries()

            # Sleep in chunks to allow for graceful shutdown
            sleep_until = time.time() + delay
            while time.time() < sleep_until:
                if shutdown_requested:
                    return None
                time.sleep(min(5, sleep_until - time.time()))

    logger.error(f"All {max_retries + 1} scrape attempts failed for {url}: {last_error}")
    return None


def process_with_retry(
    scraped: Any,
    url: str,
    model_tier: str,
    max_retries: int = 3,
    base_delay: float = 30.0,
    tracker: Optional[RobustProgressTracker] = None
) -> Optional[Any]:
    """
    Process article with LLM with exponential backoff retry logic.

    Args:
        scraped: ScrapedContent from scraper
        url: Original article URL
        model_tier: LLM tier to use
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        tracker: Progress tracker for retry counting

    Returns:
        KnowledgeArchiveEntry or None if all retries failed
    """
    last_error = "Unknown error"

    for attempt in range(max_retries + 1):
        if shutdown_requested:
            return None

        try:
            # Check connection before attempting
            if not check_internet_connection():
                if not wait_for_connection():
                    return None

            result = process_article(scraped, url, model_tier=model_tier)
            return result

        except requests.exceptions.Timeout as e:
            last_error = f"Timeout: {e}"
            logger.warning(f"LLM timeout for {url} (attempt {attempt + 1}/{max_retries + 1})")

        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"Connection error for {url} (attempt {attempt + 1}/{max_retries + 1})")

            if not wait_for_connection():
                return None

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"LLM response parse error for {url} (attempt {attempt + 1}/{max_retries + 1})")

        except Exception as e:
            last_error = f"Error: {e}"
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate" in error_str and "limit" in error_str:
                logger.warning(f"Rate limit hit for {url}, waiting longer...")
                base_delay = base_delay * 2  # Double the base delay for rate limits
            else:
                logger.warning(f"LLM error for {url}: {e} (attempt {attempt + 1}/{max_retries + 1})")

        # Retry with exponential backoff
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retrying LLM in {delay:.0f}s...")
            if tracker:
                tracker.increment_retries()

            # Sleep in chunks to allow for graceful shutdown
            sleep_until = time.time() + delay
            while time.time() < sleep_until:
                if shutdown_requested:
                    return None
                time.sleep(min(5, sleep_until - time.time()))

    logger.error(f"All {max_retries + 1} LLM attempts failed for {url}: {last_error}")
    return None


def parse_url_file(filepath: str) -> List[str]:
    """
    Parse URLs from text file, ignoring comments and blanks.

    Args:
        filepath: Path to text file with URLs

    Returns:
        List of URLs
    """
    urls = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                # Validate URL format
                if line.startswith('http://') or line.startswith('https://'):
                    urls.append(line)
                else:
                    logger.warning(f"Skipping invalid URL: {line}")
    return urls


def archive_batch_robust(
    urls: List[str],
    model_tier: str = "smart",
    delay_seconds: float = 60.0,
    verbose: bool = False,
    tracker: Optional[RobustProgressTracker] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Process a batch of URLs with robust error handling and retries.

    Args:
        urls: List of URLs to archive
        model_tier: LLM tier to use for processing
        delay_seconds: Delay between requests (default 60s for overnight)
        verbose: If True, print progress to console
        tracker: Progress tracker for real-time file updates
        max_retries: Maximum retries per article

    Returns:
        JSON-serializable report dict
    """
    global shutdown_requested

    db = KnowledgeArchiveDB()
    min_word_count = getattr(config, "knowledge_archive_min_word_count", 75)

    report = {
        "started_at": datetime.now().isoformat(),
        "total_urls": len(urls),
        "successful": 0,
        "skipped_duplicate": 0,
        "skipped_short": 0,
        "failed_scrape": 0,
        "failed_process": 0,
        "archive_org_fallbacks": 0,
        "total_retries": 0,
        "entries": [],
    }

    # If using tracker with resume, get pending URLs only
    if tracker:
        urls = tracker.get_pending_urls()
        logger.info(f"Processing {len(urls)} pending URLs")

    for i, url in enumerate(urls):
        if shutdown_requested:
            logger.info("Shutdown requested - stopping processing")
            break

        entry_report = {"url": url, "status": "unknown", "title": None, "error": None}

        # Check for duplicate
        try:
            if db.url_exists(url):
                entry_report["status"] = "skipped_duplicate"
                report["skipped_duplicate"] += 1
                report["entries"].append(entry_report)
                if tracker:
                    tracker.mark_duplicate(url)
                if verbose:
                    logger.info(f"[{i+1}/{len(urls)}] SKIP (duplicate): {url}")
                continue
        except Exception as e:
            logger.warning(f"Error checking duplicate: {e}")
            # Continue anyway - better to potentially duplicate than skip

        # Scrape with retry
        if verbose:
            logger.info(f"[{i+1}/{len(urls)}] Scraping: {url}")

        scraped = scrape_with_retry(url, max_retries=max_retries, tracker=tracker)

        if shutdown_requested:
            break

        if scraped is None:
            entry_report["status"] = "failed_scrape"
            entry_report["error"] = "All scrape attempts failed"
            report["failed_scrape"] += 1
            report["entries"].append(entry_report)
            if tracker:
                tracker.mark_failed_scrape(url, "All scrape attempts failed", max_retries)
            if verbose:
                logger.error(f"  FAILED: Could not scrape content after {max_retries + 1} attempts")
            continue

        if scraped.word_count < min_word_count:
            entry_report["status"] = "skipped_short"
            entry_report["error"] = f"Content too short ({scraped.word_count} words < {min_word_count})"
            report["skipped_short"] += 1
            report["entries"].append(entry_report)
            if tracker:
                tracker.mark_short(url, scraped.word_count)
            if verbose:
                logger.info(f"  SKIP: Content too short ({scraped.word_count} words)")
            continue

        if scraped.used_archive_org:
            report["archive_org_fallbacks"] += 1
            if verbose:
                logger.info(f"  Note: Using Archive.org fallback")

        # Process with LLM with retry
        entry = process_with_retry(
            scraped, url, model_tier,
            max_retries=max_retries,
            tracker=tracker
        )

        if shutdown_requested:
            break

        if entry is None:
            entry_report["status"] = "failed_process"
            entry_report["error"] = "All LLM attempts failed"
            report["failed_process"] += 1
            report["entries"].append(entry_report)
            if tracker:
                tracker.mark_failed_process(url, "All LLM attempts failed", max_retries)
            if verbose:
                logger.error(f"  FAILED: LLM processing failed after {max_retries + 1} attempts")
            continue

        # Save to database
        try:
            db.add_entry(entry)

            entry_report["status"] = "success"
            entry_report["title"] = entry.metadata.title
            entry_report["takeaway_count"] = entry.metadata.takeaway_count
            entry_report["tags"] = entry.metadata.tags
            entry_report["word_count"] = entry.metadata.word_count
            report["successful"] += 1

            if tracker:
                tracker.mark_success(url, entry.metadata.title, entry.metadata.tags)
            if verbose:
                logger.info(f"  SUCCESS: {entry.metadata.title}")
                logger.info(f"    Tags: {', '.join(entry.metadata.tags)}")

        except Exception as e:
            entry_report["status"] = "failed_process"
            entry_report["error"] = f"Database save failed: {e}"
            report["failed_process"] += 1
            if tracker:
                tracker.mark_failed_process(url, f"Database save failed: {e}", 0)
            if verbose:
                logger.error(f"  FAILED: Database save error: {e}")

        report["entries"].append(entry_report)

        # Delay between requests (except for last one)
        if i < len(urls) - 1 and not shutdown_requested:
            if verbose:
                logger.info(f"  Waiting {delay_seconds:.0f}s before next article...")

            # Sleep in chunks to allow for graceful shutdown
            sleep_until = time.time() + delay_seconds
            while time.time() < sleep_until:
                if shutdown_requested:
                    break
                time.sleep(min(5, sleep_until - time.time()))

    # Update report with final stats
    report["completed_at"] = datetime.now().isoformat()
    report["duration_seconds"] = (
        datetime.fromisoformat(report["completed_at"]) -
        datetime.fromisoformat(report["started_at"])
    ).total_seconds()

    if tracker:
        report["total_retries"] = tracker.total_retries

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Robust batch article archiver for overnight operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Overnight run with 1 article per minute (recommended)
    python scripts/batch_archive_articles.py urls.txt --track progress.txt -v

    # Resume from previous run
    python scripts/batch_archive_articles.py urls.txt --track progress.txt --resume -v

    # Faster processing (30s delay)
    python scripts/batch_archive_articles.py urls.txt --delay 30 --track progress.txt -v

    # With specific LLM tier and more retries
    python scripts/batch_archive_articles.py urls.txt --tier smart --retries 5 -v

Features:
    - Automatic retry with exponential backoff (default: 3 retries)
    - Internet connection monitoring and auto-reconnect
    - Resume capability (--resume flag)
    - Graceful shutdown on Ctrl+C (saves state)
    - Progress tracking with ETA calculation
"""
    )
    parser.add_argument("input_file", help="Text file with URLs (one per line)")
    parser.add_argument(
        "--tier",
        default="smart",
        choices=["fast", "smart", "strategic"],
        help="LLM tier for processing (default: smart)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON report file (default: print to stdout)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=60.0,
        help="Delay between articles in seconds (default: 60 for rate limit safety)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retry attempts per article (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to console"
    )
    parser.add_argument(
        "--track", "-t",
        default=None,
        help="Track progress to a file (updated in real-time)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from previous run (requires --track)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.resume and not args.track:
        print("Error: --resume requires --track to be specified", file=sys.stderr)
        sys.exit(1)

    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Parse URLs
    urls = parse_url_file(args.input_file)
    if not urls:
        print("No valid URLs found in input file", file=sys.stderr)
        sys.exit(1)

    # Check internet connection before starting
    if not check_internet_connection():
        print("Error: No internet connection", file=sys.stderr)
        sys.exit(1)

    # Print header
    print()
    print("=" * 60)
    print("  KNOWLEDGE ARCHIVE - ROBUST BATCH PROCESSOR")
    print("=" * 60)
    print(f"  Input file:    {args.input_file}")
    print(f"  Total URLs:    {len(urls)}")
    print(f"  LLM tier:      {args.tier}")
    print(f"  Delay:         {args.delay}s between articles")
    print(f"  Max retries:   {args.retries} per article")
    if args.track:
        print(f"  Progress file: {args.track}")
    if args.resume:
        print(f"  Resume mode:   YES")
    print()

    # Estimate time
    est_minutes = len(urls) * (args.delay / 60 + 0.5)  # delay + ~30s processing
    est_hours = est_minutes / 60
    print(f"  Estimated time: {est_hours:.1f} hours ({est_minutes:.0f} minutes)")
    print()
    print("  Press Ctrl+C to gracefully stop and save progress")
    print("=" * 60)
    print()

    # Create progress tracker if requested
    tracker = None
    if args.track:
        tracker = RobustProgressTracker(args.track, urls, resume=args.resume)
        pending = len(tracker.get_pending_urls())
        print(f"Progress tracker initialized: {pending} URLs pending")
        print()

    # Process batch
    try:
        report = archive_batch_robust(
            urls,
            model_tier=args.tier,
            delay_seconds=args.delay,
            verbose=args.verbose or args.output is None,
            tracker=tracker,
            max_retries=args.retries,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        report = {"status": "interrupted"}

    # Print summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if "total_urls" in report:
        print(f"  Total URLs:        {report['total_urls']}")
        print(f"  Successful:        {report['successful']}")
        print(f"  Skipped (dup):     {report['skipped_duplicate']}")
        print(f"  Skipped (short):   {report['skipped_short']}")
        print(f"  Failed (scrape):   {report['failed_scrape']}")
        print(f"  Failed (process):  {report['failed_process']}")
        print(f"  Archive.org used:  {report['archive_org_fallbacks']}")
        print(f"  Total retries:     {report.get('total_retries', 0)}")
        if "duration_seconds" in report:
            duration = report['duration_seconds']
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Duration:          {hours}h {minutes}m {seconds}s")
    else:
        print("  Run was interrupted before completion")
    print()

    if tracker:
        pending = len(tracker.get_pending_urls())
        if pending > 0:
            print(f"  Remaining: {pending} URLs pending")
            print(f"  Run with --resume to continue")
            print()

    # Output report
    if args.output and "total_urls" in report:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
