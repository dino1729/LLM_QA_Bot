"""
Knowledge Archive Scraper - Direct URL extraction with Archive.org fallback.

This module provides:
- Primary extraction via direct HTTP/news article parsing
- Automatic fallback to Archive.org Wayback Machine for paywalled/failed content
- Metadata extraction (title, content, word count)
- Domain extraction utility
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests

from config import config
from helper_functions.perplexity_search import extract_web_content

logger = logging.getLogger(__name__)

# Configuration
firecrawl_timeout = getattr(config, "knowledge_archive_firecrawl_timeout", 30)
archive_org_timeout = getattr(config, "knowledge_archive_archive_org_timeout", 15)
min_word_count = getattr(config, "knowledge_archive_min_word_count", 75)


@dataclass
class ScrapedContent:
    """Result of content scraping."""
    content: str
    title: Optional[str]
    author: Optional[str]
    publish_date: Optional[datetime]
    word_count: int
    used_archive_org: bool = False
    original_url_failed: bool = False


def scrape_article(url: str, timeout: int = None) -> Optional[ScrapedContent]:
    """
    Scrape article content using direct extraction with Archive.org fallback.

    Args:
        url: Article URL to scrape
        timeout: Request timeout in seconds (default from config)

    Returns:
        ScrapedContent or None if all methods failed
    """
    timeout = timeout or firecrawl_timeout

    # Try primary URL via direct extraction
    logger.info(f"Scraping {url} via direct extraction")
    content, title = _scrape_with_firecrawl(url, timeout)

    if content and len(content.split()) >= min_word_count:
        return _build_scraped_content(url, content, title, used_archive_org=False)

    # Fallback to Archive.org
    logger.info(f"Primary scrape failed for {url}, trying Archive.org")
    archive_content, archive_title = _scrape_archive_org(url, timeout)

    if archive_content and len(archive_content.split()) >= min_word_count:
        return _build_scraped_content(
            url, archive_content, archive_title or title,
            used_archive_org=True
        )

    logger.warning(f"All scrape methods failed for {url}")
    return None


def _scrape_with_firecrawl(url: str, timeout: int) -> tuple[Optional[str], Optional[str]]:
    """
    Compatibility wrapper: extract URL content directly without Firecrawl.

    Returns:
        Tuple of (content, title) or (None, None) if failed
    """
    try:
        return extract_web_content(url, timeout=timeout)
    except requests.exceptions.Timeout:
        logger.warning(f"Direct scrape timed out for {url}")
        return None, None
    except Exception as e:
        logger.warning(f"Error scraping {url} directly: {e}")
        return None, None


def _scrape_archive_org(url: str, timeout: int) -> tuple[Optional[str], Optional[str]]:
    """
    Scrape from Archive.org Wayback Machine.

    Returns:
        Tuple of (content, title) or (None, None) if failed
    """
    try:
        # Get latest snapshot
        availability_url = f"https://archive.org/wayback/available?url={url}"
        response = requests.get(availability_url, timeout=archive_org_timeout)

        if response.status_code != 200:
            logger.warning(f"Archive.org availability check failed for {url}")
            return None, None

        data = response.json()

        if "archived_snapshots" not in data:
            logger.info(f"No archived snapshots found for {url}")
            return None, None

        if "closest" not in data["archived_snapshots"]:
            logger.info(f"No closest snapshot found for {url}")
            return None, None

        snapshot = data["archived_snapshots"]["closest"]

        # Check if snapshot is available
        if not snapshot.get("available", False):
            logger.info(f"Snapshot not available for {url}")
            return None, None

        snapshot_url = snapshot["url"]
        logger.info(f"Found Archive.org snapshot: {snapshot_url}")

        # Scrape the archived version via direct extraction
        return _scrape_with_firecrawl(snapshot_url, timeout)

    except requests.exceptions.Timeout:
        logger.warning(f"Archive.org fallback timed out for {url}")
        return None, None
    except Exception as e:
        logger.warning(f"Archive.org fallback failed for {url}: {e}")
        return None, None


def _extract_title_from_url(url: str) -> Optional[str]:
    """
    Extract title from URL patterns for known newsletter formats.

    Args:
        url: Article URL

    Returns:
        Title string if pattern matched, None otherwise
    """
    url_lower = url.lower()

    # NZS Capital SITALWeek: /sitalweek/sitalweek-448 -> "SITALWeek #448"
    sitalweek_match = re.search(r'/sitalweek/sitalweek-(\d+)', url_lower)
    if sitalweek_match:
        issue_num = sitalweek_match.group(1)
        return f"SITALWeek #{issue_num}"

    # Farnam Street Brain Food: /brain-food/january-4-2026/ -> "Brain Food - January 4, 2026"
    brain_food_match = re.search(r'/brain-food/([a-z]+)-(\d+)-(\d{4})/?', url_lower)
    if brain_food_match:
        month = brain_food_match.group(1).capitalize()
        day = brain_food_match.group(2)
        year = brain_food_match.group(3)
        return f"Brain Food - {month} {day}, {year}"

    return None


def _is_bad_title(title: Optional[str]) -> bool:
    """
    Check if a title looks like a CTA, navigation element, or other non-title text.

    Args:
        title: Title string to check

    Returns:
        True if title appears to be bad/suspicious
    """
    if not title:
        return True

    title_lower = title.lower().strip()

    # Too short or too long
    if len(title_lower) < 5 or len(title_lower) > 200:
        return True

    # Common CTA/noise patterns
    bad_patterns = [
        "sign up",
        "subscribe",
        "newsletter",
        "click here",
        "read more",
        "learn more",
        "get started",
        "join now",
        "free trial",
        "download",
        "cookie",
        "privacy policy",
        "terms of service",
        "log in",
        "login",
        "register",
        "create account",
        "follow us",
        "share this",
    ]

    for pattern in bad_patterns:
        if pattern in title_lower:
            return True

    # Title is just a URL or domain
    if title_lower.startswith("http") or title_lower.startswith("www."):
        return True

    return False


def _build_scraped_content(
    url: str,
    content: str,
    title: Optional[str],
    used_archive_org: bool
) -> ScrapedContent:
    """Build ScrapedContent from raw scraped data."""
    # Clean content
    content = _clean_content(content)

    # Extract title from content if not provided or if title looks suspicious
    if not title or _is_bad_title(title):
        # Try URL-based title extraction first (for known newsletter patterns)
        url_title = _extract_title_from_url(url)
        if url_title:
            logger.info(f"Using URL-based title: {url_title}")
            title = url_title
        else:
            # Fall back to content extraction
            extracted_title = _extract_title_from_content(content)
            if extracted_title and not _is_bad_title(extracted_title):
                logger.info(f"Using extracted title instead of metadata: {extracted_title[:50]}...")
                title = extracted_title

    return ScrapedContent(
        content=content,
        title=title or "Unknown Title",
        author=None,  # Will be extracted by LLM in processor
        publish_date=None,  # Will be extracted by LLM in processor
        word_count=len(content.split()),
        used_archive_org=used_archive_org,
        original_url_failed=used_archive_org,
    )


def _clean_content(content: str) -> str:
    """Clean scraped content."""
    if not content:
        return ""

    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r' {2,}', ' ', content)

    # Remove common noise patterns
    noise_patterns = [
        r'Share this article.*?(?=\n|$)',
        r'Follow us on.*?(?=\n|$)',
        r'Subscribe to.*?(?=\n|$)',
        r'Cookie.*?(?=\n|$)',
        r'Privacy Policy.*?(?=\n|$)',
    ]

    for pattern in noise_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    return content.strip()


def _extract_title_from_content(content: str) -> Optional[str]:
    """Extract title from markdown content."""
    if not content:
        return None

    lines = content.strip().split('\n')

    for line in lines[:15]:  # Check first 15 lines
        line = line.strip()

        # Skip empty lines and links
        if not line or line.startswith('[') or line.startswith('!['):
            continue

        # Look for markdown H1
        if line.startswith('# '):
            title = line[2:].strip()
            # Clean markdown formatting
            title = _clean_markdown_title(title)
            if title and len(title) > 5:
                return title

        # Look for markdown H2 (sometimes used as main title)
        if line.startswith('## '):
            title = line[3:].strip()
            title = _clean_markdown_title(title)
            if title and len(title) > 5:
                return title

        # Look for bold text (handles **text** and **text _with_ formatting**)
        if line.startswith('**'):
            # Extract text between first ** and last **
            if '**' in line[2:]:
                end_idx = line.rfind('**')
                if end_idx > 2:
                    title = line[2:end_idx].strip()
                    title = _clean_markdown_title(title)
                    if title and len(title) > 5 and not _is_bad_title(title):
                        return title

    # Fallback: look for first substantial line that's not a link or CTA
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like links, CTAs, or navigation
        if line.startswith('[') or line.startswith('![') or line.startswith('http'):
            continue
        if line.lower().startswith('click') or line.lower().startswith('subscribe'):
            continue

        # Clean and validate
        title = _clean_markdown_title(line)
        if title and len(title) > 10 and len(title) < 150:
            if not _is_bad_title(title):
                return title

    return None


def _clean_markdown_title(title: str) -> str:
    """Remove markdown formatting from title string."""
    if not title:
        return ""

    # Remove bold markers
    title = title.replace('**', '')
    # Remove italic markers (both * and _)
    title = title.replace('_', ' ').replace('*', '')
    # Remove link syntax [text](url) -> text
    import re
    title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)
    # Clean up multiple spaces
    title = ' '.join(title.split())

    return title.strip()


def extract_domain(url: str) -> str:
    """
    Extract source domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain without www prefix (e.g., "paulgraham.com")
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def estimate_read_time(word_count: int) -> int:
    """
    Estimate reading time in minutes.

    Args:
        word_count: Number of words

    Returns:
        Estimated read time (minimum 1 minute)
    """
    # Average reading speed: 200-250 WPM, use 200 for non-fiction
    return max(1, word_count // 200)
