"""
Perplexity-backed search and direct web content extraction helpers.

This module replaces Firecrawl-based discovery in active runtime paths:
- Search/discovery uses LiteLLM's /v1/perplexity-search endpoint
- Full-page extraction uses direct HTTP fetches with newspaper3k/BeautifulSoup
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from bs4 import BeautifulSoup

from config import config

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def get_perplexity_search_url() -> str:
    """Return the configured Perplexity search endpoint."""
    configured = getattr(config, "perplexity_search_url", None)
    if configured:
        return configured.rstrip("/")
    return f"{config.litellm_base_url.rstrip('/')}/v1/perplexity-search"


def _perplexity_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {config.litellm_api_key}",
        "Content-Type": "application/json",
    }


def _extract_date(snippet: str) -> Optional[str]:
    match = DATE_RE.search(snippet or "")
    return match.group(1) if match else None


def search_with_perplexity(
    query: str,
    max_results: int = 5,
    timeout: int = 60,
) -> List[Dict[str, str]]:
    """
    Search via LiteLLM's Perplexity search endpoint.

    Returns a normalized list with title/url/snippet/date/source.
    """
    try:
        response = requests.post(
            get_perplexity_search_url(),
            headers=_perplexity_headers(),
            json={"query": query, "max_results": max_results},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("Perplexity search failed for %s: %s", query, e)
        return []

    results: List[Dict[str, str]] = []
    for item in payload.get("results", [])[:max_results]:
        url = item.get("url", "").strip()
        if not url:
            continue
        snippet = (item.get("snippet") or "").strip()
        results.append(
            {
                "title": (item.get("title") or "Untitled").strip(),
                "url": url,
                "snippet": snippet,
                "date": _extract_date(snippet) or "",
                "source": _extract_source_name(url),
            }
        )
    return results


async def search_with_perplexity_async(
    session: aiohttp.ClientSession,
    query: str,
    max_results: int = 5,
    timeout: int = 60,
) -> List[Dict[str, str]]:
    """Async Perplexity search variant for news fan-out flows."""
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.post(
            get_perplexity_search_url(),
            headers=_perplexity_headers(),
            json={"query": query, "max_results": max_results},
            timeout=client_timeout,
        ) as response:
            if response.status != 200:
                logger.error(
                    "Async Perplexity search failed for %s: status %s",
                    query,
                    response.status,
                )
                return []
            payload = await response.json()
    except Exception as e:
        logger.error("Async Perplexity search failed for %s: %s", query, e)
        return []

    results: List[Dict[str, str]] = []
    for item in payload.get("results", [])[:max_results]:
        url = item.get("url", "").strip()
        if not url:
            continue
        snippet = (item.get("snippet") or "").strip()
        results.append(
            {
                "title": (item.get("title") or "Untitled").strip(),
                "url": url,
                "snippet": snippet,
                "date": _extract_date(snippet) or "",
                "source": _extract_source_name(url),
            }
        )
    return results


def extract_web_content(url: str, timeout: int = 20) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract page text directly from the URL.

    Strategy:
    1. Try newspaper3k for article-quality extraction
    2. Fallback to direct HTTP + BeautifulSoup text extraction
    """
    if not url:
        return None, None

    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text.split()) >= 75:
            title = article.title.strip() if article.title else None
            return article.text.strip(), title
    except Exception:
        pass

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = soup.get_text(separator="\n", strip=True)
        if text and len(text.split()) >= 20:
            return text, title
    except Exception as e:
        logger.debug("Direct extraction failed for %s: %s", url, e)

    return None, None


async def extract_web_content_async(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 20,
) -> Tuple[Optional[str], Optional[str]]:
    """Async direct HTTP extraction for parallel news fetches."""
    if not url:
        return None, None

    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=client_timeout,
        ) as response:
            if response.status != 200:
                return None, None
            html = await response.text()
    except Exception as e:
        logger.debug("Async direct extraction failed for %s: %s", url, e)
        return None, None

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        text = soup.get_text(separator="\n", strip=True)
        if text and len(text.split()) >= 20:
            return text, title
    except Exception as e:
        logger.debug("Async parse failed for %s: %s", url, e)

    return None, None


def _extract_source_name(url: str) -> str:
    """Extract a readable source name from a URL."""
    try:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        domain = domain.replace("www.", "").split(".")[0]
        return domain.title()
    except Exception:
        return "Unknown"
