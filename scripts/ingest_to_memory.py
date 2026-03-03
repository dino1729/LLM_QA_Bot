"""
Unified content ingestion to Memory Palace.

Auto-detects input type (YouTube URL, article URL, local file) and routes
through the appropriate extraction pipeline, then uploads to all three
Memory Palace stores:
  1. EDITH Lessons (MemoryPalaceDB) - distilled single-line insights
  2. Local Memory Palace (save_memory) - full vector document
  3. Knowledge Archive (KnowledgeArchiveDB) - structured article data (URLs only)

Usage:
    # YouTube video
    python scripts/ingest_to_memory.py https://youtu.be/abc123

    # Article URL
    python scripts/ingest_to_memory.py https://paulgraham.com/ds.html

    # Local PDF
    python scripts/ingest_to_memory.py /path/to/paper.pdf

    # Custom takeaway count, automatic mode
    python scripts/ingest_to_memory.py https://youtu.be/abc123 --num 10 --auto

    # Skip EDITH distillation (faster, stores raw takeaways)
    python scripts/ingest_to_memory.py article.pdf --skip-distill
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from helper_functions.llm_client import get_client
from helper_functions.memory_palace_db import (
    MemoryPalaceDB,
    Lesson,
    LessonMetadata,
    LessonCategory,
    distill_lesson,
    is_objective_lesson_text,
)
from helper_functions.memory_palace_local import save_memory

logger = logging.getLogger(__name__)


# --- Input type detection ---

class InputType:
    YOUTUBE = "youtube"
    URL = "url"
    FILE = "file"


def detect_input_type(source: str) -> str:
    """Detect whether input is a YouTube URL, article URL, or local file."""
    if "youtube.com" in source or "youtu.be" in source:
        return InputType.YOUTUBE
    if source.startswith("http://") or source.startswith("https://"):
        return InputType.URL
    if os.path.exists(source):
        return InputType.FILE
    # Could be a URL without scheme
    if "." in source and "/" in source:
        return InputType.URL
    raise ValueError(f"Cannot determine input type for: {source}")


# --- Content extraction ---

@dataclass
class ExtractedContent:
    """Unified extraction result from any source."""
    text: str
    title: str
    source_type: str  # youtube, article, file
    source_ref: str   # URL or file path
    word_count: int


def _extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL (inlined to avoid heavy analyzers imports)."""
    match = re.search(r"youtu\.be\/([^?&]+)", url)
    if match:
        return match.group(1)
    if "youtube.com" in url:
        match = re.search(r"v=([^&]+)", url)
        if match:
            return match.group(1)
    return url


def _get_video_title(url: str, video_id: str) -> str:
    """Get YouTube video title via yt-dlp (graceful fallback if unavailable)."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", video_id)
    except Exception:
        return f"YouTube video {video_id}"


def extract_youtube(url: str) -> ExtractedContent:
    """Extract transcript from a YouTube video."""
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _extract_video_id(url)
    print(f"  Video ID: {video_id}")

    title = _get_video_title(url, video_id)
    print(f"  Title: {title}")

    # Extract transcript
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en-IN", "en"])
    text = " ".join([snippet.text for snippet in transcript])

    return ExtractedContent(
        text=text,
        title=title,
        source_type="video",
        source_ref=url,
        word_count=len(text.split()),
    )


def extract_article(url: str) -> ExtractedContent:
    """Extract content from an article URL via Firecrawl or newspaper3k fallback."""
    # Try Firecrawl first (best quality)
    try:
        from helper_functions.knowledge_archive_scraper import scrape_article
        scraped = scrape_article(url)
        if scraped and scraped.word_count >= 75:
            return ExtractedContent(
                text=scraped.content,
                title=scraped.title or url,
                source_type="article",
                source_ref=url,
                word_count=scraped.word_count,
            )
    except Exception as e:
        logger.info(f"Firecrawl scrape failed, trying newspaper3k: {e}")

    # Fallback to newspaper3k
    try:
        from newspaper import Article as NewsArticle
        article = NewsArticle(url)
        article.download()
        article.parse()
        if article.text and len(article.text.split()) >= 75:
            return ExtractedContent(
                text=article.text,
                title=article.title or url,
                source_type="article",
                source_ref=url,
                word_count=len(article.text.split()),
            )
    except Exception as e:
        logger.info(f"newspaper3k failed, trying requests+BS4: {e}")

    # Last resort: raw HTML scrape
    try:
        import requests as req
        from bs4 import BeautifulSoup
        resp = req.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            return ExtractedContent(
                text=text,
                title=soup.title.string if soup.title else url,
                source_type="article",
                source_ref=url,
                word_count=len(text.split()),
            )
    except Exception as e:
        logger.info(f"BS4 fallback failed: {e}")

    raise RuntimeError(f"All article extraction methods failed for: {url}")


def extract_file(path: str) -> ExtractedContent:
    """Extract text from a local file (PDF, DOCX, TXT, MD, CSV)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = file_path.suffix.lower()
    text = None

    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        import docx
        doc = docx.Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext in (".txt", ".md", ".csv"):
        text = file_path.read_text(errors="ignore")
    else:
        # Try reading as text
        text = file_path.read_text(errors="ignore")

    if not text or not text.strip():
        raise ValueError(f"No text content extracted from: {path}")

    return ExtractedContent(
        text=text,
        title=file_path.stem.replace("_", " ").replace("-", " ").title(),
        source_type="file",
        source_ref=str(file_path.resolve()),
        word_count=len(text.split()),
    )


# --- Takeaway extraction ---

def extract_takeaways(content: ExtractedContent, num_takeaways: int, tier: str) -> List[str]:
    """Use LLM to extract key takeaways from content."""
    client = get_client(provider="litellm", model_tier=tier)

    # Truncate to fit context window
    truncated = content.text[:15000] if len(content.text) > 15000 else content.text

    prompt = f"""You are analyzing content titled "{content.title}".

Extract exactly {num_takeaways} key takeaways. Each takeaway should be:
- A self-contained, actionable insight (2-3 sentences max)
- Specific enough to be useful, not generic platitudes
- Include reasoning, examples, or evidence where relevant

Format as a numbered list (1-{num_takeaways}). Each item starts with a bolded title phrase.

CONTENT:
{truncated}"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=3000,
    )

    # Parse numbered takeaways. Models sometimes emit variants like:
    # "1. ...", "1) ...", or markdown-wrapped "**1. ...**".
    numbered_prefix = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?\d+\s*[\.\)]\s*(?:\*\*)?\s*")
    takeaways = []
    current = ""
    for line in response.strip().split("\n"):
        stripped = line.strip()
        if numbered_prefix.match(stripped):
            if current.strip():
                takeaways.append(current.strip())
            current = numbered_prefix.sub("", stripped, count=1).strip()
        else:
            # Ignore preamble text before the first numbered item.
            if current.strip() and stripped:
                current += " " + stripped
    if current.strip():
        takeaways.append(current.strip())

    # Clean: remove markdown emphasis markers
    cleaned = []
    for t in takeaways:
        t = t.replace("**", "")
        if t.strip():
            cleaned.append(t.strip())

    # Filter out preamble lines that aren't actual takeaways
    preamble_patterns = re.compile(
        r"^(here are|based on|the following|below are|key takeaways|takeaways from)",
        re.IGNORECASE,
    )
    cleaned = [t for t in cleaned if len(t) > 40 and not preamble_patterns.search(t)]

    return cleaned[:num_takeaways]


def _normalize_skip_distill_takeaway(takeaway: str) -> Optional[str]:
    """Normalize raw takeaway text for skip-distill mode without calling LLM."""
    text = re.sub(r"\s+", " ", (takeaway or "")).replace("**", "").strip()
    if not text:
        return None

    rewrites = (
        (
            r"^\s*(?:based on|from)\s+(?:the\s+)?(?:video|podcast|article|book)[^:]*:\s*",
            "",
        ),
        (
            r"^\s*(?:based on|from)\s+(?:the\s+)?(?:video|podcast|article|book)\s*,\s*",
            "",
        ),
        (
            r"^\s*(?:here are|these are|the following|key takeaways|takeaways from)\b.*?:\s*",
            "",
        ),
    )
    for pattern, replacement in rewrites:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if not text.endswith((".", "!", "?")):
        text = f"{text}."
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    valid, _ = is_objective_lesson_text(text)
    if not valid:
        return None
    return text


# --- Upload to Memory Palace stores ---

def upload_to_edith(
    takeaways: List[str],
    content: ExtractedContent,
    skip_distill: bool = False,
) -> int:
    """Upload takeaways to EDITH's MemoryPalaceDB (distilled lessons)."""
    db = MemoryPalaceDB()
    successes = 0

    for i, takeaway in enumerate(takeaways, 1):
        try:
            if skip_distill:
                # Store as-is with keyword-based category
                from helper_functions.memory_palace_db import suggest_category
                normalized_takeaway = _normalize_skip_distill_takeaway(takeaway)
                if not normalized_takeaway:
                    print(f"  [{i}/{len(takeaways)}] SKIPPED: non-objective takeaway format")
                    continue

                category = suggest_category(normalized_takeaway)
                lesson = Lesson(
                    distilled_text=normalized_takeaway,
                    metadata=LessonMetadata(
                        category=LessonCategory(category),
                        source="manual",
                        original_input=f"[{content.title}] {takeaway}",
                        distilled_by_model="raw",
                        tags=[content.source_type],
                    ),
                )
            else:
                # Full LLM distillation
                few_shot = db.get_few_shot_examples(count=3)
                result = distill_lesson(takeaway, few_shot_examples=few_shot)
                valid_result, _ = is_objective_lesson_text(result.distilled_text)
                if not valid_result:
                    print(f"  [{i}/{len(takeaways)}] SKIPPED: distillation output remained source-referential")
                    continue
                lesson = Lesson(
                    distilled_text=result.distilled_text,
                    metadata=LessonMetadata(
                        category=LessonCategory(result.suggested_category),
                        source="manual",
                        original_input=f"[{content.title}] {takeaway}",
                        distilled_by_model=config.memory_palace_primary_model or "unknown",
                        tags=result.suggested_tags + [content.source_type],
                    ),
                )

            db.add_lesson(lesson)
            label = lesson.distilled_text[:70]
            print(f"  [{i}/{len(takeaways)}] {label}...")
            successes += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  [{i}/{len(takeaways)}] FAILED: {e}")

    return successes


def upload_to_local_memory(content: ExtractedContent, takeaways: List[str]):
    """Upload combined takeaways to Local Memory Palace."""
    combined = "\n".join([f"{i}. {t}" for i, t in enumerate(takeaways, 1)])
    return save_memory(
        title=content.title,
        content=combined,
        source_type=content.source_type,
        source_ref=content.source_ref,
        model_name="LITELLM",
    )


def upload_to_knowledge_archive(content: ExtractedContent):
    """Upload article to Knowledge Archive (URLs only, structured extraction)."""
    from helper_functions.knowledge_archive_scraper import scrape_article, ScrapedContent
    from helper_functions.knowledge_archive_processor import process_article
    from helper_functions.knowledge_archive_db import KnowledgeArchiveDB

    db = KnowledgeArchiveDB()

    # Check for duplicate
    if db.url_exists(content.source_ref):
        print("  Already in Knowledge Archive (skipped)")
        return

    # Build ScrapedContent from what we already have
    scraped = ScrapedContent(
        content=content.text,
        title=content.title,
        author=None,
        publish_date=None,
        word_count=content.word_count,
    )

    entry = process_article(scraped, content.source_ref)
    db.add_entry(entry)
    tags_str = ", ".join(entry.metadata.tags) if entry.metadata.tags else "none"
    print(f"  Archived: {entry.metadata.title} (tags: {tags_str})")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Ingest content into Memory Palace from YouTube, articles, or files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # YouTube video (interactive, 13 takeaways)
    python scripts/ingest_to_memory.py https://youtu.be/abc123

    # Article with 10 takeaways, automatic upload
    python scripts/ingest_to_memory.py https://paulgraham.com/ds.html --num 10 --auto

    # PDF file, skip distillation for speed
    python scripts/ingest_to_memory.py paper.pdf --skip-distill

    # Use strategic tier for better extraction
    python scripts/ingest_to_memory.py article.pdf --tier strategic
""",
    )
    parser.add_argument("source", help="YouTube URL, article URL, or local file path")
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=13,
        help="Number of takeaways to extract (default: 13)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip interactive approval, upload immediately",
    )
    parser.add_argument(
        "--skip-distill",
        action="store_true",
        help="Store takeaways as-is without EDITH distillation (faster)",
    )
    parser.add_argument(
        "--tier",
        default="smart",
        choices=["fast", "smart", "strategic"],
        help="LLM tier for takeaway extraction (default: smart)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy HTTP logs unless verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("helper_functions").setLevel(logging.WARNING)

    # Step 1: Detect input type
    try:
        input_type = detect_input_type(args.source)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"\nDetected input type: {input_type}")

    # Step 2: Extract content
    print(f"Extracting content...")
    try:
        if input_type == InputType.YOUTUBE:
            content = extract_youtube(args.source)
        elif input_type == InputType.URL:
            content = extract_article(args.source)
        else:
            content = extract_file(args.source)
    except Exception as e:
        print(f"Error extracting content: {e}")
        sys.exit(1)

    print(f"  Extracted {content.word_count} words from: {content.title}\n")

    # Step 3: Extract takeaways
    print(f"Extracting {args.num} takeaways (tier: {args.tier})...")
    try:
        takeaways = extract_takeaways(content, args.num, args.tier)
    except Exception as e:
        print(f"Error extracting takeaways: {e}")
        sys.exit(1)

    if not takeaways:
        print("No takeaways extracted. Exiting.")
        sys.exit(1)

    # Step 4: Show takeaways
    print(f"\n{'='*60}")
    print(f"  {len(takeaways)} TAKEAWAYS from: {content.title}")
    print(f"{'='*60}\n")
    for i, t in enumerate(takeaways, 1):
        print(f"  {i}. {t}\n")

    # Step 5: Get approval (unless --auto)
    if not args.auto:
        response = input("Upload to Memory Palace? [y/n]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

    # Step 6: Upload to all stores
    print(f"\n{'='*60}")

    # 6a: EDITH Lessons
    distill_label = "raw" if args.skip_distill else "with distillation"
    print(f"\n[1/3] Uploading to EDITH Lessons ({distill_label})...")
    edith_count = upload_to_edith(takeaways, content, skip_distill=args.skip_distill)
    print(f"  Done: {edith_count}/{len(takeaways)} lessons saved")

    # 6b: Local Memory Palace
    print(f"\n[2/3] Uploading to Local Memory Palace...")
    try:
        status = upload_to_local_memory(content, takeaways)
        print(f"  {status}")
    except Exception as e:
        print(f"  Failed: {e}")

    # 6c: Knowledge Archive (URLs only)
    if input_type in (InputType.YOUTUBE, InputType.URL):
        print(f"\n[3/3] Uploading to Knowledge Archive...")
        try:
            upload_to_knowledge_archive(content)
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print(f"\n[3/3] Knowledge Archive: skipped (local files not archived)")

    print(f"\n{'='*60}")
    print(f"Done! {edith_count} lessons ingested from: {content.title}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
