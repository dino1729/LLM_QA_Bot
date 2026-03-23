"""
Shared link ingestion helpers for Memory Palace workflows.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import config
from helper_functions.knowledge_archive_db import KnowledgeArchiveDB
from helper_functions.llm_client import get_client
from helper_functions.memory_palace_db import (
    Lesson,
    LessonCategory,
    LessonMetadata,
    MemoryPalaceDB,
    distill_lesson,
    is_objective_lesson_text,
)
from helper_functions.memory_palace_local import save_memory

logger = logging.getLogger(__name__)


class InputType:
    YOUTUBE = "youtube"
    URL = "url"
    FILE = "file"


@dataclass
class ExtractedContent:
    """Unified extraction result from any source."""

    text: str
    title: str
    source_type: str
    source_ref: str
    word_count: int


@dataclass
class LinkPreview:
    """Preview state for a URL or file before saving."""

    content: ExtractedContent
    takeaways: List[str]
    already_archived: bool = False


@dataclass
class LinkSaveResult:
    """Summary of save actions after user confirmation."""

    edith_count: int
    local_memory_status: str
    archive_status: str


def detect_input_type(source: str) -> str:
    """Detect whether input is a YouTube URL, article URL, or local file."""
    if "youtube.com" in source or "youtu.be" in source:
        return InputType.YOUTUBE
    if source.startswith("http://") or source.startswith("https://"):
        return InputType.URL
    if os.path.exists(source):
        return InputType.FILE
    if "." in source and "/" in source:
        return InputType.URL
    raise ValueError(f"Cannot determine input type for: {source}")


def _extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    match = re.search(r"youtu\.be\/([^?&]+)", url)
    if match:
        return match.group(1)
    if "youtube.com" in url:
        match = re.search(r"v=([^&]+)", url)
        if match:
            return match.group(1)
    return url


def _get_video_title(url: str, video_id: str) -> str:
    """Get YouTube video title via yt-dlp with graceful fallback."""
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
    title = _get_video_title(url, video_id)

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en-IN", "en"])
    text = " ".join(snippet.text for snippet in transcript)

    return ExtractedContent(
        text=text,
        title=title,
        source_type="video",
        source_ref=url,
        word_count=len(text.split()),
    )


def extract_article(url: str) -> ExtractedContent:
    """Extract content from an article URL via scraper, newspaper3k, or raw HTML."""
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
    except Exception as exc:
        logger.info("Primary article scrape failed, trying newspaper3k: %s", exc)

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
    except Exception as exc:
        logger.info("newspaper3k failed, trying requests+BS4: %s", exc)

    try:
        import requests as req
        from bs4 import BeautifulSoup

        response = req.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            return ExtractedContent(
                text=text,
                title=soup.title.string if soup.title else url,
                source_type="article",
                source_ref=url,
                word_count=len(text.split()),
            )
    except Exception as exc:
        logger.info("BS4 fallback failed: %s", exc)

    raise RuntimeError(f"All article extraction methods failed for: {url}")


def extract_file(path: str) -> ExtractedContent:
    """Extract text from a local file."""
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
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif ext in (".txt", ".md", ".csv"):
        text = file_path.read_text(errors="ignore")
    else:
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


def parse_takeaway_text(
    text: str,
    limit: Optional[int] = None,
    min_length: int = 40,
) -> List[str]:
    """Parse numbered takeaway text into a clean list."""
    if not text:
        return []

    numbered_prefix = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?\d+\s*[\.\)]\s*(?:\*\*)?\s*")
    takeaways: List[str] = []
    current = ""

    for line in text.strip().split("\n"):
        stripped = line.strip()
        if numbered_prefix.match(stripped):
            if current.strip():
                takeaways.append(current.strip())
            current = numbered_prefix.sub("", stripped, count=1).strip()
        elif current.strip() and stripped:
            current += " " + stripped

    if current.strip():
        takeaways.append(current.strip())

    if not takeaways:
        takeaways = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned = []
    for takeaway in takeaways:
        cleaned_takeaway = takeaway.replace("**", "").strip()
        if cleaned_takeaway:
            cleaned.append(cleaned_takeaway)

    preamble_patterns = re.compile(
        r"^(here are|based on|the following|below are|key takeaways|takeaways from)",
        re.IGNORECASE,
    )
    filtered = [
        takeaway
        for takeaway in cleaned
        if len(takeaway) >= min_length and not preamble_patterns.search(takeaway)
    ]

    if limit is None:
        return filtered
    return filtered[:limit]


def extract_takeaways(
    content: ExtractedContent,
    num_takeaways: int,
    tier: str,
) -> List[str]:
    """Use an LLM to extract key takeaways from content."""
    client = get_client(provider="litellm", model_tier=tier)
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

    return parse_takeaway_text(response, limit=num_takeaways)


def _normalize_skip_distill_takeaway(takeaway: str) -> Optional[str]:
    """Normalize raw takeaway text for skip-distill mode without another LLM call."""
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


def upload_to_edith(
    takeaways: List[str],
    content: ExtractedContent,
    skip_distill: bool = False,
    per_item_delay_seconds: float = 0.3,
) -> int:
    """Upload takeaways to the lesson store."""
    db = MemoryPalaceDB()
    successes = 0

    for i, takeaway in enumerate(takeaways, 1):
        try:
            if skip_distill:
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
                few_shot = db.get_few_shot_examples(count=3)
                result = distill_lesson(takeaway, few_shot_examples=few_shot)
                valid_result, _ = is_objective_lesson_text(result.distilled_text)
                if not valid_result:
                    print(
                        f"  [{i}/{len(takeaways)}] SKIPPED: distillation output remained source-referential"
                    )
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
            if per_item_delay_seconds > 0:
                time.sleep(per_item_delay_seconds)
        except Exception as exc:
            print(f"  [{i}/{len(takeaways)}] FAILED: {exc}")

    return successes


def upload_to_local_memory(content: ExtractedContent, takeaways: List[str]) -> str:
    """Upload combined takeaways to local Memory Palace storage."""
    combined = "\n".join(f"{i}. {takeaway}" for i, takeaway in enumerate(takeaways, 1))
    return save_memory(
        title=content.title,
        content=combined,
        source_type=content.source_type,
        source_ref=content.source_ref,
        model_name="LITELLM",
    )


def upload_to_knowledge_archive(content: ExtractedContent) -> str:
    """Upload a URL-backed content item to the Knowledge Archive."""
    from helper_functions.knowledge_archive_processor import process_article
    from helper_functions.knowledge_archive_scraper import ScrapedContent

    db = KnowledgeArchiveDB()
    if db.url_exists(content.source_ref):
        return "Already in Knowledge Archive (skipped)"

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
    return f"Archived: {entry.metadata.title} (tags: {tags_str})"


def prepare_link_preview(
    source: str,
    num_takeaways: int = 5,
    tier: str = "fast",
) -> LinkPreview:
    """Extract content and preview takeaways before saving."""
    input_type = detect_input_type(source)

    if input_type == InputType.URL:
        archive_db = KnowledgeArchiveDB()
        archived_entry = archive_db.get_entry_by_url(source)
        if archived_entry:
            archived_takeaways = parse_takeaway_text(
                archived_entry.takeaways,
                limit=num_takeaways,
                min_length=1,
            )
            if not archived_takeaways and archived_entry.summary:
                archived_takeaways = [archived_entry.summary]
            if not archived_takeaways:
                raise RuntimeError("No takeaways extracted.")

            archived_text = "\n\n".join(
                part
                for part in (archived_entry.summary, archived_entry.takeaways)
                if part
            )
            return LinkPreview(
                content=ExtractedContent(
                    text=archived_text,
                    title=archived_entry.metadata.title,
                    source_type="article",
                    source_ref=source,
                    word_count=archived_entry.metadata.word_count,
                ),
                takeaways=archived_takeaways,
                already_archived=True,
            )

    if input_type == InputType.YOUTUBE:
        content = extract_youtube(source)
    elif input_type == InputType.URL:
        content = extract_article(source)
    else:
        content = extract_file(source)

    takeaways = extract_takeaways(content, num_takeaways, tier)
    if not takeaways:
        raise RuntimeError("No takeaways extracted.")

    return LinkPreview(
        content=content,
        takeaways=takeaways,
        already_archived=False,
    )


def save_link_preview(
    preview: LinkPreview,
    save_archive: bool = False,
    skip_distill: bool = False,
    edith_delay_seconds: float = 0.3,
) -> LinkSaveResult:
    """Persist a confirmed preview to the configured stores."""
    edith_count = upload_to_edith(
        preview.takeaways,
        preview.content,
        skip_distill=skip_distill,
        per_item_delay_seconds=edith_delay_seconds,
    )

    try:
        local_memory_status = upload_to_local_memory(preview.content, preview.takeaways)
    except Exception as exc:
        local_memory_status = f"Local Memory upload failed: {exc}"

    archive_status = "Archive skipped"
    if save_archive:
        if preview.already_archived:
            archive_status = "Already in Knowledge Archive (skipped)"
        else:
            try:
                archive_status = upload_to_knowledge_archive(preview.content)
            except Exception as exc:
                archive_status = f"Knowledge Archive failed: {exc}"

    return LinkSaveResult(
        edith_count=edith_count,
        local_memory_status=local_memory_status,
        archive_status=archive_status,
    )
