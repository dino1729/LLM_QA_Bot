"""
Knowledge Archive LLM Processing - Two-pass extraction for articles.

Pass 1: Summary + Takeaways extraction
Pass 2: Auto-tag generation

This module provides:
- Two-pass LLM extraction for high-quality article distillation
- Configurable model tier per-run
- Structured JSON output parsing
"""
import json
import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

from config import config
from helper_functions.llm_client import get_client
from helper_functions.knowledge_archive_scraper import (
    ScrapedContent,
    extract_domain,
    estimate_read_time,
)
from helper_functions.knowledge_archive_db import (
    KnowledgeArchiveEntry,
    KnowledgeArchiveMetadata,
)

logger = logging.getLogger(__name__)

# Configuration
default_model_tier = getattr(config, "knowledge_archive_default_tier", "smart")


def process_article(
    scraped: ScrapedContent,
    url: str,
    model_tier: str = None,
    provider: str = None,
) -> KnowledgeArchiveEntry:
    """
    Process scraped article through two-pass LLM extraction.

    Args:
        scraped: ScrapedContent from scraper
        url: Original article URL
        model_tier: LLM tier to use (default: smart, can be overridden per-run)
        provider: LLM provider (default from config)

    Returns:
        KnowledgeArchiveEntry ready for storage
    """
    provider = provider or getattr(config, "memory_palace_provider", "litellm")
    model_tier = model_tier or default_model_tier

    # Determine takeaway count based on word count
    # 3 for articles < 2000 words, up to 8 for >= 2000 words
    takeaway_count = 3 if scraped.word_count < 2000 else 8

    # Pass 1: Extract summary and takeaways
    logger.info(f"Pass 1: Extracting summary and {takeaway_count} takeaways")
    summary, takeaways, author, publish_date = _extract_summary_and_takeaways(
        scraped.content,
        takeaway_count,
        provider,
        model_tier,
    )

    # Pass 2: Generate auto-tags
    logger.info("Pass 2: Generating tags")
    tags = _generate_tags(summary, takeaways, provider, model_tier)

    # Get model name for metadata
    client = get_client(provider=provider, model_tier=model_tier)
    model_name = getattr(client, "model_name", None) or f"{provider}:{model_tier}"

    # Use scraped title or extracted one
    title = scraped.title or "Unknown Title"

    # Count actual takeaways from the text
    actual_takeaway_count = len([line for line in takeaways.split('\n') if line.strip() and line.strip()[0].isdigit()])
    if actual_takeaway_count == 0:
        actual_takeaway_count = takeaway_count  # Fallback to expected count

    return KnowledgeArchiveEntry(
        summary=summary,
        takeaways=takeaways,
        content_preview=scraped.content[:500] if scraped.content else "",
        metadata=KnowledgeArchiveMetadata(
            url=url,
            title=title,
            word_count=scraped.word_count,
            takeaway_count=actual_takeaway_count,
            author=author or scraped.author,
            publish_date=publish_date or scraped.publish_date,
            source_domain=extract_domain(url),
            estimated_read_time=estimate_read_time(scraped.word_count),
            distilled_by_model=model_name,
            tags=tags,
            archive_org_fallback=scraped.used_archive_org,
            original_url_failed=scraped.original_url_failed,
        )
    )


def _extract_summary_and_takeaways(
    content: str,
    takeaway_count: int,
    provider: str,
    model_tier: str,
) -> Tuple[str, str, Optional[str], Optional[datetime]]:
    """
    Pass 1: Extract summary and takeaways from article content.

    Returns:
        (summary, takeaways, author, publish_date)
    """
    client = get_client(provider=provider, model_tier=model_tier)

    # Limit content to fit context window (roughly 15k chars is safe)
    truncated_content = content[:15000] if len(content) > 15000 else content

    prompt = f"""You are a knowledge curator distilling articles into searchable wisdom.

ARTICLE CONTENT:
{truncated_content}

TASK:
Extract the following and respond with valid JSON only:

1. "summary": A rich 3-4 sentence summary capturing the CORE IDEAS and KEY CONCEPTS
2. "takeaways": Exactly {takeaway_count} key insights, numbered 1-{takeaway_count}, each on its own line
3. "author": The article author's name if mentioned (null if not found)
4. "publish_date": The publication date in YYYY-MM-DD format if mentioned (null if not found)

CRITICAL GUIDELINES for summary:
- Get straight to the substance - NO meta-references like "This article discusses..." or "The author explores..."
- Lead with the main insight or thesis directly
- Include specific concepts, frameworks, or mental models mentioned
- Write as if explaining the core idea to someone who hasn't read it
- BAD: "This newsletter explores how writing helps reasoning"
- GOOD: "Writing forces clarity of thought - the act of putting ideas on paper reveals gaps in understanding and strengthens reasoning"

GUIDELINES for takeaways:
- Each takeaway should be a standalone, actionable insight with enough context to be useful
- Include the WHY or HOW, not just the WHAT
- Use active voice and present tense
- Be specific with concrete examples or applications when possible
- BAD: "Writing helps with thinking"
- GOOD: "Use writing as a thinking tool: before making a decision, write out your reasoning - the gaps in your logic will become visible on paper"
- Number format: "1. Insight here"

Respond with valid JSON only (no markdown code blocks):
{{"summary": "...", "takeaways": "1. ...\\n2. ...", "author": "...", "publish_date": "..."}}"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        # Parse JSON (handle markdown code blocks)
        result = _parse_json_response(response)

        author = result.get("author")
        if author == "null" or author == "":
            author = None

        publish_date = None
        publish_date_str = result.get("publish_date")
        if publish_date_str and publish_date_str != "null":
            try:
                publish_date = datetime.strptime(publish_date_str, "%Y-%m-%d")
            except ValueError:
                pass

        summary = result.get("summary", "")
        takeaways = result.get("takeaways", "")

        return summary, takeaways, author, publish_date

    except Exception as e:
        logger.error(f"Pass 1 extraction failed: {e}")
        # Fallback: use first paragraph as summary
        paragraphs = content.split('\n\n')
        summary = paragraphs[0][:500] if paragraphs else content[:500]
        return summary, "", None, None


def _generate_tags(
    summary: str,
    takeaways: str,
    provider: str,
    model_tier: str,
) -> List[str]:
    """
    Pass 2: Generate auto-tags from extracted content.

    Returns:
        List of 3-5 lowercase tags
    """
    client = get_client(provider=provider, model_tier=model_tier)

    prompt = f"""Generate 3-5 tags for this article based on its content.

SUMMARY: {summary}

KEY TAKEAWAYS: {takeaways}

RULES:
- Tags must be lowercase, single words or hyphenated (e.g., "machine-learning")
- Focus on topics, not style (avoid "interesting", "important")
- Include at least one broad category tag (e.g., "technology", "business", "science")
- Include specific topic tags (e.g., "ai", "startups", "climate-change")

Respond with JSON array only (no markdown code blocks):
["tag1", "tag2", "tag3"]"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,  # Increased to avoid truncation
        )

        tags = _parse_json_response(response)

        if isinstance(tags, list):
            return [tag.lower().strip() for tag in tags if isinstance(tag, str)][:5]
        else:
            logger.warning(f"Unexpected tag response format: {type(tags)}")
            return []

    except Exception as e:
        # Try to extract tags from truncated response
        if response:
            tag_matches = re.findall(r'"([a-z][a-z0-9-]*)"', response.lower())
            if tag_matches:
                logger.info(f"Extracted {len(tag_matches)} tags from truncated response")
                return tag_matches[:5]
        logger.warning(f"Pass 2 tag generation failed: {e}")
        return []


def _parse_json_response(response: str) -> any:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON object (dict or list)
    """
    response_text = response.strip()

    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first line (```json or ```)
        if lines:
            lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()

    # Try to parse JSON directly first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to find and extract JSON object or array
    # Find the first { or [ and match to its closing bracket
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = response_text.find(start_char)
        if start_idx == -1:
            continue

        # Find matching closing bracket by counting depth
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response_text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    # Found matching bracket
                    json_str = response_text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break

    # Last resort: try to find simple array pattern for tags
    # Handle cases like: ["tag1", "tag2", "tag3"] or truncated ["tag1", "tag2",
    array_match = re.search(r'\[([^\]]*)\]?', response_text)
    if array_match:
        array_content = array_match.group(0)
        # Try to fix truncated array by closing it
        if not array_content.endswith(']'):
            # Remove trailing comma and whitespace, then close
            array_content = array_content.rstrip(', \t\n') + ']'
        try:
            return json.loads(array_content)
        except json.JSONDecodeError:
            # Try to extract quoted strings manually
            tag_matches = re.findall(r'"([^"]+)"', array_match.group(1) if array_match.group(1) else array_match.group(0))
            if tag_matches:
                return tag_matches

    logger.error(f"Failed to parse JSON from response")
    logger.debug(f"Response was: {response_text[:500]}")
    raise json.JSONDecodeError("Could not extract JSON", response_text, 0)
