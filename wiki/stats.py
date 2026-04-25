"""
Wiki statistics for /wikistats command and monitoring.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from wiki.page_writer import PageWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WikiStats:
    """Comprehensive wiki statistics."""

    total_pages: int
    concepts: int
    entities: int
    summaries: int
    query_outputs: int
    total_wikilinks: int
    avg_sources_per_concept: float
    top_connected: List[tuple]  # [(page_name, link_count), ...]
    coverage_percent: float  # % of Palace entries in wiki
    raw_articles: int


def compute_wiki_stats(vault_root: Path, palace_entry_count: int = 575) -> WikiStats:
    """Compute comprehensive wiki stats."""
    page_writer = PageWriter(vault_root)

    counts = {}
    wiki_dirs = {
        "concepts": vault_root / "wiki" / "concepts",
        "entities": vault_root / "wiki" / "entities",
        "summaries": vault_root / "wiki" / "summaries",
    }
    for subdir, dir_path in wiki_dirs.items():
        if dir_path.exists():
            counts[subdir] = len(list(dir_path.rglob("*.md")))
        else:
            counts[subdir] = 0

    query_dir = vault_root / "outputs" / "queries"
    counts["query_outputs"] = len(list(query_dir.rglob("*.md"))) if query_dir.exists() else 0

    total = counts["concepts"] + counts["entities"] + counts["summaries"]

    # Count raw articles
    raw_dir = vault_root / "raw" / "articles"
    raw_count = len(list(raw_dir.glob("*.md"))) if raw_dir.exists() else 0

    # Count wikilinks and find top connected
    import re
    wikilink_pattern = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")
    link_counts: Dict[str, int] = {}

    total_links = 0
    total_sources_sum = 0
    concept_count = 0

    for subdir, dir_path in wiki_dirs.items():
        if not dir_path.exists():
            continue
        for page_path in dir_path.rglob("*.md"):
            content = page_path.read_text(encoding="utf-8")
            links = wikilink_pattern.findall(content)
            link_count = len(links)
            total_links += link_count
            link_counts[page_path.stem] = link_count

            if subdir == "concepts":
                metadata = page_writer.read_page_metadata(page_path)
                if metadata:
                    src = metadata.sources
                    total_sources_sum += src if isinstance(src, int) else len(src) if isinstance(src, list) else 0
                    concept_count += 1

    top_connected = sorted(link_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    avg_sources = total_sources_sum / concept_count if concept_count > 0 else 0
    coverage = (total / palace_entry_count * 100) if palace_entry_count > 0 else 0

    return WikiStats(
        total_pages=total,
        concepts=counts.get("concepts", 0),
        entities=counts.get("entities", 0),
        summaries=counts.get("summaries", 0),
        query_outputs=counts.get("query_outputs", 0),
        total_wikilinks=total_links,
        avg_sources_per_concept=avg_sources,
        top_connected=top_connected,
        coverage_percent=min(coverage, 100.0),
        raw_articles=raw_count,
    )


def format_stats_telegram(stats: WikiStats) -> str:
    """Format stats for Telegram message."""
    lines = [
        "Wiki Stats",
        f"  Total pages: {stats.total_pages}",
        f"  Concepts: {stats.concepts}",
        f"  Entities: {stats.entities}",
        f"  Summaries: {stats.summaries}",
        f"  Query outputs: {stats.query_outputs}",
        f"  Raw articles: {stats.raw_articles}",
        "",
        f"  Wikilinks: {stats.total_wikilinks}",
        f"  Avg sources/concept: {stats.avg_sources_per_concept:.1f}",
        f"  Palace coverage: {stats.coverage_percent:.0f}%",
    ]

    if stats.top_connected:
        lines.append("")
        lines.append("Top 5 most connected:")
        for name, count in stats.top_connected[:5]:
            display = name.replace("-", " ").title()
            lines.append(f"  {display}: {count} links")

    return "\n".join(lines)
