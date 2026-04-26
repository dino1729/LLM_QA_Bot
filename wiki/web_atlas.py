"""
Read-only web atlas for the generated Memory Wiki vault.

This module intentionally stays deterministic: it scans Markdown files on each
request, parses frontmatter, resolves Obsidian wikilinks, and exposes compact
JSON for the React atlas UI without invoking LLMs or mutating the vault.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import yaml
from fastapi import APIRouter, HTTPException, Query


WIKI_SECTIONS = ("concepts", "entities", "summaries")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class WikiLink:
    raw_target: str
    label: str
    resolved_id: str = ""
    resolved_title: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "target": self.raw_target,
            "label": self.label,
            "resolved": bool(self.resolved_id),
            "id": self.resolved_id,
            "title": self.resolved_title,
            "url": self.url,
        }


@dataclass
class PageRecord:
    section: str
    slug: str
    path: Path
    relative_path: str
    title: str
    page_type: str
    category: str
    tags: List[str]
    sources: List[str]
    source_count: int
    confidence: str
    last_updated: str
    body_markdown: str
    render_markdown: str = ""
    summary: str = ""
    headings: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    outgoing_links: List[WikiLink] = field(default_factory=list)
    backlinks: List["PageRecord"] = field(default_factory=list, repr=False)

    @property
    def page_id(self) -> str:
        return f"{self.section}/{self.slug}"

    @property
    def url(self) -> str:
        return f"/wiki/{self.section}/{self.slug}"

    def summary_dict(self) -> dict:
        return {
            "id": self.page_id,
            "section": self.section,
            "slug": self.slug,
            "url": self.url,
            "title": self.title,
            "type": self.page_type,
            "summary": self.summary,
            "category": self.category,
            "tags": self.tags,
            "sources": self.sources,
            "source_count": self.source_count,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "path": self.relative_path,
            "quality_flags": self.quality_flags,
            "outgoing_count": len(self.outgoing_links),
            "backlink_count": len(self.backlinks),
        }

    def detail_dict(self) -> dict:
        data = self.summary_dict()
        data.update(
            {
                "body_markdown": self.body_markdown,
                "render_markdown": self.render_markdown,
                "headings": self.headings,
                "outgoing_links": [link.to_dict() for link in self.outgoing_links],
                "backlinks": [page.summary_dict() for page in self.backlinks],
            }
        )
        return data


class WikiAtlas:
    """Live read-only view over a generated Memory Wiki vault."""

    def __init__(self, vault_root: Path | str) -> None:
        self.vault_root = Path(vault_root)

    def summary(self) -> dict:
        pages = self._load_pages()
        total_backlinks = sum(len(page.backlinks) for page in pages)
        quality_counts = Counter(flag for page in pages for flag in page.quality_flags)

        return {
            "vault_path": str(self.vault_root),
            "vault_exists": self.vault_root.exists(),
            "wiki_root_exists": (self.vault_root / "wiki").exists(),
            "total_pages": len(pages),
            "sections": {section: sum(1 for page in pages if page.section == section) for section in WIKI_SECTIONS},
            "total_source_count": sum(page.source_count for page in pages),
            "total_backlinks": total_backlinks,
            "weak_page_count": sum(1 for page in pages if page.quality_flags),
            "categories": _count_values(page.category for page in pages if page.category),
            "tags": _count_values(tag for page in pages for tag in page.tags),
            "quality_counts": dict(sorted(quality_counts.items())),
            "message": _summary_message(self.vault_root, len(pages)),
        }

    def list_pages(
        self,
        *,
        query: str = "",
        section: str = "",
        page_type: str = "",
        category: str = "",
        tag: str = "",
        quality: str = "",
        limit: int = 200,
    ) -> dict:
        pages = self._load_pages()
        query = query.strip()
        limit = max(1, min(limit, 500))

        scored: List[Tuple[float, PageRecord]] = []
        for page in pages:
            if section and section != "all" and page.section != section:
                continue
            if page_type and page_type != "all" and not _matches_type(page, page_type):
                continue
            if category and _norm(page.category) != _norm(category):
                continue
            if tag and _norm(tag) not in {_norm(value) for value in page.tags}:
                continue
            if quality and quality not in page.quality_flags:
                continue

            score = _score_page(page, query)
            if query and score <= 0:
                continue
            scored.append((score, page))

        if query:
            scored.sort(key=lambda item: (-item[0], item[1].title.lower()))
        else:
            scored.sort(key=lambda item: (item[1].category.lower(), item[1].title.lower()))

        selected = [page for _, page in scored[:limit]]
        return {
            "vault_path": str(self.vault_root),
            "vault_exists": self.vault_root.exists(),
            "query": query,
            "total": len(scored),
            "pages": [page.summary_dict() for page in selected],
            "groups": _groups_for(selected),
            "filters": {
                "sections": list(WIKI_SECTIONS),
                "categories": _count_values(page.category for page in pages if page.category),
                "tags": _count_values(tag for page in pages for tag in page.tags),
                "quality_flags": _count_values(flag for page in pages for flag in page.quality_flags),
            },
            "message": _summary_message(self.vault_root, len(pages)),
        }

    def page_detail(self, section: str, slug: str) -> dict:
        self._validate_page_address(section, slug)
        pages = self._load_pages()
        for page in pages:
            if page.section == section and page.slug == slug:
                return page.detail_dict()
        raise FileNotFoundError(f"Wiki page not found: {section}/{slug}")

    def _load_pages(self) -> List[PageRecord]:
        records = self._scan_pages()
        lookup = _build_lookup(records)
        backlink_index: Dict[str, List[PageRecord]] = defaultdict(list)

        for page in records:
            page.outgoing_links = [
                _resolve_wikilink(target, alias, lookup) for target, alias in _iter_wikilinks(page.body_markdown)
            ]
            page.render_markdown = _render_wikilinks(page.body_markdown, lookup)
            for link in page.outgoing_links:
                if link.resolved_id and link.resolved_id != page.page_id:
                    backlink_index[link.resolved_id].append(page)

        for page in records:
            page.backlinks = sorted(backlink_index.get(page.page_id, []), key=lambda item: item.title.lower())

        return records

    def _scan_pages(self) -> List[PageRecord]:
        wiki_root = self.vault_root / "wiki"
        if not wiki_root.exists():
            return []

        records: List[PageRecord] = []
        for section in WIKI_SECTIONS:
            section_root = wiki_root / section
            if not section_root.exists():
                continue
            for path in sorted(section_root.rglob("*.md")):
                text = path.read_text(encoding="utf-8")
                metadata, body = _split_frontmatter(text)
                slug = path.stem
                title = _as_string(metadata.get("title")) or _extract_h1(body) or _humanize_slug(slug)
                sources = _sources_list(metadata.get("sources"))
                source_count = _source_count(metadata.get("sources"), sources)
                summary, has_summary = _extract_summary(body)
                headings = _extract_headings(body)

                record = PageRecord(
                    section=section,
                    slug=slug,
                    path=path,
                    relative_path=str(path.relative_to(self.vault_root)),
                    title=title,
                    page_type=_as_string(metadata.get("type")) or _type_from_section(section),
                    category=_as_string(metadata.get("category")),
                    tags=_string_list(metadata.get("tags")),
                    sources=sources,
                    source_count=source_count,
                    confidence=_as_string(metadata.get("confidence")),
                    last_updated=_date_string(
                        metadata.get("last_updated") or metadata.get("updated") or metadata.get("created")
                    ),
                    body_markdown=body.strip(),
                    summary=summary,
                    headings=headings,
                    quality_flags=_quality_flags(source_count, metadata, has_summary),
                )
                records.append(record)
        return records

    @staticmethod
    def _validate_page_address(section: str, slug: str) -> None:
        if section not in WIKI_SECTIONS:
            raise ValueError(f"Unsupported wiki section: {section}")
        if not SLUG_RE.match(slug):
            raise ValueError(f"Invalid wiki slug: {slug}")


def create_wiki_router(vault_root_provider: Optional[Callable[[], Path | str]] = None) -> APIRouter:
    """Create the FastAPI router for the read-only wiki atlas API."""
    router = APIRouter(prefix="/api/wiki", tags=["wiki"])
    get_root = vault_root_provider or _default_vault_root

    def atlas() -> WikiAtlas:
        return WikiAtlas(get_root())

    @router.get("/summary")
    async def wiki_summary() -> dict:
        return atlas().summary()

    @router.get("/pages")
    async def wiki_pages(
        query: str = "",
        section: str = "",
        page_type: str = Query("", alias="type"),
        category: str = "",
        tag: str = "",
        quality: str = "",
        limit: int = 200,
    ) -> dict:
        return atlas().list_pages(
            query=query,
            section=section,
            page_type=page_type,
            category=category,
            tag=tag,
            quality=quality,
            limit=limit,
        )

    @router.get("/pages/{section}/{slug}")
    async def wiki_page(section: str, slug: str) -> dict:
        try:
            return atlas().page_detail(section, slug)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router


def _default_vault_root() -> Path:
    try:
        from config import wiki_config

        configured = getattr(wiki_config, "wiki_vault_path", "./vault")
    except Exception:
        configured = "./vault"
    return Path(configured)


def _split_frontmatter(text: str) -> Tuple[dict, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, text[match.end():]


def _extract_h1(body: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return ""


def _extract_summary(body: str) -> Tuple[str, bool]:
    for heading in ("Core Idea", "Overview", "Synthesis"):
        pattern = rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)"
        match = re.search(pattern, body, re.DOTALL | re.MULTILINE)
        if match:
            summary = _first_text_block(match.group(1))
            if summary:
                return summary, True

    without_h1 = re.sub(r"^#\s+.+?$", "", body, count=1, flags=re.MULTILINE)
    summary = _first_text_block(without_h1)
    return summary, bool(summary)


def _first_text_block(markdown: str) -> str:
    lines = []
    for raw_line in markdown.strip().splitlines():
        line = raw_line.strip()
        if not line:
            if lines:
                break
            continue
        if line.startswith("#") or line.startswith("|") or re.match(r"^[-*_]{3,}$", line):
            continue
        lines.append(line.lstrip("- ").strip())

    cleaned = _clean_markdown(" ".join(lines))
    return _truncate(cleaned, 220)


def _extract_headings(body: str) -> List[str]:
    headings = []
    for match in re.finditer(r"^(#{2,4})\s+(.+?)\s*$", body, re.MULTILINE):
        headings.append(match.group(2).strip())
    return headings


def _iter_wikilinks(markdown: str) -> Iterable[Tuple[str, str]]:
    for match in WIKILINK_RE.finditer(markdown):
        target = match.group(1).strip()
        alias = (match.group(2) or "").strip()
        yield target, alias


def _render_wikilinks(markdown: str, lookup: Dict[str, PageRecord]) -> str:
    def replace(match: re.Match) -> str:
        target = match.group(1).strip()
        alias = (match.group(2) or "").strip()
        link = _resolve_wikilink(target, alias, lookup)
        label = link.label.replace("[", "").replace("]", "")
        if link.url:
            return f"[{label}]({link.url})"
        return label

    return WIKILINK_RE.sub(replace, markdown)


def _resolve_wikilink(target: str, alias: str, lookup: Dict[str, PageRecord]) -> WikiLink:
    key = _target_key(target)
    page = lookup.get(key) or lookup.get(key.split("/")[-1])
    label = alias or (page.title if page else _humanize_slug(key.split("/")[-1]))
    if not page:
        return WikiLink(raw_target=target, label=label)
    return WikiLink(
        raw_target=target,
        label=label,
        resolved_id=page.page_id,
        resolved_title=page.title,
        url=page.url,
    )


def _build_lookup(records: List[PageRecord]) -> Dict[str, PageRecord]:
    lookup: Dict[str, PageRecord] = {}
    for page in records:
        keys = {
            page.slug,
            page.page_id,
            f"wiki/{page.page_id}",
            page.title.lower(),
            _target_key(page.title),
            page.relative_path.removesuffix(".md").lower(),
        }
        for key in keys:
            lookup.setdefault(key, page)
    return lookup


def _target_key(target: str) -> str:
    key = target.strip().replace("\\", "/").removesuffix(".md")
    key = re.sub(r"^/?wiki/", "", key, flags=re.IGNORECASE)
    key = key.lower().replace(" ", "-")
    key = re.sub(r"-+", "-", key)
    return key.strip("/")


def _score_page(page: PageRecord, query: str) -> float:
    if not query:
        return 1.0
    q = query.lower()
    words = set(re.findall(r"[a-z0-9]+", q))
    title = page.title.lower()
    slug = page.slug.lower().replace("-", " ")
    category = page.category.lower()
    tags = " ".join(page.tags).lower()
    headings = " ".join(page.headings).lower()
    body = page.body_markdown.lower()

    score = 0.0
    if q == title or q == slug:
        score += 100
    if q in title:
        score += 70
    if q in slug:
        score += 60
    if q in tags:
        score += 45
    if q in category:
        score += 35
    if q in headings:
        score += 25
    if q in body:
        score += 10

    hay_words = set(re.findall(r"[a-z0-9]+", " ".join([title, slug, category, tags, headings])))
    if words:
        score += 20 * (len(words & hay_words) / len(words))

    fuzzy = max(
        SequenceMatcher(None, q, title).ratio(),
        SequenceMatcher(None, q, slug).ratio(),
    )
    if fuzzy >= 0.72:
        score += 15 * fuzzy

    return score


def _matches_type(page: PageRecord, page_type: str) -> bool:
    normalized = page_type.lower()
    return normalized in {page.section.lower(), page.page_type.lower()}


def _quality_flags(source_count: int, metadata: dict, has_summary: bool) -> List[str]:
    flags = []
    if source_count == 0:
        flags.append("zero_sources")
    if _as_string(metadata.get("confidence")).lower() == "low":
        flags.append("low_confidence")
    if not has_summary:
        flags.append("missing_summary")
    if not _as_string(metadata.get("category")):
        flags.append("missing_category")
    return flags


def _groups_for(pages: List[PageRecord]) -> List[dict]:
    grouped: Dict[str, List[PageRecord]] = defaultdict(list)
    for page in pages:
        key = page.category or (page.tags[0] if page.tags else "Uncategorized")
        grouped[key].append(page)

    return [
        {
            "name": name,
            "count": len(items),
            "page_ids": [page.page_id for page in sorted(items, key=lambda item: item.title.lower())],
        }
        for name, items in sorted(grouped.items(), key=lambda item: item[0].lower())
    ]


def _summary_message(vault_root: Path, page_count: int) -> str:
    if page_count:
        return f"Found {page_count} generated wiki pages."
    return (
        "No generated wiki pages found. Expected Markdown files under "
        f"{vault_root}/wiki/{{concepts,entities,summaries}}."
    )


def _sources_list(raw_sources) -> List[str]:
    if raw_sources is None:
        return []
    if isinstance(raw_sources, list):
        return [_as_string(item) for item in raw_sources if _as_string(item)]
    if isinstance(raw_sources, int):
        return []
    value = _as_string(raw_sources)
    return [value] if value else []


def _source_count(raw_sources, source_list: List[str]) -> int:
    if isinstance(raw_sources, int):
        return max(0, raw_sources)
    return len(source_list)


def _string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_string(item) for item in value if _as_string(item)]
    text = _as_string(value)
    return [text] if text else []


def _date_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)


def _as_string(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _count_values(values: Iterable[str]) -> dict:
    counts = Counter(value for value in values if value)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0].lower())))


def _clean_markdown(text: str) -> str:
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`>]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _humanize_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _type_from_section(section: str) -> str:
    return {
        "concepts": "concept",
        "entities": "person",
        "summaries": "source",
    }.get(section, "concept")


def _norm(value: str) -> str:
    return value.strip().lower()
