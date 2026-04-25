"""
WikiBuilder - main orchestrator for the LLM Wiki pipeline.

Composes the council, entity registry, page writer, and indexer
into a unified pipeline. Handles the full ingest flow: entity extraction,
page identification, council convening, page updates, and index maintenance.

Thread safety: per-page file locks prevent concurrent write corruption
when newsletter cron and Telegram bot run simultaneously.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional filelock - graceful fallback if not installed
try:
    from filelock import FileLock
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    logger.warning("filelock not installed - concurrent page updates may conflict")


@dataclass
class WikiUpdateResult:
    """Result of a wiki pipeline operation."""

    pages_created: List[str] = field(default_factory=list)
    pages_updated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_title: str = ""
    operation_type: str = ""
    council_mode: str = ""
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.pages_created) + len(self.pages_updated) > 0

    @property
    def total_pages_affected(self) -> int:
        return len(self.pages_created) + len(self.pages_updated)


class WikiBuilder:
    """
    Main orchestrator for the LLM Wiki pipeline.

    Usage:
        builder = WikiBuilder(vault_root=Path("./vault"), config=wiki_config)
        result = builder.process_ingest(content, operation_type="ingest_link")
    """

    def __init__(self, vault_root: Path, config: Any) -> None:
        from wiki.council import LLMCouncil
        from wiki.entities import EntityRegistry
        from wiki.indexer import WikiIndexer
        from wiki.page_writer import PageWriter
        from wiki.rate_limiter import RateLimiter

        self.vault_root = vault_root
        self.config = config

        # Ensure vault structure exists
        self._ensure_vault_structure()

        # Initialize components
        self.rate_limiter = RateLimiter(
            getattr(config, "wiki_rate_limits", {})
        )
        self.registry = EntityRegistry(
            registry_path=vault_root / "raw" / "refs" / "legacy-schema" / "entity_registry.json"
        )
        self.registry.load()

        self.page_writer = PageWriter(vault_root)
        self.indexer = WikiIndexer(vault_root, self.page_writer)

        self.council = LLMCouncil(
            entity_extractor_model=getattr(config, "wiki_entity_extractor_model", "gemini-3.1-pro"),
            prose_synthesizer_model=getattr(config, "wiki_prose_synthesizer_model", "gpt-5.4-mini"),
            cross_connector_model=getattr(config, "wiki_cross_connector_model", "glm-5"),
            contradiction_finder_model=getattr(config, "wiki_contradiction_finder_model", "kimi-2.5"),
            chairman_model=getattr(config, "wiki_chairman_model", "claude-opus-4-6"),
            rate_limiter=self.rate_limiter,
        )

        self._lock_dir = vault_root / "raw" / "refs" / "legacy-schema" / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    def process_ingest(
        self,
        content: Any,
        operation_type: str = "ingest_link",
        dry_run: bool = False,
    ) -> WikiUpdateResult:
        """
        Full pipeline for a single new source item.

        Steps:
        1. Extract text and metadata from content
        2. Run entity extractor (council member 1)
        3. Resolve canonical entity names via registry
        4. Identify which vault pages to update
        5. Load existing page contexts for contradiction check
        6. Run remaining council members (prose, cross, contradiction)
        7. For each affected entity: chairman creates/updates page
        8. Save raw content to vault/raw/
        9. Update index.md, log.md, connections

        Args:
            content: ExtractedContent, Lesson, or dict with text/title/source_type/source_ref
            operation_type: ingest_link | ingest_newsletter | ingest_lesson
            dry_run: if True, don't write files, just return what would change
        """
        import time
        start = time.monotonic()
        result = WikiUpdateResult(operation_type=operation_type)

        try:
            # Step 1: Normalize input
            source_text, source_metadata = self._normalize_content(content)
            result.source_title = source_metadata.get("title", "Unknown")

            if not source_text.strip():
                result.errors.append("Empty source text")
                return result

            # Step 2-4: Run entity extraction + resolve names
            existing_slugs = self.registry.get_all_slugs()
            session = self.council.convene(
                source_content=source_text,
                source_metadata=source_metadata,
                existing_entities=[
                    self.registry.get_entry(s).display_name
                    for s in existing_slugs[:100]
                    if self.registry.get_entry(s)
                ],
                existing_page_summaries=self._gather_related_summaries(source_text),
                council_mode="full",
            )
            result.council_mode = "full"

            # Parse extracted entities
            entity_output = session.get_output(
                __import__("wiki.council", fromlist=["CouncilRole"]).CouncilRole.ENTITY_EXTRACTOR
            )
            extracted = []
            if entity_output and entity_output.parsed:
                extracted = entity_output.parsed.get("entities", [])

            if not extracted:
                logger.warning("No entities extracted from: %s", result.source_title)
                # Still save raw content
                if not dry_run:
                    self._save_raw(source_text, source_metadata)
                result.errors.append("No entities extracted")
                return result

            # Resolve canonical slugs
            entity_slugs = []
            for entity in extracted:
                slug = self.registry.resolve(
                    raw_name=entity.get("name", ""),
                    entity_type=entity.get("type", "concept"),
                    category=entity.get("category", ""),
                )
                entity_slugs.append(slug)
            session.affected_entities = entity_slugs

            if dry_run:
                result.pages_created = [s for s in entity_slugs if not self._page_exists(s)]
                result.pages_updated = [s for s in entity_slugs if self._page_exists(s)]
                result.duration_seconds = time.monotonic() - start
                return result

            # Step 7: For each affected entity, ask chairman to create/update
            # Pass slug|Display Name pairs so the chairman writes [[slug|Display Name]]
            # wikilinks that resolve to actual kebab-case filenames on disk.
            canonical_names = [
                f"{s}|{self.registry.get_entry(s).display_name}"
                for s in self.registry.get_all_slugs()
                if self.registry.get_entry(s)
            ][:100]

            created_paths = []
            updated_paths = []

            for i, slug in enumerate(entity_slugs):
                page_path = self.registry.get_page_path(slug, self.vault_root)
                entity_entry = self.registry.get_entry(slug)
                if not entity_entry:
                    continue

                try:
                    if page_path.exists():
                        # Update existing page
                        existing_content = page_path.read_text(encoding="utf-8")
                        xml_diff = self.council.call_chairman_update(
                            session=session,
                            existing_page_content=existing_content,
                            source_title=source_metadata.get("title", ""),
                            source_ref=source_metadata.get("source_ref", ""),
                            source_slug=f"src-{slug}",
                            canonical_entities=canonical_names,
                        )
                        with self._page_lock(page_path):
                            if self.page_writer.update_page(page_path, xml_diff):
                                updated_paths.append(page_path)
                                result.pages_updated.append(slug)
                    else:
                        # Create new page
                        entity_info = extracted[i] if i < len(extracted) else {}
                        page_content = self.council.call_chairman_create(
                            session=session,
                            entity_name=slug,
                            entity_type=entity_entry.entity_type,
                            display_name=entity_entry.display_name,
                            category=entity_entry.category or entity_info.get("category", ""),
                            tags=entity_info.get("aliases", []),
                            people=entity_output.parsed.get("people_mentioned", [])
                            if entity_output and entity_output.parsed
                            else [],
                            source_count=1,
                            confidence="medium" if session.failed_count == 0 else "low",
                            canonical_entities=canonical_names,
                        )
                        with self._page_lock(page_path):
                            self.page_writer.create_page_from_full_markdown(page_path, page_content)
                        created_paths.append(page_path)
                        result.pages_created.append(slug)

                except Exception as e:
                    logger.exception("Failed to process entity %s", slug)
                    result.errors.append(f"{slug}: {e}")

            # Step 8: Save raw content
            self._save_raw(source_text, source_metadata)

            # Step 9: Update indexes
            self.registry.save()
            self.indexer.update_index(
                updated_pages=updated_paths,
                created_pages=created_paths,
            )
            self.indexer.append_log(
                self.indexer.make_log_entry(
                    operation=operation_type,
                    pages=[p.stem for p in created_paths + updated_paths],
                    details=f"source: {result.source_title}, "
                    f"created: {len(created_paths)}, updated: {len(updated_paths)}",
                )
            )

        except Exception as e:
            logger.exception("Wiki pipeline failed for %s", operation_type)
            result.errors.append(str(e))

        result.duration_seconds = time.monotonic() - start
        logger.info(
            "Wiki %s complete: %d created, %d updated, %d errors (%.1fs)",
            operation_type,
            len(result.pages_created),
            len(result.pages_updated),
            len(result.errors),
            result.duration_seconds,
        )
        return result

    def process_answer(
        self,
        question: str,
        answer_text: str,
        sources: List[str],
    ) -> Optional[WikiUpdateResult]:
        """
        Auto-file a high-confidence EDITH answer as a wiki analysis page.

        Only called when citation_count >= auto_file_threshold.
        """
        from wiki.entities import slugify
        from wiki.page_writer import PageMetadata

        threshold = getattr(self.config, "wiki_auto_file_citation_threshold", 3)
        if len(sources) < threshold:
            return None

        slug = slugify(question[:80])
        page_path = self.vault_root / "outputs" / "queries" / f"{slug}.md"

        if page_path.exists():
            logger.info("Analysis page already exists: %s", slug)
            return None

        metadata = PageMetadata(
            title=question[:100],
            entity_type="analysis",
            category="analysis",
            tags=("auto-filed", "edith-answer"),
            sources=len(sources),
            people=(),
            last_updated=date.today().isoformat(),
            confidence="high",
        )

        body = (
            f"# {question[:100]}\n\n"
            f"## Question\n{question}\n\n"
            f"## Synthesis\n{answer_text}\n\n"
            f"## Sources Cited\n"
        )
        for src in sources:
            body += f"- {src}\n"
        body += f"\n## Filed From\nAuto-filed by EDITH on {date.today().isoformat()}\n"

        self.page_writer.create_page(page_path, metadata, body)

        # Update index
        self.indexer.update_index(created_pages=[page_path])
        self.indexer.append_log(
            self.indexer.make_log_entry(
                operation="auto_file_answer",
                pages=[slug],
                details=f"question: {question[:60]}, sources: {len(sources)}",
            )
        )

        result = WikiUpdateResult(
            pages_created=[slug],
            source_title=question[:60],
            operation_type="auto_file_answer",
        )
        logger.info("Auto-filed EDITH answer: %s", slug)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get wiki statistics."""
        return self.indexer._compute_stats()

    # ── Internal helpers ────────────────────────────────────────────────

    def _normalize_content(self, content: Any) -> tuple:
        """
        Extract text and metadata from various input types.
        Supports: ExtractedContent, Lesson, dict, str.
        """
        if isinstance(content, str):
            return content, {"title": "Unknown", "source_type": "text", "source_ref": ""}

        if isinstance(content, dict):
            return (
                content.get("text", ""),
                {
                    "title": content.get("title", "Unknown"),
                    "source_type": content.get("source_type", "article"),
                    "source_ref": content.get("source_ref", ""),
                },
            )

        # ExtractedContent dataclass
        if hasattr(content, "text") and hasattr(content, "title"):
            return (
                content.text,
                {
                    "title": content.title,
                    "source_type": getattr(content, "source_type", "article"),
                    "source_ref": getattr(content, "source_ref", ""),
                },
            )

        # Lesson model
        if hasattr(content, "distilled_text") and hasattr(content, "metadata"):
            meta = content.metadata
            return (
                content.distilled_text,
                {
                    "title": content.distilled_text[:60],
                    "source_type": "lesson",
                    "source_ref": getattr(meta, "source", "telegram"),
                    "category": getattr(meta, "category", ""),
                },
            )

        raise TypeError(f"Unsupported content type: {type(content)}")

    def _page_exists(self, slug: str) -> bool:
        """Check if a page exists for an entity slug."""
        page_path = self.registry.get_page_path(slug, self.vault_root)
        return page_path.exists()

    def _gather_related_summaries(self, source_text: str, max_pages: int = 10) -> str:
        """
        Gather summaries from existing pages that might be related to the source.
        Simple keyword-based matching against page titles for now.
        """
        summaries = []
        words = set(source_text.lower().split()[:200])  # Top 200 words

        for page_path in self.indexer._scan_all_pages():
            page_name = page_path.stem.replace("-", " ")
            # Check if any page name words appear in source
            page_words = set(page_name.split())
            if page_words & words:
                body = self.page_writer.read_page_body(page_path)
                # Take first 500 chars as summary
                summary = body[:500].strip()
                if summary:
                    summaries.append(f"### {page_name}\n{summary}")
                    if len(summaries) >= max_pages:
                        break

        return "\n\n".join(summaries)

    def _save_raw(self, source_text: str, metadata: Dict[str, str]) -> None:
        """Save raw source content to vault/raw/articles/."""
        from wiki.entities import slugify
        from wiki.page_writer import _atomic_write

        raw_dir = self.vault_root / "raw" / "articles"
        raw_dir.mkdir(parents=True, exist_ok=True)

        slug = slugify(metadata.get("title", "untitled"))
        raw_path = raw_dir / f"{slug}.md"

        # Don't overwrite existing raw files (immutable)
        if raw_path.exists():
            return

        content = (
            f"---\n"
            f"title: {metadata.get('title', 'Untitled')}\n"
            f"source_type: {metadata.get('source_type', 'article')}\n"
            f"source_ref: {metadata.get('source_ref', '')}\n"
            f"ingested_at: {datetime.now().isoformat()}\n"
            f"---\n\n"
            f"{source_text}\n"
        )
        _atomic_write(raw_path, content)

    def _page_lock(self, page_path: Path):
        """Get a file lock for a specific page."""
        if _HAS_FILELOCK:
            lock_path = self._lock_dir / f"{page_path.stem}.lock"
            return FileLock(lock_path, timeout=30)
        # Fallback: no-op context manager
        return _NoOpLock()

    def _ensure_vault_structure(self) -> None:
        """Create vault directory structure if it doesn't exist."""
        for subdir in [
            "concepts",
            "people",
            "sources",
            "analyses",
            "raw/articles",
            "raw/assets",
            "schema/council_cache",
            "schema/.locks",
        ]:
            (self.vault_root / subdir).mkdir(parents=True, exist_ok=True)


class _NoOpLock:
    """Fallback context manager when filelock is not installed."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
