"""
Bulk migrator for converting existing Memory Palace entries into wiki pages.

Processes 575 entries (164 lessons + 325 archive + 86 link memories)
in category-grouped batches. Uses migration council mode (entity extractor
+ chairman only) to reduce cost. Pages marked confidence: low for later upgrade.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from wiki.council import CouncilRole

logger = logging.getLogger(__name__)

_PAGE_BREAK = "===PAGE_BREAK==="


@dataclass
class MigrationReport:
    """Report from a migration run."""

    source: str  # "lessons" | "archive" | "link_memories"
    total_entries: int = 0
    batches_processed: int = 0
    pages_created: int = 0
    pages_updated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    resumed_from_batch: int = 0


class BulkMigrator:
    """
    Processes existing Memory Palace data into wiki pages.

    Migration order:
    1. lessons_index (164 entries) - grouped by 10 categories
    2. archive_index (325 articles) - grouped by source domain
    3. link memories (86 entries) - grouped by source_type

    Uses migration council mode: entity_extractor + chairman only.
    """

    def __init__(self, builder: Any) -> None:
        self.builder = builder
        self.vault_root = builder.vault_root
        self.migration_log_path = self.vault_root / "raw" / "refs" / "legacy-schema" / "migration_log.json"
        self._migration_state = self._load_state()

    def migrate_lessons(
        self,
        batch_size: int = 10,
        category_filter: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationReport:
        """Migrate lessons from the Memory Palace lessons_index."""
        report = MigrationReport(source="lessons")
        start = time.monotonic()

        try:
            lessons_by_category = self._load_lessons()
            if category_filter:
                lessons_by_category = {
                    k: v for k, v in lessons_by_category.items()
                    if k == category_filter
                }

            total = sum(len(v) for v in lessons_by_category.values())
            report.total_entries = total
            logger.info("Migrating %d lessons across %d categories",
                        total, len(lessons_by_category))

            for category, lessons in lessons_by_category.items():
                # Check if already migrated
                state_key = f"lessons:{category}"
                if self._migration_state.get(state_key, {}).get("completed"):
                    logger.info("Skipping already-migrated category: %s", category)
                    continue

                # Process in batches
                for i in range(0, len(lessons), batch_size):
                    batch = lessons[i:i + batch_size]
                    batch_num = i // batch_size + 1

                    if dry_run:
                        report.batches_processed += 1
                        report.pages_created += len(batch)
                        continue

                    try:
                        created, updated, errors = self._migrate_lesson_batch(
                            batch, category
                        )
                        report.pages_created += created
                        report.pages_updated += updated
                        report.errors.extend(errors)
                        report.batches_processed += 1

                        logger.info(
                            "Batch %d/%d for %s: %d created, %d updated",
                            batch_num,
                            (len(lessons) + batch_size - 1) // batch_size,
                            category,
                            created,
                            updated,
                        )
                    except Exception as e:
                        logger.exception("Batch %d failed for %s", batch_num, category)
                        report.errors.append(f"{category} batch {batch_num}: {e}")

                # Mark category as migrated
                self._migration_state[state_key] = {
                    "completed": True,
                    "timestamp": date.today().isoformat(),
                    "count": len(lessons),
                }
                self._save_state()

        except Exception as e:
            logger.exception("Lesson migration failed")
            report.errors.append(str(e))

        report.duration_seconds = time.monotonic() - start
        return report

    def migrate_archive(
        self,
        batch_size: int = 5,
        domain_filter: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationReport:
        """Migrate articles from the Knowledge Archive."""
        report = MigrationReport(source="archive")
        start = time.monotonic()

        try:
            articles_by_domain = self._load_archive_entries()
            if domain_filter:
                articles_by_domain = {
                    k: v for k, v in articles_by_domain.items()
                    if k == domain_filter
                }

            total = sum(len(v) for v in articles_by_domain.values())
            report.total_entries = total
            logger.info("Migrating %d archive articles across %d domains",
                        total, len(articles_by_domain))

            for domain, articles in articles_by_domain.items():
                state_key = f"archive:{domain}"
                if self._migration_state.get(state_key, {}).get("completed"):
                    logger.info("Skipping already-migrated domain: %s", domain)
                    continue

                for i in range(0, len(articles), batch_size):
                    batch = articles[i:i + batch_size]

                    if dry_run:
                        report.batches_processed += 1
                        continue

                    for article in batch:
                        try:
                            result = self.builder.process_ingest(
                                content=article,
                                operation_type="migration_archive",
                            )
                            report.pages_created += len(result.pages_created)
                            report.pages_updated += len(result.pages_updated)
                            report.errors.extend(result.errors)
                        except Exception as e:
                            report.errors.append(f"archive {article.get('title', '?')}: {e}")
                    report.batches_processed += 1

                self._migration_state[state_key] = {
                    "completed": True,
                    "timestamp": date.today().isoformat(),
                    "count": len(articles),
                }
                self._save_state()

        except Exception as e:
            logger.exception("Archive migration failed")
            report.errors.append(str(e))

        report.duration_seconds = time.monotonic() - start
        return report

    def migrate_link_memories(
        self,
        batch_size: int = 5,
        dry_run: bool = False,
    ) -> MigrationReport:
        """Migrate link memories from the litellm embedding store."""
        report = MigrationReport(source="link_memories")
        start = time.monotonic()

        try:
            memories = self._load_link_memories()
            report.total_entries = len(memories)
            logger.info("Migrating %d link memories", len(memories))

            state_key = "link_memories"
            if self._migration_state.get(state_key, {}).get("completed"):
                logger.info("Link memories already migrated")
                return report

            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]

                if dry_run:
                    report.batches_processed += 1
                    continue

                for memory in batch:
                    try:
                        result = self.builder.process_ingest(
                            content=memory,
                            operation_type="migration_link",
                        )
                        report.pages_created += len(result.pages_created)
                        report.pages_updated += len(result.pages_updated)
                        report.errors.extend(result.errors)
                    except Exception as e:
                        report.errors.append(f"link {memory.get('title', '?')}: {e}")
                report.batches_processed += 1

            self._migration_state[state_key] = {
                "completed": True,
                "timestamp": date.today().isoformat(),
                "count": len(memories),
            }
            self._save_state()

        except Exception as e:
            logger.exception("Link memory migration failed")
            report.errors.append(str(e))

        report.duration_seconds = time.monotonic() - start
        return report

    def _migrate_lesson_batch(
        self, lessons: List[Dict], category: str
    ) -> tuple:
        """
        Migrate a batch of lessons using migration council mode.

        Groups related lessons and asks the chairman to create concept pages.
        Returns (created_count, updated_count, errors).
        """
        created = 0
        updated = 0
        errors = []

        # Build the lessons text for the chairman
        lessons_text = ""
        for i, lesson in enumerate(lessons, 1):
            text = lesson.get("distilled_text", lesson.get("text", ""))
            source = lesson.get("source", "migration")
            lessons_text += f"{i}. {text} [source: {source}]\n"

        # Run entity extraction on the batch
        existing_slugs = self.builder.registry.get_all_slugs()
        session = self.builder.council.convene(
            source_content=lessons_text,
            source_metadata={"title": f"Lesson batch: {category}", "source_type": "lesson"},
            existing_entities=[
                self.builder.registry.get_entry(s).display_name
                for s in existing_slugs[:100]
                if self.builder.registry.get_entry(s)
            ],
            council_mode="migration",
        )

        # Ask chairman to create pages from the batch
        canonical_names = [
            self.builder.registry.get_entry(s).display_name
            for s in existing_slugs[:100]
            if self.builder.registry.get_entry(s)
        ]

        chairman_output = self.builder.council.call_chairman_migration(
            session=session,
            lessons_text=lessons_text,
            category=category,
            batch_size=len(lessons),
            canonical_entities=canonical_names,
        )

        # Parse the chairman's output into individual pages
        pages = chairman_output.split(_PAGE_BREAK)
        for page_content in pages:
            page_content = page_content.strip()
            if not page_content:
                continue

            try:
                # Extract title from frontmatter or first heading
                title = self._extract_title(page_content)
                if not title:
                    continue

                from wiki.entities import slugify
                slug = self.builder.registry.resolve(
                    raw_name=title,
                    entity_type="concept",
                    category=category,
                )
                page_path = self.builder.registry.get_page_path(slug, self.vault_root)

                if page_path.exists():
                    # Page already exists from earlier batch - skip for now
                    updated += 1
                else:
                    self.builder.page_writer.create_page_from_full_markdown(
                        page_path, page_content
                    )
                    created += 1

            except Exception as e:
                errors.append(f"Page creation failed: {e}")

        # Save registry after batch
        self.builder.registry.save()

        # Update index for created pages
        created_paths = []
        for subdir in ["concepts", "people", "sources"]:
            dir_path = self.vault_root / subdir
            if dir_path.exists():
                created_paths.extend(dir_path.glob("*.md"))
        self.builder.indexer.update_index(created_pages=created_paths)

        return created, updated, errors

    def _extract_title(self, page_content: str) -> Optional[str]:
        """Extract title from page markdown (frontmatter or first heading)."""
        import re
        import yaml as yaml_mod

        # Try frontmatter
        fm_match = re.match(r"^---\n(.*?)\n---", page_content, re.DOTALL)
        if fm_match:
            try:
                data = yaml_mod.safe_load(fm_match.group(1))
                if data and "title" in data:
                    return data["title"]
            except Exception:
                pass

        # Try first heading
        heading_match = re.search(r"^#\s+(.+)$", page_content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        return None

    def _load_lessons(self) -> Dict[str, List[Dict]]:
        """Load lessons from the Memory Palace docstore, grouped by category."""
        import os
        lessons_by_category: Dict[str, List[Dict]] = {}

        # Try loading from the LlamaIndex docstore
        docstore_path = Path("./memory_palace/lessons_index/docstore.json")
        if docstore_path.exists():
            data = json.loads(docstore_path.read_text(encoding="utf-8"))
            doc_store = data.get("docstore/data", data.get("docstore", {}))

            for doc_id, doc_data in doc_store.items():
                if isinstance(doc_data, dict):
                    # Handle both direct and nested formats
                    node = doc_data.get("__data__", doc_data)
                    text = node.get("text", "")
                    metadata = node.get("metadata", {})
                    category = metadata.get("category", "observations")

                    lessons_by_category.setdefault(category, []).append({
                        "distilled_text": text,
                        "source": metadata.get("source", "migration"),
                        "category": category,
                        "tags": metadata.get("tags", []),
                    })

        # Fallback: load from seed JSON files
        if not lessons_by_category:
            for json_file in Path("./memory_palace").glob("*.json"):
                if json_file.name in ("shown_history.json",):
                    continue
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if "categories" in data:
                        for cat, items in data["categories"].items():
                            for item in items:
                                lessons_by_category.setdefault(cat, []).append({
                                    "distilled_text": item if isinstance(item, str) else str(item),
                                    "source": data.get("source", json_file.stem),
                                    "category": cat,
                                })
                except (json.JSONDecodeError, KeyError):
                    continue

        return lessons_by_category

    def _load_archive_entries(self) -> Dict[str, List[Dict]]:
        """Load archive entries grouped by source domain."""
        articles_by_domain: Dict[str, List[Dict]] = {}
        docstore_path = Path("./memory_palace/archive_index/docstore.json")

        if docstore_path.exists():
            data = json.loads(docstore_path.read_text(encoding="utf-8"))
            doc_store = data.get("docstore/data", data.get("docstore", {}))

            for doc_id, doc_data in doc_store.items():
                if isinstance(doc_data, dict):
                    node = doc_data.get("__data__", doc_data)
                    text = node.get("text", "")
                    metadata = node.get("metadata", {})
                    domain = metadata.get("source_domain", "unknown")

                    articles_by_domain.setdefault(domain, []).append({
                        "text": text,
                        "title": metadata.get("title", "Unknown"),
                        "source_type": "article",
                        "source_ref": metadata.get("url", ""),
                    })

        return articles_by_domain

    def _load_link_memories(self) -> List[Dict]:
        """Load link memories from the SimpleVectorStore."""
        memories = []
        store_dir = Path("./memory_palace/litellm__text-embedding-3-large")

        if not store_dir.exists():
            return memories

        vector_store_path = store_dir / "vector_store.json"
        if vector_store_path.exists():
            data = json.loads(vector_store_path.read_text(encoding="utf-8"))
            # SimpleVectorStore format: {uuid: {text, metadata, embedding}}
            for entry_id, entry_data in data.items():
                if isinstance(entry_data, dict) and "text" in entry_data:
                    metadata = entry_data.get("metadata", {})
                    memories.append({
                        "text": entry_data["text"],
                        "title": metadata.get("source_title", "Unknown"),
                        "source_type": metadata.get("source_type", "article"),
                        "source_ref": metadata.get("source_ref", ""),
                    })

        return memories

    def _load_state(self) -> Dict:
        """Load migration state for resume capability."""
        if self.migration_log_path.exists():
            try:
                return json.loads(self.migration_log_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_state(self) -> None:
        """Persist migration state."""
        self.migration_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.migration_log_path.write_text(
            json.dumps(self._migration_state, indent=2), encoding="utf-8"
        )
