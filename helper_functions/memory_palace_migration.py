"""
Memory Palace Migration Script

One-time import script to migrate existing JSON lesson files into the
LlamaIndex-based Memory Palace database.

Features:
- Maps 23 old categories to 10 consolidated categories
- Deduplicates by tracking seen text (normalized)
- Supports dry-run mode for preview
- Skips itl.json (duplicate of interesting_things_learnt.json)

Usage:
    # Preview what would be migrated
    python -m helper_functions.memory_palace_migration --dry-run

    # Execute migration
    python -m helper_functions.memory_palace_migration

    # Force re-migration (clears existing index)
    python -m helper_functions.memory_palace_migration --force
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import config
from helper_functions.memory_palace_db import (
    CATEGORIES,
    Lesson,
    LessonCategory,
    LessonMetadata,
    MemoryPalaceDB,
)

logger = logging.getLogger(__name__)

# Mapping from old category names (23 categories) to new consolidated categories (10)
CATEGORY_MAPPING: Dict[str, str] = {
    # Strategy
    "strategy_and_game_theory": "strategy",
    "strategy_and_decision_making": "strategy",
    "negotiation_and_persuasion": "strategy",

    # Psychology
    "psychology_and_cognitive_biases": "psychology",
    "psychology_and_philosophy": "psychology",
    "philosophy_and_psychology": "psychology",

    # History
    "history_and_civilization": "history",
    "history_and_society": "history",

    # Science
    "science_and_physics": "science",
    "science_and_engineering": "science",

    # Technology
    "technology_and_computing": "technology",
    "technology_and_ai": "technology",

    # Economics
    "economics_and_finance": "economics",
    "wealth_and_investing": "economics",
    "economics_and_society": "economics",

    # Engineering
    "engineering_and_systems_thinking": "engineering",

    # Biology
    "biology_and_health": "biology",

    # Leadership
    "career_and_leadership": "leadership",
    "personal_growth_and_mindset": "leadership",

    # Observations (catch-all)
    "miscellaneous_facts": "observations",
    "miscellaneous_observations": "observations",
    "future_predictions_and_ideas": "observations",
    "philosophy_and_life_lessons": "observations",
}

# Files to migrate (exclude itl.json as it's a duplicate)
MIGRATION_FILES = [
    "high_iq_lessons.json",
    "interesting_things_learnt.json",
    "little_observations.json",
    "rules_of_life.json",
]

# Files to skip (known duplicates)
SKIP_FILES = [
    "itl.json",  # Duplicate of interesting_things_learnt.json
]


def normalize_text(text: str) -> str:
    """Normalize text for deduplication comparison."""
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    """Generate a hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()


def map_category(old_category: str) -> str:
    """
    Map an old category name to the new consolidated category.

    Returns 'observations' if the category is unknown.
    """
    mapped = CATEGORY_MAPPING.get(old_category)
    if mapped:
        return mapped

    # Try case-insensitive match
    old_lower = old_category.lower()
    for old_cat, new_cat in CATEGORY_MAPPING.items():
        if old_cat.lower() == old_lower:
            return new_cat

    # Fallback to observations
    logger.warning(f"Unknown category '{old_category}', mapping to 'observations'")
    return "observations"


def parse_json_file(file_path: Path) -> Tuple[str, str, Dict[str, List[str]]]:
    """
    Parse a Memory Palace JSON file.

    Returns:
        Tuple of (title, source, categories_dict)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data.get("title", file_path.stem)
    source = data.get("source", file_path.name)
    categories = data.get("categories", {})

    return title, source, categories


def migrate_file(
    db: MemoryPalaceDB,
    file_path: Path,
    seen_hashes: Set[str],
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Migrate a single JSON file to the database.

    Args:
        db: MemoryPalaceDB instance
        file_path: Path to JSON file
        seen_hashes: Set of already-seen text hashes (for deduplication)
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration statistics
    """
    stats = {
        "total": 0,
        "added": 0,
        "skipped_duplicate": 0,
        "by_category": {},
    }

    title, source, categories = parse_json_file(file_path)
    logger.info(f"Processing '{title}' from {file_path.name}")

    for old_category, lessons in categories.items():
        new_category = map_category(old_category)

        if new_category not in stats["by_category"]:
            stats["by_category"][new_category] = 0

        for lesson_text in lessons:
            stats["total"] += 1

            # Check for duplicate
            h = text_hash(lesson_text)
            if h in seen_hashes:
                stats["skipped_duplicate"] += 1
                logger.debug(f"Skipping duplicate: {lesson_text[:50]}...")
                continue

            seen_hashes.add(h)

            if not dry_run:
                # Create and add the lesson
                lesson = Lesson(
                    distilled_text=lesson_text,
                    metadata=LessonMetadata(
                        category=LessonCategory(new_category),
                        source=f"migration:{source}",
                        original_input=lesson_text,
                        distilled_by_model="migration",
                        tags=[old_category],  # Preserve old category as tag
                    )
                )
                db.add_lesson(lesson)

            stats["added"] += 1
            stats["by_category"][new_category] += 1

    return stats


def migrate_all(
    dry_run: bool = False,
    force: bool = False,
    memory_palace_folder: Optional[str] = None,
) -> Dict[str, any]:
    """
    Migrate all JSON files to the Memory Palace database.

    Args:
        dry_run: If True, only report what would be done
        force: If True, clear existing index before migration
        memory_palace_folder: Override folder path (for testing)

    Returns:
        Dict with overall migration statistics
    """
    # Determine folder path
    folder = Path(memory_palace_folder or config.MEMORY_PALACE_FOLDER or "./memory_palace")

    if not folder.exists():
        raise FileNotFoundError(f"Memory Palace folder not found: {folder}")

    # Handle force mode
    if force and not dry_run:
        index_folder = Path(config.memory_palace_index_folder)
        if index_folder.exists():
            logger.warning(f"Force mode: removing existing index at {index_folder}")
            shutil.rmtree(index_folder)

    # Initialize database (creates fresh index if needed)
    db = MemoryPalaceDB() if not dry_run else None

    # Check if already migrated
    if db and db.get_lesson_count() > 0 and not force:
        logger.warning(
            f"Database already contains {db.get_lesson_count()} lessons. "
            "Use --force to clear and re-migrate."
        )
        return {"status": "skipped", "reason": "database_not_empty"}

    # Track overall stats
    overall_stats = {
        "status": "success",
        "dry_run": dry_run,
        "files_processed": 0,
        "total_lessons": 0,
        "added": 0,
        "skipped_duplicate": 0,
        "skipped_files": [],
        "by_category": {},
        "by_file": {},
    }

    seen_hashes: Set[str] = set()

    # Process each migration file
    for filename in MIGRATION_FILES:
        file_path = folder / filename

        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            overall_stats["skipped_files"].append(filename)
            continue

        stats = migrate_file(db, file_path, seen_hashes, dry_run)

        overall_stats["files_processed"] += 1
        overall_stats["total_lessons"] += stats["total"]
        overall_stats["added"] += stats["added"]
        overall_stats["skipped_duplicate"] += stats["skipped_duplicate"]
        overall_stats["by_file"][filename] = stats

        # Merge category stats
        for cat, count in stats["by_category"].items():
            overall_stats["by_category"][cat] = (
                overall_stats["by_category"].get(cat, 0) + count
            )

    # Log skipped files
    for filename in SKIP_FILES:
        file_path = folder / filename
        if file_path.exists():
            overall_stats["skipped_files"].append(f"{filename} (duplicate)")

    return overall_stats


def print_migration_report(stats: Dict[str, any]) -> None:
    """Print a formatted migration report."""
    if stats.get("status") == "skipped":
        print(f"\nMigration skipped: {stats.get('reason')}")
        return

    mode = "[DRY RUN] " if stats.get("dry_run") else ""

    print(f"\n{mode}Migration Report")
    print("=" * 50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total lessons found: {stats['total_lessons']}")
    print(f"Lessons added: {stats['added']}")
    print(f"Duplicates skipped: {stats['skipped_duplicate']}")

    if stats.get("skipped_files"):
        print(f"\nSkipped files: {', '.join(stats['skipped_files'])}")

    print("\nLessons by category:")
    for cat, count in sorted(stats.get("by_category", {}).items()):
        display_name = CATEGORIES.get(cat, {}).get("display", cat)
        print(f"  {display_name}: {count}")

    print("\nLessons by file:")
    for filename, file_stats in stats.get("by_file", {}).items():
        print(f"  {filename}: {file_stats['added']} added, "
              f"{file_stats['skipped_duplicate']} duplicates")

    print("=" * 50)


def main():
    """CLI entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate Memory Palace JSON files to LlamaIndex database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing index before migration"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        stats = migrate_all(dry_run=args.dry_run, force=args.force)
        print_migration_report(stats)

        if stats.get("status") == "success" and not args.dry_run:
            print("\nMigration completed successfully!")
        elif args.dry_run:
            print("\nDry run complete. Use without --dry-run to execute migration.")

    except Exception as e:
        logger.exception("Migration failed")
        print(f"\nMigration failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
