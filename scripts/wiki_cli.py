#!/usr/bin/env python3
"""
CLI runner for wiki operations.

Usage:
    python scripts/wiki_cli.py migrate --source lessons
    python scripts/wiki_cli.py migrate --source archive
    python scripts/wiki_cli.py migrate --source links
    python scripts/wiki_cli.py migrate --all
    python scripts/wiki_cli.py lint
    python scripts/wiki_cli.py backup
    python scripts/wiki_cli.py stats
    python scripts/wiki_cli.py rebuild-index
    python scripts/wiki_cli.py ingest <url>
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import wiki_config as wc

logger = logging.getLogger(__name__)


def cmd_migrate(args: argparse.Namespace) -> int:
    """Run bulk migration of Memory Palace data into wiki pages."""
    from wiki import get_wiki_builder
    from wiki.migration import BulkMigrator

    builder = get_wiki_builder()
    if builder is None:
        print("Wiki is not enabled. Set wiki.enabled: true in config.yml")
        return 1

    migrator = BulkMigrator(builder)

    sources = []
    if args.all:
        sources = ["lessons", "archive", "links"]
    elif args.source:
        sources = [args.source]
    else:
        print("Specify --source (lessons|archive|links) or --all")
        return 1

    total_created = 0
    total_updated = 0
    total_errors = 0

    for source in sources:
        print(f"\nMigrating: {source}...")
        start = time.monotonic()

        if source == "lessons":
            report = migrator.migrate_lessons(
                batch_size=args.batch_size,
                category_filter=args.category,
                dry_run=args.dry_run,
            )
        elif source == "archive":
            report = migrator.migrate_archive(
                batch_size=args.batch_size,
                domain_filter=args.domain,
                dry_run=args.dry_run,
            )
        elif source == "links":
            report = migrator.migrate_link_memories(
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
        else:
            print(f"Unknown source: {source}")
            continue

        elapsed = time.monotonic() - start
        print(f"  Entries: {report.total_entries}")
        print(f"  Batches: {report.batches_processed}")
        print(f"  Created: {report.pages_created}")
        print(f"  Updated: {report.pages_updated}")
        print(f"  Errors: {len(report.errors)}")
        print(f"  Time: {elapsed:.1f}s")

        if report.errors:
            print(f"  Error details:")
            for err in report.errors[:10]:
                print(f"    - {err}")

        total_created += report.pages_created
        total_updated += report.pages_updated
        total_errors += len(report.errors)

    print(f"\nMigration complete: {total_created} created, {total_updated} updated, {total_errors} errors")

    if not args.dry_run:
        # Rebuild index after migration
        print("Rebuilding index...")
        builder.indexer.update_index()
        builder.indexer.update_connections()
        print("Done.")

    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    """Run wiki linter."""
    from wiki.entities import EntityRegistry
    from wiki.lint import WikiLinter, format_lint_telegram

    vault_root = Path(wc.wiki_vault_path)
    registry = EntityRegistry(vault_root / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
    registry.load()

    linter = WikiLinter(vault_root, registry)
    report = linter.run(auto_fix=not args.no_fix)

    print(format_lint_telegram(report))

    if report.issues:
        print("\nAll issues:")
        for issue in report.issues:
            status = "FIXED" if issue.fixed else ("REVIEW" if not issue.auto_fixable else "PENDING")
            print(f"  [{issue.severity.upper()}] {issue.page}: {issue.message} ({status})")

    return 0 if report.high_count == 0 else 1


def cmd_backup(args: argparse.Namespace) -> int:
    """Create a vault backup."""
    from wiki.backup import WikiBackup

    vault_root = Path(wc.wiki_vault_path)
    backup_dir = Path(wc.wiki_backup_dir)
    keep_n = wc.wiki_backup_keep_last_n

    backup = WikiBackup(vault_root, backup_dir, keep_n)
    tarball = backup.create_tarball()
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"Backup created: {tarball} ({size_mb:.1f} MB)")

    existing = backup.list_backups()
    print(f"Total backups: {len(existing)} (keeping last {keep_n})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show wiki statistics."""
    from wiki.stats import compute_wiki_stats, format_stats_telegram

    vault_root = Path(wc.wiki_vault_path)
    stats = compute_wiki_stats(vault_root)
    print(format_stats_telegram(stats))
    return 0


def cmd_rebuild_index(args: argparse.Namespace) -> int:
    """Rebuild index.md, CONNECTIONS.md, and TIMELINE.md from scratch."""
    from wiki import get_wiki_builder

    builder = get_wiki_builder()
    if builder is None:
        print("Wiki is not enabled.")
        return 1

    print("Rebuilding index.md...")
    builder.indexer.update_index()

    print("Rebuilding CONNECTIONS.md...")
    builder.indexer.update_connections()

    print("Updating TIMELINE.md...")
    builder.indexer.update_timeline()

    print("Done.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a single URL into the wiki."""
    from wiki import get_wiki_builder

    builder = get_wiki_builder()
    if builder is None:
        print("Wiki is not enabled.")
        return 1

    # Extract article content
    try:
        from helper_functions.link_ingestion import extract_article, extract_takeaways
        content = extract_article(args.url)
        print(f"Extracted: {content.title} ({content.word_count} words)")
    except Exception as e:
        print(f"Failed to extract: {e}")
        return 1

    result = builder.process_ingest(content, operation_type="cli_ingest", dry_run=args.dry_run)

    if result.success:
        print(f"Created: {result.pages_created}")
        print(f"Updated: {result.pages_updated}")
    else:
        print(f"Failed: {result.errors}")

    return 0 if result.success else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM Wiki CLI - manage the Obsidian knowledge wiki",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Bulk migrate Palace data to wiki")
    p_migrate.add_argument("--source", choices=["lessons", "archive", "links"])
    p_migrate.add_argument("--all", action="store_true", help="Migrate all sources")
    p_migrate.add_argument("--batch-size", type=int, default=10)
    p_migrate.add_argument("--category", help="Filter lessons by category")
    p_migrate.add_argument("--domain", help="Filter archive by source domain")
    p_migrate.add_argument("--dry-run", action="store_true")

    # lint
    p_lint = subparsers.add_parser("lint", help="Run wiki health check")
    p_lint.add_argument("--no-fix", action="store_true", help="Don't auto-fix issues")

    # backup
    subparsers.add_parser("backup", help="Create vault backup")

    # stats
    subparsers.add_parser("stats", help="Show wiki statistics")

    # rebuild-index
    subparsers.add_parser("rebuild-index", help="Rebuild index and connections")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a single URL")
    p_ingest.add_argument("url", help="URL to ingest")
    p_ingest.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not wc.wiki_enabled:
        print("Wiki is disabled. Set wiki.enabled: true in config/config.yml")
        print("\nAdd this block to your config.yml:\n")
        print("wiki:")
        print("  enabled: true")
        print("  vault_path: \"./vault\"")
        print("  council:")
        print("    entity_extractor_model: \"gemini-3.1-pro\"")
        print("    prose_synthesizer_model: \"gpt-5.4-mini\"")
        print("    cross_connector_model: \"glm-5\"")
        print("    contradiction_finder_model: \"kimi-2.5\"")
        print("    chairman_model: \"claude-opus-4-6\"")
        return 1

    commands = {
        "migrate": cmd_migrate,
        "lint": cmd_lint,
        "backup": cmd_backup,
        "stats": cmd_stats,
        "rebuild-index": cmd_rebuild_index,
        "ingest": cmd_ingest,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
