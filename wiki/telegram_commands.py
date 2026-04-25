"""
Telegram command handlers for wiki operations.

New commands: /wiki, /lint, /wikistats
Also provides the post-save hook for auto-filing and wiki updates.

These handlers are registered by memory_palace_bot.py's build_application().
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def wiki_command(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """
    /wiki <query> - Search wiki pages and return excerpts.

    Uses simple text matching against page titles and tags.
    No embeddings needed - fast file-based search.
    """
    if not update.message or not update.message.text:
        return

    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text(
            "Usage: /wiki <search query>\n\n"
            "Example: /wiki compound interest"
        )
        return

    query = parts[1].strip().lower()

    try:
        results = await asyncio.to_thread(_search_wiki_pages, query)
    except Exception as e:
        logger.exception("Wiki search failed")
        await update.message.reply_text(f"Wiki search failed: {e}")
        return

    if not results:
        await update.message.reply_text(
            f"No wiki pages found matching '{query}'.\n"
            "The wiki may not have been populated yet."
        )
        return

    # Format results
    response_parts = [f"Wiki results for '{query}':\n"]
    for title, summary, page_type, source_count in results[:5]:
        response_parts.append(
            f"\n{title} ({page_type})\n"
            f"{summary}\n"
            f"Sources: {source_count}"
        )

    await update.message.reply_text("\n".join(response_parts))


async def lint_command(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """
    /lint - Trigger wiki health check and return summary.
    """
    await update.message.reply_text("Running wiki lint check...")

    try:
        report_text = await asyncio.to_thread(_run_lint)
    except Exception as e:
        logger.exception("Wiki lint failed")
        await update.message.reply_text(f"Wiki lint failed: {e}")
        return

    await update.message.reply_text(report_text)


async def wikistats_command(update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
    """
    /wikistats - Show wiki statistics.
    """
    try:
        stats_text = await asyncio.to_thread(_compute_stats)
    except Exception as e:
        logger.exception("Wiki stats failed")
        await update.message.reply_text(f"Wiki stats failed: {e}")
        return

    await update.message.reply_text(stats_text)


def trigger_wiki_ingest(content, operation_type: str) -> None:
    """
    Synchronous hook called from EDITH post-save paths.
    Wraps WikiBuilder.process_ingest() with error handling.

    Called via asyncio.to_thread() from the async bot handlers.
    Never raises - logs errors silently to avoid breaking the main pipeline.
    """
    try:
        from wiki import get_wiki_builder

        builder = get_wiki_builder()
        if builder is None:
            return

        result = builder.process_ingest(content, operation_type=operation_type)
        if result.success:
            logger.info(
                "Wiki hook (%s): created %d, updated %d pages",
                operation_type,
                len(result.pages_created),
                len(result.pages_updated),
            )
        elif result.errors:
            logger.warning("Wiki hook (%s) errors: %s", operation_type, result.errors)

    except Exception:
        logger.exception("Wiki hook failed (non-fatal) for %s", operation_type)


def trigger_wiki_auto_file(
    question: str,
    answer_text: str,
    sources: list,
    bot_send_message=None,
    chat_id: Optional[int] = None,
) -> None:
    """
    Auto-file a high-confidence EDITH answer into the wiki.

    Called when citation_count >= threshold. Optionally sends a
    Telegram notification about the new wiki page.
    """
    try:
        from wiki import get_wiki_builder

        builder = get_wiki_builder()
        if builder is None:
            return

        result = builder.process_answer(question, answer_text, sources)
        if result and result.success:
            logger.info("Auto-filed wiki analysis: %s", result.pages_created)

            # Send Telegram notification if configured
            if bot_send_message and chat_id:
                try:
                    from config import wiki_config
                    if getattr(wiki_config, "wiki_telegram_notify_on_auto_file", True):
                        page_name = result.pages_created[0] if result.pages_created else "unknown"
                        display = page_name.replace("-", " ").title()
                        asyncio.get_event_loop().create_task(
                            bot_send_message(
                                chat_id=chat_id,
                                text=f"Filed new wiki page: {display}",
                            )
                        )
                except Exception:
                    logger.debug("Could not send auto-file notification", exc_info=True)

    except Exception:
        logger.exception("Wiki auto-file failed (non-fatal)")


# ── Internal helpers ────────────────────────────────────────────────────


def _get_vault_root() -> Optional[Path]:
    """Get the vault root path from config."""
    try:
        from config import wiki_config
        if not getattr(wiki_config, "wiki_enabled", False):
            return None
        return Path(getattr(wiki_config, "wiki_vault_path", "./vault"))
    except (ImportError, AttributeError):
        return None


def _search_wiki_pages(query: str, max_results: int = 5) -> list:
    """
    Search wiki pages by title/tag matching.
    Returns list of (title, summary, page_type, source_count) tuples.
    """
    vault_root = _get_vault_root()
    if vault_root is None:
        return []

    from wiki.page_writer import PageWriter
    writer = PageWriter(vault_root)

    results = []
    query_words = set(query.lower().split())

    for subdir in ["concepts", "entities", "summaries"]:
        dir_path = vault_root / "wiki" / subdir
        if not dir_path.exists():
            continue

        for page_path in dir_path.rglob("*.md"):
            metadata = writer.read_page_metadata(page_path)
            if metadata is None:
                continue

            # Score by matching query words against title, tags, category
            title_words = set(metadata.title.lower().split())
            tag_words = set(t.lower() for t in metadata.tags)
            slug_words = set(page_path.stem.replace("-", " ").split())
            all_words = title_words | tag_words | slug_words | {metadata.category.lower()}

            overlap = query_words & all_words
            if not overlap:
                # Also check substring match in title
                if query not in metadata.title.lower():
                    continue

            score = len(overlap) / len(query_words) if query_words else 0
            summary = writer.get_one_line_summary(page_path)

            results.append((score, metadata.title, summary, metadata.entity_type, metadata.sources))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    return [(title, summary, ptype, sc) for _, title, summary, ptype, sc in results[:max_results]]


def _run_lint() -> str:
    """Run the wiki linter and return formatted report."""
    vault_root = _get_vault_root()
    if vault_root is None:
        return "Wiki is not enabled."

    from wiki.entities import EntityRegistry
    from wiki.lint import WikiLinter, format_lint_telegram

    registry = EntityRegistry(vault_root / "raw" / "refs" / "legacy-schema" / "entity_registry.json")
    registry.load()

    linter = WikiLinter(vault_root, registry)
    report = linter.run(auto_fix=True)
    return format_lint_telegram(report)


def _compute_stats() -> str:
    """Compute and format wiki stats."""
    vault_root = _get_vault_root()
    if vault_root is None:
        return "Wiki is not enabled."

    from wiki.stats import compute_wiki_stats, format_stats_telegram

    stats = compute_wiki_stats(vault_root)
    return format_stats_telegram(stats)
