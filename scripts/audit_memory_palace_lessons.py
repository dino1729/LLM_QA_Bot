#!/usr/bin/env python3
"""
Audit Memory Palace lessons for source-referential phrasing and optionally rewrite them.

Default mode is dry-run. Use --apply to mutate the index.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root for local imports when executed as script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from helper_functions.memory_palace_db import (  # noqa: E402
    MemoryPalaceDB,
    Lesson,
    is_objective_lesson_text,
    rewrite_to_objective_lesson,
)

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _load_processed_ids(journal_path: Optional[Path]) -> Set[str]:
    if not journal_path or not journal_path.exists():
        return set()

    processed: Set[str] = set()
    with open(journal_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lesson_id = row.get("lesson_id")
            status = row.get("status")
            if not lesson_id:
                continue
            if status in {"unchanged", "rewritten", "skipped", "skipped_limit"}:
                processed.add(lesson_id)
    return processed


def _backup_vector_store(index_folder: Path, backup_dir: Path) -> Optional[Path]:
    source = index_folder / "vector_store.json"
    if not source.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"vector_store.backup.{_utc_timestamp()}.json"
    shutil.copy2(source, backup_path)
    return backup_path


def _iter_lessons(
    db: MemoryPalaceDB,
    include_forgotten: bool,
    source_filter: Optional[str],
) -> List[Lesson]:
    lessons = db.get_all_lessons(include_forgotten=include_forgotten)
    if source_filter:
        source_filter_norm = source_filter.lower()
        lessons = [
            lesson
            for lesson in lessons
            if source_filter_norm in (lesson.metadata.source or "").lower()
        ]
    lessons.sort(key=lambda lesson: (lesson.metadata.created_at, lesson.id))
    return lessons


def _coerce_positive(value: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback


def audit_lessons(
    db: MemoryPalaceDB,
    *,
    apply: bool,
    include_forgotten: bool,
    source_filter: Optional[str],
    max_rewrites: int,
    batch_size: int,
    sleep_ms: int,
    provider: Optional[str],
    model_tier: Optional[str],
    model_name: Optional[str],
    journal_path: Path,
    resume_journal: Optional[Path],
) -> Dict:
    lessons = _iter_lessons(db, include_forgotten=include_forgotten, source_filter=source_filter)
    resumed_ids = _load_processed_ids(resume_journal)
    rewritten_in_batch = 0
    rewrites_done = 0
    rewrite_latency_total = 0.0
    rewrite_latency_count = 0

    stats = {
        "total_scanned": 0,
        "flagged": 0,
        "unchanged": 0,
        "rewritten": 0,
        "skipped": 0,
        "skipped_limit": 0,
        "errors": 0,
        "resumed_skips": 0,
    }

    for lesson in lessons:
        stats["total_scanned"] += 1

        if lesson.id in resumed_ids:
            stats["resumed_skips"] += 1
            continue

        valid, reasons = is_objective_lesson_text(lesson.distilled_text)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "lesson_id": lesson.id,
            "source": lesson.metadata.source,
            "category": str(lesson.metadata.category),
            "status": None,
            "reasons": reasons,
            "before_text": lesson.distilled_text,
        }

        if valid:
            row["status"] = "unchanged"
            stats["unchanged"] += 1
            _append_jsonl(journal_path, row)
            continue

        stats["flagged"] += 1
        if not apply:
            row["status"] = "flagged"
            _append_jsonl(journal_path, row)
            continue

        if rewrites_done >= max_rewrites:
            row["status"] = "skipped_limit"
            stats["skipped_limit"] += 1
            _append_jsonl(journal_path, row)
            continue

        try:
            t_start = time.perf_counter()
            rewritten = rewrite_to_objective_lesson(
                lesson.distilled_text,
                provider=provider,
                model_tier=model_tier,
                model_name=model_name,
            )
            rewrite_latency_total += time.perf_counter() - t_start
            rewrite_latency_count += 1

            rewritten_valid, rewritten_reasons = is_objective_lesson_text(rewritten)
            if not rewritten_valid:
                row["status"] = "skipped"
                row["error"] = f"rewrite_not_objective:{','.join(rewritten_reasons)}"
                stats["skipped"] += 1
                _append_jsonl(journal_path, row)
                continue

            updated = db.update_lesson_text(
                lesson.id,
                rewritten,
                rewritten_by_model=model_name or config.memory_palace_primary_model or "memory-cleanup",
                preserve_category=True,
                append_tags=["auto-cleanup"],
                persist=False,
            )
            if not updated:
                row["status"] = "error"
                row["error"] = "update_failed"
                stats["errors"] += 1
                _append_jsonl(journal_path, row)
                continue

            row["status"] = "rewritten"
            row["after_text"] = rewritten
            stats["rewritten"] += 1
            rewrites_done += 1
            rewritten_in_batch += 1
            _append_jsonl(journal_path, row)

            if rewritten_in_batch >= batch_size:
                db.persist()
                rewritten_in_batch = 0

            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            stats["errors"] += 1
            _append_jsonl(journal_path, row)

    if apply and rewritten_in_batch > 0:
        db.persist()

    avg_rewrite_latency_ms = (
        (rewrite_latency_total / rewrite_latency_count) * 1000.0
        if rewrite_latency_count
        else 0.0
    )
    stats["avg_rewrite_latency_ms"] = round(avg_rewrite_latency_ms, 2)
    stats["estimated_token_cost_input_chars"] = sum(
        len(lesson.distilled_text) for lesson in lessons
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Memory Palace lessons and optionally rewrite source-referential phrasing",
    )
    parser.add_argument("--apply", action="store_true", help="Apply rewrites (default is dry-run)")
    parser.add_argument("--include-forgotten", action="store_true", help="Include forgotten lessons in audit")
    parser.add_argument("--source-filter", type=str, default=None, help="Only audit lessons whose source contains this substring")
    parser.add_argument("--max-rewrites", type=int, default=200, help="Maximum rewrites in one run")
    parser.add_argument("--batch-size", type=int, default=20, help="Persist to disk after this many rewrites")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between rewrites to reduce rate-limit pressure")
    parser.add_argument("--provider", type=str, default=None, help="Override provider for LLM rewrites")
    parser.add_argument("--model-tier", type=str, default=None, help="Override model tier for LLM rewrites")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name for LLM rewrites")
    parser.add_argument("--index-folder", type=str, default=None, help="Override Memory Palace index folder")
    parser.add_argument("--backup-dir", type=str, default=None, help="Where to place vector_store backups")
    parser.add_argument("--journal-file", type=str, default=None, help="JSONL journal path for this run")
    parser.add_argument("--resume-journal", type=str, default=None, help="JSONL journal to resume from")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    max_rewrites = _coerce_positive(args.max_rewrites, 200)
    batch_size = _coerce_positive(args.batch_size, 20)
    sleep_ms = max(0, int(args.sleep_ms))

    timestamp = _utc_timestamp()
    base_folder = Path(getattr(config, "MEMORY_PALACE_FOLDER", None) or "./memory_palace")
    reports_dir = base_folder / "audit_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    journal_path = Path(args.journal_file) if args.journal_file else (reports_dir / f"audit_{timestamp}.jsonl")
    report_path = reports_dir / f"audit_report_{timestamp}.json"
    resume_journal = Path(args.resume_journal) if args.resume_journal else None

    db = MemoryPalaceDB(index_folder=args.index_folder) if args.index_folder else MemoryPalaceDB()

    backup_path = None
    if args.apply:
        backup_dir = Path(args.backup_dir) if args.backup_dir else (base_folder / "backups")
        backup_path = _backup_vector_store(Path(db.index_folder), backup_dir)
        if backup_path:
            logger.info("Backed up vector store to %s", backup_path)
        else:
            logger.warning("No vector_store.json found for backup under %s", db.index_folder)

    stats = audit_lessons(
        db,
        apply=args.apply,
        include_forgotten=args.include_forgotten,
        source_filter=args.source_filter,
        max_rewrites=max_rewrites,
        batch_size=batch_size,
        sleep_ms=sleep_ms,
        provider=args.provider,
        model_tier=args.model_tier,
        model_name=args.model_name,
        journal_path=journal_path,
        resume_journal=resume_journal,
    )

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "index_folder": db.index_folder,
        "journal_path": str(journal_path),
        "report_path": str(report_path),
        "backup_path": str(backup_path) if backup_path else None,
        "options": {
            "include_forgotten": args.include_forgotten,
            "source_filter": args.source_filter,
            "max_rewrites": max_rewrites,
            "batch_size": batch_size,
            "sleep_ms": sleep_ms,
            "provider": args.provider,
            "model_tier": args.model_tier,
            "model_name": args.model_name,
            "resume_journal": str(resume_journal) if resume_journal else None,
        },
        "stats": stats,
    }
    _write_json(report_path, summary)

    print("\n=== Memory Palace Lesson Audit ===")
    print(f"Mode: {summary['mode']}")
    print(f"Scanned: {stats['total_scanned']}")
    print(f"Flagged: {stats['flagged']}")
    print(f"Rewritten: {stats['rewritten']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Skipped: {stats['skipped']} (limit: {stats['skipped_limit']})")
    print(f"Errors: {stats['errors']}")
    print(f"Resumed skips: {stats['resumed_skips']}")
    print(f"Avg rewrite latency (ms): {stats['avg_rewrite_latency_ms']}")
    print(f"Journal: {journal_path}")
    print(f"Report: {report_path}")
    if backup_path:
        print(f"Backup: {backup_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
