#!/usr/bin/env python3
"""
Recover Memory Palace lessons whose distilled_text is a placeholder (e.g. "...").

Background: a bulk ingest on 2026-06-09 stored 61 lessons whose distilled_text was
the literal "..." (the small distill model returned it verbatim). These rendered as
an empty Memory Palace section in the daily newsletter. Each row still holds a rich
``original_input``, so we re-distill from that source text.

Default mode is dry-run. Use --apply to mutate the index.

Usage:
    python scripts/redistill_placeholder_lessons.py                 # dry-run report
    python scripts/redistill_placeholder_lessons.py --apply         # recover in place
    python scripts/redistill_placeholder_lessons.py --apply --model gemini-3-flash-preview
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config  # noqa: E402
from helper_functions.memory_palace_db import (  # noqa: E402
    MemoryPalaceDB,
    distill_lesson,
    is_meaningful_lesson_text,
    is_objective_lesson_text,
)

# original_input is stored as "[Title] takeaway text"; recover the takeaway.
_TITLE_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")


def _source_text(original_input: str) -> str:
    """Strip the leading "[Title] " prefix to recover the raw takeaway."""
    return _TITLE_PREFIX.sub("", original_input or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry-run)")
    parser.add_argument(
        "--model",
        default=config.memory_palace_fallback_model,
        help="Model to use for re-distillation (default: configured fallback model)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    args = parser.parse_args()

    db = MemoryPalaceDB()

    # Find placeholder rows the newsletter would otherwise skip. We must opt out of
    # the placeholder filter to see them.
    all_rows = db.get_all_lessons(include_forgotten=True, exclude_placeholders=False)
    placeholders = [
        lesson
        for lesson in all_rows
        if not is_meaningful_lesson_text(lesson.distilled_text)[0]
    ]

    print(f"Store rows (incl. forgotten): {len(all_rows)}")
    print(f"Placeholder rows to recover:  {len(placeholders)}")
    if args.limit:
        placeholders = placeholders[: args.limit]
        print(f"Limited to:                   {len(placeholders)}")
    print(f"Re-distill model:             {args.model}")
    print(f"Mode:                         {'APPLY' if args.apply else 'DRY-RUN'}")
    print("-" * 70)

    recovered = 0
    unrecoverable = 0
    for i, lesson in enumerate(placeholders, 1):
        source = _source_text(lesson.metadata.original_input)
        if len(source) < 20:
            print(f"[{i}/{len(placeholders)}] {lesson.id[:8]} UNRECOVERABLE: original_input too short")
            unrecoverable += 1
            continue

        result = distill_lesson(
            source,
            model_name=args.model,
        )
        new_text = result.distilled_text
        meaningful, reason = is_meaningful_lesson_text(new_text)
        objective, _ = is_objective_lesson_text(new_text)

        if not meaningful:
            print(f"[{i}/{len(placeholders)}] {lesson.id[:8]} FAILED: re-distill not meaningful ({reason})")
            unrecoverable += 1
            continue

        status = "OK" if objective else "OK(non-objective)"
        print(f"[{i}/{len(placeholders)}] {lesson.id[:8]} {status}: {new_text[:80]}")

        if args.apply:
            ok = db.update_lesson_text(
                lesson.id,
                new_text,
                rewritten_by_model=f"redistill:{args.model}",
                preserve_category=False,
                append_tags=["recovered-placeholder"],
                persist=True,
            )
            if not ok:
                print(f"          WARN: update_lesson_text returned False for {lesson.id[:8]}")
                continue
        recovered += 1

    print("-" * 70)
    print(f"Recovered:     {recovered}")
    print(f"Unrecoverable: {unrecoverable}")
    if not args.apply:
        print("\nDRY-RUN only. Re-run with --apply to persist changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
