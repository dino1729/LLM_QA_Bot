"""
Unified content ingestion to Memory Palace.

Auto-detects input type (YouTube URL, article URL, local file) and routes
through the shared helper pipeline, then uploads to the configured stores.
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helper_functions.link_ingestion import (
    InputType,
    detect_input_type,
    prepare_link_preview,
    upload_to_edith,
    upload_to_knowledge_archive,
    upload_to_local_memory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest content into Memory Palace from YouTube, articles, or files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ingest_to_memory.py https://youtu.be/abc123
    python scripts/ingest_to_memory.py https://paulgraham.com/ds.html --num 10 --auto
    python scripts/ingest_to_memory.py paper.pdf --skip-distill
    python scripts/ingest_to_memory.py article.pdf --tier strategic
""",
    )
    parser.add_argument("source", help="YouTube URL, article URL, or local file path")
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=13,
        help="Number of takeaways to extract (default: 13)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip interactive approval, upload immediately",
    )
    parser.add_argument(
        "--skip-distill",
        action="store_true",
        help="Store takeaways as-is without EDITH distillation (faster)",
    )
    parser.add_argument(
        "--tier",
        default="smart",
        choices=["fast", "smart", "strategic"],
        help="LLM tier for takeaway extraction (default: smart)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("helper_functions").setLevel(logging.WARNING)

    try:
        input_type = detect_input_type(args.source)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"\nDetected input type: {input_type}")
    print("Extracting content...")

    try:
        preview = prepare_link_preview(
            args.source,
            num_takeaways=args.num,
            tier=args.tier,
        )
    except Exception as exc:
        print(f"Error extracting content: {exc}")
        raise SystemExit(1) from exc

    print(f"  Extracted {preview.content.word_count} words from: {preview.content.title}\n")
    print(f"Extracted {len(preview.takeaways)} takeaways (tier: {args.tier})...")

    if not preview.takeaways:
        print("No takeaways extracted. Exiting.")
        raise SystemExit(1)

    print(f"\n{'=' * 60}")
    print(f"  {len(preview.takeaways)} TAKEAWAYS from: {preview.content.title}")
    print(f"{'=' * 60}\n")
    for i, takeaway in enumerate(preview.takeaways, 1):
        print(f"  {i}. {takeaway}\n")

    if not args.auto:
        response = input("Upload to Memory Palace? [y/n]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            raise SystemExit(0)

    print(f"\n{'=' * 60}")
    distill_label = "raw" if args.skip_distill else "with distillation"
    print(f"\n[1/3] Uploading to EDITH Lessons ({distill_label})...")
    edith_count = upload_to_edith(
        preview.takeaways,
        preview.content,
        skip_distill=args.skip_distill,
    )
    print(f"  Done: {edith_count}/{len(preview.takeaways)} lessons saved")

    print("\n[2/3] Uploading to Local Memory Palace...")
    try:
        status = upload_to_local_memory(preview.content, preview.takeaways)
        print(f"  {status}")
    except Exception as exc:
        print(f"  Failed: {exc}")

    if input_type in (InputType.YOUTUBE, InputType.URL):
        print("\n[3/3] Uploading to Knowledge Archive...")
        try:
            archive_status = upload_to_knowledge_archive(preview.content)
            print(f"  {archive_status}")
        except Exception as exc:
            print(f"  Failed: {exc}")
    else:
        print("\n[3/3] Knowledge Archive: skipped (local files not archived)")

    print(f"\n{'=' * 60}")
    print(f"Done! {edith_count} lessons ingested from: {preview.content.title}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
