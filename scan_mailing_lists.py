"""Retroactive mailing list backfill for LKML Daily Activity Tracker.

Reads all existing reports/daily/*.json files, fetches the raw List-Id header
for each unique message ID from lore.kernel.org, and populates
data/mailing_lists.json.

This is a one-time (or occasional) script.  Once generate_report.py has run
with the updated mailing_list_tracker integration, new contributions are
recorded automatically on every daily run.

Usage:
    python scan_mailing_lists.py                   # Full backfill
    python scan_mailing_lists.py --dry-run -v      # Preview without writing
    python scan_mailing_lists.py --rate-limit 1.0  # Be more polite to lore
    python scan_mailing_lists.py --daily-dir PATH  # Custom reports/daily/ path
    python scan_mailing_lists.py --tracker-file PATH  # Custom tracker path
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from lkml_client import LKMLClient, LKMLAPIError
import mailing_list_tracker as mlt

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 50  # Save state to disk after every N fetched messages


# ---------------------------------------------------------------------------
# Message ID collection from existing daily JSON files
# ---------------------------------------------------------------------------

def _iter_items(dev_report: dict):
    """Yield raw item dicts from all activity categories of a developer report."""
    for key in ("patches_submitted", "patches_reviewed", "patches_acked", "discussions_posted"):
        for item in dev_report.get(key, []):
            yield item
            # Also include individual series patches
            for si in item.get("series_items", []):
                yield si


def collect_message_ids(daily_dir: Path) -> list[tuple[str, str]]:
    """Scan all daily JSON reports and return de-duplicated (message_id, obs_date) pairs.

    obs_date is taken from the item's ``date`` field (first 10 chars, YYYY-MM-DD).
    Falls back to the filename date if the field is absent or malformed.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    file_count = 0

    for json_file in sorted(daily_dir.glob("*.json")):
        filename_date = json_file.stem[:10]  # e.g. "2026-02-22"
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", json_file, e)
            continue

        file_count += 1
        for dr in data.get("developer_reports", []):
            for item in _iter_items(dr):
                msg_id = item.get("message_id", "").strip("<>")
                if not msg_id or msg_id in seen:
                    continue
                seen.add(msg_id)
                # Prefer the item's own date; fall back to filename date
                raw_date = item.get("date", "")
                obs_date = raw_date[:10] if len(raw_date) >= 10 else filename_date
                pairs.append((msg_id, obs_date))

    logger.info(
        "Collected %d unique message IDs from %d daily JSON files.",
        len(pairs), file_count,
    )
    return pairs


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan(
    daily_dir: Path,
    tracker_path: Path,
    rate_limit: float,
    dry_run: bool,
) -> None:
    state = mlt.load_state(tracker_path)
    existing_count = len(state.get("lists", {}))
    logger.info(
        "Tracker loaded: %d existing lists from %s",
        existing_count, tracker_path,
    )

    message_ids = collect_message_ids(daily_dir)
    if not message_ids:
        print("No message IDs found. Is the --daily-dir correct?")
        return

    client = LKMLClient(rate_limit_delay=rate_limit, timeout=30)

    new_lists = 0
    total_recorded = 0
    total_skipped = 0
    errors = 0
    fetched = 0

    print(f"Scanning {len(message_ids)} unique message IDs...\n")

    for i, (msg_id, obs_date) in enumerate(message_ids, start=1):
        logger.debug("[%d/%d] %s (%s)", i, len(message_ids), msg_id, obs_date)

        try:
            list_id_raw, x_ml_raw = client.get_list_id_from_raw(msg_id)
        except LKMLAPIError as e:
            logger.warning("  Fetch error for %s: %s", msg_id, e)
            errors += 1
            continue

        raw = list_id_raw or x_ml_raw
        if not raw:
            logger.debug("  No List-Id header — skipping.")
            total_skipped += 1
            continue

        normalized = mlt.normalize_list_id(raw)
        if not normalized:
            logger.debug("  Not a tracked domain: %r — skipping.", raw)
            total_skipped += 1
            continue

        display_name = mlt._extract_display_name(list_id_raw)
        fetched += 1

        if dry_run:
            logger.info("  [dry-run] would record: %s on %s (from %r)", normalized, obs_date, raw)
            continue

        is_new = mlt.record_contribution(state, normalized, obs_date, display_name)
        if is_new:
            new_lists += 1
            print(f"  + NEW: {normalized}")
        total_recorded += 1

        # Checkpoint save every N fetched messages to survive interruptions
        if not dry_run and fetched % CHECKPOINT_EVERY == 0:
            mlt.save_state(state, tracker_path)
            logger.debug("Checkpoint: saved after %d fetched messages.", fetched)

    # Final save
    if not dry_run:
        pruned = mlt.prune_stale(state)
        mlt.save_state(state, tracker_path)
        if pruned:
            print(f"\nPruned {len(pruned)} stale list(s): {', '.join(pruned)}")
        print(f"\nSaved tracker: {tracker_path}")

    # Summary
    print(f"""
Backfill complete:
  Message IDs processed : {len(message_ids)}
  Lists recorded        : {total_recorded}{"  (dry-run)" if dry_run else ""}
  Newly discovered      : {new_lists}{"  (dry-run)" if dry_run else ""}
  Skipped (unknown domain) : {total_skipped}
  Fetch errors          : {errors}
  Active lists now      : {len(state.get("lists", {}))}
""")

    if not dry_run and (new_lists or total_recorded):
        print("Run  python mailing_list_tracker.py --list  to view results.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retroactively populate data/mailing_lists.json from existing daily reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scan_mailing_lists.py                    # Full backfill (default paths)
  python scan_mailing_lists.py --dry-run -v       # Preview without writing
  python scan_mailing_lists.py --rate-limit 1.0   # Slower, more polite to lore
  python scan_mailing_lists.py --daily-dir reports/daily
  python scan_mailing_lists.py --tracker-file data/mailing_lists.json
""",
    )
    p.add_argument(
        "--daily-dir", type=str, default="reports/daily",
        metavar="PATH",
        help="Directory containing daily JSON report files (default: reports/daily).",
    )
    p.add_argument(
        "--tracker-file", type=str, default=str(mlt.DEFAULT_TRACKER_PATH),
        metavar="PATH",
        help=f"Path to the tracking JSON file (default: {mlt.DEFAULT_TRACKER_PATH}).",
    )
    p.add_argument(
        "--rate-limit", type=float, default=0.5,
        metavar="SECS",
        help="Seconds between HTTP requests to lore.kernel.org (default: 0.5).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and parse but do not write to the tracker file.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    daily_dir = Path(args.daily_dir)
    tracker_path = Path(args.tracker_file)

    if not daily_dir.exists():
        print(f"Error: daily reports directory not found: {daily_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    scan(daily_dir, tracker_path, args.rate_limit, args.dry_run)


if __name__ == "__main__":
    main()
