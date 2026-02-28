#!/usr/bin/env python3
"""
migrate_llm_cache.py  — Re-key v5 cache entries to their thread root IDs.

Background
----------
Before the thread-root cache key fix, `_compute_cache_key()` used
`activity_item.message_id` as the key base.  For activity items that were
*not* the thread root (e.g. a review reply, a Gmail-originated patch, or any
message that is a descendant of the real cover letter), each item produced its
own unique cache key even when the full thread was shared.  This caused
redundant LLM calls and a bloated cache.

After the fix, `analyze_thread_llm()` accepts `thread_root_id` and all
activity items in the same LKML thread share one canonical cache entry.

This script migrates existing v5-format cache entries to the canonical key so
that future runs get cache hits against the new key and duplicate entries for
the same thread are pruned, shrinking the cache.

Which entries are migrated
--------------------------
Only entries whose key hash resolves with the current v5 key formula:
    sha256("{message_id}|{backend_tag}|v5")[:16]
are candidates.  Entries written with the old formula (which included
`len(messages)` in the hash, used before prompt version v5) cannot be
migrated without knowing the original thread length at analysis time, so they
are left untouched.

For each candidate, `get_thread()` is called once per unique message_id to
obtain the real thread root.  If the message_id already IS the thread root the
entry is left in place.  Otherwise the entry is copied to the canonical key
and the old key is removed.  If `get_thread()` fails the entry is skipped.

Usage
-----
    # Preview changes (no writes):
    python migrate_llm_cache.py --dry-run

    # Migrate a single date:
    python migrate_llm_cache.py --date 2026-02-24

    # Migrate all cache files:
    python migrate_llm_cache.py

    # Custom cache directory:
    python migrate_llm_cache.py --cache-dir /path/to/.llm_cache
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Must match current llm_summarizer.py values
PROMPT_VERSION = "v5"
# All backends whose v5 entries can be identified and migrated
KNOWN_BACKENDS = [
    "ollama:llama3.1:8b",
    "ollama:llama3.2:1b",
    "anthropic:claude-3-5-haiku-20241022",
    "anthropic:claude-3-5-sonnet-20241022",
    "anthropic:claude-3-haiku-20240307",
]

# Cache key format: {message_id}_{16_hex_hash}  or  {message_id}_{16_hex_hash}_pr_{suffix}
_KEY_RE = re.compile(r"^(.+)_([0-9a-f]{16})(_pr_.+)?$")


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _extract_v5_key_parts(key: str):
    """Parse a cache key and verify it was produced with the v5 formula.

    Returns (message_id, backend_tag, pr_suffix) on success, or None if the
    key is not a v5-format entry (e.g. produced with the old len(messages)
    formula or with an unknown backend).
    """
    m = _KEY_RE.match(key)
    if not m:
        return None
    msg_id = m.group(1)
    stored_hash = m.group(2)
    pr_suffix = m.group(3) or ""

    for bt in KNOWN_BACKENDS:
        content_str = "|".join([msg_id, bt, PROMPT_VERSION])
        computed = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        if computed == stored_hash:
            return msg_id, bt, pr_suffix

    return None  # old-format key (message count was in hash) — leave it


def _canonical_key(thread_root_id: str, backend_tag: str, pr_suffix: str) -> str:
    """Compute the canonical cache key for a thread root."""
    content_str = "|".join([thread_root_id, backend_tag, PROMPT_VERSION])
    h = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    return f"{thread_root_id}_{h}{pr_suffix}"


# ---------------------------------------------------------------------------
# Per-file migration
# ---------------------------------------------------------------------------

def migrate_cache_file(
    cache_file: Path,
    dry_run: bool = False,
) -> dict:
    """Migrate a single cache file.  Returns a stats dict."""
    raw = cache_file.read_text(encoding="utf-8")
    data: dict = json.loads(raw)

    # Identify all v5-format entries (any message_id, not just gmail)
    v5_entries: dict = {}  # key → (msg_id, backend_tag, pr_suffix, value)
    for key, value in data.items():
        parts = _extract_v5_key_parts(key)
        if parts:
            msg_id, backend_tag, pr_suffix = parts
            v5_entries[key] = (msg_id, backend_tag, pr_suffix, value)

    if not v5_entries:
        logger.debug("%s: no v5-format entries, skipping", cache_file.name)
        return {"file": cache_file.name, "skipped": True}

    unique_msg_ids = {v[0] for v in v5_entries.values()}
    logger.info(
        "%s: %d v5 entries, %d unique message_ids to check",
        cache_file.name, len(v5_entries), len(unique_msg_ids),
    )

    # For each unique message_id, fetch the thread to find its root.
    # If the message_id IS already the root, no migration is needed for those entries.
    from lkml_client import LKMLClient, LKMLAPIError
    client = LKMLClient()

    msg_to_root: dict = {}  # message_id → thread_root_id (None if same as msg_id or failed)
    for msg_id in sorted(unique_msg_ids):
        try:
            result = client.get_thread(msg_id)
            messages = result.get("messages", [])
            if messages:
                root_id = messages[0].get("message_id", "").strip("<>")
                if root_id and root_id != msg_id:
                    msg_to_root[msg_id] = root_id
                    logger.debug("  non-root: %s → %s", msg_id[:50], root_id[:50])
                else:
                    logger.debug("  root:     %s (no change needed)", msg_id[:50])
            else:
                logger.warning("  Empty thread returned for %s", msg_id[:50])
        except LKMLAPIError as e:
            logger.warning("  get_thread failed for %s: %s", msg_id[:50], e)

    non_root_count = len(msg_to_root)
    if non_root_count == 0:
        logger.info("  All message_ids are thread roots — nothing to migrate")
        return {"file": cache_file.name, "skipped": False, "entries": len(v5_entries),
                "non_root_ids": 0, "migrated": 0, "already_existed": 0, "failed": 0}

    logger.info("  %d / %d message_ids are non-root (need migration)", non_root_count, len(unique_msg_ids))

    # Re-key entries for non-root message_ids
    updated_data = dict(data)
    migrated = 0
    already_existed = 0
    failed = 0
    not_needed = 0

    for key, (msg_id, backend_tag, pr_suffix, value) in v5_entries.items():
        thread_root = msg_to_root.get(msg_id)
        if thread_root is None:
            # Either already a root or fetch failed — leave as-is
            not_needed += 1
            continue

        new_key = _canonical_key(thread_root, backend_tag, pr_suffix)

        if new_key == key:
            # Shouldn't happen (msg_id != root but key is same), but guard anyway
            not_needed += 1
            continue

        if new_key in updated_data:
            # Canonical entry already present; remove the duplicate non-root entry
            already_existed += 1
            logger.debug("  [dup] removing %s (canonical key already present)", key[:70])
        else:
            # First entry for this thread root — copy value to canonical key
            if not dry_run:
                updated_data[new_key] = value
            migrated += 1
            logger.debug("  [new] %s → %s", key[:60], new_key[:60])

        # Remove old non-root-keyed entry
        if not dry_run:
            updated_data.pop(key, None)

    if not dry_run and (migrated or already_existed):
        before = len(data)
        after = len(updated_data)
        cache_file.write_text(
            json.dumps(updated_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "  Saved %s: %d → %d entries (%+d)",
            cache_file.name, before, after, after - before,
        )
    elif dry_run:
        before = len(data)
        after = before - already_existed  # approximate post-dedup count
        logger.info(
            "  [DRY RUN] %s: %d → ~%d entries  "
            "(migrated=%d  deduped=%d  unchanged=%d)",
            cache_file.name, before, after,
            migrated, already_existed, not_needed,
        )

    return {
        "file": cache_file.name,
        "skipped": False,
        "entries": len(v5_entries),
        "non_root_ids": non_root_count,
        "migrated": migrated,
        "already_existed": already_existed,
        "failed": failed,
        "not_needed": not_needed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate LLM cache keys from per-message IDs to thread root IDs. "
            "Only v5-format entries (current key formula) are migrated; "
            "old-format entries (which included len(messages) in the hash) are left untouched."
        ),
    )
    parser.add_argument(
        "--cache-dir", default=".llm_cache",
        help="Path to the LLM cache directory (default: .llm_cache)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without writing any files",
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Only process the cache file for this specific date",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error("Cache directory not found: %s", cache_dir)
        sys.exit(1)

    if args.date:
        files = [cache_dir / f"{args.date}.json"]
        files = [f for f in files if f.exists()]
        if not files:
            logger.error("No cache file found for date %s in %s", args.date, cache_dir)
            sys.exit(1)
    else:
        files = sorted(cache_dir.glob("*.json"))

    if args.dry_run:
        logger.info("=== DRY RUN — no files will be modified ===")

    totals = {
        "entries": 0, "non_root_ids": 0,
        "migrated": 0, "already_existed": 0, "failed": 0,
    }

    for cache_file in files:
        try:
            stats = migrate_cache_file(cache_file, dry_run=args.dry_run)
            if not stats.get("skipped"):
                for k in totals:
                    totals[k] += stats.get(k, 0)
        except Exception as exc:
            logger.error("Error processing %s: %s", cache_file.name, exc, exc_info=True)

    logger.info(
        "=== Total: %d entries examined, %d non-root IDs resolved, "
        "%d migrated (new canonical key), %d deduplicated, %d failed ===",
        totals["entries"], totals["non_root_ids"],
        totals["migrated"], totals["already_existed"], totals["failed"],
    )


if __name__ == "__main__":
    main()
