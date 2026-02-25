"""Disk-based JSON cache for LLM analysis results.

Structure on disk:
    .llm_cache/
        2025-02-12.json    <-- one file per message date (NOT analysis-run date)
        2025-02-11.json

Each file is a JSON dict mapping cache_key -> parsed LLM response dict.

Cache files are keyed by the date of the original message being analysed
(e.g. the patch submission date), not by the date of the analysis run.
This means that if a thread from 2026-02-19 is analysed again on 2026-02-23,
the cache entry is found in 2026-02-19.json and no redundant LLM call is made.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".llm_cache"


class LLMCache:
    """File-based cache for LLM thread analysis results.

    Each cache file covers one calendar date (the date of the message being
    analysed).  Reads and writes are routed to the file matching the message
    date so that repeated analysis of the same thread on different days always
    hits the cache.

    Usage
    -----
    cache = LLMCache()
    result = cache.get(key, message_date="2026-02-19")
    if result is None:
        result = run_llm(...)
        cache.put(key, result, message_date="2026-02-19")

    If *message_date* is omitted, *date_str* (passed at construction time) is
    used as a fallback so the class is backwards-compatible.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, date_str: str = ""):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.date_str = date_str          # fallback / "current run" date
        # Per-date in-memory caches.  Lazily populated on first access.
        self._date_caches: Dict[str, Dict[str, Any]] = {}
        # Pre-load the fallback date if provided.
        if date_str:
            self._date_caches[date_str] = self._load_file(date_str)
            logger.debug(
                "LLM cache: loaded %d entries from %s.json",
                len(self._date_caches[date_str]),
                date_str,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, date_str: str) -> Dict[str, Any]:
        """Read a date cache file from disk. Returns {} on miss or error."""
        cache_file = self.cache_dir / f"{date_str}.json"
        if not cache_file.exists():
            return {}
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Cache file has unexpected format, ignoring: %s", cache_file)
                return {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache file unreadable, ignoring: %s: %s", cache_file.name, e)
            return {}

    def _ensure_loaded(self, date_str: str) -> Dict[str, Any]:
        """Return (and lazily load) the in-memory dict for *date_str*."""
        if date_str not in self._date_caches:
            self._date_caches[date_str] = self._load_file(date_str)
        return self._date_caches[date_str]

    def _save_date(self, date_str: str) -> None:
        """Flush the in-memory dict for *date_str* to disk."""
        data = self._date_caches.get(date_str, {})
        cache_file = self.cache_dir / f"{date_str}.json"
        try:
            cache_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to write cache %s: %s", cache_file.name, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, message_date: str = "") -> Optional[Dict]:
        """Retrieve a cached result.

        Lookup order:
        1. *message_date* file (if provided and differs from fallback date)
        2. Fallback *date_str* file (current analysis-run date)

        Returns None if not found in either location.
        """
        # 1. Check the message's own date file first.
        if message_date and message_date != self.date_str:
            result = self._ensure_loaded(message_date).get(key)
            if result is not None:
                return result
        # 2. Fall back to the current-run date file.
        if self.date_str:
            return self._date_caches.get(self.date_str, {}).get(key)
        return None

    def put(self, key: str, value: Dict, message_date: str = "") -> None:
        """Store a result.

        Writes to *message_date* file if provided, otherwise to the fallback
        *date_str* file.  Changes are flushed to disk immediately.
        """
        target = message_date if message_date else self.date_str
        if not target:
            logger.warning("LLMCache.put() called with no date context; entry not stored.")
            return
        self._ensure_loaded(target)[key] = value
        self._save_date(target)
        logger.info(
            "Cache written: %s.json (message_date=%s) ← %s",
            target, message_date if message_date else f"fallback:{self.date_str}", key,
        )

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = sum(len(d) for d in self._date_caches.values())
        current = len(self._date_caches.get(self.date_str, {}))
        return {"entries": current, "total_entries": total}
