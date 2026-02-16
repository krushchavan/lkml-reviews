"""Disk-based JSON cache for LLM analysis results.

Structure on disk:
    .llm_cache/
        2025-02-12.json    <-- one file per report date
        2025-02-11.json

Each file is a JSON dict mapping cache_key -> parsed LLM response dict.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".llm_cache"


class LLMCache:
    """Simple file-based cache for LLM thread analysis results."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, date_str: str = ""):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.date_str = date_str
        self._filename = self.cache_dir / f"{date_str}.json" if date_str else None
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load existing cache from disk."""
        if self._filename and self._filename.exists():
            try:
                self._data = json.loads(self._filename.read_text(encoding="utf-8"))
                logger.debug(
                    "Loaded %d cached LLM results from %s",
                    len(self._data),
                    self._filename,
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cache file corrupted, starting fresh: %s", e)
                self._data = {}

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve a cached result, or None if not cached."""
        return self._data.get(key)

    def put(self, key: str, value: Dict):
        """Store a result and persist to disk."""
        self._data[key] = value
        self._save()

    def _save(self):
        """Write cache to disk."""
        if self._filename:
            try:
                self._filename.write_text(
                    json.dumps(self._data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as e:
                logger.warning("Failed to write cache: %s", e)

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {"entries": len(self._data)}
