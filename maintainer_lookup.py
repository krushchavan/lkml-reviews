"""Maintainer lookup utilities for LKML Daily Activity Tracker.

Loads data/maintainers.json and provides helpers to:
  - Build a per-list set of maintainer emails and short names.
  - Mark ReviewComment instances whose author is a subsystem maintainer.

Matching strategy (applied in order):
  1. Email match  — exact, lowercase (used for new reports with email field).
  2. Name match   — short "First Last" form, case-insensitive (fallback for
                    older JSON reports where email was not yet captured).

Usage:
    import maintainer_lookup

    lookup = maintainer_lookup.load()
    maintainer_lookup.mark_maintainers(item, lookup)
"""

import json
import logging
import re
from pathlib import Path

from models import ActivityItem

logger = logging.getLogger(__name__)

DEFAULT_MAINTAINERS_PATH = Path("data/maintainers.json")

_EMAIL_RE = re.compile(r"<([^>]+)>")
_NAME_RE = re.compile(r"^(.*?)\s*<")
# Strip parenthetical suffixes the analyzer appends: "(author)", "(Google)", "(Arm)", …
_PAREN_SUFFIX_RE = re.compile(r"\s*\(.*?\)\s*$")


def _parse_email(name_email: str) -> str:
    """Extract lowercase email from 'Name <email>' string."""
    m = _EMAIL_RE.search(name_email)
    return m.group(1).lower() if m else ""


def _short_name(full_name: str) -> str:
    """Return the 'First Last' short form used by _extract_author_short().

    Takes the first word and the last word of the name, mirroring the logic
    in thread_analyzer._extract_author_short() so name lookups stay in sync.
    """
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}".lower()
    return full_name.strip().lower()


def _normalise_author(author: str) -> str:
    """Normalise an rc.author string for name-based lookup.

    Strips parenthetical suffixes added by the analyzer:
      "Steven Rostedt"          → "steven rostedt"
      "Masami (Google)"         → "masami"   (no last name — single token)
      "Vlastimil (SUSE)"        → "vlastimil"
      "Dmitry Ilvokhin (author)"→ "dmitry ilvokhin"
    """
    stripped = _PAREN_SUFFIX_RE.sub("", author).strip()
    return stripped.lower()


# ---------------------------------------------------------------------------
# Lookup data structure
# ---------------------------------------------------------------------------

class MaintainerLookup:
    """Holds both email-based and name-based maintainer sets per list."""

    def __init__(self) -> None:
        # list_addr → frozenset of lowercase maintainer emails
        self.by_email: dict[str, frozenset[str]] = {}
        # list_addr → frozenset of lowercase short names ("first last")
        self.by_name: dict[str, frozenset[str]] = {}
        # Global sets (union across all lists) used when list_id is unknown
        self._global_emails: frozenset[str] = frozenset()
        self._global_names: frozenset[str] = frozenset()

    def _build_globals(self) -> None:
        self._global_emails = frozenset(
            e for emails in self.by_email.values() for e in emails
        )
        self._global_names = frozenset(
            n for names in self.by_name.values() for n in names
        )

    def is_maintainer(self, list_id: str, email: str, author: str) -> bool:
        """Return True if the reviewer is a subsystem maintainer.

        When list_id is known, matches against that list's maintainer set
        first.  If not found there, falls back to the global set so that
        maintainers of related subsystems (e.g. david@kernel.org indexed under
        linux-mm@kvack.org but reviewing a thread fetched via linux-kernel@
        vger.kernel.org) are still recognised.
        When list_id is empty (old JSON reports), only the global set is used.
        """
        norm = _normalise_author(author)
        if list_id:
            emails = self.by_email.get(list_id, frozenset())
            names = self.by_name.get(list_id, frozenset())
            if email and email in emails:
                return True
            if norm and norm in names:
                return True
            # Fall back to global: catches maintainers whose subsystem list
            # differs from the list_id captured from the lore.kernel.org thread
            # (e.g. fetched from /all/ which returns linux-kernel as list_id).
        if email and email in self._global_emails:
            return True
        if norm and norm in self._global_names:
            return True
        return False


def load(path: Path = DEFAULT_MAINTAINERS_PATH) -> MaintainerLookup:
    """Load maintainers.json and return a MaintainerLookup.

    Only M: (maintainer) entries are included — R: (reviewer) entries are
    intentionally excluded so the badge is reserved for official maintainers.
    Returns an empty lookup if the file is missing or malformed.
    """
    result = MaintainerLookup()

    if not path.exists():
        logger.debug("maintainers.json not found at %s; maintainer lookup disabled.", path)
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load maintainers.json: %s", e)
        return result

    for list_addr, subsystem_entries in data.items():
        emails: set[str] = set()
        names: set[str] = set()
        for entry in subsystem_entries:
            for m_str in entry.get("maintainers", []):
                email = _parse_email(m_str)
                if email:
                    emails.add(email)
                # Also index by short name extracted from the full "Name <email>" string
                name_match = _NAME_RE.match(m_str)
                if name_match:
                    full = name_match.group(1).strip()
                    sn = _short_name(full)
                    if sn:
                        names.add(sn)
        result.by_email[list_addr] = frozenset(emails)
        result.by_name[list_addr] = frozenset(names)

    result._build_globals()
    total_emails = sum(len(v) for v in result.by_email.values())
    total_names = sum(len(v) for v in result.by_name.values())
    logger.debug(
        "Maintainer lookup loaded: %d lists, %d email entries, %d name entries, "
        "%d global emails, %d global names.",
        len(result.by_email), total_emails, total_names,
        len(result._global_emails), len(result._global_names),
    )
    return result


def _mark_conv(conv, list_id: str, lookup: MaintainerLookup) -> None:
    for rc in conv.review_comments:
        if lookup.is_maintainer(list_id, rc.email, rc.author):
            rc.is_maintainer = True


def mark_maintainers(item: ActivityItem, lookup: MaintainerLookup) -> None:
    """Set is_maintainer=True on ReviewComments whose author is a subsystem maintainer.

    Uses item.list_id to find the relevant maintainer set.  Tries email match
    first (new reports), falls back to short-name match (older JSON reports).
    Also recurses into llm_analyses and series_items.
    """
    list_id = getattr(item, "list_id", "")

    conv = getattr(item, "conversation", None)
    if conv:
        _mark_conv(conv, list_id, lookup)

    # Recurse into LLM multi-analysis results
    for analysis in getattr(item, "llm_analyses", []):
        _mark_conv(analysis.conversation, list_id, lookup)

    # Recurse into series items (individual patches)
    for si in getattr(item, "series_items", []):
        mark_maintainers(si, lookup)
