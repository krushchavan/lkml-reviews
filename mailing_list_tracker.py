"""Mailing list activity tracker for LKML Daily Activity Tracker.

Maintains a persistent JSON database (data/mailing_lists.json) recording every
vger.kernel.org mailing list to which the tracked engineers have contributed.
Lists with no observed activity for 6+ months are automatically pruned.

Standalone usage:
    python mailing_list_tracker.py --list
    python mailing_list_tracker.py --stats
    python mailing_list_tracker.py --prune [--cutoff-days 180]
    python mailing_list_tracker.py -v

Programmatic usage:
    import mailing_list_tracker as mlt

    state = mlt.load_state()
    mlt.record_contribution(state, "linux-mm@vger.kernel.org", "2026-03-06")
    mlt.prune_stale(state)
    mlt.save_state(state)
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRACKER_PATH = Path("data/mailing_lists.json")
SCHEMA_VERSION = 1
VGER_DOMAIN = "vger.kernel.org"   # kept for reference / backward compat
# All kernel-development mailing list domains to track.
# Add new entries here to expand coverage.
TRACKED_DOMAINS: frozenset[str] = frozenset({
    "vger.kernel.org",    # classic kernel mailing lists
    "lists.linux.dev",    # Linux Foundation mailing lists (modern)
})

_EMPTY_STATE: dict = {
    "schema_version": SCHEMA_VERSION,
    "last_updated": "",
    "lists": {},
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_list_id(raw: str) -> str | None:
    """Convert a raw mailing-list header value to a canonical email address.

    Accepts domains listed in TRACKED_DOMAINS (vger.kernel.org and lists.linux.dev
    by default); returns None for all other domains.  Add to TRACKED_DOMAINS to
    expand coverage.

    Handles two input formats:

    * RFC 2919 List-Id format: ``Display Name <local.domain>`` or ``<local.domain>``
      — the angle-bracket portion is ``local.domain`` (dot-separated, no ``@``).
    * Plain email format: ``local@domain`` — used by X-Mailing-List and similar
      headers that already carry a proper email address.

    Examples:
        "Linux-MM <linux-mm.vger.kernel.org>"  -> "linux-mm@vger.kernel.org"
        "<netdev.vger.kernel.org>"             -> "netdev@vger.kernel.org"
        "<damon.lists.linux.dev>"              -> "damon@lists.linux.dev"
        "damon@lists.linux.dev"               -> "damon@lists.linux.dev"
        "<lists.freedesktop.org>"             -> None
        ""                                    -> None
    """
    if not raw:
        return None

    raw = raw.strip()

    # Fast-path: already a plain email address (X-Mailing-List style, no angle brackets).
    # e.g. "damon@lists.linux.dev" or "linux-mm@vger.kernel.org"
    if "@" in raw and "<" not in raw:
        at = raw.rfind("@")
        local = raw[:at].strip().lower()
        domain = raw[at + 1:].strip().lower()
        if local and domain in TRACKED_DOMAINS:
            return f"{local}@{domain}"
        return None

    # RFC 2919 List-Id format: extract the angle-bracket portion.
    # e.g. "Linux-MM <linux-mm.vger.kernel.org>" -> "linux-mm.vger.kernel.org"
    m = re.search(r"<([^>]+)>", raw)
    if m:
        domain_str = m.group(1).strip()
    else:
        # Bare domain string (no angle brackets) — try interpreting directly
        domain_str = raw.strip()

    if not domain_str:
        return None

    # Split on the first dot: "linux-mm" + "vger.kernel.org"
    #                     or  "damon"    + "lists.linux.dev"
    dot_pos = domain_str.find(".")
    if dot_pos == -1:
        return None

    local_part = domain_str[:dot_pos]
    domain_remainder = domain_str[dot_pos + 1:]

    if domain_remainder not in TRACKED_DOMAINS:
        return None

    if not local_part:
        return None

    return f"{local_part}@{domain_remainder}"


def _extract_display_name(raw: str) -> str:
    """Return the human-readable label before '<' in a List-Id header, or ''."""
    if not raw:
        return ""
    lt = raw.find("<")
    if lt > 0:
        return raw[:lt].strip()
    return ""


# ---------------------------------------------------------------------------
# State load / save
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    """Return a fresh, valid empty state dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "last_updated": "",
        "lists": {},
    }


def load_state(path: Path = DEFAULT_TRACKER_PATH) -> dict:
    """Load the tracking state from *path*.

    Returns a valid empty state if the file does not exist or cannot be parsed.
    Never raises an exception.
    """
    if not path.exists():
        logger.debug("Tracker file not found at %s; starting fresh.", path)
        return _empty_state()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "lists" not in data:
            logger.warning("Tracker file %s has unexpected format; starting fresh.", path)
            return _empty_state()
        if data.get("schema_version", 0) != SCHEMA_VERSION:
            logger.warning(
                "Tracker schema_version mismatch (got %s, expected %s); loading anyway.",
                data.get("schema_version"), SCHEMA_VERSION,
            )
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load tracker file %s: %s; starting fresh.", path, e)
        return _empty_state()


def save_state(state: dict, path: Path = DEFAULT_TRACKER_PATH) -> None:
    """Save *state* to *path*, updating last_updated to current UTC time.

    Creates parent directories as needed. Logs a warning on OS errors but
    never raises, so a tracker failure never breaks the main report pipeline.
    """
    state["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Tracker saved: %d lists -> %s", len(state.get("lists", {})), path)
    except OSError as e:
        logger.warning("Failed to save tracker file %s: %s", path, e)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def record_contribution(
    state: dict,
    list_addr: str,
    obs_date: str,
    display_name: str = "",
) -> bool:
    """Record one observed contribution to *list_addr* on *obs_date*.

    Args:
        state:        The in-memory tracker state (mutated in-place).
        list_addr:    Normalized list address, e.g. "linux-mm@vger.kernel.org".
        obs_date:     Observation date as "YYYY-MM-DD".
        display_name: Optional human label from the List-Id header.

    Returns:
        True if this is a newly discovered list, False if it already existed.

    Note: Does NOT call save_state. Callers should batch multiple calls and
          save once to avoid repeated disk writes.
    """
    lists = state.setdefault("lists", {})
    if list_addr in lists:
        entry = lists[list_addr]
        if obs_date > entry.get("last_seen", ""):
            entry["last_seen"] = obs_date
        entry["contribution_count"] = entry.get("contribution_count", 0) + 1
        if not entry.get("display_name") and display_name:
            entry["display_name"] = display_name
        return False
    else:
        lists[list_addr] = {
            "address": list_addr,
            "display_name": display_name,
            "first_seen": obs_date,
            "last_seen": obs_date,
            "contribution_count": 1,
        }
        logger.info("New mailing list discovered: %s", list_addr)
        return True


def prune_stale(state: dict, cutoff_days: int = 180) -> list[str]:
    """Remove lists whose last_seen date is older than *cutoff_days*.

    Args:
        state:       In-memory tracker state (mutated in-place).
        cutoff_days: Entries with last_seen older than this many days are removed.

    Returns:
        List of removed mailing list addresses (empty list if nothing pruned).

    Note: Does NOT call save_state. Caller decides whether to save.
    """
    cutoff = (datetime.now() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
    lists = state.get("lists", {})
    stale = [addr for addr, entry in lists.items() if entry.get("last_seen", "") < cutoff]
    for addr in stale:
        del lists[addr]
        logger.info("Pruned stale list: %s (last seen before %s)", addr, cutoff)
    return stale


def get_active_lists(state: dict) -> list[dict]:
    """Return all tracked list entries, sorted by last_seen desc, then count desc."""
    entries = list(state.get("lists", {}).values())
    entries.sort(
        key=lambda e: (e.get("last_seen", ""), e.get("contribution_count", 0)),
        reverse=True,
    )
    return entries


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _cmd_list(state: dict) -> None:
    """Print a formatted table of all active mailing lists."""
    entries = get_active_lists(state)
    if not entries:
        print("No mailing lists tracked yet.")
        print(f"Run {_c('scan_mailing_lists.py', _DIM)} to populate from existing reports,")
        print(f"or run {_c('generate_report.py', _DIM)} to populate going forward.")
        return

    # Header
    print(f"\n{_c('Active Mailing Lists', _BOLD)}  ({len(entries)} lists)\n")
    hdr = (
        f"  {'Mailing List':<38} {'Display Name':<16} "
        f"{'First Seen':>12} {'Last Seen':>12} {'Count':>7}"
    )
    print(_c(hdr, _DIM))
    print("  " + "-" * 90)

    today = datetime.now().strftime("%Y-%m-%d")
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for e in entries:
        addr = e.get("address", "")
        name = (e.get("display_name") or "")[:15]
        first = e.get("first_seen", "")
        last = e.get("last_seen", "")
        count = e.get("contribution_count", 0)

        # Color last_seen based on recency
        if last >= ninety_days_ago:
            last_colored = _c(last, _GREEN)
        elif last >= (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"):
            last_colored = _c(last, _YELLOW)
        else:
            last_colored = _c(last, _RED)

        print(
            f"  {addr:<38} {name:<16} {first:>12} {last_colored:>12}   {count:>6}"
        )
    print()


def _cmd_stats(state: dict) -> None:
    """Print summary statistics about the tracked lists."""
    entries = get_active_lists(state)
    total = len(entries)

    print(f"\n{_c('Mailing List Tracker Stats', _BOLD)}\n")
    print(f"  Total tracked lists:   {_c(str(total), _BOLD)}")

    if not entries:
        return

    most_active = max(entries, key=lambda e: e.get("contribution_count", 0))
    most_recent = entries[0]  # already sorted by last_seen desc
    oldest = min(entries, key=lambda e: e.get("first_seen", "9999"))

    cutoff_90 = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    quiet = [e for e in entries if e.get("last_seen", "") < cutoff_90]

    print(f"  Most active list:      {_c(most_active.get('address',''), _GREEN)} "
          f"({most_active.get('contribution_count', 0)} contributions)")
    print(f"  Most recently active:  {_c(most_recent.get('address',''), _CYAN)} "
          f"(last seen {most_recent.get('last_seen', '')})")
    print(f"  Oldest list:           {most_oldest_addr(oldest)} "
          f"(since {oldest.get('first_seen', '')})")
    print(f"  Lists quiet >90 days:  {len(quiet)}"
          + (f" (would be pruned at 180 days)" if quiet else ""))
    print(f"  Last file update:      {state.get('last_updated', '—')}")
    print()


def most_oldest_addr(entry: dict) -> str:
    return entry.get("address", "")


def _cmd_prune(state: dict, cutoff_days: int, path: Path) -> None:
    """Run prune, save, and report."""
    before = len(state.get("lists", {}))
    removed = prune_stale(state, cutoff_days=cutoff_days)
    after = len(state.get("lists", {}))

    if removed:
        print(f"\nPruned {len(removed)} stale list(s) (last seen > {cutoff_days} days ago):\n")
        for addr in removed:
            print(f"  {_c('✗', _RED)} {addr}")
    else:
        print(f"\nNo lists to prune (all active within {cutoff_days} days).")

    save_state(state, path)
    print(f"\n{before} → {after} lists. Saved to {path}\n")


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manage the mailing list tracking database (data/mailing_lists.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mailing_list_tracker.py --list
  python mailing_list_tracker.py --stats
  python mailing_list_tracker.py --prune
  python mailing_list_tracker.py --prune --cutoff-days 90
  python mailing_list_tracker.py --list --tracker-file /path/to/custom.json
""",
    )
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List all active mailing lists.")
    action.add_argument("--stats", action="store_true", help="Print summary statistics.")
    action.add_argument("--prune", action="store_true",
                        help="Remove stale lists (last seen > cutoff) and save.")
    p.add_argument(
        "--cutoff-days", type=int, default=180,
        help="Days of inactivity before a list is considered stale (default: 180). "
             "Used with --prune.",
    )
    p.add_argument(
        "--tracker-file", type=str, default=str(DEFAULT_TRACKER_PATH),
        metavar="PATH",
        help=f"Path to the tracking JSON file (default: {DEFAULT_TRACKER_PATH}).",
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

    tracker_path = Path(args.tracker_file)
    state = load_state(tracker_path)

    if args.list:
        _cmd_list(state)
    elif args.stats:
        _cmd_stats(state)
    elif args.prune:
        _cmd_prune(state, args.cutoff_days, tracker_path)


if __name__ == "__main__":
    main()
