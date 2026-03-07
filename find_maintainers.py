"""Find kernel maintainers for tracked mailing lists.

Downloads the Linux kernel MAINTAINERS file from kernel.org, parses it, and
produces data/maintainers.json mapping each tracked mailing list address to
the subsystems and maintainers responsible for it.

Usage:
    python find_maintainers.py                          # fetch + parse + write
    python find_maintainers.py --cache                  # use cached MAINTAINERS file
    python find_maintainers.py --output data/maintainers.json
    python find_maintainers.py --list-file data/mailing_lists.json
    python find_maintainers.py --show                   # pretty-print to stdout only

The output JSON has this shape:
    {
      "linux-nfs@vger.kernel.org": [
        {
          "subsystem": "NFS AND RELATED FILESYSTEMS",
          "maintainers": ["Trond Myklebust <trond.myklebust@hammerspace.com>", ...],
          "reviewers":   ["Anna Schumaker <anna@kernel.org>", ...],
          "status":      "Maintained"
        },
        ...
      ],
      ...
    }
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests

logger = logging.getLogger(__name__)

MAINTAINERS_URL = (
    "https://raw.githubusercontent.com/torvalds/linux/master/MAINTAINERS"
)
DEFAULT_CACHE_PATH = Path("data/MAINTAINERS.cache")
DEFAULT_OUTPUT_PATH = Path("data/maintainers.json")
DEFAULT_LIST_FILE = Path("data/mailing_lists.json")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def fetch_maintainers(url: str, cache_path: Path, use_cache: bool) -> str:
    """Return the MAINTAINERS file contents as a string."""
    if use_cache and cache_path.exists():
        logger.info("Using cached MAINTAINERS file: %s", cache_path)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    logger.info("Downloading MAINTAINERS from %s …", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.text

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    logger.info("Cached MAINTAINERS to %s (%d bytes)", cache_path, len(text))
    return text


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Lines in a MAINTAINERS block that we care about
_LINE_RE = re.compile(r"^([A-Z]):\s+(.*)")

# Strip modifiers like "(moderated)" or "(patches)" from mailing list lines
_LIST_MODIFIER_RE = re.compile(r"\s*\(.*?\)\s*$")


def _normalise_list_addr(raw: str) -> str:
    """Lower-case and strip modifiers from an L: line value."""
    addr = _LIST_MODIFIER_RE.sub("", raw).strip().lower()
    return addr


def parse_maintainers(text: str) -> list[dict]:
    """Parse MAINTAINERS text into a list of subsystem dicts.

    Each dict has keys:
        subsystem   – name string
        maintainers – list of "Name <email>" strings (M: lines)
        reviewers   – list of "Name <email>" strings (R: lines)
        lists       – list of normalised mailing-list addresses (L: lines)
        status      – string from S: line (or "")
    """
    subsystems: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        if not line.strip():
            # Blank line → end of block
            if current is not None:
                subsystems.append(current)
                current = None
            continue

        m = _LINE_RE.match(line)
        if m:
            if current is None:
                # A field line without a preceding title — skip (preamble)
                continue
            tag, value = m.group(1), m.group(2).strip()
            if tag == "M":
                current["maintainers"].append(value)
            elif tag == "R":
                current["reviewers"].append(value)
            elif tag == "L":
                current["lists"].append(_normalise_list_addr(value))
            elif tag == "S":
                current["status"] = value
        else:
            # No recognised prefix → treat as subsystem title
            # (Could also be blank-stripped comment lines, but those are blank)
            if current is not None:
                subsystems.append(current)
            current = {
                "subsystem": line.strip(),
                "maintainers": [],
                "reviewers": [],
                "lists": [],
                "status": "",
            }

    if current is not None:
        subsystems.append(current)

    return subsystems


# ---------------------------------------------------------------------------
# Match subsystems to tracked lists
# ---------------------------------------------------------------------------

def build_mapping(subsystems: list[dict], tracked: set[str]) -> dict[str, list[dict]]:
    """Return a dict mapping tracked mailing list addr → list of matching subsystem dicts."""
    mapping: dict[str, list[dict]] = {addr: [] for addr in tracked}

    for sub in subsystems:
        for addr in sub["lists"]:
            if addr in tracked:
                mapping[addr].append({
                    "subsystem": sub["subsystem"],
                    "maintainers": sub["maintainers"],
                    "reviewers": sub["reviewers"],
                    "status": sub["status"],
                })

    return mapping


# ---------------------------------------------------------------------------
# Load tracked mailing lists
# ---------------------------------------------------------------------------

def load_tracked_lists(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("lists", {}).keys())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(mapping: dict[str, list[dict]]) -> None:
    total_subs = sum(len(v) for v in mapping.values())
    print(f"\nFound {total_subs} subsystem entries across {len(mapping)} mailing lists.\n")
    for addr in sorted(mapping):
        entries = mapping[addr]
        if not entries:
            print(f"  {addr}  — NO MATCH IN MAINTAINERS")
            continue
        print(f"  {addr}  ({len(entries)} subsystem(s))")
        for e in entries:
            print(f"    • {e['subsystem']}  [{e['status']}]")
            for m in e["maintainers"]:
                print(f"        M: {m}")
            for r in e["reviewers"]:
                print(f"        R: {r}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map tracked mailing lists to kernel MAINTAINERS entries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--list-file", default=str(DEFAULT_LIST_FILE), metavar="PATH",
        help=f"Tracked mailing lists JSON (default: {DEFAULT_LIST_FILE}).",
    )
    p.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH), metavar="PATH",
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--cache-file", default=str(DEFAULT_CACHE_PATH), metavar="PATH",
        help=f"Local cache for MAINTAINERS (default: {DEFAULT_CACHE_PATH}).",
    )
    p.add_argument(
        "--cache", action="store_true",
        help="Use cached MAINTAINERS file if it exists (avoids network request).",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Pretty-print results to stdout (in addition to writing JSON).",
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

    list_path = Path(args.list_file)
    if not list_path.exists():
        print(f"Error: mailing list file not found: {list_path}", file=sys.stderr)
        sys.exit(1)

    tracked = load_tracked_lists(list_path)
    logger.info("Tracking %d mailing lists.", len(tracked))

    text = fetch_maintainers(
        MAINTAINERS_URL,
        cache_path=Path(args.cache_file),
        use_cache=args.cache,
    )

    logger.info("Parsing MAINTAINERS …")
    subsystems = parse_maintainers(text)
    logger.info("Parsed %d subsystem blocks.", len(subsystems))

    mapping = build_mapping(subsystems, tracked)

    if args.show:
        _print_summary(mapping)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {out_path}")

    # Quick coverage report
    matched = sum(1 for v in mapping.values() if v)
    unmatched = [addr for addr, v in mapping.items() if not v]
    print(f"Coverage: {matched}/{len(tracked)} lists matched in MAINTAINERS.")
    if unmatched:
        print("No match found for:")
        for addr in sorted(unmatched):
            print(f"  {addr}")


if __name__ == "__main__":
    main()
