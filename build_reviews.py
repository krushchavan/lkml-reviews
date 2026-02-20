"""Build per-patchset review HTML files from JSON data.

Scans reports/reviews/*.json and generates a corresponding HTML file for each
patchset showing review comments as a threaded conversation tree.  Dates are
embedded as chips on individual cards rather than used as section dividers.
Each date has a named anchor so the main report can deep-link to new activity
on a specific day.

Usage:
    python build_reviews.py                       # Uses reports/ in CWD
    python build_reviews.py --reports-dir /app/reports
"""

import argparse
import html
import json
from pathlib import Path


_SENTIMENT_COLORS = {
    "positive": ("#155724", "#d4edda"),
    "needs_work": ("#856404", "#fff3cd"),
    "contentious": ("#721c24", "#f8d7da"),
    "neutral": ("#383d41", "#e2e3e5"),
}

_ANALYSIS_SOURCE_STYLES = {
    "heuristic": ("#6c4b00", "#ffeeba", "Heuristic"),
    "llm": ("#004085", "#cce5ff", "LLM"),
    "llm-per-reviewer": ("#004085", "#cce5ff", "LLM"),
    "llm-fallback-heuristic": ("#721c24", "#f8d7da", "LLM \u2192 Heuristic"),
}

_SENTIMENT_LABELS = {
    "positive": "Positive",
    "needs_work": "Needs Work",
    "contentious": "Contentious",
    "neutral": "Neutral",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _sentiment_badge(sentiment: str) -> str:
    color, bg = _SENTIMENT_COLORS.get(sentiment, ("#383d41", "#e2e3e5"))
    label = _SENTIMENT_LABELS.get(sentiment, "Neutral")
    return (
        f'<span class="badge" style="color:{color};background:{bg}">'
        f"{label}</span>"
    )


def _analysis_source_badge(source: str) -> str:
    color, bg, label = _ANALYSIS_SOURCE_STYLES.get(
        source, ("#383d41", "#e2e3e5", source)
    )
    return (
        f'<span class="analysis-source-badge" style="color:{color};background:{bg}"'
        f' title="Analysis source: {_esc(label)}">'
        f'{label}</span>'
    )


# ---------------------------------------------------------------------------
# Cross-date merge
# ---------------------------------------------------------------------------

_SOURCE_RANK = {
    "llm": 3, "llm-per-reviewer": 3,
    "llm-fallback-heuristic": 2,
    "heuristic": 1,
}


def _merge_reviews_across_dates(dates: dict) -> tuple[list[dict], str]:
    """Collect review blocks from all dates, one card per (author, reply_to, date).

    Each date in the JSON is a full snapshot of all reviews known at that point
    in time.  Rather than deduplicating by (author, reply_to) — which would
    hide updates from later dates — we emit one card per unique
    (author, reply_to, date) triple.  This means:

    - A reviewer who first commented on Feb 13 and added a follow-up on Feb 17
      will appear as TWO separate cards, each showing its own date chip.
    - Within a single date's snapshot, if the same (author, reply_to) appears
      more than once the later entry wins (shouldn't happen in practice).

    Each block carries:
    - ``first_seen``: the date this specific entry comes from (date chip)
    - ``report_file``: report HTML filename for the date-chip link
    - ``anchor_date``: stamped on the first card that is new for each date
      (i.e. first card whose date == date_str) so that ``id="{date}"`` deep-links
      from the main report land on the correct card.

    Reviews within a date that are identical in content to a previous date's
    entry for the same (author, reply_to) are skipped to avoid noise from
    unchanged carry-over entries.  Two entries are considered identical if
    their ``summary`` strings match exactly.

    Also picks the best patch_summary (LLM preferred over heuristic).

    Returns: (reviews_list, patch_summary)
    """
    sorted_dates = sorted(dates.keys())  # oldest → newest

    best_patch_summary = ""
    best_patch_source_rank = -1

    # Track the last-seen summary for each (author, reply_to) so we can skip
    # exact duplicates carried forward across dates.
    last_summary: dict[tuple, str] = {}

    # Result list — one entry per card
    result: list[dict] = []

    # For anchor placement: first card index per date
    first_idx_per_date: dict[str, int | None] = {d: None for d in sorted_dates}

    for date_str in sorted_dates:
        date_data = dates[date_str]
        report_file = date_data.get("report_file", "")

        # Best patch_summary
        ps = date_data.get("patch_summary", "")
        src = date_data.get("analysis_source", "heuristic")
        src_rank = _SOURCE_RANK.get(src, 0)
        if ps and src_rank >= best_patch_source_rank:
            best_patch_summary = ps
            best_patch_source_rank = src_rank

        for review in date_data.get("reviews", []):
            if not isinstance(review, dict):
                continue
            author = str(review.get("author") or "").strip()
            reply_to = str(review.get("reply_to") or "").strip()
            dedup_key = (author, reply_to)
            summary = str(review.get("summary") or "")

            # Skip if content is identical to what we already showed for this
            # (author, reply_to) on a previous date — pure carry-over noise.
            if last_summary.get(dedup_key) == summary:
                continue

            last_summary[dedup_key] = summary

            entry = dict(review)
            entry["first_seen"] = date_str
            entry["report_file"] = report_file

            # Record the first new card for this date (for anchor placement)
            if first_idx_per_date[date_str] is None:
                first_idx_per_date[date_str] = len(result)

            result.append(entry)

    # Stamp anchor_date onto the first card new for each date.
    for date_str, idx in first_idx_per_date.items():
        if idx is not None:
            result[idx]["anchor_date"] = date_str

    return result, best_patch_summary


# ---------------------------------------------------------------------------
# Thread tree construction
# ---------------------------------------------------------------------------

def _build_tree(reviews: list[dict]) -> list[dict]:
    """Arrange flat review list into a reply tree.

    Each node gets a ``children`` list.  Matching is done by author short name
    against ``reply_to`` values (case-insensitive, first-match wins when there
    are multiple blocks by the same author).

    Root nodes are blocks whose ``reply_to`` is empty or doesn't match any
    known author.  Children are sorted by ``first_seen`` date then by their
    original order so the tree reads chronologically top-to-bottom.
    """
    # Build a name → list[node] map (one author may have multiple blocks with
    # different reply_to values; we pick the first matching node as parent)
    nodes = [dict(r, children=[]) for r in reviews]

    # Index: lowercased author name → list of node indices
    name_index: dict[str, list[int]] = {}
    for i, node in enumerate(nodes):
        name = node.get("author", "").lower().strip()
        # Strip " (author)" suffix that the LLM sometimes appends
        name = name.replace(" (author)", "").strip()
        name_index.setdefault(name, []).append(i)

    roots: list[dict] = []
    placed = set()

    for i, node in enumerate(nodes):
        reply_to = node.get("reply_to", "").lower().strip()
        if not reply_to:
            roots.append(node)
            placed.add(i)
            continue

        # Find the best parent: the first node (by order) whose author name
        # matches reply_to and is not the node itself
        parent_indices = name_index.get(reply_to, [])
        parent = None
        for pi in parent_indices:
            if pi != i:
                parent = nodes[pi]
                break

        if parent is not None:
            parent["children"].append(node)
            placed.add(i)
        else:
            # reply_to didn't resolve — treat as root
            roots.append(node)
            placed.add(i)

    # Anything not placed yet (shouldn't happen, but guard)
    for i, node in enumerate(nodes):
        if i not in placed:
            roots.append(node)

    return roots


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_node(node: dict, depth: int = 0) -> str:
    """Render a single review node and its children recursively."""
    parts = []

    # Emit id="{date}" on the wrapper div when this card is the first
    # new review for that date — makes deep-links from the main report land here.
    indent_class = f"depth-{min(depth, 4)}"
    anchor_date = node.get("anchor_date", "")
    id_attr = f' id="{_esc(anchor_date)}"' if anchor_date else ""
    parts.append(f'<div class="thread-node {indent_class}"{id_attr}>')

    # ── Card ──
    parts.append('<div class="review-comment">')

    # Header row
    parts.append('<div class="review-comment-header">')
    parts.append(f'<span class="review-author">{_esc(node.get("author", ""))}</span>')

    # Date chip
    first_seen = node.get("first_seen", "")
    report_file = node.get("report_file", "")
    if first_seen:
        if report_file:
            parts.append(
                f'<a class="date-chip" href="../{_esc(report_file)}" '
                f'title="First appeared in report for {_esc(first_seen)}">'
                f'{_esc(first_seen)}</a>'
            )
        else:
            parts.append(f'<span class="date-chip">{_esc(first_seen)}</span>')

    if node.get("has_inline_review"):
        parts.append('<span class="inline-review-badge">Inline Review</span>')

    for tag in node.get("tags_given", []):
        parts.append(f'<span class="review-tag-badge">{_esc(tag)}</span>')

    parts.append(_sentiment_badge(node.get("sentiment", "neutral")))

    rc_source = node.get("analysis_source", "")
    if rc_source:
        parts.append(_analysis_source_badge(rc_source))

    parts.append('</div>')  # review-comment-header

    # Summary
    summary = node.get("summary", "")
    if summary:
        parts.append(f'<div class="review-comment-text">{_esc(summary)}</div>')

    # Collapsible raw body
    raw_body = node.get("raw_body", "")
    if raw_body:
        parts.append('<details class="raw-body-toggle">')
        parts.append('<summary>Show original comment</summary>')
        parts.append(f'<pre class="raw-body-text">{_esc(raw_body)}</pre>')
        parts.append('</details>')

    # Sentiment signals
    signals = node.get("sentiment_signals", [])
    if signals:
        signals_str = ", ".join(signals[:3])
        parts.append(f'<div class="review-comment-signals">Signals: {_esc(signals_str)}</div>')

    parts.append('</div>')  # review-comment

    # ── Children ──
    children = node.get("children", [])
    if children:
        parts.append('<div class="thread-children">')
        for child in children:
            parts.append(_render_node(child, depth + 1))
        parts.append('</div>')

    parts.append('</div>')  # thread-node
    return "\n".join(parts)


def _render_thread_tree(reviews: list[dict]) -> str:
    """Render the full thread tree HTML from a flat review list."""
    roots = _build_tree(reviews)
    if not roots:
        return '<div class="no-reviews">No review comments yet.</div>'
    parts = ['<div class="thread-tree">']
    for root in roots:
        parts.append(_render_node(root, depth=0))
    parts.append('</div>')
    return "\n".join(parts)


def build_review_html(data: dict) -> str:
    """Generate a self-contained HTML page for one patchset's review comments."""
    subject = data.get("subject", "Unknown Patch")
    url = data.get("url", "")
    dates = data.get("dates", {})

    # Merge across dates and reconstruct tree
    merged_reviews, patch_summary = _merge_reviews_across_dates(dates)
    thread_html = _render_thread_tree(merged_reviews)

    # Patch summary block (above the tree)
    if patch_summary:
        patch_summary_html = (
            f'<div class="patch-summary-block">'
            f'<div class="patch-summary-label">Patch summary</div>'
            f'<div class="patch-summary-text">{_esc(patch_summary)}</div>'
            f'</div>'
        )
    else:
        patch_summary_html = ""

    # Date list for the page subtitle (newest first, comma-separated)
    sorted_dates = sorted(dates.keys(), reverse=True)
    date_range = ""
    if sorted_dates:
        date_range = (
            f'<div class="date-range">Active on: '
            + " &bull; ".join(
                f'<a href="#{_esc(d)}">{_esc(d)}</a>' for d in sorted_dates
            )
            + "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Review Comments: {_esc(subject)}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}
        .home-link {{ margin-bottom: 12px; display: block; }}
        .home-link a {{ color: #0366d6; text-decoration: none; font-size: 0.9em; }}
        .home-link a:hover {{ text-decoration: underline; }}

        h1 {{ font-size: 1.3em; margin-bottom: 2px; color: #1a1a1a; line-height: 1.3; }}

        .lore-link {{ font-size: 0.85em; margin: 4px 0 6px; display: block; }}
        .lore-link a {{ color: #0366d6; text-decoration: none; }}
        .lore-link a:hover {{ text-decoration: underline; }}

        .date-range {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 16px;
        }}
        .date-range a {{ color: #0366d6; text-decoration: none; }}
        .date-range a:hover {{ text-decoration: underline; }}

        /* thread-node scroll margin so the card isn't clipped at the top */
        .thread-node {{ scroll-margin-top: 8px; }}

        /* ── Patch summary ──────────────────────────────────────────── */
        .patch-summary-block {{
            background: #fff;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border-left: 3px solid #4a90d9;
        }}
        .patch-summary-label {{
            font-size: 0.72em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #4a90d9;
            margin-bottom: 4px;
        }}
        .patch-summary-text {{
            font-size: 0.88em;
            color: #444;
            line-height: 1.55;
        }}

        /* ── Thread tree ────────────────────────────────────────────── */
        .thread-tree {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        /* Depth indentation via left border */
        .thread-node {{ position: relative; }}
        .thread-children {{
            margin-left: 20px;
            padding-left: 12px;
            border-left: 2px solid #e0e0e0;
            margin-top: 6px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        /* ── Review comment card ────────────────────────────────────── */
        .review-comment {{
            background: #fff;
            border-radius: 6px;
            padding: 10px 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            font-size: 0.88em;
        }}
        .review-comment-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            margin-bottom: 5px;
        }}
        .review-author {{
            font-weight: 700;
            color: #1a1a1a;
            font-size: 0.95em;
        }}

        /* Date chip — links back to the daily report */
        .date-chip {{
            font-size: 0.75em;
            color: #777;
            background: #f0f0f0;
            border-radius: 10px;
            padding: 1px 7px;
            text-decoration: none;
            white-space: nowrap;
        }}
        a.date-chip:hover {{ background: #e0e8f5; color: #0366d6; }}

        .badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 600;
        }}
        .inline-review-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.78em;
            font-weight: 500;
            background: #e3f2fd;
            color: #1565c0;
        }}
        .review-tag-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.78em;
            font-weight: 500;
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .analysis-source-badge {{
            display: inline-block;
            padding: 1px 7px;
            border-radius: 10px;
            font-size: 0.72em;
            font-weight: 600;
            border: 1px solid rgba(0,0,0,0.1);
        }}

        .review-comment-text {{
            color: #444;
            line-height: 1.55;
            margin-bottom: 4px;
        }}
        .review-comment-signals {{
            margin-top: 3px;
            font-size: 0.85em;
            color: #aaa;
            font-style: italic;
        }}

        /* ── Collapsible raw body ───────────────────────────────────── */
        .raw-body-toggle {{
            margin-top: 5px;
            font-size: 0.85em;
        }}
        .raw-body-toggle summary {{
            cursor: pointer;
            color: #888;
            padding: 2px 0;
            font-weight: 500;
            font-size: 0.9em;
            list-style: none;
        }}
        .raw-body-toggle summary::-webkit-details-marker {{ display: none; }}
        .raw-body-toggle summary::before {{ content: "▶ "; font-size: 0.7em; }}
        .raw-body-toggle[open] summary::before {{ content: "▼ "; }}
        .raw-body-toggle summary:hover {{ color: #555; }}
        .raw-body-text {{
            white-space: pre-wrap;
            font-size: 0.95em;
            background: #f8f8f8;
            padding: 8px 10px;
            border-radius: 4px;
            max-height: 360px;
            overflow-y: auto;
            margin-top: 4px;
            line-height: 1.5;
            color: #444;
            border: 1px solid #e8e8e8;
        }}

        .no-reviews {{
            color: #aaa;
            font-size: 0.85em;
            font-style: italic;
            padding: 8px 0;
        }}

        footer {{
            text-align: center;
            color: #bbb;
            font-size: 0.78em;
            margin-top: 36px;
            padding: 16px;
        }}
    </style>
</head>
<body>
    <div class="home-link"><a href="../">&larr; Back to reports</a></div>
    <h1>{_esc(subject)}</h1>
    {'<div class="lore-link"><a href="' + _esc(url) + '" target="_blank">View on lore.kernel.org &rarr;</a></div>' if url else ''}
    {date_range}
    {patch_summary_html}
    {thread_html}

    <footer>LKML Daily Activity Tracker</footer>
    <script>
    // When arriving via a date anchor (e.g. #2026-02-15 from a daily report),
    // scroll the anchor into view after a brief delay so layout is complete.
    (function () {{
        var hash = window.location.hash;
        if (!hash) return;
        var target = document.getElementById(hash.slice(1));
        if (!target) return;
        setTimeout(function () {{
            target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
        }}, 80);
    }})();
    </script>
</body>
</html>"""


def build_all_reviews(reports_dir: Path) -> int:
    """Build HTML review pages for all JSON files in reports/reviews/.

    Returns the number of review pages generated.
    """
    reviews_dir = reports_dir / "reviews"
    if not reviews_dir.exists():
        return 0

    count = 0
    for json_path in sorted(reviews_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {json_path}: {e}")
            continue

        html_content = build_review_html(data)
        html_path = json_path.with_suffix(".html")
        html_path.write_text(html_content, encoding="utf-8")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Build per-patchset review HTML files.")
    parser.add_argument("--reports-dir", default="reports",
                        help="Reports directory containing reviews/ subdir. Default: reports/")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    count = build_all_reviews(reports_dir)
    print(f"Review pages generated: {count}")


if __name__ == "__main__":
    main()
