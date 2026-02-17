"""Build per-patchset review HTML files from JSON data.

Scans reports/reviews/*.json and generates a corresponding HTML file for each
patchset, with review comments grouped by date (newest first). Each date section
has an anchor so the main report can deep-link to it.

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


def _render_review(review: dict) -> str:
    """Render a single review comment block."""
    parts = []
    parts.append('<div class="review-comment">')

    # Header: author + badges
    parts.append('<div class="review-comment-header">')
    parts.append(f'<span class="review-author">{_esc(review["author"])}</span>')

    if review.get("has_inline_review"):
        parts.append('<span class="inline-review-badge">Inline Review</span>')

    for tag in review.get("tags_given", []):
        parts.append(f'<span class="review-tag-badge">{_esc(tag)}</span>')

    parts.append(_sentiment_badge(review.get("sentiment", "neutral")))
    parts.append('</div>')

    # Summary text
    if review.get("summary"):
        parts.append(f'<div class="review-comment-text">{_esc(review["summary"])}</div>')

    # Signals
    signals = review.get("sentiment_signals", [])
    if signals:
        signals_str = ", ".join(signals[:3])
        parts.append(f'<div class="review-comment-signals">Signals: {_esc(signals_str)}</div>')

    parts.append('</div>')
    return "\n".join(parts)


def build_review_html(data: dict) -> str:
    """Generate a self-contained HTML page for one patchset's review comments."""
    subject = data.get("subject", "Unknown Patch")
    url = data.get("url", "")
    dates = data.get("dates", {})

    # Sort dates newest first
    sorted_dates = sorted(dates.keys(), reverse=True)

    # Build date sections
    date_sections = []
    for date_str in sorted_dates:
        date_data = dates[date_str]
        report_file = date_data.get("report_file", "")
        developer = date_data.get("developer", "")
        reviews = date_data.get("reviews", [])

        parts = []
        parts.append(f'<div class="date-section" id="{_esc(date_str)}">')
        parts.append(f'<div class="date-header">')
        parts.append(f'<h2>{_esc(date_str)}</h2>')
        if report_file:
            parts.append(f'<a href="../{_esc(report_file)}" class="back-link">'
                         f'&larr; Back to report</a>')
        parts.append('</div>')

        if developer:
            parts.append(f'<div class="developer-name">Developer: {_esc(developer)}</div>')

        if reviews:
            parts.append('<div class="review-comments">')
            for review in reviews:
                parts.append(_render_review(review))
            parts.append('</div>')
        else:
            parts.append('<div class="no-reviews">No review comments</div>')

        parts.append('</div>')
        date_sections.append("\n".join(parts))

    sections_html = "\n".join(date_sections)

    # Date navigation bar
    date_nav = ""
    if len(sorted_dates) > 1:
        links = [f'<a href="#{_esc(d)}">{_esc(d)}</a>' for d in sorted_dates]
        date_nav = f'<div class="date-nav">{" &bull; ".join(links)}</div>'

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
        h1 {{
            font-size: 1.4em;
            margin-bottom: 4px;
            color: #1a1a1a;
        }}
        .lore-link {{
            font-size: 0.85em;
            margin-bottom: 16px;
            display: block;
        }}
        .lore-link a {{
            color: #0366d6;
            text-decoration: none;
        }}
        .lore-link a:hover {{
            text-decoration: underline;
        }}
        .date-nav {{
            margin-bottom: 20px;
            padding: 10px 16px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.85em;
        }}
        .date-nav a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }}
        .date-nav a:hover {{
            text-decoration: underline;
        }}
        .home-link {{
            margin-bottom: 16px;
            display: block;
        }}
        .home-link a {{
            color: #0366d6;
            text-decoration: none;
            font-size: 0.9em;
        }}
        .home-link a:hover {{
            text-decoration: underline;
        }}
        .date-section {{
            background: #fff;
            border-radius: 8px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .date-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }}
        .date-header h2 {{
            font-size: 1.1em;
            color: #333;
        }}
        .back-link {{
            color: #0366d6;
            text-decoration: none;
            font-size: 0.82em;
        }}
        .back-link:hover {{
            text-decoration: underline;
        }}
        .developer-name {{
            padding: 8px 20px;
            font-size: 0.85em;
            color: #666;
            font-weight: 600;
            border-bottom: 1px solid #f0f0f0;
        }}
        .review-comments {{
            padding: 12px 20px;
        }}
        .review-comment {{
            margin-bottom: 10px;
            padding: 8px 12px;
            background: #fafbfc;
            border-radius: 4px;
            font-size: 0.88em;
        }}
        .review-comment:last-child {{
            margin-bottom: 0;
        }}
        .review-comment-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
        }}
        .review-author {{
            font-weight: 600;
            color: #333;
        }}
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
            font-size: 0.8em;
            font-weight: 500;
            background: #e3f2fd;
            color: #1565c0;
        }}
        .review-tag-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 500;
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .review-comment-text {{
            color: #555;
            line-height: 1.5;
        }}
        .review-comment-signals {{
            margin-top: 3px;
            font-size: 0.9em;
            color: #999;
            font-style: italic;
        }}
        .no-reviews {{
            padding: 12px 20px;
            color: #aaa;
            font-size: 0.85em;
            font-style: italic;
        }}
        footer {{
            text-align: center;
            color: #aaa;
            font-size: 0.8em;
            margin-top: 32px;
            padding: 16px;
        }}
    </style>
</head>
<body>
    <div class="home-link"><a href="../">&larr; Back to reports</a></div>
    <h1>Review Comments</h1>
    <h1 style="font-size:1.1em;color:#555;font-weight:normal;margin-bottom:8px">{_esc(subject)}</h1>
    {'<div class="lore-link"><a href="' + _esc(url) + '" target="_blank">View on lore.kernel.org &rarr;</a></div>' if url else ''}

    {date_nav}

    {sections_html}

    <footer>LKML Daily Activity Tracker</footer>
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
