"""Build an index.html for the reports directory.

Scans reports/ and logs/ directories, groups by date, and generates a styled
index page with links to reports and their associated log files.

Usage:
    python build_index.py                    # Uses reports/ and logs/ in CWD
    python build_index.py --reports-dir /app/reports --logs-dir /app/logs
"""

import argparse
import html
import os
import re
from collections import defaultdict
from pathlib import Path


def _extract_date(filename: str) -> str | None:
    """Extract YYYY-MM-DD date from a report or log filename."""
    # Reports: 2025-02-13.html or 2025-02-13_ollama_llama3.1-8b.html
    # Logs: lkml_2025-02-13_143000.log
    m = re.match(r"(?:lkml_)?(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


def _report_label(filename: str) -> str:
    """Build a human-readable label for a report file."""
    # Strip .html extension
    name = filename.removesuffix(".html")
    # Extract date prefix
    m = re.match(r"(\d{4}-\d{2}-\d{2})[_-](.*)", name)
    if m:
        backends = m.group(2)
        # Convert "ollama_llama3.1-8b_anthropic_claude-haiku-4-5" to readable label
        # Split on known backend prefixes
        parts = []
        for segment in re.split(r"(ollama_|anthropic_)", backends):
            if segment in ("ollama_", "anthropic_"):
                parts.append(segment.rstrip("_") + "/")
            elif segment:
                # Strip trailing underscores from model names
                cleaned = segment.rstrip("_")
                if parts:
                    parts[-1] = parts[-1] + cleaned
                else:
                    parts.append(cleaned)
        return " + ".join(parts) if parts else backends
    return "Heuristic"


def _log_label(filename: str) -> str:
    """Build a human-readable label for a log file."""
    # New format: lkml_2025-02-13.log -> "Log"
    # Legacy format: lkml_2025-02-13_143000.log -> "14:30:00"
    m = re.match(r"lkml_(\d{4}-\d{2}-\d{2})\.log$", filename)
    if m:
        return "Log"
    m = re.match(r"lkml_\d{4}-\d{2}-\d{2}_(\d{2})(\d{2})(\d{2})\.log", filename)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return filename


def build_index(reports_dir: Path, logs_dir: Path) -> str:
    """Generate index.html content."""
    # Collect reports
    reports_by_date: dict[str, list[str]] = defaultdict(list)
    if reports_dir.exists():
        for f in sorted(reports_dir.iterdir()):
            if f.suffix == ".html" and f.name != "index.html":
                date = _extract_date(f.name)
                if date:
                    reports_by_date[date].append(f.name)

    # Collect logs
    logs_by_date: dict[str, list[str]] = defaultdict(list)
    if logs_dir.exists():
        for f in sorted(logs_dir.iterdir()):
            if f.suffix == ".log":
                date = _extract_date(f.name)
                if date:
                    logs_by_date[date].append(f.name)

    # All dates, newest first
    all_dates = sorted(set(reports_by_date.keys()) | set(logs_by_date.keys()), reverse=True)

    # Build date rows
    rows: list[str] = []
    for date in all_dates:
        reports = reports_by_date.get(date, [])
        logs = logs_by_date.get(date, [])

        report_links = []
        for r in reports:
            label = html.escape(_report_label(r))
            report_links.append(f'<a href="{html.escape(r)}" class="report-link">{label}</a>')

        log_links = []
        for lg in logs:
            label = html.escape(_log_label(lg))
            log_links.append(f'<a href="/logs/{html.escape(lg)}" class="log-link">{label}</a>')

        reports_html = "\n".join(report_links) if report_links else '<span class="none">No reports</span>'
        logs_html = "\n".join(log_links) if log_links else '<span class="none">No logs</span>'

        rows.append(f"""
        <tr>
            <td class="date-cell">{html.escape(date)}</td>
            <td class="reports-cell">{reports_html}</td>
            <td class="logs-cell">{logs_html}</td>
        </tr>""")

    rows_html = "\n".join(rows)
    total_reports = sum(len(v) for v in reports_by_date.values())
    total_logs = sum(len(v) for v in logs_by_date.values())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LKML Activity Reports</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 1.8em;
            margin-bottom: 4px;
            color: #1a1a1a;
        }}
        .subtitle {{
            font-size: 1.0em;
            color: #888;
            margin-bottom: 24px;
        }}
        .stats {{
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .stat {{
            background: #fff;
            border-radius: 8px;
            padding: 16px 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 1.8em;
            font-weight: 700;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #888;
        }}
        table {{
            width: 100%;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-collapse: collapse;
            overflow: hidden;
        }}
        thead th {{
            background: #f8f9fa;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            color: #555;
            border-bottom: 2px solid #eee;
        }}
        tbody tr {{
            border-bottom: 1px solid #f0f0f0;
        }}
        tbody tr:last-child {{
            border-bottom: none;
        }}
        tbody tr:hover {{
            background: #fafbfc;
        }}
        td {{
            padding: 12px 16px;
            vertical-align: top;
        }}
        .date-cell {{
            font-weight: 600;
            font-size: 1.0em;
            white-space: nowrap;
            width: 130px;
        }}
        .reports-cell {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .logs-cell {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            width: 160px;
        }}
        .report-link {{
            display: inline-block;
            padding: 2px 10px;
            background: #e3f2fd;
            color: #1565c0;
            border-radius: 12px;
            text-decoration: none;
            font-size: 0.82em;
            font-weight: 500;
        }}
        .report-link:hover {{
            background: #bbdefb;
        }}
        .log-link {{
            display: inline-block;
            padding: 2px 10px;
            background: #f3e5f5;
            color: #7b1fa2;
            border-radius: 12px;
            text-decoration: none;
            font-size: 0.82em;
            font-weight: 500;
        }}
        .log-link:hover {{
            background: #e1bee7;
        }}
        .none {{
            color: #ccc;
            font-size: 0.82em;
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
    <h1>LKML Activity Reports</h1>
    <p class="subtitle">Linux Kernel Mailing List &mdash; Daily Developer Activity Tracker</p>

    <div class="stats">
        <div class="stat">
            <div class="stat-number">{len(all_dates)}</div>
            <div class="stat-label">Dates</div>
        </div>
        <div class="stat">
            <div class="stat-number">{total_reports}</div>
            <div class="stat-label">Reports</div>
        </div>
        <div class="stat">
            <div class="stat-number">{total_logs}</div>
            <div class="stat-label">Log Files</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Reports</th>
                <th>Logs</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <footer>LKML Daily Activity Tracker</footer>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Build index.html for LKML reports.")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory. Default: reports/")
    parser.add_argument("--logs-dir", default="logs", help="Logs directory. Default: logs/")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    logs_dir = Path(args.logs_dir)

    index_html = build_index(reports_dir, logs_dir)
    output = reports_dir / "index.html"
    output.write_text(index_html, encoding="utf-8")
    print(f"Index generated: {output} ({len(index_html)} bytes)")


if __name__ == "__main__":
    main()
