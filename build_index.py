"""Build an index.html for the reports directory.

Scans reports/ and logs/ directories, groups by date, and generates a styled
index page with links to reports and their 1:1 associated log files.

Usage:
    python build_index.py                    # Uses reports/ and logs/ in CWD
    python build_index.py --reports-dir /app/reports --logs-dir /app/logs
"""

import argparse
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _extract_date(filename: str) -> str | None:
    """Extract YYYY-MM-DD date from a report or log filename."""
    # Reports: 2025-02-13.html or 2025-02-13_ollama_llama3.1-8b.html
    # Logs: 2025-02-13.log or lkml_2025-02-13.log
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


def _match_log(report_filename: str, log_files: set[str], logs_by_date: dict[str, list[str]]) -> Optional[str]:
    """Find the log file matching a report.

    Primary: match by shared filename stem (e.g. 2026-02-15.html <-> 2026-02-15.log).
    Fallback: match legacy log formats by date (lkml_YYYY-MM-DD.log, range logs).
    """
    # Primary: exact stem match
    stem = report_filename.removesuffix(".html")
    candidate = f"{stem}.log"
    if candidate in log_files:
        return candidate

    # Fallback: legacy date-based match
    date = _extract_date(report_filename)
    if date and date in logs_by_date:
        return logs_by_date[date][0]

    return None


def _build_run_history_html(logs_dir: Path) -> str:
    """Build the run history section HTML from logs/run_history.json."""
    history_path = logs_dir / "run_history.json"
    if not history_path.exists():
        return ""

    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    if not history:
        return ""

    # Sort newest first
    history.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    rows = []
    for entry in history:
        ts = html.escape(entry.get("timestamp", ""))
        report_date = html.escape(entry.get("report_date", ""))
        report_file = entry.get("report_file", "")
        log_file = entry.get("log_file", "")
        status = entry.get("status", "unknown")
        error = entry.get("error", "")
        duration = entry.get("duration_seconds", 0)
        patches = entry.get("patches", 0)
        reviews = entry.get("reviews", 0)
        acks = entry.get("acks", 0)

        if status == "success":
            status_html = '<span class="status-ok">OK</span>'
        else:
            title = html.escape(error) if error else "Failed"
            status_html = f'<span class="status-fail" title="{title}">FAIL</span>'

        report_link = f'<a href="{html.escape(report_file)}">{html.escape(report_date)}</a>' if report_file else report_date
        log_link = f'<a href="/logs/{html.escape(log_file)}" class="log-link">Log</a>' if log_file else "--"

        rows.append(
            f"<tr>"
            f"<td>{ts}</td>"
            f"<td>{report_link}</td>"
            f"<td>{status_html}</td>"
            f"<td>{duration:.0f}s</td>"
            f"<td>{patches}p / {reviews}r / {acks}a</td>"
            f"<td>{log_link}</td>"
            f"</tr>"
        )

    rows_html = "\n            ".join(rows)

    return f"""
    <details class="history-details">
        <summary class="history-summary">
            <span class="history-summary-title">Run History (Last 14 Days)</span>
            <span class="history-summary-hint">click to expand</span>
        </summary>
        <table class="history-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Report Date</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Activity</th>
                    <th>Log</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </details>"""


def build_index(reports_dir: Path, logs_dir: Path) -> str:
    """Generate index.html content."""
    # Collect all log filenames into a set for O(1) lookup
    log_files: set[str] = set()
    logs_by_date: dict[str, list[str]] = defaultdict(list)
    if logs_dir.exists():
        for f in sorted(logs_dir.iterdir()):
            if f.suffix == ".log":
                log_files.add(f.name)
                date = _extract_date(f.name)
                if date:
                    logs_by_date[date].append(f.name)

    # Collect reports (exclude index.html, reviews/ subdir entries)
    reports_by_date: dict[str, list[str]] = defaultdict(list)
    if reports_dir.exists():
        for f in sorted(reports_dir.iterdir()):
            if not f.is_file() or f.suffix != ".html":
                continue
            if f.name == "index.html":
                continue
            date = _extract_date(f.name)
            if date:
                reports_by_date[date].append(f.name)

    # Check if reviews/ subdir exists for the "Review Comments" link
    has_reviews = (reports_dir / "reviews").exists() and any(
        (reports_dir / "reviews").glob("*.html")
    )

    # All dates, newest first
    all_dates = sorted(reports_by_date.keys(), reverse=True)

    # Track which log files are matched to a report
    matched_logs: set[str] = set()

    # Build table rows: one row per report, date cell uses rowspan
    rows: list[str] = []
    for date in all_dates:
        reports = reports_by_date[date]
        num_reports = len(reports)

        for i, report_file in enumerate(reports):
            report_label_text = html.escape(_report_label(report_file))
            report_link = f'<a href="{html.escape(report_file)}" class="report-link">{report_label_text}</a>'

            log_file = _match_log(report_file, log_files, logs_by_date)
            if log_file:
                matched_logs.add(log_file)
                log_link = f'<a href="/logs/{html.escape(log_file)}" class="log-link">Log</a>'
            else:
                log_link = '<span class="none">--</span>'

            if i == 0:
                rowspan = f' rowspan="{num_reports}"' if num_reports > 1 else ""
                rows.append(
                    f'<tr>'
                    f'<td class="date-cell"{rowspan}>{html.escape(date)}</td>'
                    f'<td class="reports-cell">{report_link}</td>'
                    f'<td class="logs-cell">{log_link}</td>'
                    f'</tr>'
                )
            else:
                rows.append(
                    f'<tr>'
                    f'<td class="reports-cell">{report_link}</td>'
                    f'<td class="logs-cell">{log_link}</td>'
                    f'</tr>'
                )

    rows_html = "\n            ".join(rows)
    total_reports = sum(len(v) for v in reports_by_date.values())
    total_logs = len(matched_logs)

    # Build run history section
    run_history_html = _build_run_history_html(logs_dir)

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
            width: 100px;
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
        .reviews-link {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 8px 16px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }}
        .reviews-link a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }}
        .reviews-link a:hover {{
            text-decoration: underline;
        }}
        .history-details {{
            margin-top: 32px;
        }}
        .history-summary {{
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            list-style: none;
            user-select: none;
            margin-bottom: 12px;
        }}
        .history-summary::-webkit-details-marker {{ display: none; }}
        .history-summary::before {{
            content: "▶";
            font-size: 0.75em;
            color: #888;
            transition: transform 0.15s ease;
        }}
        .history-details[open] .history-summary::before {{
            transform: rotate(90deg);
        }}
        .history-summary-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #1a1a1a;
        }}
        .history-summary-hint {{
            font-size: 0.8em;
            color: #aaa;
            font-style: italic;
        }}
        .history-details[open] .history-summary-hint {{
            display: none;
        }}
        .history-table {{
            margin-top: 0;
        }}
        .history-table td {{
            font-size: 0.85em;
        }}
        .status-ok {{
            display: inline-block;
            padding: 1px 8px;
            background: #e8f5e9;
            color: #2e7d32;
            border-radius: 10px;
            font-size: 0.82em;
            font-weight: 600;
        }}
        .status-fail {{
            display: inline-block;
            padding: 1px 8px;
            background: #ffebee;
            color: #c62828;
            border-radius: 10px;
            font-size: 0.82em;
            font-weight: 600;
            cursor: help;
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

    {'<div class="reviews-link"><a href="reviews/">Browse Review Comments &rarr;</a></div>' if has_reviews else ''}

    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Report</th>
                <th>Log</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    {run_history_html}

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
