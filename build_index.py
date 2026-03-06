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
import time
from collections import defaultdict
from datetime import datetime, timezone
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


def _build_date_tooltip(reports_dir: Path, date: str) -> str:
    """Build a hover tooltip showing active contributors for a date.

    Reads reports/daily/{date}.json. Returns empty string if not available.
    """
    daily_json = reports_dir / "daily" / f"{date}.json"
    if not daily_json.exists():
        return ""
    try:
        data = json.loads(daily_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""

    rows = []
    for dr in data.get("developer_reports", []):
        p  = len(dr.get("patches_submitted", []))
        rv = len(dr.get("patches_reviewed", []))
        ak = len(dr.get("patches_acked", []))
        di = len(dr.get("discussions_posted", []))
        if p == 0 and rv == 0 and ak == 0 and di == 0:
            continue
        parts = []
        if p:  parts.append(f"{p} {'Patch' if p == 1 else 'Patches'}")
        if rv: parts.append(f"{rv} {'Review' if rv == 1 else 'Reviews'}")
        if ak: parts.append(f"{ak} {'Ack' if ak == 1 else 'Acks'}")
        if di: parts.append(f"{di} {'Discussion' if di == 1 else 'Discussions'}")
        badge = ", ".join(parts)
        rows.append(
            f'<tr>'
            f'<td class="tt-name">{html.escape(dr["name"])}</td>'
            f'<td class="tt-badge">{html.escape(badge)}</td>'
            f'</tr>'
        )

    if not rows:
        return ""

    return (
        '<div class="date-tooltip">'
        '<table class="tt-table">'
        '<thead><tr>'
        '<th>Developer</th>'
        '<th>Activity</th>'
        '</tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody>'
        '</table>'
        '</div>'
    )


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
        log_link = f'<a href="log_viewer.html?log={html.escape(log_file)}" class="log-link">Log</a>' if log_file else "--"

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


def _infer_report_status(reports_dir: Path, logs_dir: Path, date_str: str) -> dict:
    """Determine whether a report date is in-progress or complete.

    Returns a dict with keys:
      - ``status``:       "complete" or "in_progress"
      - ``last_updated``: human-readable UTC timestamp string, or "" if unknown

    Priority:
    1. ``status`` / ``last_updated`` fields in reports/daily/{date}.json  — written by new runs
    2. Last log file for that date — search for the completion marker logged by
       generate_single_report ("Report generation complete: {date}")
       Falls back to "in_progress" only if the log is less than 3 hours old
       (guards against showing a stale badge for a crashed run)
    3. Default: "complete"
    """
    # 1. Check daily JSON status field
    daily_json = reports_dir / "daily" / f"{date_str}.json"
    if daily_json.exists():
        try:
            data = json.loads(daily_json.read_text(encoding="utf-8"))
            if "status" in data:
                return {
                    "status": data["status"],
                    "last_updated": data.get("last_updated", ""),
                }
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Check last log file for completion marker
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob(f"{date_str}_*.log"))
        if log_files:
            last_log = log_files[-1]
            try:
                text = last_log.read_text(encoding="utf-8", errors="replace")
                if f"Report generation complete: {date_str}" in text:
                    return {"status": "complete", "last_updated": ""}
                # Log exists but no completion marker — flag as in_progress only
                # if the log was modified recently (< 3 hours), to avoid a
                # permanent badge for a crashed run
                age_hours = (time.time() - last_log.stat().st_mtime) / 3600
                if age_hours < 3:
                    mtime_str = datetime.fromtimestamp(
                        last_log.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                    return {"status": "in_progress", "last_updated": mtime_str}
            except OSError:
                pass

    # 3. Default: assume complete
    return {"status": "complete", "last_updated": ""}


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

    # All dates, newest first
    all_dates = sorted(reports_by_date.keys(), reverse=True)

    # Track which log files are matched to a report
    matched_logs: set[str] = set()

    # Build table rows: one row per report, date cell uses rowspan
    rows: list[str] = []
    has_inprogress = False
    for date in all_dates:
        reports = reports_by_date[date]
        num_reports = len(reports)
        date_tooltip = _build_date_tooltip(reports_dir, date)
        report_info = _infer_report_status(reports_dir, logs_dir, date)
        report_status   = report_info["status"]
        report_last_upd = report_info["last_updated"]
        title_attr = (
            f' title="Last updated: {html.escape(report_last_upd)}"'
            if report_last_upd else ""
        )
        if report_status == "in_progress":
            has_inprogress = True
        inprogress_badge = (
            f' <span class="inprogress-badge"{title_attr}>&#x27F3; In Progress</span>'
            if report_status == "in_progress" else ""
        )

        for i, report_file in enumerate(reports):
            report_label_text = html.escape(_report_label(report_file))
            report_link = f'<a href="{html.escape(report_file)}" class="report-link">{report_label_text}</a>'

            log_file = _match_log(report_file, log_files, logs_by_date)
            if log_file:
                matched_logs.add(log_file)
                log_link = f'<a href="log_viewer.html?log={html.escape(log_file)}" class="log-link">Log</a>'
            else:
                log_link = '<span class="none">--</span>'

            if i == 0:
                rowspan = f' rowspan="{num_reports}"' if num_reports > 1 else ""
                rows.append(
                    f'<tr>'
                    f'<td class="date-cell"{rowspan}>{html.escape(date)}{inprogress_badge}{date_tooltip}</td>'
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

    refresh_meta = '    <meta http-equiv="refresh" content="120">\n' if has_inprogress else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}    <title>LKML Activity Reports</title>
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
            position: relative;
        }}
        .date-cell:hover .date-tooltip {{
            display: block;
        }}
        .date-tooltip {{
            display: none;
            position: absolute;
            top: 50%;
            left: calc(100% + 10px);
            transform: translateY(-50%);
            background: #1a1a1a;
            color: #f0f0f0;
            border-radius: 8px;
            padding: 10px 14px;
            min-width: 220px;
            z-index: 100;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            font-size: 0.82em;
            line-height: 1.5;
            font-weight: normal;
            white-space: normal;
        }}
        .date-tooltip::before {{
            content: "";
            position: absolute;
            top: 50%;
            right: 100%;
            transform: translateY(-50%);
            border: 6px solid transparent;
            border-right-color: #1a1a1a;
        }}
        .tt-table {{
            width: 100%;
            border-collapse: collapse;
            background: transparent;
            box-shadow: none;
            border-radius: 0;
        }}
        .tt-table thead th {{
            background: transparent;
            color: #aaa;
            font-size: 0.78em;
            padding: 0 6px 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.15);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .tt-table tbody tr {{
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .tt-table tbody tr:last-child {{
            border-bottom: none;
        }}
        .tt-table tbody tr:hover {{
            background: rgba(255,255,255,0.06);
        }}
        .tt-name {{
            padding: 3px 8px 3px 0;
            color: #f0f0f0;
        }}
        .tt-badge {{
            padding: 3px 0;
            color: #7ec8e3;
            font-weight: 700;
            text-align: right;
            white-space: nowrap;
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
        .inprogress-badge {{
            display: inline-block;
            padding: 1px 7px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 6px;
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
            vertical-align: middle;
            animation: pulse 2s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50%       {{ opacity: 0.45; }}
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


LOG_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log Viewer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 13px;
      display: flex;
      flex-direction: column;
    }
    /* ── Toolbar ── */
    #toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      position: sticky;
      top: 0;
      z-index: 10;
      flex-shrink: 0;
      flex-wrap: wrap;
    }
    #back-link {
      color: #58a6ff;
      text-decoration: none;
      font-size: 0.85em;
      white-space: nowrap;
    }
    #back-link:hover { text-decoration: underline; }
    #logname {
      flex: 1;
      color: #8b949e;
      font-size: 0.82em;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    #linecount { color: #6e7681; font-size: 0.78em; white-space: nowrap; }
    #status {
      font-size: 0.78em;
      font-weight: 600;
      padding: 2px 9px;
      border-radius: 10px;
      white-space: nowrap;
    }
    #status.following { background: #122d12; color: #56d364; border: 1px solid #238636; }
    #status.paused    { background: #3d1a00; color: #f0883e; border: 1px solid #9e3d00; }
    .sep { color: #30363d; }
    /* ── Buttons ── */
    .btn {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 3px 10px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 0.8em;
      font-family: inherit;
      white-space: nowrap;
    }
    .btn:hover { background: #30363d; }
    .btn.active { background: #1f6feb; border-color: #388bfd; color: #fff; }
    /* ── Log lines area ── */
    #log-output {
      flex: 1;
      overflow-y: auto;
      padding: 6px 14px 40px;
    }
    .line {
      display: block;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-all;
      padding: 1px 0;
    }
    .line:hover { background: rgba(255,255,255,0.04); }
    /* timestamp + level coloring */
    .ts  { color: #484f58; }
    .lvl-debug   { color: #484f58; }
    .lvl-info    { color: #58a6ff; }
    .lvl-warning { color: #e3b341; }
    .lvl-error   { color: #f85149; font-weight: 600; }
    /* full-line dim for DEBUG lines */
    .line-debug  { color: #484f58; }
    .line-info   { color: #c9d1d9; }
    .line-warning { color: #e3b341; }
    .line-error  { color: #f85149; }
    /* empty / loading states */
    #empty-msg {
      margin-top: 60px;
      text-align: center;
      color: #484f58;
      font-size: 0.9em;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <a id="back-link" href="index.html">&#8592; Index</a>
    <span class="sep">|</span>
    <span id="logname">Loading…</span>
    <span id="linecount"></span>
    <span id="status" class="following">&#x2B07; Following</span>
    <span class="sep">|</span>
    <button class="btn" id="btn-tail" onclick="toggleTail()">Pause</button>
    <span class="sep">|</span>
    <span style="color:#6e7681;font-size:0.78em">Refresh:</span>
    <button class="btn active" id="iv-2" onclick="setIv(2000,this)">2s</button>
    <button class="btn" id="iv-5"  onclick="setIv(5000,this)">5s</button>
    <button class="btn" id="iv-10" onclick="setIv(10000,this)">10s</button>
    <button class="btn" id="iv-off" onclick="setIv(0,this)">Off</button>
  </div>
  <div id="log-output">
    <div id="empty-msg">Loading log…</div>
  </div>

  <script>
    // ── Query param ──────────────────────────────────────────────
    const params  = new URLSearchParams(location.search);
    const logName = params.get("log") || "";
    const logUrl  = logName ? "logs/" + logName : "";

    document.title = logName ? "Log: " + logName : "Log Viewer";
    document.getElementById("logname").textContent = logName || "(no log specified)";

    // ── State ────────────────────────────────────────────────────
    let knownLines   = 0;   // number of lines already rendered
    let tailing      = true;
    let intervalMs   = 2000;
    let timerId      = null;
    let firstLoad    = true;

    const output = document.getElementById("log-output");
    const empty  = document.getElementById("empty-msg");

    // ── Line parser / renderer ───────────────────────────────────
    const LINE_RE = /^(\d{2}:\d{2}:\d{2}\s+\S+)\s+\[(DEBUG|INFO|WARNING|ERROR)\]\s*([\s\S]*)$/i;

    function renderLine(raw) {
      const span = document.createElement("span");
      span.className = "line";
      const m = LINE_RE.exec(raw);
      if (m) {
        const [, ts, lvl, msg] = m;
        const lvlLow = lvl.toLowerCase();
        span.classList.add("line-" + lvlLow);
        span.innerHTML =
          '<span class="ts">' + escHtml(ts) + "</span> " +
          '<span class="lvl-' + lvlLow + '">[' + escHtml(lvl) + "]</span> " +
          escHtml(msg);
      } else {
        span.textContent = raw;
      }
      return span;
    }

    function escHtml(s) {
      return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    // ── Fetch + update ───────────────────────────────────────────
    async function fetchLog() {
      if (!logUrl) return;
      try {
        const resp = await fetch(logUrl, { cache: "no-store" });
        if (!resp.ok) {
          showError("HTTP " + resp.status + " — " + resp.statusText);
          return;
        }
        const text = await resp.text();
        const lines = text.split("\\n");
        // strip trailing empty line from final newline
        if (lines.length && lines[lines.length - 1] === "") lines.pop();

        if (lines.length > knownLines) {
          if (firstLoad && empty) empty.remove();
          firstLoad = false;

          const frag = document.createDocumentFragment();
          for (let i = knownLines; i < lines.length; i++) {
            frag.appendChild(renderLine(lines[i]));
            frag.appendChild(document.createTextNode("\\n"));
          }
          output.appendChild(frag);
          knownLines = lines.length;

          document.getElementById("linecount").textContent =
            knownLines.toLocaleString() + " lines";

          if (tailing) scrollToBottom();
        }
      } catch (err) {
        showError(err.message);
      }
    }

    function showError(msg) {
      if (empty) {
        empty.textContent = "Error loading log: " + msg;
        empty.style.color = "#f85149";
      }
    }

    // ── Scroll detection (auto-pause / auto-resume) ──────────────
    function isAtBottom() {
      return output.scrollHeight - output.scrollTop - output.clientHeight < 30;
    }

    output.addEventListener("scroll", () => {
      if (isAtBottom()) {
        if (!tailing) resumeTail();
      } else {
        if (tailing) pauseTail();
      }
    }, { passive: true });

    function scrollToBottom() {
      output.scrollTop = output.scrollHeight;
    }

    // ── Tail controls ─────────────────────────────────────────────
    function pauseTail() {
      tailing = false;
      const st = document.getElementById("status");
      st.textContent = "\\u23F8 Paused";
      st.className = "paused";
      document.getElementById("btn-tail").textContent = "Resume";
    }

    function resumeTail() {
      tailing = true;
      const st = document.getElementById("status");
      st.textContent = "\\u2B07 Following";
      st.className = "following";
      document.getElementById("btn-tail").textContent = "Pause";
      scrollToBottom();
    }

    function toggleTail() {
      if (tailing) pauseTail(); else resumeTail();
    }

    // ── Interval selector ─────────────────────────────────────────
    function setIv(ms, btn) {
      intervalMs = ms;
      document.querySelectorAll(".btn[id^='iv-']").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      clearInterval(timerId);
      if (ms > 0) timerId = setInterval(fetchLog, ms);
    }

    // ── Boot ──────────────────────────────────────────────────────
    fetchLog();
    timerId = setInterval(fetchLog, intervalMs);
  </script>
</body>
</html>
"""


def write_log_viewer(reports_dir: Path) -> None:
    """Write (or overwrite) log_viewer.html into reports_dir."""
    out = reports_dir / "log_viewer.html"
    out.write_text(LOG_VIEWER_HTML, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build index.html for LKML reports.")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory. Default: reports/")
    parser.add_argument("--logs-dir", default=None, help="Logs directory. Default: <reports-dir>/logs/")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    logs_dir = Path(args.logs_dir) if args.logs_dir else reports_dir / "logs"

    write_log_viewer(reports_dir)
    index_html = build_index(reports_dir, logs_dir)
    output = reports_dir / "index.html"
    output.write_text(index_html, encoding="utf-8")
    print(f"Index generated: {output} ({len(index_html)} bytes)")


if __name__ == "__main__":
    main()
