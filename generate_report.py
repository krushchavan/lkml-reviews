"""LKML Daily Activity Tracker - generates HTML reports of kernel developer activity.

Usage:
    python generate_report.py                              # Yesterday's report
    python generate_report.py --date 2025-02-12            # Specific date
    python generate_report.py --days 7                     # Last 7 days
    python generate_report.py --date 2025-02-15 --days 7   # Feb 9-15 (7 days ending Feb 15)
    python generate_report.py --people custom.csv -v       # Custom CSV, verbose
    python generate_report.py --skip-threads               # Fast mode, no summaries

LLM-enriched summaries (single backend):
    python generate_report.py --llm                        # LLM via Ollama (default)
    python generate_report.py --llm --llm-backend anthropic  # LLM via Claude API
    python generate_report.py --llm --llm-model llama3.2   # Custom model
    python generate_report.py --llm --llm-no-cache         # Skip result cache

Multi-LLM comparison (both Ollama and Anthropic):
    python generate_report.py --llm-all                    # Both backends, default models
    python generate_report.py --llm-all --ollama-model llama3.2 --anthropic-model claude-sonnet-4-5

Publishing to GitHub Pages:
    python generate_report.py --publish-github             # Push reports/ to GitHub (uses GITHUB_REPO env var)
    python generate_report.py --publish-github --github-repo owner/repo
    python generate_report.py --publish-github --github-branch gh-pages
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from build_index import build_index
from build_reviews import build_all_reviews


def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """Load key=value pairs from a .env file into os.environ.

    Only sets variables that are NOT already present in the environment so
    that real environment variables and Docker --env flags always take
    precedence over the .env file.  Lines starting with '#' and blank lines
    are ignored.  Values may optionally be quoted with single or double quotes.
    """
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass  # Non-fatal — .env is optional


_load_dotenv()

from activity_classifier import (
    check_thread_activity_on_date,
    classify_messages,
    extract_patch_submissions,
    _deduplicate_patches,
)
from lkml_client import LKMLAPIError, LKMLClient
from llm_cache import LLMCache
from llm_summarizer import (
    AnthropicBackend,
    LLMBackend,
    OllamaBackend,
    analyze_thread_llm,
)
from models import (
    ActivityItem, ActivityType, ConversationSummary, DailyReport,
    Developer, DeveloperReport, LLMAnalysis, ReviewComment, Sentiment,
)
from report_generator import extract_reviews_data, generate_html_report, message_id_to_slug
from thread_analyzer import analyze_thread

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate daily LKML activity report for tracked kernel developers."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to report on (YYYY-MM-DD format). Defaults to yesterday.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help=(
            "Number of days to generate reports for (default: 1). "
            "Reports are generated for each day ending on --date (or yesterday). "
            "E.g., --date 2026-02-15 --days 7 generates reports from Feb 9–15."
        ),
    )
    parser.add_argument(
        "--people",
        type=str,
        default="kernel_developers_emails_1.csv",
        help="Path to CSV file with developer names and emails.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for HTML output files. Default: reports/",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory for per-report log files. Default: logs/",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests. Default: 1.0",
    )
    parser.add_argument(
        "--skip-threads",
        action="store_true",
        help="Skip thread fetching (faster, but no conversation summaries).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    # LLM options
    llm_group = parser.add_argument_group("LLM summarization")
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-enriched summaries (single backend, default: ollama).",
    )
    llm_group.add_argument(
        "--llm-all",
        action="store_true",
        help="Run BOTH Ollama and Anthropic backends and show side-by-side results.",
    )
    llm_group.add_argument(
        "--llm-backend",
        type=str,
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend for --llm mode. Default: ollama.",
    )
    llm_group.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=(
            "Override the default model (for --llm single-backend mode). "
            "Ollama default: llama3.1:8b, Anthropic default: claude-haiku-4-5."
        ),
    )
    llm_group.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use with --llm-all. Default: llama3.1:8b.",
    )
    llm_group.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-haiku-4-5",
        help="Anthropic model to use with --llm-all. Default: claude-haiku-4-5.",
    )
    llm_group.add_argument(
        "--llm-no-cache",
        action="store_true",
        help="Disable disk caching of LLM results.",
    )
    llm_group.add_argument(
        "--llm-dump",
        type=str,
        default=None,
        metavar="DIR",
        help="Dump all raw LLM responses to DIR (errors always dumped to logs/llm_dumps/).",
    )
    llm_group.add_argument(
        "--llm-monolithic",
        action="store_true",
        help="Force monolithic LLM prompts (disable per-reviewer decomposition for Ollama).",
    )

    # Purge options
    purge_group = parser.add_argument_group("Purge / retention")
    purge_group.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Delete reports, logs, and review data older than this many days. Default: 30.",
    )
    purge_group.add_argument(
        "--no-purge",
        action="store_true",
        help="Skip automatic purge after report generation.",
    )
    purge_group.add_argument(
        "--purge-only",
        action="store_true",
        help="Only run the purge (no report generation).",
    )

    # GitHub publishing options
    github_group = parser.add_argument_group("GitHub publishing")
    github_group.add_argument(
        "--publish-github",
        action="store_true",
        help=(
            "Push the reports directory to a GitHub repository after generation. "
            "Requires git to be installed and the repo to be configured via "
            "--github-repo or the GITHUB_REPO environment variable."
        ),
    )
    github_group.add_argument(
        "--github-repo",
        type=str,
        default=None,
        help=(
            "GitHub repository to publish to, in 'owner/repo' format "
            "(e.g. 'myorg/lkml-reports'). Falls back to GITHUB_REPO env var. "
            "The reports directory will be pushed as the root of this repo."
        ),
    )
    github_group.add_argument(
        "--github-branch",
        type=str,
        default="main",
        help="Branch to push reports to. Default: main.",
    )
    github_group.add_argument(
        "--github-token",
        type=str,
        default=None,
        help=(
            "GitHub personal access token (or fine-grained token) used to "
            "authenticate the push. Falls back to the GITHUB_TOKEN environment "
            "variable. The token is spliced into the remote URL so no separate "
            "credential helper is needed — safe for Docker/CI use."
        ),
    )
    github_group.add_argument(
        "--publish-only",
        action="store_true",
        help=(
            "Skip report generation and push the existing reports directory to "
            "GitHub immediately. Useful for testing the GitHub flow or for "
            "re-publishing after a manual edit. Implies --publish-github."
        ),
    )

    parser.add_argument(
        "--rebuild-html",
        action="store_true",
        help=(
            "Re-render HTML reports from the existing persisted review JSON files "
            "without any web fetches or LLM calls. Useful after template or CSS "
            "changes. Uses --date / --days to select which dates to rebuild."
        ),
    )

    parser.add_argument(
        "--message-id",
        type=str,
        default=None,
        metavar="MSG_ID",
        help=(
            "With --rebuild-html: render only the patchset whose message-id contains "
            "this string (case-insensitive partial match, e.g. '20260126065242'). "
            "Writes a standalone preview file (reports/preview_<slug>.html) without "
            "touching the full report or index."
        ),
    )

    return parser.parse_args()


def load_developers(csv_path: str) -> list[Developer]:
    """Load developer list from CSV file."""
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    developers = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            primary = row.get("Primary Email", "").strip()
            secondary = row.get("Secondary Email", "").strip() or None

            if not name or not primary:
                logger.warning("Skipping invalid row: %s", row)
                continue

            developers.append(Developer(
                name=name,
                primary_email=primary,
                secondary_email=secondary,
            ))

    logger.info("Loaded %d developers from %s", len(developers), csv_path)
    return developers


def _parse_submission_date(updated_str: str) -> str:
    """Extract YYYY-MM-DD from an Atom feed updated timestamp like 2025-02-11T16:38:25Z."""
    if not updated_str:
        return ""
    try:
        return updated_str[:10]  # "2025-02-11T..." -> "2025-02-11"
    except Exception:
        return ""


def process_developer(
    client: LKMLClient,
    developer: Developer,
    date_str: str,
    skip_threads: bool = False,
    lookback_days: int = 14,
    llm_backends: Optional[dict[str, LLMBackend]] = None,
    llm_cache: Optional[LLMCache] = None,
    llm_dump_dir: Optional[Path] = None,
    force_monolithic: bool = False,
) -> DeveloperReport:
    """Fetch and classify all activity for one developer on one date.

    Also finds patches submitted in the last `lookback_days` days that have
    thread activity on the report date.
    """
    report = DeveloperReport(developer=developer)
    all_entries = []
    seen_msg_ids = set()

    # Fetch messages for all emails on the report date
    for email_addr in developer.all_emails():
        try:
            entries = client.get_user_messages_for_date(email_addr, date_str)
            for entry in entries:
                msg_id = entry.get("message_id", "")
                if msg_id and msg_id not in seen_msg_ids:
                    seen_msg_ids.add(msg_id)
                    all_entries.append(entry)
            logger.info(
                "  %s (%s): %d messages",
                developer.name,
                email_addr,
                len(entries),
            )
        except LKMLAPIError as e:
            err_msg = f"Failed to fetch messages for {email_addr}: {e}"
            logger.warning("  %s", err_msg)
            report.errors.append(err_msg)

    # Classify messages for the report date
    def raw_fetcher(msg_id: str) -> str:
        return client.get_raw_message(msg_id)

    if all_entries:
        try:
            patches, reviews, acks, discussions = classify_messages(
                all_entries, developer, raw_fetcher
            )
            report.patches_submitted = patches
            report.patches_reviewed = reviews
            report.patches_acked = acks
            report.discussions_posted = discussions
        except Exception as e:
            err_msg = f"Classification failed for {developer.name}: {e}"
            logger.warning("  %s", err_msg)
            report.errors.append(err_msg)
    else:
        patches = []
        reviews = []
        acks = []

    logger.info(
        "  %s: %d patches, %d reviews, %d acks (%s)",
        developer.name,
        len(patches),
        len(reviews),
        len(acks),
        date_str,
    )

    # --- 14-day lookback: find recent patches with activity on the report date ---
    today_patch_ids = {p.message_id for p in report.patches_submitted}
    # Also collect timestamp-prefix keys from the report date's patches to avoid dupes
    today_prefix_keys = set()
    for p in report.patches_submitted:
        pfx = re.match(r"^(\d+\.\d+)", p.message_id)
        if pfx:
            today_prefix_keys.add(pfx.group(1))

    # Compute lookback date range
    target_date_obj = datetime.strptime(date_str, "%Y%m%d")
    lookback_start = (target_date_obj - timedelta(days=lookback_days)).strftime("%Y%m%d")
    # End date is the day before the report date (report date's patches already captured)
    lookback_end_obj = target_date_obj - timedelta(days=1)
    if lookback_end_obj < datetime.strptime(lookback_start, "%Y%m%d"):
        lookback_end = lookback_start
    else:
        lookback_end = lookback_end_obj.strftime("%Y%m%d")

    recent_patch_entries = []
    recent_seen_ids = set(seen_msg_ids)  # avoid dupes with report date's messages

    for email_addr in developer.all_emails():
        try:
            range_entries = client.get_user_messages_for_range(
                email_addr, lookback_start, lookback_end
            )
            # Filter to patch submissions only
            patch_entries = extract_patch_submissions(range_entries)
            for entry in patch_entries:
                msg_id = entry.get("message_id", "")
                if msg_id and msg_id not in recent_seen_ids:
                    # Also skip if the series prefix matches a report date patch
                    pfx = re.match(r"^(\d+\.\d+)", msg_id)
                    if pfx and pfx.group(1) in today_prefix_keys:
                        continue
                    recent_seen_ids.add(msg_id)
                    recent_patch_entries.append(entry)
            logger.debug(
                "  %s (%s): %d patch submissions in last %d days",
                developer.name,
                email_addr,
                len(patch_entries),
                lookback_days,
            )
        except LKMLAPIError as e:
            logger.debug("  Lookback fetch failed for %s: %s", email_addr, e)

    # Deduplicate recent patch series (same logic as the report date's patches)
    if recent_patch_entries:
        # Convert to ActivityItems for deduplication, then check threads
        recent_items = []
        for entry in recent_patch_entries:
            title = entry.get("title", "")
            msg_id = entry.get("message_id", "")
            url = entry.get("url", "")
            updated = entry.get("updated", "")
            if url and not url.startswith("http"):
                url = f"https://lore.kernel.org{url}"
            if not url:
                url = f"https://lore.kernel.org/all/{msg_id}/"
            recent_items.append(ActivityItem(
                activity_type=ActivityType.PATCH_SUBMITTED,
                subject=title,
                message_id=msg_id,
                url=url,
                date=updated,
                is_ongoing=True,
                submitted_date=_parse_submission_date(updated),
            ))

        recent_items = _deduplicate_patches(recent_items)
        date_display = target_date_obj.strftime("%Y-%m-%d")
        logger.info(
            "  %s: %d recent patch series to check for activity on %s",
            developer.name,
            len(recent_items),
            date_display,
        )

        # For each recent patch, fetch thread and check for activity on report date
        # We need thread data anyway for conversation summaries, so cache it
        thread_cache: dict[str, list[dict]] = {}
        ongoing_patches = []

        for item in recent_items:
            msg_id = item.message_id
            try:
                result = client.get_thread(msg_id)
                thread_messages = result.get("messages", [])
                thread_cache[msg_id] = thread_messages
            except LKMLAPIError as e:
                logger.debug("  Thread fetch failed for %s: %s", msg_id, e)
                thread_messages = []
                thread_cache[msg_id] = thread_messages

            if check_thread_activity_on_date(thread_messages, date_str):
                ongoing_patches.append(item)
                logger.debug("  ONGOING: %s", item.subject)

        if ongoing_patches:
            logger.info(
                "  %s: %d ongoing patches with activity on %s",
                developer.name,
                len(ongoing_patches),
                date_display,
            )
            # Append ongoing patches after the report date's patches
            report.patches_submitted.extend(ongoing_patches)
    else:
        thread_cache = {}

    # Fetch threads for conversation summaries on all items
    if not skip_threads:
        all_items = report.patches_submitted + reviews + acks + report.discussions_posted
        total_items = len(all_items)

        for item_idx, item in enumerate(all_items, 1):
            msg_id = item.message_id
            if item.conversation is not None:
                continue  # Already analyzed (cache hit)
            short_subject = item.subject[:70] + ("\u2026" if len(item.subject) > 70 else "")
            logger.info("  [%d/%d] %s", item_idx, total_items, short_subject)
            if msg_id in thread_cache:
                thread_messages = thread_cache[msg_id]
            else:
                try:
                    result = client.get_thread(msg_id)
                    thread_messages = result.get("messages", [])
                    thread_cache[msg_id] = thread_messages
                except LKMLAPIError as e:
                    logger.debug("  Thread fetch failed for %s: %s", msg_id, e)
                    thread_messages = []
                    thread_cache[msg_id] = thread_messages

            if llm_backends:
                # Run each backend and collect attributed analyses
                for backend_name, backend in llm_backends.items():
                    summary = analyze_thread_llm(
                        thread_messages, item, backend, llm_cache,
                        dump_dir=llm_dump_dir,
                        force_monolithic=force_monolithic,
                    )
                    item.llm_analyses.append(LLMAnalysis(
                        backend=backend_name,
                        model=backend.model,
                        conversation=summary,
                    ))
                # Use the first backend's result as the primary conversation
                # (backward compat for single-backend rendering)
                item.conversation = item.llm_analyses[0].conversation
            else:
                item.conversation = analyze_thread(thread_messages, item)

    return report


def _serialize_daily_report(
    report: "DailyReport", report_filename: str, status: str = "complete"
) -> dict:
    """Serialize a DailyReport to a plain dict for JSON persistence.

    Captures enough information to fully reconstruct the DailyReport for
    --rebuild-html without any web fetches or LLM calls.

    ``status`` is "in_progress" during incremental writes and "complete" once
    the full report has been generated.
    """
    def _serialize_item(item: "ActivityItem") -> dict:
        conv = item.conversation
        rc_list = []
        if conv:
            for rc in conv.review_comments:
                rc_list.append({
                    "author": rc.author,
                    "summary": rc.summary,
                    "sentiment": rc.sentiment.value.upper(),
                    "sentiment_signals": rc.sentiment_signals,
                    "has_inline_review": rc.has_inline_review,
                    "tags_given": rc.tags_given,
                    "raw_body": rc.raw_body,
                    "reply_to": rc.reply_to,
                    "message_date": rc.message_date,
                    "message_id": rc.message_id,
                    "analysis_source": rc.analysis_source,
                })
        return {
            "activity_type": item.activity_type.value,
            "subject": item.subject,
            "message_id": item.message_id,
            "url": item.url,
            "date": item.date,
            "in_reply_to": item.in_reply_to,
            "ack_type": item.ack_type,
            "is_ongoing": item.is_ongoing,
            "submitted_date": item.submitted_date,
            "patch_summary": conv.patch_summary if conv else "",
            "analysis_source": conv.analysis_source if conv else "heuristic",
            "review_comments": rc_list,
        }

    dev_reports = []
    for dr in report.developer_reports:
        dev_reports.append({
            "name": dr.developer.name,
            "primary_email": dr.developer.primary_email,
            "patches_submitted": [_serialize_item(i) for i in dr.patches_submitted],
            "patches_reviewed": [_serialize_item(i) for i in dr.patches_reviewed],
            "patches_acked": [_serialize_item(i) for i in dr.patches_acked],
            "discussions_posted": [_serialize_item(i) for i in dr.discussions_posted],
            "errors": dr.errors,
        })

    return {
        "date": report.date,
        "report_file": report_filename,
        "status": status,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "llm_backends": report.llm_backends,
        "generation_time_seconds": report.generation_time_seconds,
        "developer_reports": dev_reports,
    }


def _load_daily_report(daily_json_path: Path) -> "DailyReport":
    """Reconstruct a DailyReport from a persisted daily summary JSON file."""
    data = json.loads(daily_json_path.read_text(encoding="utf-8"))

    def _load_item(d: dict) -> "ActivityItem":
        try:
            act_type = ActivityType(d["activity_type"])
        except (KeyError, ValueError):
            act_type = ActivityType.PATCH_SUBMITTED

        rc_list = []
        for rc in d.get("review_comments", []):
            try:
                sentiment = Sentiment[rc.get("sentiment", "NEUTRAL").upper()]
            except KeyError:
                sentiment = Sentiment.NEUTRAL
            rc_list.append(ReviewComment(
                author=rc.get("author", ""),
                summary=rc.get("summary", ""),
                sentiment=sentiment,
                sentiment_signals=rc.get("sentiment_signals", []),
                has_inline_review=rc.get("has_inline_review", False),
                tags_given=rc.get("tags_given", []),
                raw_body=rc.get("raw_body", ""),
                reply_to=rc.get("reply_to", ""),
                message_date=rc.get("message_date", ""),
                message_id=rc.get("message_id", ""),
                analysis_source=rc.get("analysis_source", "heuristic"),
            ))

        conv = ConversationSummary(
            review_comments=rc_list,
            patch_summary=d.get("patch_summary", ""),
            analysis_source=d.get("analysis_source", "heuristic"),
        )

        return ActivityItem(
            activity_type=act_type,
            subject=d.get("subject", ""),
            message_id=d.get("message_id", ""),
            url=d.get("url", ""),
            date=d.get("date", ""),
            in_reply_to=d.get("in_reply_to"),
            ack_type=d.get("ack_type"),
            is_ongoing=d.get("is_ongoing", False),
            submitted_date=d.get("submitted_date"),
            conversation=conv,
        )

    report = DailyReport(
        date=data["date"],
        llm_backends=[tuple(b) for b in data.get("llm_backends", [])],
        generation_time_seconds=data.get("generation_time_seconds", 0.0),
    )

    for dr_data in data.get("developer_reports", []):
        dev = Developer(name=dr_data["name"], primary_email=dr_data.get("primary_email", ""))
        dr = DeveloperReport(developer=dev, errors=dr_data.get("errors", []))
        dr.patches_submitted = [_load_item(i) for i in dr_data.get("patches_submitted", [])]
        dr.patches_reviewed = [_load_item(i) for i in dr_data.get("patches_reviewed", [])]
        dr.patches_acked = [_load_item(i) for i in dr_data.get("patches_acked", [])]
        dr.discussions_posted = [_load_item(i) for i in dr_data.get("discussions_posted", [])]
        report.developer_reports.append(dr)
        report.total_patches += len(dr.patches_submitted)
        report.total_reviews += len(dr.patches_reviewed)
        report.total_acks += len(dr.patches_acked)

    return report


def _write_review_jsons(
    daily_report: "DailyReport",
    report_filename: str,
    reviews_dir: Path,
) -> dict[str, str]:
    """Write per-patchset review JSON files and return a review_links mapping.

    Extracts review comment data from *daily_report*, writes one JSON file per
    patchset that has review comments (merging with any existing file to
    accumulate data across report dates), and returns a dict mapping
    ``message_id -> slug`` for every item that has reviews.

    Extracted so that both ``generate_single_report()`` and
    ``rebuild_html_from_json()`` can call the same logic.
    """
    reviews_data = extract_reviews_data(daily_report, report_filename)
    review_links: dict[str, str] = {}

    for item_data in reviews_data:
        slug = item_data["slug"]
        msg_id = item_data["message_id"]
        review_links[msg_id] = slug

        json_path = reviews_dir / f"{slug}.json"

        # Merge with existing JSON (accumulate across dates)
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        else:
            existing = {}

        dates = existing.get("dates", {})
        report_date = item_data["date"]
        report_file = item_data["report_file"]

        # Group reviews by their individual message_date
        reviews_by_date: dict[str, list] = {}
        for review in item_data["reviews"]:
            bucket = review.get("message_date") or report_date
            reviews_by_date.setdefault(bucket, []).append(review)

        # Merge each date bucket into the JSON, preserving existing dates
        for bucket_date, bucket_reviews in reviews_by_date.items():
            link = report_file if bucket_date == report_date else dates.get(
                bucket_date, {}
            ).get("report_file", report_file)
            date_entry = {
                "report_file": link,
                "developer": item_data["developer"],
                "reviews": bucket_reviews,
            }
            if item_data.get("analysis_source"):
                date_entry["analysis_source"] = item_data["analysis_source"]
            if item_data.get("patch_summary"):
                date_entry["patch_summary"] = item_data["patch_summary"]
            dates[bucket_date] = date_entry

        # Always ensure the report-run date entry exists
        if report_date not in dates:
            dates[report_date] = {
                "report_file": report_file,
                "developer": item_data["developer"],
                "reviews": [],
            }
            if item_data.get("analysis_source"):
                dates[report_date]["analysis_source"] = item_data["analysis_source"]
            if item_data.get("patch_summary"):
                dates[report_date]["patch_summary"] = item_data["patch_summary"]

        merged = {
            "thread_id": msg_id,
            "subject": item_data["subject"],
            "url": item_data["url"],
            "dates": dates,
        }
        json_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if reviews_data:
        logger.info(
            "Saved review data for %d patchsets to %s", len(reviews_data), reviews_dir
        )

    return review_links


def _write_incremental_update(
    daily_report: "DailyReport",
    output_path: Path,
    reports_dir: Path,
    logs_dir: Path,
    report_filename: str,
    progress_status: dict,
    publish: bool = False,
    publish_repo: str = "",
    publish_branch: str = "main",
    publish_token: str = "",
) -> None:
    """Write a partial HTML report + daily JSON + index and optionally push to GitHub.

    Called before the developer loop (placeholder) and after each developer completes.
    The daily JSON is written with status="in_progress" so build_index can show the
    animated badge. The final complete report is written separately by
    generate_single_report after the loop.
    """
    # 1. Save partial daily JSON with status="in_progress"
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_json_path = daily_dir / f"{daily_report.date}.json"
    try:
        daily_json_path.write_text(
            json.dumps(
                _serialize_daily_report(daily_report, report_filename, status="in_progress"),
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Incremental update: failed to write daily JSON: %s", e)
        return

    # 2. Build review_links for items processed so far
    all_items = [
        item
        for dr in daily_report.developer_reports
        for item in (dr.patches_submitted + dr.patches_reviewed
                     + dr.patches_acked + dr.discussions_posted)
        if item.message_id
    ]
    review_links = (
        {item.message_id: message_id_to_slug(item.message_id) for item in all_items}
        or None
    )

    # 3. Write partial HTML report with progress banner (include last_updated timestamp)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    full_progress_status = {**progress_status, "last_updated": now_str}
    try:
        html_content = generate_html_report(
            daily_report,
            review_links=review_links,
            log_filename=None,
            progress_status=full_progress_status,
        )
        output_path.write_text(html_content, encoding="utf-8")
    except OSError as e:
        logger.warning("Incremental update: failed to write HTML: %s", e)
        return

    # 4. Rebuild index so the in-progress badge appears
    try:
        index_html = build_index(reports_dir, logs_dir)
        (reports_dir / "index.html").write_text(index_html, encoding="utf-8")
    except Exception as e:
        logger.warning("Incremental update: failed to rebuild index: %s", e)

    # 5. Push to GitHub if enabled
    if publish and publish_repo:
        done = progress_status.get("done", "?")
        total = progress_status.get("total", "?")
        logger.info("Incremental push to GitHub (%s/%s developers)...", done, total)
        publish_to_github(reports_dir, publish_repo,
                          branch=publish_branch, token=publish_token)


def generate_single_report(
    args: argparse.Namespace,
    target_date: datetime,
    developers: list[Developer],
    client: LKMLClient,
    llm_backends: dict[str, "LLMBackend"],
    publish: bool = False,
    publish_repo: str = "",
    publish_branch: str = "main",
    publish_token: str = "",
) -> Path:
    """Generate a report for a single date. Returns the output file path."""
    date_display = target_date.strftime("%Y-%m-%d")
    date_lore = target_date.strftime("%Y%m%d")

    # --- Compute output filename early (needed for log filename) ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if llm_backends:
        parts = [date_display]
        for backend_name, backend in llm_backends.items():
            safe_model = backend.model.replace(":", "-").replace("/", "-")
            parts.append(f"{backend_name}_{safe_model}")
        output_path = output_dir / f"{'_'.join(parts)}.html"
    else:
        output_path = output_dir / f"{date_display}.html"

    # --- Set up per-report log file ---
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{output_path.stem}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S %Z")
    )
    file_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    status = "success"
    error_msg = ""
    start_time = time.time()
    daily_report = DailyReport(date=date_display)

    try:
        logger.info("Generating report for %s", date_display)
        logger.info("Log file: %s", log_path)

        # Initialize LLM cache per-date
        llm_cache: Optional[LLMCache] = None
        if llm_backends and not args.llm_no_cache:
            llm_cache = LLMCache(date_str=date_display)
            logger.info(
                "LLM cache: enabled (%d cached entries)",
                llm_cache.stats()["entries"],
            )

        # Record LLM info for the HTML header
        for backend_name, backend in llm_backends.items():
            daily_report.llm_backends.append((backend_name, backend.model))

        # --- Placeholder: write "in progress" report before processing starts ---
        dev_names = [d.name for d in developers]
        _write_incremental_update(
            daily_report, output_path, output_dir, logs_dir, output_path.name,
            progress_status={
                "done": 0, "total": len(developers),
                "current": dev_names[0] if dev_names else "",
                "pending": dev_names,
            },
            publish=publish, publish_repo=publish_repo,
            publish_branch=publish_branch, publish_token=publish_token,
        )

        for i, dev in enumerate(developers, 1):
            logger.info("[%d/%d] Processing %s for %s...", i, len(developers), dev.name, date_display)
            try:
                llm_dump_dir = Path(args.llm_dump) if args.llm_dump else None
                dev_report = process_developer(
                    client, dev, date_lore, args.skip_threads,
                    llm_backends=llm_backends if llm_backends else None,
                    llm_cache=llm_cache,
                    llm_dump_dir=llm_dump_dir,
                    force_monolithic=getattr(args, 'llm_monolithic', False),
                )
            except Exception as e:
                logger.error("Unexpected error processing %s: %s", dev.name, e)
                dev_report = DeveloperReport(
                    developer=dev, errors=[f"Unexpected error: {e}"]
                )

            daily_report.developer_reports.append(dev_report)
            daily_report.total_patches += len(dev_report.patches_submitted)
            daily_report.total_reviews += len(dev_report.patches_reviewed)
            daily_report.total_acks += len(dev_report.patches_acked)
            # discussions_posted not counted in totals (separate stat in HTML)

            # --- Incremental update: push partial results after each developer ---
            remaining = [d.name for d in developers[i:]]
            _write_incremental_update(
                daily_report, output_path, output_dir, logs_dir, output_path.name,
                progress_status={
                    "done": i, "total": len(developers),
                    "current": remaining[0] if remaining else "",
                    "pending": remaining,
                },
                publish=publish, publish_repo=publish_repo,
                publish_branch=publish_branch, publish_token=publish_token,
            )

        daily_report.generation_time_seconds = time.time() - start_time

        # --- Save per-patchset review JSON and build review_links ---
        reviews_dir = output_dir / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)

        review_links = _write_review_jsons(daily_report, output_path.name, reviews_dir)

        # --- Save daily activity summary for --rebuild-html ---
        daily_dir = output_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_json_path = daily_dir / f"{date_display}.json"
        daily_json_path.write_text(
            json.dumps(
                _serialize_daily_report(daily_report, output_path.name, status="complete"),
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.debug("Saved daily summary: %s", daily_json_path)

        # Generate HTML report with review links and log filename
        html_content = generate_html_report(
            daily_report,
            review_links=review_links if review_links else None,
            log_filename=log_path.name,
        )
        output_path.write_text(html_content, encoding="utf-8")

        logger.info(
            "Report generated: %s (%d patches, %d reviews, %d acks in %.1fs)",
            output_path,
            daily_report.total_patches,
            daily_report.total_reviews,
            daily_report.total_acks,
            daily_report.generation_time_seconds,
        )
        # Completion marker — build_index uses this to infer status from old logs
        logger.info("Report generation complete: %s", date_display)

    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        logger.error("Report generation failed: %s", exc)
        raise

    finally:
        file_handler.close()
        root_logger.removeHandler(file_handler)

        # Record run in history log
        elapsed = time.time() - start_time
        _record_run(logs_dir, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_date": date_display,
            "report_file": output_path.name,
            "log_file": log_path.name,
            "status": status,
            "error": error_msg,
            "duration_seconds": round(elapsed, 1),
            "patches": daily_report.total_patches,
            "reviews": daily_report.total_reviews,
            "acks": daily_report.total_acks,
        })

    return output_path


def purge_old_files(
    reports_dir: Path,
    logs_dir: Path,
    retention_days: int = 30,
) -> dict[str, int]:
    """Remove reports, logs, and review data older than retention_days.

    Deletes:
      - reports/*.html whose date prefix is older than the cutoff
      - logs/*.log whose date prefix is older than the cutoff
      - review JSON/HTML entries: removes date entries from JSON files for dates
        older than the cutoff, and deletes the JSON+HTML pair if no dates remain

    Returns a dict with counts: {"reports", "logs", "reviews"}.
    """
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    counts = {"reports": 0, "logs": 0, "reviews": 0}

    date_pattern = re.compile(r"(?:lkml_)?(\d{4}-\d{2}-\d{2})")

    def _file_date(filename: str) -> str | None:
        m = date_pattern.match(filename)
        return m.group(1) if m else None

    # --- Purge old reports ---
    if reports_dir.exists():
        for f in list(reports_dir.iterdir()):
            if not f.is_file() or f.suffix != ".html" or f.name == "index.html":
                continue
            file_date = _file_date(f.name)
            if file_date and file_date < cutoff:
                f.unlink()
                counts["reports"] += 1
                logger.info("Purged report: %s", f.name)

    # --- Purge old logs ---
    if logs_dir.exists():
        for f in list(logs_dir.iterdir()):
            if not f.is_file() or f.suffix != ".log":
                continue
            file_date = _file_date(f.name)
            if file_date and file_date < cutoff:
                f.unlink()
                counts["logs"] += 1
                logger.info("Purged log: %s", f.name)

    # --- Purge old review data ---
    reviews_dir = reports_dir / "reviews"
    if reviews_dir.exists():
        for json_path in list(reviews_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            dates = data.get("dates", {})
            original_count = len(dates)
            # Remove date entries older than cutoff
            dates = {d: v for d, v in dates.items() if d >= cutoff}

            if not dates:
                # No dates remain — delete both JSON and HTML
                json_path.unlink()
                html_path = json_path.with_suffix(".html")
                if html_path.exists():
                    html_path.unlink()
                counts["reviews"] += 1
                logger.info("Purged review (all dates expired): %s", json_path.name)
            elif len(dates) < original_count:
                # Some dates removed — rewrite JSON (HTML rebuilt by build_reviews.py)
                data["dates"] = dates
                json_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(
                    "Pruned %d old date(s) from review: %s",
                    original_count - len(dates),
                    json_path.name,
                )

    # --- Purge old run_history entries ---
    history_path = logs_dir / "run_history.json"
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            before = len(history)
            history = [e for e in history if e.get("timestamp", "")[:10] >= cutoff]
            if len(history) < before:
                history_path.write_text(
                    json.dumps(history, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(
                    "Pruned %d old run history entries",
                    before - len(history),
                )
        except (json.JSONDecodeError, OSError):
            pass

    logger.info(
        "Purge complete: %d reports, %d logs, %d reviews removed (cutoff: %s)",
        counts["reports"],
        counts["logs"],
        counts["reviews"],
        cutoff,
    )
    return counts


def _record_run(logs_dir: Path, entry: dict) -> None:
    """Append a run entry to logs/run_history.json, keeping the last 14 days."""
    history_path = logs_dir / "run_history.json"
    try:
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        else:
            history = []
    except (json.JSONDecodeError, OSError):
        history = []

    history.append(entry)

    # Keep only entries from the last 14 days
    cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    history = [e for e in history if e.get("timestamp", "")[:10] >= cutoff]

    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def publish_to_github(
    reports_dir: Path,
    repo: str,
    branch: str = "main",
    token: str = "",
) -> bool:
    """Commit all changes in reports_dir and push to a GitHub repository.

    The reports directory is treated as a self-contained git repository.
    On first use it will be initialised with ``git init`` and the remote added.
    On subsequent runs it re-uses the existing repo.

    If ``token`` is provided it is embedded in the remote URL as
    ``https://x-access-token:<token>@github.com/owner/repo.git`` so that no
    separate credential helper or SSH key is required.  This is the standard
    approach for Docker / CI environments.  The token-bearing URL is stored
    only in the local ``.git/config`` of the (ephemeral) reports directory and
    is never logged.

    Args:
        reports_dir: Path to the reports directory to publish.
        repo: GitHub repository slug, e.g. ``"owner/repo"``.
        branch: Branch name to push to. Default: ``"main"``.
        token: GitHub personal access token or fine-grained token. Optional —
            falls back to whatever git credential helper is configured.

    Returns:
        True on success, False if any git command fails.
    """
    def _run(cmd: list[str], cwd: Path, redact: str = "") -> subprocess.CompletedProcess:
        display = " ".join(cmd)
        if redact:
            display = display.replace(redact, "***")
        logger.debug("git: %s (cwd=%s)", display, cwd)
        return subprocess.run(
            cmd, cwd=str(cwd),
            capture_output=True, text=True,
        )

    # Normalise repo to "owner/repo" slug regardless of whether a full URL was given.
    # Accepted formats:
    #   https://github.com/owner/repo.git  →  owner/repo
    #   https://github.com/owner/repo      →  owner/repo
    #   git@github.com:owner/repo.git      →  owner/repo
    #   owner/repo                         →  owner/repo  (passthrough)
    slug = repo.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", slug)
    if m:
        slug = m.group(1)

    # Build URLs — one safe (for logging), one with embedded token (for git)
    public_url = f"https://github.com/{slug}.git"
    if token:
        push_url = f"https://x-access-token:{token}@github.com/{slug}.git"
    else:
        push_url = public_url

    git_dir = reports_dir / ".git"

    # --- Initialise repo if not already a git repo ---
    if not git_dir.exists():
        logger.info("GitHub publish: initialising git repo in %s", reports_dir)
        r = _run(["git", "init", "-b", branch], reports_dir)
        if r.returncode != 0:
            # Older git versions don't support -b; init then rename
            _run(["git", "init"], reports_dir)
            _run(["git", "checkout", "-b", branch], reports_dir)

        # Set a minimal identity so commits work in CI / Docker environments
        # that have no global git config.
        _run(["git", "config", "user.email", "lkml-tracker@localhost"], reports_dir)
        _run(["git", "config", "user.name", "LKML Tracker"], reports_dir)

    # --- Ensure remote is configured with the correct (token-bearing) URL ---
    r = _run(["git", "remote", "get-url", "origin"], reports_dir)
    if r.returncode != 0:
        logger.info("GitHub publish: adding remote origin → %s", public_url)
        r = _run(["git", "remote", "add", "origin", push_url], reports_dir,
                 redact=token)
        if r.returncode != 0:
            logger.error("GitHub publish: failed to add remote: %s", r.stderr.strip())
            return False
    else:
        # Update URL whenever token or repo slug may have changed
        _run(["git", "remote", "set-url", "origin", push_url], reports_dir,
             redact=token)
        logger.debug("GitHub publish: remote origin set to %s", public_url)

    # --- Stage all changes ---
    r = _run(["git", "add", "-A"], reports_dir)
    if r.returncode != 0:
        logger.error("GitHub publish: git add failed: %s", r.stderr.strip())
        return False

    # Check if there's actually anything to commit and log a clear summary
    r = _run(["git", "status", "--porcelain"], reports_dir)
    has_changes = bool(r.stdout.strip())

    if not has_changes:
        # Nothing new to commit — but we may still need to push a previous
        # commit that failed to reach the remote (e.g. after a 403 that was
        # since fixed). Check whether HEAD is ahead of the remote tracking
        # branch and push if so, otherwise we are genuinely up to date.
        r_ahead = _run(
            ["git", "rev-list", "--count", "@{u}..HEAD"],
            reports_dir,
        )
        if r_ahead.returncode != 0 or r_ahead.stdout.strip() == "0":
            logger.info("GitHub publish: nothing to commit, reports are up to date.")
            return True
        ahead = r_ahead.stdout.strip()
        logger.info(
            "GitHub publish: nothing new to commit but %s unpushed commit(s) found — pushing now",
            ahead,
        )

    if has_changes:
        # Parse and log a human-readable diff summary (added / modified / deleted)
        added, modified, deleted = [], [], []
        for line in r.stdout.splitlines():
            if len(line) < 3:
                continue
            xy = line[:2]
            path = line[3:]
            if "A" in xy:
                added.append(path)
            elif "D" in xy:
                deleted.append(path)
            elif "M" in xy or "R" in xy:
                modified.append(path)
        logger.info(
            "GitHub publish: %d added, %d modified, %d deleted",
            len(added), len(modified), len(deleted),
        )
        for f in added:
            logger.info("  + %s", f)
        for f in modified:
            logger.info("  ~ %s", f)
        for f in deleted:
            logger.info("  - %s", f)

        # --- Commit ---
        commit_msg = f"LKML reports update {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        r = _run(["git", "commit", "-m", commit_msg], reports_dir)
        if r.returncode != 0:
            logger.error("GitHub publish: git commit failed: %s", r.stderr.strip())
            return False
        logger.info("GitHub publish: committed — %s", commit_msg)

    # --- Push ---
    # Check whether our branch already tracks the remote. If it does we use
    # --force-with-lease (safe: aborts if someone else pushed since we last
    # fetched). If there is no upstream yet — e.g. the remote was initialised
    # with a GitHub README that we don't have locally — we force-push to make
    # this repo the authoritative source of truth for generated reports.
    logger.info("GitHub publish: pushing to %s (branch: %s)…", slug, branch)
    r_upstream = _run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        reports_dir,
    )
    has_upstream = r_upstream.returncode == 0 and r_upstream.stdout.strip()

    if has_upstream:
        push_cmd = ["git", "push", "-u", "origin", branch, "--force-with-lease"]
    else:
        # No upstream yet — remote may have divergent history (e.g. auto-README).
        # Force-push to establish this repo as the source of truth.
        logger.info("GitHub publish: no upstream set, using --force for initial push")
        push_cmd = ["git", "push", "-u", "origin", branch, "--force"]

    r = _run(push_cmd, reports_dir)
    if r.returncode != 0:
        logger.error(
            "GitHub publish: git push failed:\n%s\n%s",
            r.stdout.strip(), r.stderr.strip(),
        )
        return False

    logger.info("GitHub publish: pushed successfully to %s/%s", repo, branch)
    return True


def _rebuild_single_patchset(args: argparse.Namespace, target_dates: list) -> None:
    """Find a single patchset by message-id fragment and write a standalone preview HTML.

    Searches all target_dates' daily JSON files for an item whose message_id contains
    args.message_id (case-insensitive partial match). When found, builds a mini
    DailyReport containing only that item and renders it to
    reports/preview_<slug>.html — without touching the live report or index.
    """
    msg_filter = args.message_id.strip("<>").lower()
    reports_dir = Path(args.output_dir)
    daily_dir = reports_dir / "daily"
    found = False

    for target_date in target_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        daily_json = daily_dir / f"{date_str}.json"
        if not daily_json.exists():
            continue
        try:
            daily_report = _load_daily_report(daily_json)
        except Exception as e:
            logger.error("Failed to load %s: %s", daily_json, e)
            continue

        for dr in daily_report.developer_reports:
            categories = [
                ("patches_submitted",  dr.patches_submitted),
                ("patches_reviewed",   dr.patches_reviewed),
                ("patches_acked",      dr.patches_acked),
                ("discussions_posted", dr.discussions_posted),
            ]
            for cat_name, items in categories:
                for item in items:
                    if msg_filter not in item.message_id.lower():
                        continue
                    # Build a mini DailyReport containing only this one item
                    mini_dr = DeveloperReport(developer=dr.developer)
                    getattr(mini_dr, cat_name).append(item)
                    mini_report = DailyReport(
                        date=date_str,
                        llm_backends=daily_report.llm_backends,
                        generation_time_seconds=daily_report.generation_time_seconds,
                    )
                    mini_report.developer_reports.append(mini_dr)
                    mini_report.total_patches = len(mini_dr.patches_submitted)
                    mini_report.total_reviews = len(mini_dr.patches_reviewed)
                    mini_report.total_acks    = len(mini_dr.patches_acked)

                    slug = message_id_to_slug(item.message_id)
                    preview_path = reports_dir / f"preview_{slug}.html"
                    html_content = generate_html_report(mini_report, review_links=None)
                    preview_path.write_text(html_content, encoding="utf-8")
                    logger.info("Preview written: %s", preview_path)
                    found = True

    if not found:
        logger.error(
            "No patchset found matching --message-id '%s' in dates: %s",
            args.message_id,
            [d.strftime("%Y-%m-%d") for d in target_dates],
        )


def rebuild_html_from_json(args: argparse.Namespace, target_dates: list) -> None:
    """Re-render HTML reports from persisted daily summary JSON files.

    Reads reports/daily/{date}.json (saved during the original run) to
    reconstruct a full DailyReport, then re-generates the HTML report,
    review pages, and index — without any web fetches or LLM calls.
    """
    if getattr(args, "message_id", None):
        _rebuild_single_patchset(args, target_dates)
        return

    reports_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    daily_dir = reports_dir / "daily"

    for target_date in target_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        daily_json = daily_dir / f"{date_str}.json"

        if not daily_json.exists():
            logger.warning(
                "No daily summary found for %s (%s) — skipping. "
                "Run without --rebuild-html first to generate the summary.",
                date_str, daily_json,
            )
            continue

        logger.info("Rebuilding HTML for %s from %s...", date_str, daily_json)
        try:
            daily_report = _load_daily_report(daily_json)
        except Exception as e:
            logger.error("Failed to load daily summary for %s: %s", date_str, e)
            continue

        # Determine output path from the saved report_file name
        saved_data = json.loads(daily_json.read_text(encoding="utf-8"))
        report_filename = saved_data.get("report_file", f"{date_str}.html")
        output_path = reports_dir / report_filename

        # Write review JSON files (creates missing ones, merges with existing)
        # and get the review_links mapping for items that actually have reviews.
        reviews_dir = reports_dir / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        review_links = _write_review_jsons(daily_report, report_filename, reviews_dir)

        # Re-render the daily HTML report
        html_content = generate_html_report(
            daily_report,
            review_links=review_links or None,
            log_filename=None,
        )
        output_path.write_text(html_content, encoding="utf-8")
        logger.info("Rebuilt report: %s", output_path)

        # Rebuild review pages and index
        logger.info("Rebuilding review pages...")
        build_all_reviews(reports_dir)

        logger.info("Rebuilding index...")
        index_html = build_index(reports_dir, logs_dir)
        (reports_dir / "index.html").write_text(index_html, encoding="utf-8")
        logger.info("Index updated.")


def main():
    args = parse_args()

    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S %Z")
    _console = logging.StreamHandler()
    _console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    _console.setFormatter(_fmt)
    # Root logger must be DEBUG so per-report file handlers (which are always DEBUG)
    # actually receive DEBUG records. Console threshold is controlled separately above.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(_console)

    # --- Purge-only mode ---
    if args.purge_only:
        reports_dir = Path(args.output_dir)
        logs_dir = Path(args.logs_dir)
        purge_old_files(reports_dir, logs_dir, args.retention_days)
        return

    # --- Publish-only mode (test/re-publish without regenerating reports) ---
    if args.publish_only:
        repo = args.github_repo or os.environ.get("GITHUB_REPO", "")
        if not repo:
            logger.error(
                "GitHub publish: no repository specified. "
                "Use --github-repo owner/repo or set the GITHUB_REPO environment variable."
            )
            sys.exit(1)
        token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
        if not token:
            logger.warning(
                "GitHub publish: no token provided. Falling back to git credential "
                "helper. Set --github-token or GITHUB_TOKEN for Docker/CI environments."
            )
        branch = args.github_branch
        if branch == "main":
            branch = os.environ.get("GITHUB_BRANCH", branch)
        reports_dir = Path(args.output_dir)
        ok = publish_to_github(reports_dir, repo, branch=branch, token=token)
        if not ok:
            logger.error("GitHub publish: failed — see errors above.")
            sys.exit(1)
        return

    # Determine target end date
    if args.date:
        try:
            end_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format: %s (expected YYYY-MM-DD)", args.date)
            sys.exit(1)
    else:
        end_date = datetime.now() - timedelta(days=1)

    # Validate --days
    num_days = args.days
    if num_days < 1:
        logger.error("--days must be at least 1 (got %d)", num_days)
        sys.exit(1)

    # Build the list of dates (oldest first)
    target_dates = [
        end_date - timedelta(days=i) for i in range(num_days - 1, -1, -1)
    ]

    if num_days > 1:
        logger.info(
            "Generating reports for %d days: %s to %s",
            num_days,
            target_dates[0].strftime("%Y-%m-%d"),
            target_dates[-1].strftime("%Y-%m-%d"),
        )

    # --- Rebuild-HTML mode (no web fetches or LLM calls) ---
    if args.rebuild_html:
        rebuild_html_from_json(args, target_dates)
        return

    # Load developers
    developers = load_developers(args.people)
    if not developers:
        logger.error("No developers loaded. Check CSV file.")
        sys.exit(1)

    # Initialize client
    client = LKMLClient(rate_limit_delay=args.rate_limit)

    # Initialize LLM backends (if requested)
    llm_backends: dict[str, LLMBackend] = {}

    if args.llm_all:
        # --llm-all: initialize BOTH Ollama and Anthropic
        # Ollama
        try:
            ollama_model = args.ollama_model
            llm_backends["ollama"] = OllamaBackend(model=ollama_model)
            logger.info("LLM backend: Ollama (%s)", ollama_model)
        except RuntimeError as e:
            logger.error("Failed to initialize Ollama backend: %s", e)
            logger.info("Ollama will be skipped.")

        # Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Required for --llm-all."
            )
            sys.exit(1)
        try:
            anthropic_model = args.anthropic_model
            llm_backends["anthropic"] = AnthropicBackend(
                api_key=api_key, model=anthropic_model,
            )
            logger.info("LLM backend: Anthropic (%s)", anthropic_model)
        except RuntimeError as e:
            logger.error("Failed to initialize Anthropic backend: %s", e)
            logger.info("Anthropic will be skipped.")

        if not llm_backends:
            logger.error("No LLM backends could be initialized. Exiting.")
            sys.exit(1)

    elif args.llm:
        # --llm: single backend mode (backward compat)
        try:
            if args.llm_backend == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    logger.error(
                        "ANTHROPIC_API_KEY environment variable not set. "
                        "Set it or use --llm-backend ollama."
                    )
                    sys.exit(1)
                model = args.llm_model or "claude-haiku-4-5"
                llm_backends["anthropic"] = AnthropicBackend(
                    api_key=api_key, model=model,
                )
                logger.info("LLM backend: Anthropic (%s)", model)
            else:
                model = args.llm_model or "llama3.1:8b"
                llm_backends["ollama"] = OllamaBackend(model=model)
                logger.info("LLM backend: Ollama (%s)", model)
        except RuntimeError as e:
            logger.error("Failed to initialize LLM backend: %s", e)
            logger.info("Falling back to heuristic analysis.")

    # Resolve publish settings once (used inside the per-day loop)
    reports_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    publish = args.publish_github
    publish_repo = ""
    publish_token = ""
    publish_branch = ""
    if publish:
        publish_repo = args.github_repo or os.environ.get("GITHUB_REPO", "")
        if not publish_repo:
            logger.error(
                "GitHub publish: no repository specified. "
                "Use --github-repo owner/repo or set the GITHUB_REPO environment variable."
            )
            sys.exit(1)
        publish_token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
        if not publish_token:
            logger.warning(
                "GitHub publish: no token provided. Falling back to git credential "
                "helper. Set --github-token or GITHUB_TOKEN for Docker/CI environments."
            )
        publish_branch = args.github_branch
        if publish_branch == "main":  # argparse default — check if env var overrides it
            publish_branch = os.environ.get("GITHUB_BRANCH", publish_branch)

    # Generate a report for each date, rebuilding index + publishing after each one
    total_start = time.time()
    for day_num, target_date in enumerate(target_dates, 1):
        if num_days > 1:
            logger.info(
                "=== Day %d/%d: %s ===",
                day_num, num_days, target_date.strftime("%Y-%m-%d"),
            )
        generate_single_report(
            args, target_date, developers, client, llm_backends,
            publish=publish,
            publish_repo=publish_repo,
            publish_branch=publish_branch,
            publish_token=publish_token,
        )

        # --- Rebuild review HTML pages and index after every day ---
        logger.info("Rebuilding review pages...")
        build_all_reviews(reports_dir)

        logger.info("Rebuilding index...")
        index_html = build_index(reports_dir, logs_dir)
        index_path = reports_dir / "index.html"
        index_path.write_text(index_html, encoding="utf-8")
        logger.info("Index updated: %s", index_path)

        # --- Publish to GitHub after every day (if requested) ---
        if publish:
            ok = publish_to_github(reports_dir, publish_repo, branch=publish_branch, token=publish_token)
            if not ok:
                logger.error("GitHub publish: failed for %s — see errors above.", target_date.strftime("%Y-%m-%d"))
                # Don't exit — continue processing remaining days

    if num_days > 1:
        total_elapsed = time.time() - total_start
        logger.info(
            "All %d reports generated in %.1fs",
            num_days, total_elapsed,
        )

    # --- Automatic purge (unless --no-purge) ---
    # Purge runs once at the end: no benefit to purging after every day, and
    # doing it mid-run could remove data still needed for subsequent days.
    if not args.no_purge:
        purge_old_files(reports_dir, logs_dir, args.retention_days)
        # Rebuild index once more after purge to reflect any removed entries
        logger.info("Rebuilding index after purge...")
        index_html = build_index(reports_dir, logs_dir)
        index_path.write_text(index_html, encoding="utf-8")
        if publish:
            publish_to_github(reports_dir, publish_repo, branch=publish_branch, token=publish_token)


if __name__ == "__main__":
    main()
