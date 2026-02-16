"""LKML Daily Activity Tracker - generates HTML reports of kernel developer activity.

Usage:
    python generate_report.py                              # Yesterday's report
    python generate_report.py --date 2025-02-12            # Specific date
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
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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
from models import ActivityItem, ActivityType, DailyReport, Developer, DeveloperReport, LLMAnalysis
from report_generator import generate_html_report
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

    # Classify today's messages
    def raw_fetcher(msg_id: str) -> str:
        return client.get_raw_message(msg_id)

    if all_entries:
        try:
            patches, reviews, acks = classify_messages(
                all_entries, developer, raw_fetcher
            )
            report.patches_submitted = patches
            report.patches_reviewed = reviews
            report.patches_acked = acks
        except Exception as e:
            err_msg = f"Classification failed for {developer.name}: {e}"
            logger.warning("  %s", err_msg)
            report.errors.append(err_msg)
    else:
        patches = []
        reviews = []
        acks = []

    logger.info(
        "  %s: %d patches, %d reviews, %d acks (today)",
        developer.name,
        len(patches),
        len(reviews),
        len(acks),
    )

    # --- 14-day lookback: find recent patches with activity today ---
    today_patch_ids = {p.message_id for p in report.patches_submitted}
    # Also collect timestamp-prefix keys from today's patches to avoid dupes
    today_prefix_keys = set()
    for p in report.patches_submitted:
        pfx = re.match(r"^(\d+\.\d+)", p.message_id)
        if pfx:
            today_prefix_keys.add(pfx.group(1))

    # Compute lookback date range
    target_date_obj = datetime.strptime(date_str, "%Y%m%d")
    lookback_start = (target_date_obj - timedelta(days=lookback_days)).strftime("%Y%m%d")
    # End date is the day before the report date (today's patches already captured)
    lookback_end_obj = target_date_obj - timedelta(days=1)
    if lookback_end_obj < datetime.strptime(lookback_start, "%Y%m%d"):
        lookback_end = lookback_start
    else:
        lookback_end = lookback_end_obj.strftime("%Y%m%d")

    recent_patch_entries = []
    recent_seen_ids = set(seen_msg_ids)  # avoid dupes with today's messages

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
                    # Also skip if the series prefix matches a today patch
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

    # Deduplicate recent patch series (same logic as today's patches)
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
        logger.info(
            "  %s: %d recent patch series to check for activity today",
            developer.name,
            len(recent_items),
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
                "  %s: %d ongoing patches with activity today",
                developer.name,
                len(ongoing_patches),
            )
            # Append ongoing patches after today's patches
            report.patches_submitted.extend(ongoing_patches)
    else:
        thread_cache = {}

    # Fetch threads for conversation summaries on all items
    if not skip_threads:
        all_items = report.patches_submitted + reviews + acks

        for item in all_items:
            msg_id = item.message_id
            if item.conversation is not None:
                continue  # Already analyzed
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
                        thread_messages, item, backend, llm_cache
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


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine target date
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format: %s (expected YYYY-MM-DD)", args.date)
            sys.exit(1)
    else:
        target_date = datetime.now() - timedelta(days=1)

    date_display = target_date.strftime("%Y-%m-%d")
    date_lore = target_date.strftime("%Y%m%d")

    logger.info("Generating report for %s", date_display)

    # Load developers
    developers = load_developers(args.people)
    if not developers:
        logger.error("No developers loaded. Check CSV file.")
        sys.exit(1)

    # Initialize client
    client = LKMLClient(rate_limit_delay=args.rate_limit)

    # Initialize LLM backends (if requested)
    llm_backends: dict[str, LLMBackend] = {}
    llm_cache: Optional[LLMCache] = None

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

    if llm_backends and not args.llm_no_cache:
        llm_cache = LLMCache(date_str=date_display)
        logger.info(
            "LLM cache: enabled (%d cached entries)",
            llm_cache.stats()["entries"],
        )

    # Process each developer
    start_time = time.time()
    daily_report = DailyReport(date=date_display)

    # Record LLM info for the report filename and HTML header
    for backend_name, backend in llm_backends.items():
        daily_report.llm_backends.append((backend_name, backend.model))

    for i, dev in enumerate(developers, 1):
        logger.info("[%d/%d] Processing %s...", i, len(developers), dev.name)
        try:
            dev_report = process_developer(
                client, dev, date_lore, args.skip_threads,
                llm_backends=llm_backends if llm_backends else None,
                llm_cache=llm_cache,
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

    daily_report.generation_time_seconds = time.time() - start_time

    # Generate HTML
    html_content = generate_html_report(daily_report)

    # Write output — include LLM backend/model in filename when enabled
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if daily_report.llm_backends:
        # Build filename with all backends, e.g. "2025-02-11_ollama_llama3.1-8b_anthropic_claude-haiku-4-5.html"
        parts = [date_display]
        for backend_name, model_name in daily_report.llm_backends:
            safe_model = model_name.replace(":", "-").replace("/", "-")
            parts.append(f"{backend_name}_{safe_model}")
        output_path = output_dir / f"{'_'.join(parts)}.html"
    else:
        output_path = output_dir / f"{date_display}.html"
    output_path.write_text(html_content, encoding="utf-8")

    logger.info(
        "Report generated: %s (%d patches, %d reviews, %d acks in %.1fs)",
        output_path,
        daily_report.total_patches,
        daily_report.total_reviews,
        daily_report.total_acks,
        daily_report.generation_time_seconds,
    )
    print(f"\nReport: {output_path.resolve()}")


if __name__ == "__main__":
    main()
