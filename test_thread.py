"""Test single-thread analysis — heuristic and/or LLM.

Usage:
    python test_thread.py <message-id>                            # Heuristic only
    python test_thread.py <message-id> --llm                      # Heuristic + LLM (Ollama)
    python test_thread.py <message-id> --llm --llm-backend anthropic  # + Anthropic
    python test_thread.py <message-id> --llm-all                  # Heuristic + Ollama + Anthropic
    python test_thread.py <message-id> --llm-all --ollama-model llama3.2 --anthropic-model claude-sonnet-4-5
    python test_thread.py <message-id> --llm --llm-no-cache -v    # No cache, verbose
"""

import argparse
import io
import logging
import os
import re
import sys
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from lkml_client import LKMLClient, LKMLAPIError
from llm_cache import LLMCache
from llm_summarizer import (
    AnthropicBackend,
    OllamaBackend,
    analyze_thread_llm,
    _build_thread_text,
    _build_analysis_prompt,
)
from models import ActivityItem, ActivityType, ConversationSummary, Sentiment
from thread_analyzer import analyze_thread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

_SENTIMENT_COLORS = {
    Sentiment.POSITIVE: "\033[32m",      # green
    Sentiment.NEEDS_WORK: "\033[33m",    # yellow
    Sentiment.CONTENTIOUS: "\033[31m",   # red
    Sentiment.NEUTRAL: "\033[90m",       # grey
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _sentiment_str(sentiment: Sentiment) -> str:
    color = _SENTIMENT_COLORS.get(sentiment, "")
    return _colored(sentiment.value.upper(), color)


def _print_summary(summary: ConversationSummary, label: str) -> None:
    """Pretty-print a ConversationSummary to the terminal."""
    bar = "\u2501" * 50  # ━
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  {label}{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")

    # Sentiment
    signals = ", ".join(f'"{s}"' for s in summary.sentiment_signals[:5])
    print(f"  Sentiment:  {_sentiment_str(summary.sentiment)}", end="")
    if signals:
        print(f"  {_DIM}({signals}){_RESET}")
    else:
        print()

    # Progress
    if summary.discussion_progress:
        prog = summary.discussion_progress.value.upper()
        detail = f" \u2014 {summary.progress_detail}" if summary.progress_detail else ""
        print(f"  Progress:   {_BOLD}{prog}{_RESET}{detail}")

    # Patch summary
    if summary.patch_summary:
        print(f"\n  {_BOLD}Patch Summary:{_RESET}")
        for para in summary.patch_summary.split("\n\n"):
            wrapped = para.strip()
            if wrapped:
                # Indent each line
                for line in wrapped.split("\n"):
                    print(f"    {line}")
                print()

    # Review comments
    if summary.review_comments:
        print(f"  {_BOLD}Review Comments ({len(summary.review_comments)}):{_RESET}")
        for rc in summary.review_comments:
            badges = []
            badges.append(f"[{_sentiment_str(rc.sentiment)}]")
            if rc.has_inline_review:
                badges.append(_colored("[inline]", "\033[36m"))
            for tag in rc.tags_given:
                badges.append(_colored(f"[{tag}]", "\033[35m"))
            badge_str = " ".join(badges)

            print(f"    \u2022 {_BOLD}{rc.author}{_RESET} {badge_str}")
            if rc.summary:
                # Wrap summary text with indent
                lines = rc.summary.strip().split("\n")
                for line in lines:
                    print(f"      {line}")
            if rc.sentiment_signals:
                sig = ", ".join(f'"{s}"' for s in rc.sentiment_signals[:3])
                print(f"      {_DIM}Signals: {sig}{_RESET}")
            print()

    # Key points (fallback when no review comments)
    if summary.key_points and not summary.review_comments:
        print(f"  {_BOLD}Key Points:{_RESET}")
        for point in summary.key_points:
            print(f"    \u2022 {point}")
        print()

    # Participant count
    if summary.participant_count:
        print(f"  {_DIM}Participants: {summary.participant_count}{_RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test single-thread analysis (heuristic and/or LLM)."
    )
    parser.add_argument(
        "message_id",
        help="Message-ID of the thread root (e.g., '20250211063601.3530292-1-alexghiti@rivosinc.com').",
    )
    parser.add_argument(
        "--type",
        choices=["patch", "review", "ack", "auto"],
        default="auto",
        help="Activity type. Default: auto-detect from subject.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    # LLM options
    llm_group = parser.add_argument_group("LLM analysis")
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Also run LLM analysis and show side-by-side comparison.",
    )
    llm_group.add_argument(
        "--llm-all",
        action="store_true",
        help="Run BOTH Ollama and Anthropic and show all results.",
    )
    llm_group.add_argument(
        "--llm-backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend for --llm mode. Default: ollama.",
    )
    llm_group.add_argument(
        "--llm-model",
        default=None,
        help="Override default model (for --llm single-backend mode).",
    )
    llm_group.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Ollama model for --llm-all. Default: llama3.1:8b.",
    )
    llm_group.add_argument(
        "--anthropic-model",
        default="claude-haiku-4-5",
        help="Anthropic model for --llm-all. Default: claude-haiku-4-5.",
    )
    llm_group.add_argument(
        "--llm-no-cache",
        action="store_true",
        help="Disable LLM result caching.",
    )
    llm_group.add_argument(
        "--llm-raw",
        action="store_true",
        help="Print raw LLM JSON response for debugging.",
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
        help="Force monolithic LLM prompts (disable per-reviewer decomposition).",
    )

    return parser.parse_args()


def _detect_activity_type(subject: str, explicit: str) -> ActivityType:
    """Determine activity type from subject or explicit flag."""
    if explicit == "patch":
        return ActivityType.PATCH_SUBMITTED
    if explicit == "review":
        return ActivityType.PATCH_REVIEWED
    if explicit == "ack":
        return ActivityType.PATCH_ACKED
    # Auto-detect
    if re.search(r"\[PATCH", subject, re.IGNORECASE):
        return ActivityType.PATCH_SUBMITTED
    if subject.lower().startswith("re:"):
        return ActivityType.PATCH_REVIEWED
    return ActivityType.PATCH_SUBMITTED


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    msg_id = args.message_id.strip("<>")

    # Early validation for LLM backends
    if args.llm_all or (args.llm and args.llm_backend == "anthropic"):
        if not os.environ.get("ANTHROPIC_API_KEY", ""):
            print(
                "Error: ANTHROPIC_API_KEY not set. "
                "Export it or use --llm-backend ollama.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fetch thread
    print(f"Fetching thread for {msg_id}...")
    client = LKMLClient(rate_limit_delay=0.5)
    try:
        result = client.get_thread(msg_id)
    except LKMLAPIError as e:
        print(f"Error fetching thread: {e}", file=sys.stderr)
        sys.exit(1)

    thread_messages = result.get("messages", [])
    if not thread_messages:
        print(f"No messages found in thread for {msg_id}", file=sys.stderr)
        sys.exit(1)

    # Thread info
    first_msg = thread_messages[0]
    subject = first_msg.get("subject", "(no subject)")
    activity_type = _detect_activity_type(subject, args.type)

    print(f"\n{_BOLD}Thread:{_RESET} {subject}")
    print(f"{_BOLD}URL:{_RESET}    https://lore.kernel.org/r/{msg_id}/")
    print(f"{_BOLD}Messages:{_RESET} {len(thread_messages)}  |  "
          f"{_BOLD}Type:{_RESET} {activity_type.value}")

    # Build ActivityItem
    activity_item = ActivityItem(
        activity_type=activity_type,
        subject=subject,
        message_id=msg_id,
        url=f"https://lore.kernel.org/r/{msg_id}/",
        date=first_msg.get("date", ""),
    )

    # --- Heuristic analysis ---
    heuristic = analyze_thread(thread_messages, activity_item)
    _print_summary(heuristic, "HEURISTIC ANALYSIS")

    # --- Build list of backends to run ---
    backends_to_run: list[tuple[str, object]] = []  # (label, backend_instance)

    if args.llm_all:
        # Both backends
        try:
            ollama_model = args.ollama_model
            ollama_backend = OllamaBackend(model=ollama_model)
            backends_to_run.append((f"ollama/{ollama_model}", ollama_backend))
        except RuntimeError as e:
            print(f"\nOllama init failed (skipping): {e}", file=sys.stderr)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        try:
            anthropic_model = args.anthropic_model
            anthropic_backend = AnthropicBackend(api_key=api_key, model=anthropic_model)
            backends_to_run.append((f"anthropic/{anthropic_model}", anthropic_backend))
        except RuntimeError as e:
            print(f"\nAnthropic init failed (skipping): {e}", file=sys.stderr)

        if not backends_to_run:
            print("Error: No LLM backends could be initialized.", file=sys.stderr)
            sys.exit(1)

    elif args.llm:
        # Single backend
        try:
            if args.llm_backend == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                model = args.llm_model or "claude-haiku-4-5"
                backend = AnthropicBackend(api_key=api_key, model=model)
                backends_to_run.append((f"anthropic/{model}", backend))
            else:
                model = args.llm_model or "llama3.1:8b"
                backend = OllamaBackend(model=model)
                backends_to_run.append((f"ollama/{model}", backend))
        except RuntimeError as e:
            print(f"\nLLM backend init failed: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Run LLM analyses ---
    if backends_to_run:
        from pathlib import Path

        cache = None
        if not args.llm_no_cache:
            today = datetime.now().strftime("%Y-%m-%d")
            cache = LLMCache(date_str=today)

        dump_dir = Path(args.llm_dump) if args.llm_dump else None

        for backend_label, backend in backends_to_run:
            print(f"\nRunning LLM analysis ({backend_label})...")

            if args.llm_raw:
                import json
                is_patch = activity_type == ActivityType.PATCH_SUBMITTED
                thread_text = _build_thread_text(thread_messages)
                prompt = _build_analysis_prompt(thread_text, subject, is_patch)
                print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
                raw_response = backend.complete(prompt)
                print(f"\n{_BOLD}{'=' * 50}{_RESET}")
                print(f"{_BOLD}  RAW LLM RESPONSE ({backend_label}){_RESET}")
                print(f"{_BOLD}{'=' * 50}{_RESET}")
                try:
                    parsed = json.loads(raw_response.strip().strip("`").strip())
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(raw_response)
                print()

            llm_result = analyze_thread_llm(
                thread_messages, activity_item, backend, cache,
                dump_dir=dump_dir,
                force_monolithic=getattr(args, 'llm_monolithic', False),
            )
            _print_summary(llm_result, f"LLM ANALYSIS ({backend_label})")

    print()


if __name__ == "__main__":
    main()
