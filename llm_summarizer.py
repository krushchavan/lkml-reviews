"""LLM-based thread analysis: enhanced summarization using Ollama or Anthropic.

Provides a drop-in replacement for thread_analyzer.analyze_thread() that produces
richer summaries by sending thread content to an LLM and parsing structured JSON
responses.  Falls back to heuristic analysis on any failure.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from llm_cache import LLMCache
from models import (
    ActivityItem,
    ActivityType,
    ConversationSummary,
    DiscussionProgress,
    ReviewComment,
    Sentiment,
)
from thread_analyzer import (
    analyze_thread,  # heuristic fallback
    _extract_author_short,
    _determine_discussion_progress,
    _has_inline_review,
    _extract_tags_from_body,
    _is_trivial_message,
    _extract_comment_body,
    _determine_message_sentiment,
    _extract_patch_summary,
)

logger = logging.getLogger(__name__)

# Increment when prompt templates, _sentence_range(), or _scale_max_tokens() change.
# This is included in cache keys so that stale cached responses (generated with
# different prompt instructions) are automatically invalidated.
_PROMPT_VERSION = "v4"


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class LLMBackend:
    """Abstract base for LLM backends."""

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Ollama local inference via HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "llama3.1:8b",
        auto_pull: bool = True,
    ):
        if base_url is None:
            base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.auto_pull = auto_pull
        self._ensure_model()

    def _ensure_model(self):
        """Check if model exists; pull it if auto_pull is enabled."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check exact name and base name (without tag)
            if self.model in model_names:
                return
            base_name = self.model.split(":")[0]
            if any(n.startswith(base_name) for n in model_names):
                return
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running?\n"
                "  Install: https://ollama.ai\n"
                "  Start:   ollama serve"
            )
        except Exception as e:
            logger.warning("Could not check Ollama models: %s", e)
            return

        if not self.auto_pull:
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama. "
                f"Run: ollama pull {self.model}"
            )

        logger.info("Pulling model '%s' from Ollama (first run, may take a few minutes)...", self.model)
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=600,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "verifying" in status or "writing" in status:
                        logger.info("  Ollama: %s", status)
            logger.info("Model '%s' pulled successfully.", self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{self.model}': {e}")

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        # Scale timeout based on prompt size — large models on CPU need
        # significant time just for prompt evaluation before the first token.
        # A 27B model on CPU can take 10-30 minutes to evaluate a 30K prompt.
        # Base: 600s, +60s per 5K chars above 5K, hard cap 3600s (1 hour).
        prompt_len = len(prompt)
        extra_chunks = max(0, (prompt_len - 5000)) // 5000
        timeout = 600 + extra_chunks * 60
        timeout = min(timeout, 3600)  # hard cap at 1 hour
        logger.info(
            "Ollama request: model=%s, prompt=%d chars, max_tokens=%d, timeout=%ds",
            self.model, prompt_len, max_tokens, timeout,
        )

        # Use streaming to accumulate the full response reliably.
        # Even with stream=False, Ollama can send chunked HTTP responses
        # that may be truncated.  Streaming token-by-token avoids this.
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                },
            },
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()

        # Accumulate streamed response tokens
        full_response = []
        done = False
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ollama: unparseable chunk: %s", line[:200])
                continue
            token = chunk.get("response", "")
            if token:
                full_response.append(token)
            if chunk.get("done", False):
                done = True
                # Log generation stats if available
                total_duration = chunk.get("total_duration", 0)
                eval_count = chunk.get("eval_count", 0)
                if total_duration:
                    secs = total_duration / 1e9
                    tps = eval_count / secs if secs > 0 else 0
                    logger.info(
                        "Ollama done: %d tokens in %.1fs (%.1f tok/s)",
                        eval_count, secs, tps,
                    )
                break

        result = "".join(full_response)
        if not done:
            logger.warning(
                "Ollama stream ended without done=true, response may be truncated (%d chars so far)",
                len(result),
            )
        return result


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5",
        max_input_tokens: int = 8000,
    ):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic backend.\n"
                "Install it with: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_input_tokens = max_input_tokens

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > self.max_input_tokens:
            prompt = self._trim_prompt(prompt, self.max_input_tokens * 4)
            logger.debug(
                "Trimmed prompt from ~%d to ~%d estimated tokens",
                estimated_tokens,
                len(prompt) // 4,
            )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg.lower():
                raise RuntimeError(
                    f"Model '{self.model}' not found (404). "
                    f"Try a current model like 'claude-haiku-4-5' or 'claude-sonnet-4-5'. "
                    f"Full error: {error_msg}"
                )
            raise

    @staticmethod
    def _trim_prompt(prompt: str, char_limit: int) -> str:
        """Trim prompt while preserving instruction header and most recent messages."""
        marker = "--- THREAD MESSAGES ---"
        header_end = prompt.find(marker)
        if header_end == -1:
            header_end = min(500, len(prompt) // 4)
        else:
            header_end += len(marker) + 1

        header = prompt[:header_end]
        remaining_budget = char_limit - len(header) - 200
        if remaining_budget < 500:
            remaining_budget = 500
        tail = prompt[-remaining_budget:]
        return header + "\n[... earlier messages trimmed for length ...]\n" + tail


# ---------------------------------------------------------------------------
# Thread text preparation
# ---------------------------------------------------------------------------

def _build_thread_text(messages: List[Dict], max_chars: int = 30000) -> str:
    """Serialize thread messages into a text block for the LLM prompt.

    Strips quoted text and trims long messages to reduce token usage.
    If total text exceeds max_chars, keeps the first message (patch
    description) and the most recent messages, dropping the middle.

    For local models (Ollama), callers should pass a lower max_chars
    (e.g. 15000) to avoid timeouts on large threads.
    """
    parts = []
    for i, msg in enumerate(messages):
        from_field = msg.get("from", "unknown")
        date = msg.get("date", "")
        subject = msg.get("subject", "")
        body = msg.get("body", "")

        # Strip quoted lines to save tokens
        lines = body.split("\n")
        filtered_lines: List[str] = []
        for line in lines:
            if line.strip().startswith(">"):
                if not filtered_lines or filtered_lines[-1] != "[quoted text omitted]":
                    filtered_lines.append("[quoted text omitted]")
            else:
                filtered_lines.append(line)
        clean_body = "\n".join(filtered_lines)

        # Truncate extremely long individual messages
        if len(clean_body) > 3000:
            clean_body = clean_body[:2800] + "\n[... message truncated ...]"

        header = f"=== Message {i + 1} | From: {from_field} | Date: {date} ==="
        if subject:
            header += f"\nSubject: {subject}"
        parts.append(f"{header}\n{clean_body}")

    full_text = "\n\n".join(parts)

    if len(full_text) > max_chars:
        # Keep first message (patch description) and as many recent as fit
        first_msg = parts[0]
        budget = max_chars - len(first_msg) - 200
        tail_parts: List[str] = []
        tail_len = 0
        for part in reversed(parts[1:]):
            if tail_len + len(part) > budget:
                break
            tail_parts.insert(0, part)
            tail_len += len(part)
        full_text = (
            first_msg
            + "\n\n[... middle messages omitted for length ...]\n\n"
            + "\n\n".join(tail_parts)
        )

    return full_text


# ---------------------------------------------------------------------------
# Proportional sizing helpers
# ---------------------------------------------------------------------------

def _sentence_range(input_chars: int, summary_type: str) -> str:
    """Return a sentence-count instruction proportional to input size.

    Args:
        input_chars: Length of the input text being summarized.
        summary_type: "patch" or "reviewer".
    """
    if summary_type == "patch":
        if input_chars < 1000:
            return "2-3 sentence"
        elif input_chars < 2000:
            return "3-4 sentence"
        elif input_chars < 4000:
            return "4-6 sentence"
        else:
            return "5-7 sentence"
    else:  # reviewer
        if input_chars < 500:
            return "2-3 sentence"
        elif input_chars < 2000:
            return "3-5 sentence"
        elif input_chars < 4000:
            return "4-6 sentence"
        else:
            return "5-7 sentence"


def _scale_max_tokens(prompt_len: int, summary_type: str) -> int:
    """Return max_tokens proportional to prompt size.

    Args:
        prompt_len: Character length of the full prompt.
        summary_type: "patch_summary" or "reviewer".
    """
    if summary_type == "patch_summary":
        return max(400, min(1200, prompt_len // 4))
    else:  # reviewer
        return max(800, min(2048, prompt_len // 2))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_analysis_prompt(
    thread_text: str,
    subject: str,
    is_patch: bool,
) -> str:
    """Build the full analysis prompt for the LLM."""

    sentiment_values = "POSITIVE, NEEDS_WORK, CONTENTIOUS, NEUTRAL"
    progress_values = (
        "ACCEPTED, CHANGES_REQUESTED, UNDER_REVIEW, "
        "NEW_VERSION_EXPECTED, WAITING_FOR_REVIEW, SUPERSEDED, RFC"
    )

    patch_instruction = ""
    if is_patch:
        patch_sr = _sentence_range(len(thread_text), "patch")
        patch_instruction = f"""
- "patch_summary": A {patch_sr} summary of what the patch/series does technically.
  Focus on the problem it solves and the approach taken. Do NOT just repeat the subject line."""

    reviewer_sr = _sentence_range(len(thread_text), "reviewer")

    return f"""You are a senior Linux kernel developer analyzing an LKML email thread.
Your job is to produce a concise analytical summary — NOT to quote or copy text from the emails.

Thread subject: {subject}

Return a JSON object with exactly these fields:

{{
  "patch_summary": "...",
  "overall_sentiment": "one of: {sentiment_values}",
  "overall_sentiment_signals": ["signal1", "signal2"],
  "discussion_progress": "one of: {progress_values}",
  "progress_detail": "one sentence describing where things stand",
  "review_comments": [
    {{
      "author": "First Last",
      "reply_to": "Name of the person this message replies to, or empty string if replying to the patch author / root message",
      "summary": "your own {reviewer_sr} analytical summary",
      "sentiment": "one of: {sentiment_values}",
      "sentiment_signals": ["signal1"],
      "has_inline_review": true,
      "tags_given": ["Reviewed-by"]
    }}
  ]
}}

Field rules:
{patch_instruction}
- "overall_sentiment": CONTENTIOUS = strong disagreement/NACK.
  NEEDS_WORK = reviewer requested changes. POSITIVE = LGTM/applied/merged. NEUTRAL = no clear signal.
- "discussion_progress":
  ACCEPTED = applied/merged/queued by a maintainer.
  CHANGES_REQUESTED = reviewer asked for revisions, author hasn't addressed yet.
  NEW_VERSION_EXPECTED = author said they'll post updated version.
  UNDER_REVIEW = active discussion, no verdict yet.
  WAITING_FOR_REVIEW = no substantive replies yet.
  SUPERSEDED = replaced by a newer version.
  RFC = Request For Comments, not yet intended for merge.
- "review_comments": One entry per RESPONSE MESSAGE (not per reviewer). If a reviewer sends
  three separate messages in the thread, produce three entries for them, each with its own
  summary, sentiment, and reply_to. Do NOT merge multiple messages from the same author into one.
  Exclude the original patch author's root message. Include the patch author only if they post
  a substantive follow-up reply. Set has_inline_review=true if they quoted code and commented inline.
- "reply_to": The short name of the person this message is directly replying to. Use an empty
  string if the message replies to the patch root / original author. This is used to reconstruct
  the thread tree structure. Examples: "David Hildenbrand", "Zi Yan", "".
- "tags_given": Only formal kernel tags (Reviewed-by, Acked-by, Tested-by) that the
  reviewer explicitly gave in that specific message.
- Empty thread (no replies): empty review_comments, WAITING_FOR_REVIEW, NEUTRAL.
- Non-patch thread: use empty string for patch_summary.

CRITICAL RULES FOR SUMMARIES:
- Every "summary" field MUST be written in your own words as an analytical summary.
- DO NOT copy, quote, or paraphrase sentences from the emails.
- DO NOT include email headers, attribution lines ("On Mon, ... wrote:"), or greetings.
- Instead, describe WHAT the reviewer raised, WHETHER they approved or objected, and
  WHAT specific technical concerns or suggestions they made.
- Good example: "Identified a double-put bug in fuse_try_move_page() where the folio refcount drops to zero prematurely. Suggested adding an extra folio_get() before replace_page_cache_folio() to maintain the reference."
- Bad example: "On Mon, 10 Feb 2025, Vlastimil wrote: I think you're right that there is a double put..."

IMPORTANT:
- Return ONLY valid JSON. No markdown fences, no explanation text before or after.
- Use EXACTLY the field names shown above.
- All sentiment/progress values must be UPPERCASE.
- has_inline_review must be a JSON boolean (true/false), not a string.
- Your response MUST contain these top-level keys: "patch_summary", "overall_sentiment",
  "overall_sentiment_signals", "discussion_progress", "progress_detail", "review_comments".
- Do NOT use any other top-level keys such as "thread_id", "topic", "participants", "messages",
  "thread", "summary", or "analysis". Only the six keys listed above are allowed.

--- THREAD MESSAGES ---
{thread_text}

--- END OF THREAD ---
Produce your analytical JSON now. Your response MUST be a single JSON object with EXACTLY these
six top-level keys: "patch_summary", "overall_sentiment", "overall_sentiment_signals",
"discussion_progress", "progress_detail", "review_comments". No other keys. Start with {{ end with }}."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_response: str) -> Optional[Dict]:
    """Parse the LLM JSON response, handling common formatting issues.

    Handles: markdown fences, preamble/postamble text, minor key name
    variations between models (Ollama vs Anthropic), and string-booleans.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Find JSON object boundaries (skip any preamble text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    else:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON response: %s", e)
        logger.debug("Raw response (first 500 chars): %s", raw_response[:500])
        return None

    # Sanity check: the parsed result should be a flat dict with expected
    # analysis fields.  If the model returned a completely different schema
    # (e.g., a "threads" list), reject it.
    if not isinstance(parsed, dict):
        logger.warning("LLM returned non-dict JSON: %s", type(parsed).__name__)
        return None

    # Check for at least one expected field to avoid using garbage output
    expected_keys = {
        "patch_summary", "summary", "overall_sentiment", "sentiment",
        "discussion_progress", "progress", "review_comments", "reviews",
    }
    if not expected_keys.intersection(parsed.keys()):
        logger.warning(
            "LLM JSON missing expected fields. Got keys: %s",
            list(parsed.keys())[:10],
        )
        return None

    # Normalize the parsed dict to handle model-to-model variations
    normalized = _normalize_llm_output(parsed)

    # Validate value types — reject structurally broken responses where the
    # model invented its own schema (e.g. patch_summary is a dict instead of
    # a string, or sentiment values are not in the allowed set).
    _VALID_SENTIMENTS = {"POSITIVE", "NEEDS_WORK", "CONTENTIOUS", "NEUTRAL"}
    _VALID_PROGRESS = {
        "ACCEPTED", "CHANGES_REQUESTED", "UNDER_REVIEW",
        "NEW_VERSION_EXPECTED", "WAITING_FOR_REVIEW", "SUPERSEDED", "RFC",
    }

    issues = []
    if not isinstance(normalized.get("patch_summary"), str):
        issues.append(f"patch_summary is {type(normalized.get('patch_summary')).__name__}, not str")
    if normalized.get("overall_sentiment") not in _VALID_SENTIMENTS:
        issues.append(f"overall_sentiment '{normalized.get('overall_sentiment')}' not in {_VALID_SENTIMENTS}")
    if not isinstance(normalized.get("overall_sentiment_signals"), list):
        issues.append("overall_sentiment_signals is not a list")
    if normalized.get("discussion_progress") not in _VALID_PROGRESS and normalized.get("discussion_progress") != "":
        issues.append(f"discussion_progress '{normalized.get('discussion_progress')}' not in {_VALID_PROGRESS}")
    if not isinstance(normalized.get("progress_detail"), str):
        issues.append(f"progress_detail is {type(normalized.get('progress_detail')).__name__}, not str")
    if not isinstance(normalized.get("review_comments"), list):
        issues.append("review_comments is not a list")
    else:
        for i, rc in enumerate(normalized["review_comments"]):
            if not isinstance(rc, dict):
                issues.append(f"review_comments[{i}] is not a dict")
            elif not rc.get("summary"):
                issues.append(f"review_comments[{i}] (author={rc.get('author', '?')}) has empty summary")

    if issues:
        logger.warning(
            "LLM response has structural issues: %s",
            "; ".join(issues),
        )
        return None

    return normalized


def _coerce_bool(value) -> bool:
    """Convert various truthy representations to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower().strip() in ("true", "yes", "1")
    return bool(value)


def _coerce_str_list(value) -> list:
    """Ensure value is a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _clean_summary_text(text: str) -> str:
    """Clean up LLM-generated summary text that may contain quoted email artifacts.

    Smaller models (Llama) sometimes copy text from the input instead of
    summarizing. This function detects and strips common artifacts:
    - Attribution lines ("On Mon, 10 Feb 2025, X wrote:")
    - Quoted text lines starting with ">"
    - Email greetings/valedictions
    - Raw email headers
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned: list = []

    for line in lines:
        stripped = line.strip()

        # Skip attribution lines
        if re.match(r"^On\s+\w{3},?\s+\d", stripped, re.IGNORECASE):
            continue
        if re.search(r"wrote:\s*$", stripped):
            continue
        if re.search(r"<[^>]+>\s+writes?:\s*$", stripped):
            continue

        # Skip quoted text
        if stripped.startswith(">"):
            continue

        # Skip raw email headers
        if re.match(r"^(From|To|Cc|Date|Subject|Message-Id|In-Reply-To):\s", stripped):
            continue

        # Skip greetings/valedictions
        lower = stripped.lower()
        if re.match(r"^(hi|hello|hey|dear)\s", lower):
            continue
        if re.match(r"^(thanks|thank you|cheers|regards|best|sincerely)\s*[,.]?\s*$", lower):
            continue

        # Skip tag-only lines
        if re.match(
            r"^(Acked-by|Reviewed-by|Tested-by|Signed-off-by):\s",
            stripped,
            re.IGNORECASE,
        ):
            continue

        # Skip signature markers
        if stripped in ("--", "-- "):
            continue

        # Skip single-word name sign-offs (e.g., "Miklos", "Josef")
        if re.match(r"^[A-Z][a-z]+$", stripped) and len(stripped) < 20:
            continue

        if stripped:
            cleaned.append(stripped)

    result = " ".join(cleaned)

    # If we stripped everything, return the original (don't make it worse)
    if not result.strip():
        return text

    return result


def _normalize_llm_output(parsed: Dict) -> Dict:
    """Normalize field names and types across different LLM backends.

    Handles common variations:
    - sentiment_signal vs sentiment_signals (singular vs plural)
    - inline_review vs has_inline_review
    - string booleans ("true") vs real booleans
    - missing fields filled with safe defaults
    - nested review_comments normalization
    - Cleans up quoted-text artifacts from weaker models
    """
    # Top-level field name aliases
    result = {}

    # patch_summary — may also be "summary" or "patch_description"
    result["patch_summary"] = _clean_summary_text(str(
        parsed.get("patch_summary")
        or parsed.get("summary", "")
        or parsed.get("patch_description", "")
        or ""
    ))

    # overall_sentiment — may also be just "sentiment"
    result["overall_sentiment"] = str(
        parsed.get("overall_sentiment")
        or parsed.get("sentiment", "NEUTRAL")
        or "NEUTRAL"
    ).upper().strip()

    # overall_sentiment_signals — many possible names
    result["overall_sentiment_signals"] = _coerce_str_list(
        parsed.get("overall_sentiment_signals")
        or parsed.get("sentiment_signals")
        or parsed.get("overall_signals")
        or []
    )

    # discussion_progress — may also be just "progress" or "status"
    result["discussion_progress"] = str(
        parsed.get("discussion_progress")
        or parsed.get("progress")
        or parsed.get("status", "")
        or ""
    ).upper().strip()

    # progress_detail — may also be "progress_description" or "status_detail"
    result["progress_detail"] = _clean_summary_text(str(
        parsed.get("progress_detail")
        or parsed.get("progress_description")
        or parsed.get("status_detail", "")
        or ""
    ))

    # review_comments — normalize each entry
    raw_comments = parsed.get("review_comments") or parsed.get("reviews") or []
    if not isinstance(raw_comments, list):
        raw_comments = []

    normalized_comments = []
    for rc in raw_comments:
        if not isinstance(rc, dict):
            continue
        normalized_comments.append({
            "author": str(
                rc.get("author")
                or rc.get("reviewer")
                or rc.get("name", "Unknown")
            ),
            "reply_to": str(rc.get("reply_to") or "").strip(),
            "summary": _clean_summary_text(str(
                rc.get("summary")
                or rc.get("comment")
                or rc.get("review", "")
                or ""
            )),
            "sentiment": str(
                rc.get("sentiment", "NEUTRAL") or "NEUTRAL"
            ).upper().strip(),
            "sentiment_signals": _coerce_str_list(
                rc.get("sentiment_signals")
                or rc.get("sentiment_signal")
                or rc.get("signals")
                or []
            ),
            "has_inline_review": _coerce_bool(
                rc.get("has_inline_review")
                if rc.get("has_inline_review") is not None
                else rc.get("inline_review", False)
            ),
            "tags_given": _coerce_str_list(
                rc.get("tags_given")
                or rc.get("tags")
                or []
            ),
        })
    result["review_comments"] = normalized_comments

    return result


def _map_sentiment(value: str) -> Sentiment:
    """Map a string sentiment value to the Sentiment enum."""
    mapping = {
        "POSITIVE": Sentiment.POSITIVE,
        "NEEDS_WORK": Sentiment.NEEDS_WORK,
        "CONTENTIOUS": Sentiment.CONTENTIOUS,
        "NEUTRAL": Sentiment.NEUTRAL,
    }
    return mapping.get(value.upper().strip(), Sentiment.NEUTRAL)


def _map_progress(value: str) -> Optional[DiscussionProgress]:
    """Map a string progress value to the DiscussionProgress enum."""
    mapping = {
        "ACCEPTED": DiscussionProgress.ACCEPTED,
        "CHANGES_REQUESTED": DiscussionProgress.CHANGES_REQUESTED,
        "UNDER_REVIEW": DiscussionProgress.UNDER_REVIEW,
        "NEW_VERSION_EXPECTED": DiscussionProgress.NEW_VERSION_EXPECTED,
        "WAITING_FOR_REVIEW": DiscussionProgress.WAITING_FOR_REVIEW,
        "SUPERSEDED": DiscussionProgress.SUPERSEDED,
        "RFC": DiscussionProgress.RFC,
    }
    return mapping.get(value.upper().strip())


def _build_conversation_summary(
    parsed: Dict,
    participant_count: int,
    thread_messages: Optional[List[Dict]] = None,
) -> ConversationSummary:
    """Convert parsed LLM JSON into a ConversationSummary."""
    # Build author→raw_body and author→earliest_date lookups from thread messages
    author_bodies: Dict[str, List[str]] = {}
    author_earliest_date: Dict[str, str] = {}
    if thread_messages:
        from email.utils import parsedate
        import time as _time
        for msg in thread_messages[1:]:  # skip root
            from_field = msg.get("from", "")
            short_name = _extract_author_short(from_field)
            key = short_name.lower()
            body = msg.get("body", "")
            if body.strip():
                author_bodies.setdefault(key, []).append(body)
            raw_date = msg.get("date", "")
            if raw_date:
                try:
                    parsed_date = parsedate(raw_date)
                    if parsed_date:
                        ymd = _time.strftime("%Y-%m-%d", parsed_date)
                        if key not in author_earliest_date or ymd < author_earliest_date[key]:
                            author_earliest_date[key] = ymd
                except Exception:
                    pass

    review_comments = []
    for rc_data in parsed.get("review_comments", []):
        author = rc_data.get("author", "Unknown")
        # Prefer cached raw_body (from per-reviewer cache); fall back to
        # reconstructing from thread messages (monolithic cache or first run).
        raw_body = rc_data.get("raw_body", "")
        if not raw_body:
            raw_bodies = author_bodies.get(author.lower(), [])
            raw_body = "\n\n---\n\n".join(raw_bodies) if raw_bodies else ""

        review_comments.append(ReviewComment(
            author=author,
            summary=rc_data.get("summary", ""),
            sentiment=_map_sentiment(rc_data.get("sentiment", "NEUTRAL")),
            sentiment_signals=rc_data.get("sentiment_signals", []),
            has_inline_review=bool(rc_data.get("has_inline_review", False)),
            tags_given=rc_data.get("tags_given", []),
            analysis_source="llm",
            raw_body=raw_body,
            reply_to=rc_data.get("reply_to", ""),
            message_date=rc_data.get("message_date", "") or author_earliest_date.get(author.lower(), ""),
        ))

    overall_sentiment = _map_sentiment(
        parsed.get("overall_sentiment", "NEUTRAL")
    )
    progress = _map_progress(
        parsed.get("discussion_progress", "")
    )

    return ConversationSummary(
        participant_count=participant_count,
        key_points=[],  # LLM mode uses review_comments instead
        sentiment=overall_sentiment,
        sentiment_signals=parsed.get("overall_sentiment_signals", []),
        patch_summary=parsed.get("patch_summary", ""),
        discussion_progress=progress,
        progress_detail=parsed.get("progress_detail", ""),
        review_comments=review_comments,
        analysis_source="llm",
    )


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------

def _compute_cache_key(
    message_id: str,
    messages: List[Dict],
    backend: Optional[LLMBackend] = None,
) -> str:
    """Compute a cache key from message_id + content hash + backend/model.

    The hash includes:
    - The count and message-ids of all thread messages, so the cache is
      invalidated if the thread grows (new replies).
    - The backend type and model name, so switching between e.g.
      ollama/llama3.1:8b and anthropic/claude-haiku-4-5 produces separate
      cache entries.
    """
    # Identify backend + model so different LLMs don't share cached results
    if isinstance(backend, OllamaBackend):
        backend_tag = f"ollama:{backend.model}"
    elif isinstance(backend, AnthropicBackend):
        backend_tag = f"anthropic:{backend.model}"
    elif backend is not None:
        backend_tag = type(backend).__name__
    else:
        backend_tag = "unknown"

    content_parts = [message_id, backend_tag, _PROMPT_VERSION, str(len(messages))]
    for msg in messages:
        content_parts.append(msg.get("message_id", ""))
    content_str = "|".join(content_parts)
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    return f"{message_id}_{content_hash}"


def _count_participants_simple(messages: List[Dict]) -> int:
    """Count unique participants by email address."""
    participants = set()
    for msg in messages:
        from_field = msg.get("from", "").lower()
        match = re.search(r"<([^>]+)>", from_field)
        if match:
            participants.add(match.group(1))
        else:
            participants.add(from_field.strip())
    return len(participants)


# ---------------------------------------------------------------------------
# Per-reviewer decomposition (for Ollama on large threads)
# ---------------------------------------------------------------------------

def _extract_email(from_field: str) -> str:
    """Extract lowercase email from a 'Name <email>' field."""
    match = re.search(r"<([^>]+)>", from_field.lower())
    if match:
        return match.group(1)
    return from_field.strip().lower()


def _should_use_per_reviewer_mode(
    backend: "LLMBackend",
    messages: List[Dict],
    force_monolithic: bool = False,
) -> bool:
    """Decide whether to decompose into per-reviewer prompts.

    Triggers when: Ollama backend + thread has reviewers + total body > 8K chars.
    Always false for Anthropic (handles large prompts fine).
    """
    if force_monolithic:
        return False
    if not isinstance(backend, OllamaBackend):
        return False
    if len(messages) <= 1:
        return False

    root_email = _extract_email(messages[0].get("from", ""))
    reviewer_emails = set()
    for msg in messages[1:]:
        email = _extract_email(msg.get("from", ""))
        if email and email != root_email:
            reviewer_emails.add(email)

    if len(reviewer_emails) < 1:
        return False

    total_chars = sum(len(msg.get("body", "")) for msg in messages)
    return total_chars > 8000


def _split_body_into_segments(body: str) -> List[str]:
    """Split a message body into author-only text segments.

    An inline review email typically interleaves the author's own comments
    with quoted text from the previous message (lines starting with ">").
    This function strips all quoted lines and returns the remaining runs of
    the author's own text as separate segments — one segment per contiguous
    block of non-quoted lines.

    Example input::

        > void foo() {
        > }
        I think we should rename this to bar().

        > int x = compute();
        This variable is never used — should be removed.

    Returns::

        ["I think we should rename this to bar().",
         "This variable is never used — should be removed."]

    A single segment is returned when there are no embedded quotes.
    Empty segments (whitespace only) are discarded.
    """
    segments: List[str] = []
    current: List[str] = []

    for line in body.split("\n"):
        if line.strip().startswith(">"):
            # Quoted line — flush the current author segment if non-empty
            if any(l.strip() for l in current):
                segments.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)

    # Flush final segment
    if any(l.strip() for l in current):
        segments.append("\n".join(current).strip())

    # If nothing survived (e.g. all-quoted message), return the original body
    # stripped of quote lines so the LLM still gets something useful.
    if not segments:
        plain = "\n".join(
            l for l in body.split("\n") if not l.strip().startswith(">")
        ).strip()
        if plain:
            segments = [plain]

    return segments


def _group_messages_by_reviewer(
    messages: List[Dict],
) -> Dict[str, Dict]:
    """Return one entry per author-text segment within each message.

    For each non-root message, the body is split into author-only segments
    (contiguous blocks of non-quoted text separated by ">"-prefixed quoted
    lines).  Each segment becomes its own independent entry so the LLM
    produces a focused summary for that specific point rather than a blended
    summary across all of the author's inline comments.

    Returns dict mapping a unique per-segment key -> {
        "name": "First Last",
        "email": "user@domain.com",
        "messages": [msg_dict],        # synthetic msg with segment body
        "is_author": bool,
        "reply_to_name": "David Hildenbrand",
        "message_date": "YYYY-MM-DD",  # date of the originating message
        "raw_body": str,               # the segment text (no quoted lines)
    }
    """
    from email.utils import parsedate as _parsedate
    import time as _time2

    def _msg_ymd(msg: Dict) -> str:
        raw = msg.get("date", "")
        if not raw:
            return ""
        try:
            pd = _parsedate(raw)
            return _time2.strftime("%Y-%m-%d", pd) if pd else ""
        except Exception:
            return ""

    root_email = _extract_email(messages[0].get("from", ""))

    # Build a message-id -> (email, short_name) lookup for reply-to resolution
    msgid_to_author: Dict[str, tuple] = {}
    for msg in messages:
        mid = msg.get("message_id", "").strip("<>")
        from_field = msg.get("from", "")
        if mid:
            msgid_to_author[mid] = (
                _extract_email(from_field),
                _extract_author_short(from_field),
            )

    groups: Dict[str, Dict] = {}
    for idx, msg in enumerate(messages[1:], start=1):  # Skip root message
        from_field = msg.get("from", "")
        email = _extract_email(from_field)
        name = _extract_author_short(from_field)
        ymd = _msg_ymd(msg)
        msg_id = msg.get("message_id", "").strip("<>") or str(idx)

        # Resolve who this message is replying to
        irt = msg.get("in_reply_to", "").strip("<>")
        reply_to_name = ""
        if irt and irt in msgid_to_author:
            reply_to_email, reply_to_name = msgid_to_author[irt]
            # Don't label the reply-to as the reviewer replying to themselves
            if reply_to_email == email:
                reply_to_name = ""

        # Split the body into author-only segments
        body = msg.get("body", "")
        segments = _split_body_into_segments(body)

        if len(segments) <= 1:
            # Single segment (or no splits) — one entry for the whole message
            segment_text = segments[0] if segments else body.strip()
            # Synthesise a msg dict with only the author's own text
            synthetic_msg = dict(msg)
            synthetic_msg["body"] = segment_text
            group_key = msg_id
            groups[group_key] = {
                "name": name,
                "email": email,
                "messages": [synthetic_msg],
                "is_author": (email == root_email),
                "reply_to_name": reply_to_name,
                "message_date": ymd,
                "raw_body": segment_text,
            }
        else:
            # Multiple segments — one entry per segment
            for seg_idx, segment_text in enumerate(segments):
                group_key = f"{msg_id}_seg{seg_idx}"
                synthetic_msg = dict(msg)
                synthetic_msg["body"] = segment_text
                groups[group_key] = {
                    "name": name,
                    "email": email,
                    "messages": [synthetic_msg],
                    "is_author": (email == root_email),
                    "reply_to_name": reply_to_name,
                    "message_date": ymd,
                    "raw_body": segment_text,
                }

    return groups


def _build_patch_context_text(messages: List[Dict], max_chars: int = 3000) -> str:
    """Build text from the first (root) message only, for patch context."""
    if not messages:
        return ""
    return _build_thread_text(messages[:1], max_chars=max_chars)


def _build_patch_summary_prompt(patch_text: str, subject: str) -> str:
    """Build a focused prompt for summarizing the patch description only."""
    sr = _sentence_range(len(patch_text), "patch")
    return f"""You are a senior Linux kernel developer. Summarize what this patch does.

Thread subject: {subject}

Return a JSON object with exactly one field:
{{
  "patch_summary": "{sr} technical summary of what the patch does, the problem it solves, and the approach taken."
}}

IMPORTANT:
- Return ONLY valid JSON. No markdown fences, no other text.
- Write in your own words. Do NOT copy text from the patch.

--- PATCH DESCRIPTION ---
{patch_text}

--- END ---
Return your JSON now. Start with {{ end with }}."""


def _parse_patch_summary_response(raw: str) -> Optional[str]:
    """Parse the LLM response from the patch summary prompt."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    else:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    summary = parsed.get("patch_summary", "")
    if isinstance(summary, str) and summary.strip():
        return _clean_summary_text(summary)
    return None


def _summarize_patch_series_chunked(
    messages: List[Dict],
    activity_item: ActivityItem,
    backend: "LLMBackend",
    cache: Optional[LLMCache] = None,
    dump_dir: Optional[Path] = None,
) -> Optional[str]:
    """Summarize a multi-message patch series by calling the LLM per-message.

    For a series with a cover letter + N patches, each message gets its own
    small LLM call (~3K prompt, max_tokens=512).  The individual summaries
    are combined into one overall patch summary.

    Returns the combined summary string, or None if all calls failed.
    """
    backend_label = f"{type(backend).__name__}({backend.model})"
    summaries: List[str] = []

    for i, msg in enumerate(messages):
        suffix = f"patch_msg_{i}"
        msg_cache_key = _compute_per_reviewer_cache_key(
            activity_item.message_id, messages, backend, suffix,
        )
        cached = cache.get(msg_cache_key) if cache else None

        if cached is not None:
            s = cached.get("patch_summary", "")
            if s:
                summaries.append(s)
            continue

        # Build a small prompt from this single message
        msg_text = _build_thread_text([msg], max_chars=3000)
        subject = msg.get("subject", activity_item.subject)
        prompt = _build_patch_summary_prompt(msg_text, subject)

        logger.info(
            "Patch series chunk %d/%d for %s — calling %s (%d chars)",
            i + 1, len(messages), activity_item.message_id,
            backend_label, len(prompt),
        )

        try:
            tokens = _scale_max_tokens(len(prompt), "patch_summary")
            raw = backend.complete(prompt, max_tokens=tokens)
            parsed_summary = _parse_patch_summary_response(raw)

            if dump_dir:
                _dump_llm_response(
                    dump_dir,
                    f"{activity_item.message_id}_patch_msg_{i}",
                    backend_label, raw, prompt,
                    is_error=(parsed_summary is None),
                )

            if parsed_summary:
                summaries.append(parsed_summary)
                if cache:
                    cache.put(msg_cache_key, {"patch_summary": parsed_summary})
            else:
                logger.warning(
                    "Patch series chunk %d/%d parse failed for %s",
                    i + 1, len(messages), activity_item.message_id,
                )
        except Exception as e:
            logger.warning(
                "Patch series chunk %d/%d LLM call failed for %s: %s",
                i + 1, len(messages), activity_item.message_id, e,
            )

    if not summaries:
        return None

    # For a single message (or single success), just return it directly
    if len(summaries) == 1:
        return summaries[0]

    # Combine: first summary is the series overview, rest are per-patch details
    combined = summaries[0]
    if len(summaries) > 1:
        patch_details = " ".join(summaries[1:])
        combined = f"{combined} Individual patches: {patch_details}"

    # Trim scales with number of patches — more patches = longer combined summary
    max_combined = 1000 + (len(summaries) * 500)
    if len(combined) > max_combined:
        combined = combined[:max_combined].rsplit(" ", 1)[0] + "..."

    return combined


def _build_reviewer_prompt(
    patch_context: str,
    reviewer_name: str,
    reviewer_messages: List[Dict],
    subject: str,
    reply_to_name: str = "",
    segment_text: str = "",
    is_author: bool = False,
) -> str:
    """Build a focused prompt for analyzing a single reviewer comment segment.

    ``segment_text`` is the author's own text for this segment, already
    stripped of quoted (">") lines by ``_split_body_into_segments``.  When
    provided it is used directly; otherwise the messages are serialised via
    ``_build_thread_text`` (legacy fallback for callers that don't split).

    When ``is_author`` is True the prompt is reframed: instead of treating
    the text as a reviewer comment we tell the LLM this is the patch author
    responding to prior feedback, and ask it to identify what feedback is
    being addressed and how.
    """
    sentiment_values = "POSITIVE, NEEDS_WORK, CONTENTIOUS, NEUTRAL"

    # Use pre-extracted segment text when available — it contains only the
    # author's own words with no "> quoted" lines, giving the LLM a clean,
    # focused view of exactly what this reviewer said at this point in the thread.
    if segment_text:
        reviewer_text = segment_text[:5000]
        if len(segment_text) > 5000:
            reviewer_text += "\n[... truncated ...]"
    else:
        reviewer_text = _build_thread_text(reviewer_messages, max_chars=5000)

    sr = _sentence_range(len(reviewer_text), "reviewer")

    reply_context = ""
    if reply_to_name:
        reply_context = f"\nThis author is responding to feedback from: {reply_to_name}\n" if is_author else f"\nThis reviewer is replying to: {reply_to_name}\n"

    if is_author:
        return f"""You are a senior Linux kernel developer analyzing ONE specific reply from the patch AUTHOR in a review thread.

Thread subject: {subject}
Patch author: {reviewer_name}{reply_context}

The original patch description is provided for context, followed by the author's reply.
The reply text below contains ONLY the author's own words — quoted lines from prior messages have been removed.

The author is responding to reviewer feedback. Your job is to identify:
- What prior feedback or concern is being addressed (infer from context if needed)
- How the author responded: did they agree and promise a fix, push back, ask a clarifying question, or explain their reasoning?
- Whether the author's response suggests the patch will need further revision

Return a JSON object with exactly these fields:
{{
  "summary": "{sr} analytical summary of what feedback the author is addressing and how they responded",
  "sentiment": "one of: {sentiment_values}",
  "sentiment_signals": ["signal1", "signal2"],
  "has_inline_review": false,
  "tags_given": []
}}

Field rules:
- "summary": Written in YOUR OWN words. Be SPECIFIC and DETAILED — name the exact technical
  issue or concern being addressed, describe the author's specific response or explanation,
  and state whether a fix is planned. For example: instead of "author addressed a concern
  about locking", write "author acknowledged that the swapoff path needs to drop the per-vswap
  spinlock before calling try_to_unmap(), agreed to restructure in v2". Do NOT quote emails.
  Do NOT use vague phrases like "raised concerns about" or "suggested improvements to".
- "sentiment": From the author's perspective on the state of the review —
  NEEDS_WORK = author acknowledges a fix is needed. POSITIVE = author confirms the issue is
  resolved or agrees with the approach. CONTENTIOUS = author strongly pushes back or disputes
  the feedback. NEUTRAL = clarification or explanation with no clear resolution signal.
- "has_inline_review": false (the author is replying, not reviewing).
- "tags_given": Empty — patch authors do not give review tags.

IMPORTANT:
- Return ONLY valid JSON. No markdown fences, no explanation.
- All sentiment values must be UPPERCASE.
- has_inline_review must be a JSON boolean (true/false).
- The summary MUST include specific technical details from the reply text, not generic statements.

--- PATCH DESCRIPTION (for context) ---
{patch_context}

--- AUTHOR REPLY ({reviewer_name}) ---
{reviewer_text}

--- END ---
Return your JSON now. Start with {{ end with }}."""

    return f"""You are a senior Linux kernel developer analyzing ONE specific comment from a reviewer on a patch.

Thread subject: {subject}
Reviewer: {reviewer_name}{reply_context}

The original patch description is provided for context, followed by the reviewer's comment.
The comment text below contains ONLY this reviewer's own words — quoted lines have been removed.

Return a JSON object with exactly these fields:
{{
  "summary": "{sr} analytical summary of what this reviewer raised",
  "sentiment": "one of: {sentiment_values}",
  "sentiment_signals": ["signal1", "signal2"],
  "has_inline_review": true,
  "tags_given": ["Reviewed-by"]
}}

Field rules:
- "summary": Written in YOUR OWN words. Be SPECIFIC and DETAILED — name the exact technical
  issue raised, explain the reviewer's specific concern or objection, and describe any
  concrete suggestion they made. For example: instead of "reviewer raised concerns about
  locking", write "reviewer noted that vswap_free() acquires the per-vswap spinlock while
  holding the folio lock, creating a lock ordering violation with reclaim paths, and requested
  the lock be dropped before calling try_to_unmap()". Do NOT quote emails directly.
  Do NOT use vague phrases like "raised concerns about X" or "suggested improvements to Y"
  without explaining the actual substance of the concern or suggestion.
- "sentiment": CONTENTIOUS = strong disagreement/NACK. NEEDS_WORK = requested changes.
  POSITIVE = LGTM/approved. NEUTRAL = no clear signal.
- "has_inline_review": true if this comment was made in response to specific quoted code.
- "tags_given": Only formal kernel tags (Reviewed-by, Acked-by, Tested-by) that the
  reviewer explicitly gave in this comment. Empty list if none.

IMPORTANT:
- Return ONLY valid JSON. No markdown fences, no explanation.
- All sentiment values must be UPPERCASE.
- has_inline_review must be a JSON boolean (true/false).
- The summary MUST include specific technical details from the comment, not generic statements.

--- PATCH DESCRIPTION (for context) ---
{patch_context}

--- REVIEWER COMMENT ({reviewer_name}) ---
{reviewer_text}

--- END ---
Return your JSON now. Start with {{ end with }}."""


def _parse_reviewer_response(raw: str, reviewer_name: str) -> Optional[Dict]:
    """Parse a single reviewer's LLM response into a dict."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    else:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    _VALID = {"POSITIVE", "NEEDS_WORK", "CONTENTIOUS", "NEUTRAL"}

    summary = _clean_summary_text(str(
        parsed.get("summary") or parsed.get("comment") or parsed.get("review", "")
    ))
    sentiment = str(parsed.get("sentiment", "NEUTRAL")).upper().strip()
    if sentiment not in _VALID:
        sentiment = "NEUTRAL"

    if not summary:
        return None

    return {
        "author": reviewer_name,
        "summary": summary,
        "sentiment": sentiment,
        "sentiment_signals": _coerce_str_list(
            parsed.get("sentiment_signals") or parsed.get("signals") or []
        ),
        "has_inline_review": _coerce_bool(
            parsed.get("has_inline_review", False)
        ),
        "tags_given": _coerce_str_list(
            parsed.get("tags_given") or parsed.get("tags") or []
        ),
        "analysis_source": "llm",
    }


def _derive_overall_sentiment(
    reviewer_results: List[Dict],
) -> Tuple[str, List[str]]:
    """Derive overall sentiment from per-reviewer results.

    Priority: CONTENTIOUS > NEEDS_WORK > POSITIVE > NEUTRAL.
    """
    all_signals: List[str] = []
    sentiments: set = set()

    for rc in reviewer_results:
        sent = rc.get("sentiment", "NEUTRAL")
        sentiments.add(sent)
        all_signals.extend(rc.get("sentiment_signals", []))

    # Deduplicate signals preserving order
    seen: set = set()
    unique_signals: List[str] = []
    for s in all_signals:
        if s not in seen:
            seen.add(s)
            unique_signals.append(s)

    if "CONTENTIOUS" in sentiments:
        return "CONTENTIOUS", unique_signals
    if "NEEDS_WORK" in sentiments:
        return "NEEDS_WORK", unique_signals
    if "POSITIVE" in sentiments:
        return "POSITIVE", unique_signals
    return "NEUTRAL", unique_signals


def _heuristic_fallback_for_reviewer(
    reviewer_name: str,
    reviewer_msgs: List[Dict],
    is_author: bool,
    reply_to_name: str = "",
) -> Optional[Dict]:
    """Build a heuristic ReviewComment dict for a single reviewer (fallback)."""
    all_bodies: List[str] = []
    has_inline = False
    all_tags: List[str] = []

    for msg in reviewer_msgs:
        body = msg.get("body", "")
        if _is_trivial_message(body):
            all_tags.extend(_extract_tags_from_body(body))
            continue
        all_bodies.append(body)
        if _has_inline_review(body):
            has_inline = True
        all_tags.extend(_extract_tags_from_body(body))

    if not all_bodies and not all_tags:
        return None

    # Preserve raw body text
    raw_body = "\n\n---\n\n".join(all_bodies)

    # Build summary — budget scales with input size
    total_body_chars = sum(len(b) for b in all_bodies)
    comment_parts: List[str] = []
    for i, body in enumerate(all_bodies):
        budget = max(200, min(800, len(body) // 3))
        extracted = _extract_comment_body(body, max_chars=budget)
        if extracted:
            comment_parts.append(extracted)

    if comment_parts:
        combined_cap = max(300, min(2000, total_body_chars // 3))
        combined = " ".join(comment_parts)
        if len(combined) > combined_cap:
            combined = combined[:combined_cap].rsplit(" ", 1)[0] + "..."
        summary = combined
    elif all_tags:
        unique_tags = list(dict.fromkeys(all_tags))
        summary = f"Gave {', '.join(unique_tags)}"
    else:
        return None

    all_text = " ".join(all_bodies)
    sentiment, signals = _determine_message_sentiment(all_text)

    label = f"{reviewer_name} (author)" if is_author else reviewer_name
    unique_tags = list(dict.fromkeys(all_tags))

    return {
        "author": label,
        "summary": summary,
        "sentiment": sentiment.value.upper(),
        "sentiment_signals": signals,
        "has_inline_review": has_inline,
        "tags_given": unique_tags,
        "analysis_source": "heuristic",
        "raw_body": raw_body,
        "reply_to": reply_to_name,
    }


def _compute_per_reviewer_cache_key(
    message_id: str,
    messages: List[Dict],
    backend: Optional["LLMBackend"],
    suffix: str,
) -> str:
    """Cache key for a per-reviewer sub-call."""
    base_key = _compute_cache_key(message_id, messages, backend)
    return f"{base_key}_pr_{suffix}"


def _analyze_per_reviewer(
    messages: List[Dict],
    activity_item: ActivityItem,
    backend: "LLMBackend",
    cache: Optional[LLMCache] = None,
    dump_dir: Optional[Path] = None,
) -> ConversationSummary:
    """Analyze a thread by decomposing into per-reviewer LLM calls.

    1. Summarize the patch (one small LLM call)
    2. Analyze each reviewer's messages individually (N small LLM calls)
    3. Derive overall sentiment/progress programmatically
    4. Assemble into ConversationSummary

    Individual reviewer failures fall back to heuristic for that reviewer only.
    """
    is_patch = activity_item.activity_type == ActivityType.PATCH_SUBMITTED
    backend_label = f"{type(backend).__name__}({backend.model})"
    participant_count = _count_participants_simple(messages)

    # --- Step 1: Patch Summary ---
    patch_summary = ""
    if is_patch:
        ps_cache_key = _compute_per_reviewer_cache_key(
            activity_item.message_id, messages, backend, "patch_summary"
        )
        cached_ps = cache.get(ps_cache_key) if cache else None

        if cached_ps is not None:
            patch_summary = cached_ps.get("patch_summary", "")
            logger.debug("Per-reviewer cache hit for patch_summary: %s", activity_item.message_id)
        else:
            patch_context_text = _build_patch_context_text(messages)
            ps_prompt = _build_patch_summary_prompt(patch_context_text, activity_item.subject)

            logger.info(
                "Per-reviewer: calling %s for patch_summary (%d chars prompt)",
                backend_label, len(ps_prompt),
            )
            try:
                ps_tokens = _scale_max_tokens(len(ps_prompt), "patch_summary")
                raw = backend.complete(ps_prompt, max_tokens=ps_tokens)
                patch_summary = _parse_patch_summary_response(raw) or ""

                if patch_summary and cache:
                    cache.put(ps_cache_key, {"patch_summary": patch_summary})

                if dump_dir:
                    _dump_llm_response(
                        dump_dir, activity_item.message_id + "_patch_summary",
                        backend_label, raw, ps_prompt, is_error=(not patch_summary),
                    )
                logger.info("Per-reviewer: patch_summary OK (%d chars)", len(patch_summary))
            except Exception as e:
                logger.warning("Patch summary LLM call failed: %s — using heuristic", e)
                patch_summary = _extract_patch_summary(messages, activity_item.subject)

    # --- Step 2: Per-Reviewer Analysis ---
    reviewer_groups = _group_messages_by_reviewer(messages)
    patch_context_text = _build_patch_context_text(messages, max_chars=2000)

    reviewer_results: List[Dict] = []

    for group_key, group_info in reviewer_groups.items():
        reviewer_name = group_info["name"]
        reviewer_msgs = group_info["messages"]
        is_author = group_info["is_author"]
        reply_to_name = group_info.get("reply_to_name", "")
        # group_key is the message-id — use it directly for cache/dump naming
        email = group_info["email"]
        reviewer_earliest_date = group_info.get("message_date", "")
        # Make cache/dump suffix unique per message (group_key == message-id)
        cache_suffix = group_key

        # Check for substantive messages — tag-only contributors (e.g. a message
        # that is just "Reviewed-by: X") are captured via heuristic since there
        # is no prose for the LLM to analyse.
        has_substance = any(
            not _is_trivial_message(m.get("body", "")) for m in reviewer_msgs
        )
        if not has_substance:
            # Tag-only contributor — capture tags via heuristic, no LLM needed
            tags: List[str] = []
            for m in reviewer_msgs:
                tags.extend(_extract_tags_from_body(m.get("body", "")))
            if tags:
                unique_tags = list(dict.fromkeys(tags))
                label = f"{reviewer_name} (author)" if is_author else reviewer_name
                reviewer_results.append({
                    "author": label,
                    "summary": f"Gave {', '.join(unique_tags)}",
                    "sentiment": "POSITIVE",
                    "sentiment_signals": [],
                    "has_inline_review": False,
                    "tags_given": unique_tags,
                    "analysis_source": "heuristic",
                    "reply_to": reply_to_name,
                    "message_date": reviewer_earliest_date,
                })
            continue

        # Check per-reviewer cache
        rv_cache_key = _compute_per_reviewer_cache_key(
            activity_item.message_id, messages, backend, f"reviewer_{cache_suffix}"
        )
        cached_rv = cache.get(rv_cache_key) if cache else None

        if cached_rv is not None:
            reviewer_results.append(cached_rv)
            logger.debug("Per-reviewer cache hit for %s: %s", reviewer_name, activity_item.message_id)
            continue

        # The segment text is already quote-stripped (set by _group_messages_by_reviewer).
        # Use it directly as the raw_body so "Show original" shows only this author's words.
        segment_text = group_info.get("raw_body", "")
        reviewer_raw_body = segment_text or "\n\n---\n\n".join(
            m.get("body", "") for m in reviewer_msgs if m.get("body", "").strip()
        )

        # Build and send reviewer prompt
        label = f"{reviewer_name} (author)" if is_author else reviewer_name
        rv_prompt = _build_reviewer_prompt(
            patch_context_text, label, reviewer_msgs, activity_item.subject,
            reply_to_name=reply_to_name,
            segment_text=segment_text,
            is_author=is_author,
        )

        logger.info(
            "Per-reviewer: calling %s for reviewer '%s'%s (%d chars prompt, %d msgs)",
            backend_label, reviewer_name,
            f" (replying to {reply_to_name})" if reply_to_name else "",
            len(rv_prompt), len(reviewer_msgs),
        )

        try:
            rv_tokens = _scale_max_tokens(len(rv_prompt), "reviewer")
            raw = backend.complete(rv_prompt, max_tokens=rv_tokens)
            parsed = _parse_reviewer_response(raw, label)

            if dump_dir:
                safe_suffix = re.sub(r"[^\w.-]", "_", cache_suffix)[:60]
                _dump_llm_response(
                    dump_dir, f"{activity_item.message_id}_reviewer_{safe_suffix}",
                    backend_label, raw, rv_prompt, is_error=(parsed is None),
                )

            if parsed is not None:
                parsed["raw_body"] = reviewer_raw_body
                parsed["reply_to"] = reply_to_name
                parsed["message_date"] = reviewer_earliest_date
                reviewer_results.append(parsed)
                if cache:
                    cache.put(rv_cache_key, parsed)
                logger.info(
                    "Per-reviewer LLM OK: %s -> %s (%s)",
                    reviewer_name, parsed["sentiment"], activity_item.message_id,
                )
            else:
                logger.warning(
                    "Per-reviewer LLM parse failed for %s — falling back to heuristic",
                    reviewer_name,
                )
                heuristic_rc = _heuristic_fallback_for_reviewer(
                    reviewer_name, reviewer_msgs, is_author, reply_to_name=reply_to_name
                )
                if heuristic_rc:
                    heuristic_rc["message_date"] = reviewer_earliest_date
                    reviewer_results.append(heuristic_rc)

        except Exception as e:
            logger.warning(
                "Per-reviewer LLM call failed for %s: %s — falling back to heuristic",
                reviewer_name, e,
            )
            heuristic_rc = _heuristic_fallback_for_reviewer(
                reviewer_name, reviewer_msgs, is_author, reply_to_name=reply_to_name
            )
            if heuristic_rc:
                heuristic_rc["message_date"] = reviewer_earliest_date
                reviewer_results.append(heuristic_rc)

    # --- Step 3: Assemble Overall Summary ---
    overall_sentiment, overall_signals = _derive_overall_sentiment(reviewer_results)

    # Use heuristic for discussion progress (reliable, no LLM needed)
    progress, progress_detail = _determine_discussion_progress(
        messages, activity_item.subject
    )

    # Build ReviewComment objects from results
    review_comments: List[ReviewComment] = []
    for rc_data in reviewer_results:
        review_comments.append(ReviewComment(
            author=rc_data.get("author", "Unknown"),
            summary=rc_data.get("summary", ""),
            sentiment=_map_sentiment(rc_data.get("sentiment", "NEUTRAL")),
            sentiment_signals=rc_data.get("sentiment_signals", []),
            has_inline_review=bool(rc_data.get("has_inline_review", False)),
            tags_given=rc_data.get("tags_given", []),
            analysis_source=rc_data.get("analysis_source", "heuristic"),
            raw_body=rc_data.get("raw_body", ""),
            reply_to=rc_data.get("reply_to", ""),
            message_date=rc_data.get("message_date", ""),
        ))

    logger.info(
        "Per-reviewer analysis complete for %s: %d reviewers (%d LLM, %d heuristic), sentiment=%s",
        activity_item.message_id,
        len(review_comments),
        sum(1 for rc in review_comments if rc.analysis_source == "llm"),
        sum(1 for rc in review_comments if rc.analysis_source == "heuristic"),
        overall_sentiment,
    )

    return ConversationSummary(
        participant_count=participant_count,
        key_points=[],
        sentiment=_map_sentiment(overall_sentiment),
        sentiment_signals=overall_signals,
        patch_summary=patch_summary,
        discussion_progress=progress,
        progress_detail=progress_detail,
        review_comments=review_comments,
        analysis_source="llm-per-reviewer",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _dump_llm_response(
    dump_dir: Path,
    message_id: str,
    backend_label: str,
    raw_response: str,
    prompt: str,
    is_error: bool = False,
) -> Path:
    """Write the raw LLM response (and optionally the prompt) to a dump file.

    Files are named: <message_id_safe>_<backend>_<timestamp>[_ERROR].json
    Returns the path of the written file.
    """
    from datetime import datetime as _dt

    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w.-]", "_", message_id)[:80]
    safe_backend = re.sub(r"[^\w.-]", "_", backend_label)
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_ERROR" if is_error else ""
    filename = f"{safe_id}_{safe_backend}_{ts}{suffix}.json"
    dump_path = dump_dir / filename

    dump_data = {
        "message_id": message_id,
        "backend": backend_label,
        "timestamp": _dt.now().isoformat(),
        "is_error": is_error,
        "raw_response": raw_response,
        "raw_response_length": len(raw_response),
        "prompt_length": len(prompt),
        "prompt": prompt,
    }
    dump_path.write_text(
        json.dumps(dump_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dump_path


def _merge_monolithic_review_comments(
    raw_comments: List[Dict],
    thread_messages: List[Dict],
) -> List[Dict]:
    """Process per-message review comment entries from the monolithic LLM output.

    The monolithic prompt asks the LLM to produce one entry per response
    message with a ``reply_to`` field, mirroring the per-reviewer decomposition.
    This function keeps each entry as its own distinct card — we do NOT combine
    multiple messages from the same author into a single block.  Each message
    retains its own individual LLM summary, sentiment, reply_to, and date.

    Post-processing:
    1. Attaching ``raw_body`` text from the thread messages, indexed by author
       name in encounter order so the "Show original" expander works.
    2. Attaching ``message_date`` (YYYY-MM-DD) from the corresponding thread
       message so each card shows the correct send date, not the author's
       earliest date.
    3. Ensuring all required fields are present with safe defaults.

    Returns the list in the original encounter order (as the LLM produced it).
    """
    from email.utils import parsedate as _parsedate
    import time as _time2

    def _msg_ymd(msg: Dict) -> str:
        raw = msg.get("date", "")
        if not raw:
            return ""
        try:
            pd = _parsedate(raw)
            return _time2.strftime("%Y-%m-%d", pd) if pd else ""
        except Exception:
            return ""

    # Build per-author queues of (raw_body, message_date) in thread order
    # so we can pop one entry per LLM comment in encounter order.
    author_msg_queues: Dict[str, List[tuple]] = {}
    if thread_messages:
        for msg in thread_messages[1:]:  # skip root
            short = _extract_author_short(msg.get("from", ""))
            body = msg.get("body", "")
            ymd = _msg_ymd(msg)
            if body.strip():
                author_msg_queues.setdefault(short.lower(), []).append((body, ymd))

    # Keep a per-author cursor so successive entries by the same author each
    # get a different raw body and date.
    author_cursor: Dict[str, int] = {}

    result: List[Dict] = []

    for rc in raw_comments:
        if not isinstance(rc, dict):
            continue
        author = str(rc.get("author") or "Unknown").strip()
        reply_to = str(rc.get("reply_to") or "").strip()
        summary = rc.get("summary", "")
        sentiment = str(rc.get("sentiment") or "NEUTRAL").upper().strip()
        signals = list(rc.get("sentiment_signals") or [])
        inline = bool(rc.get("has_inline_review", False))
        tags = list(rc.get("tags_given") or [])

        # Attach the next available raw body + date for this author
        author_key = author.lower()
        msgs = author_msg_queues.get(author_key, [])
        cursor = author_cursor.get(author_key, 0)
        if cursor < len(msgs):
            raw_body, message_date = msgs[cursor]
        else:
            raw_body, message_date = "", ""
        author_cursor[author_key] = cursor + 1

        result.append({
            "author": author,
            "reply_to": reply_to,
            "summary": summary,
            "sentiment": sentiment,
            "sentiment_signals": signals,
            "has_inline_review": inline,
            "tags_given": tags,
            "analysis_source": rc.get("analysis_source", "llm"),
            "raw_body": raw_body,
            "message_date": message_date,
        })

    return result


def analyze_thread_llm(
    thread_messages: List[Dict],
    activity_item: ActivityItem,
    backend: LLMBackend,
    cache: Optional[LLMCache] = None,
    dump_dir: Optional[Path] = None,
    force_monolithic: bool = False,
) -> ConversationSummary:
    """Analyze a thread using an LLM backend with heuristic fallback.

    Args:
        thread_messages: List of message dicts from LKMLClient.get_thread().
        activity_item: The activity item this thread relates to.
        backend: The LLM backend to use (Ollama or Anthropic).
        cache: Optional disk cache for LLM results.
        dump_dir: If set, dump every raw LLM response to this directory.
            Parse failures are always dumped with an _ERROR suffix.
        force_monolithic: If True, skip per-reviewer decomposition and use
            the monolithic prompt even for Ollama on large threads.

    Returns:
        ConversationSummary with all analysis fields populated.
    """
    # Empty threads fall through to heuristic immediately
    if not thread_messages:
        summary = analyze_thread(thread_messages, activity_item)
        summary.analysis_source = "heuristic"
        return summary

    # Single-participant patch threads (e.g. patch series with no replies):
    # use chunked LLM calls for the patch summary, then fall through to the
    # normal per-reviewer LLM path below for any remaining messages.
    # Non-patch single-participant threads also fall through to the per-reviewer
    # path — heuristic is only used when LLM actually fails.
    participant_count = _count_participants_simple(thread_messages)
    if participant_count <= 1:
        is_patch = activity_item.activity_type == ActivityType.PATCH_SUBMITTED
        if is_patch:
            logger.info(
                "Single-participant patch %s (%d msgs) — chunked patch summary",
                activity_item.message_id, len(thread_messages),
            )
            llm_summary = _summarize_patch_series_chunked(
                thread_messages, activity_item, backend, cache, dump_dir,
            )
            if not llm_summary:
                logger.warning(
                    "All chunked patch summary calls failed for %s — keeping heuristic patch summary",
                    activity_item.message_id,
                )
            # Fall through to per-reviewer analysis below (even with 1 participant,
            # the author's own messages are worth summarising with LLM)
        else:
            logger.info(
                "Single-participant non-patch %s — proceeding with per-reviewer LLM analysis",
                activity_item.message_id,
            )

    # Check cache first — key includes backend+model so different LLMs
    # don't share cached results.
    cache_key = _compute_cache_key(activity_item.message_id, thread_messages, backend)
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("LLM cache hit for %s", activity_item.message_id)
            participant_count = _count_participants_simple(thread_messages)
            return _build_conversation_summary(cached, participant_count, thread_messages)

    # --- Per-reviewer decomposition for Ollama on large threads ---
    if _should_use_per_reviewer_mode(backend, thread_messages, force_monolithic):
        logger.info(
            "Using per-reviewer decomposition for %s (%d messages, %s)",
            activity_item.message_id, len(thread_messages),
            f"{type(backend).__name__}({backend.model})",
        )
        try:
            summary = _analyze_per_reviewer(
                thread_messages, activity_item, backend, cache, dump_dir,
            )
            # Cache the assembled result under the main cache key too
            if cache and summary.analysis_source == "llm-per-reviewer":
                assembled = {
                    "patch_summary": summary.patch_summary,
                    "overall_sentiment": summary.sentiment.value.upper(),
                    "overall_sentiment_signals": summary.sentiment_signals,
                    "discussion_progress": (
                        summary.discussion_progress.value.upper()
                        if summary.discussion_progress else ""
                    ),
                    "progress_detail": summary.progress_detail,
                    "review_comments": [
                        {
                            "author": rc.author,
                            "summary": rc.summary,
                            "sentiment": rc.sentiment.value.upper(),
                            "sentiment_signals": rc.sentiment_signals,
                            "has_inline_review": rc.has_inline_review,
                            "tags_given": rc.tags_given,
                            "raw_body": rc.raw_body,
                            "reply_to": rc.reply_to,
                            "message_date": rc.message_date,
                        }
                        for rc in summary.review_comments
                    ],
                }
                cache.put(cache_key, assembled)
            return summary
        except Exception as e:
            logger.error(
                "Per-reviewer analysis failed for %s: %s — trying monolithic fallback",
                activity_item.message_id, e,
            )
            # Fall through to the existing monolithic approach below

    is_patch = activity_item.activity_type == ActivityType.PATCH_SUBMITTED
    backend_label = f"{type(backend).__name__}({backend.model})"

    # Use a single context size appropriate for the backend.
    # Anthropic handles large contexts well; Ollama on CPU benefits from
    # smaller prompts for better output quality (per-reviewer mode handles
    # the normal case — this monolithic path is either Anthropic or forced).
    if isinstance(backend, OllamaBackend):
        max_thread_chars = 10000  # keep small for quality on local models
    else:
        max_thread_chars = 30000  # Anthropic/cloud handles large prompts fine

    try:
        thread_text = _build_thread_text(thread_messages, max_chars=max_thread_chars)
        prompt = _build_analysis_prompt(thread_text, activity_item.subject, is_patch)

        logger.info(
            "Calling %s for %s (monolithic, %d chars prompt, %d char context)",
            backend_label, activity_item.message_id,
            len(prompt), max_thread_chars,
        )
        raw_response = backend.complete(prompt)
        logger.info(
            "%s responded with %d chars for %s",
            backend_label, len(raw_response), activity_item.message_id,
        )
        parsed = _parse_llm_response(raw_response)

        if parsed is None:
            # Dump the failed response
            error_dump_dir = dump_dir or Path("logs/llm_dumps")
            dump_path = _dump_llm_response(
                error_dump_dir, activity_item.message_id,
                backend_label, raw_response, prompt, is_error=True,
            )
            logger.warning(
                "LLM returned unparseable response for %s — "
                "FALLING BACK TO HEURISTIC. Dump: %s",
                activity_item.message_id, dump_path,
            )
            summary = analyze_thread(thread_messages, activity_item)
            summary.analysis_source = "llm-fallback-heuristic"
            return summary

        # Dump successful response if dump_dir is explicitly set
        if dump_dir:
            dump_path = _dump_llm_response(
                dump_dir, activity_item.message_id,
                backend_label, raw_response, prompt, is_error=False,
            )
            logger.debug("LLM response dumped to: %s", dump_path)

        # Merge per-message entries into per-(author, reply_to) blocks,
        # mirroring the tree structure produced by per-reviewer mode.
        merged_comments = _merge_monolithic_review_comments(
            parsed.get("review_comments", []), thread_messages,
        )
        parsed["review_comments"] = merged_comments

        # Override overall sentiment/progress with heuristic derivations so
        # that both code paths behave consistently:
        #   - overall_sentiment: worst-case across merged reviewer sentiments
        #   - discussion_progress: heuristic (reliable, no LLM needed)
        overall_sentiment, overall_signals = _derive_overall_sentiment(merged_comments)
        parsed["overall_sentiment"] = overall_sentiment
        parsed["overall_sentiment_signals"] = overall_signals
        heuristic_progress, heuristic_progress_detail = _determine_discussion_progress(
            thread_messages, activity_item.subject
        )
        parsed["discussion_progress"] = (
            heuristic_progress.value.upper() if heuristic_progress else ""
        )
        parsed["progress_detail"] = heuristic_progress_detail

        # Cache the merged+normalised result
        if cache:
            cache.put(cache_key, parsed)

        participant_count = _count_participants_simple(thread_messages)
        summary = _build_conversation_summary(parsed, participant_count, thread_messages)
        logger.info(
            "LLM analysis complete for %s: sentiment=%s, progress=%s, %d review blocks",
            activity_item.message_id,
            summary.sentiment.value,
            summary.discussion_progress.value if summary.discussion_progress else "none",
            len(summary.review_comments),
        )
        return summary

    except Exception as e:
        logger.error(
            "%s call FAILED for %s: %s — FALLING BACK TO HEURISTIC",
            backend_label, activity_item.message_id, e,
        )
        summary = analyze_thread(thread_messages, activity_item)
        summary.analysis_source = "llm-fallback-heuristic"
        return summary
