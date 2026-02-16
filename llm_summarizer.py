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
from thread_analyzer import analyze_thread  # heuristic fallback

logger = logging.getLogger(__name__)


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

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        # Scale timeout based on prompt size — local models need more time
        # for large inputs.  Rough estimates for 8B models on consumer GPUs:
        #   - Prefill: ~500-1000 tok/s → 5K chars ≈ 1-3s
        #   - Generation: ~20-50 tok/s → 2048 tokens ≈ 40-100s
        # Base: 180s, +30s per 5K chars of prompt above 5K.
        prompt_len = len(prompt)
        extra_chunks = max(0, (prompt_len - 5000)) // 5000
        timeout = 180 + extra_chunks * 30
        timeout = min(timeout, 600)  # hard cap at 10 minutes
        logger.debug(
            "Ollama request: model=%s, prompt=%d chars, timeout=%ds",
            self.model, prompt_len, timeout,
        )

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "format": "json",  # Force JSON output mode
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


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
        patch_instruction = """
- "patch_summary": A 2-4 sentence summary of what the patch/series does technically.
  Focus on the problem it solves and the approach taken. Do NOT just repeat the subject line."""

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
      "summary": "your own 2-3 sentence analytical summary",
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
- "review_comments": One entry per reviewer (NOT the patch author, unless they posted
  substantive follow-up). Set has_inline_review=true if they quoted code and commented inline.
- "tags_given": Only formal kernel tags (Reviewed-by, Acked-by, Tested-by) that the
  reviewer explicitly gave.
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

--- THREAD MESSAGES ---
{thread_text}

--- END OF THREAD ---
REMINDER: Respond with ONLY a JSON object. No explanatory text. Start with {{ and end with }}."""


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
    return _normalize_llm_output(parsed)


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
) -> ConversationSummary:
    """Convert parsed LLM JSON into a ConversationSummary."""
    review_comments = []
    for rc_data in parsed.get("review_comments", []):
        review_comments.append(ReviewComment(
            author=rc_data.get("author", "Unknown"),
            summary=rc_data.get("summary", ""),
            sentiment=_map_sentiment(rc_data.get("sentiment", "NEUTRAL")),
            sentiment_signals=rc_data.get("sentiment_signals", []),
            has_inline_review=bool(rc_data.get("has_inline_review", False)),
            tags_given=rc_data.get("tags_given", []),
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

    content_parts = [message_id, backend_tag, str(len(messages))]
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
# Main entry point
# ---------------------------------------------------------------------------

def analyze_thread_llm(
    thread_messages: List[Dict],
    activity_item: ActivityItem,
    backend: LLMBackend,
    cache: Optional[LLMCache] = None,
) -> ConversationSummary:
    """Analyze a thread using an LLM backend with heuristic fallback.

    Args:
        thread_messages: List of message dicts from LKMLClient.get_thread().
        activity_item: The activity item this thread relates to.
        backend: The LLM backend to use (Ollama or Anthropic).
        cache: Optional disk cache for LLM results.

    Returns:
        ConversationSummary with all analysis fields populated.
    """
    # Empty threads fall through to heuristic immediately
    if not thread_messages:
        return analyze_thread(thread_messages, activity_item)

    # Check cache first — key includes backend+model so different LLMs
    # don't share cached results.
    cache_key = _compute_cache_key(activity_item.message_id, thread_messages, backend)
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("LLM cache hit for %s", activity_item.message_id)
            participant_count = _count_participants_simple(thread_messages)
            return _build_conversation_summary(cached, participant_count)

    # Build prompt — use smaller context window for local models to avoid
    # timeouts.  Ollama models typically have 8K-128K context but generate
    # slowly on large inputs; Anthropic handles 200K easily.
    is_patch = activity_item.activity_type == ActivityType.PATCH_SUBMITTED
    max_thread_chars = 12000 if isinstance(backend, OllamaBackend) else 30000
    thread_text = _build_thread_text(thread_messages, max_chars=max_thread_chars)
    prompt = _build_analysis_prompt(thread_text, activity_item.subject, is_patch)

    try:
        logger.debug("Calling LLM for %s (%d chars prompt)", activity_item.message_id, len(prompt))
        raw_response = backend.complete(prompt)
        parsed = _parse_llm_response(raw_response)

        if parsed is None:
            logger.warning(
                "LLM returned unparseable response for %s, falling back to heuristic",
                activity_item.message_id,
            )
            return analyze_thread(thread_messages, activity_item)

        # Cache the parsed result
        if cache:
            cache.put(cache_key, parsed)

        participant_count = _count_participants_simple(thread_messages)
        summary = _build_conversation_summary(parsed, participant_count)
        logger.debug(
            "LLM analysis complete for %s: sentiment=%s, progress=%s, %d reviews",
            activity_item.message_id,
            summary.sentiment.value,
            summary.discussion_progress.value if summary.discussion_progress else "none",
            len(summary.review_comments),
        )
        return summary

    except Exception as e:
        logger.warning(
            "LLM call failed for %s: %s — falling back to heuristic",
            activity_item.message_id,
            e,
        )
        return analyze_thread(thread_messages, activity_item)
