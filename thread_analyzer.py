"""Thread analysis: conversation summaries, patch summaries, and discussion progress."""

import logging
import re
from typing import Dict, List, Optional, Tuple

from models import (
    ActivityItem,
    ActivityType,
    ConversationSummary,
    DiscussionProgress,
    ReviewComment,
    Sentiment,
)

logger = logging.getLogger(__name__)

# Sentiment patterns, checked in priority order
_CONTENTIOUS_SIGNALS = [
    (r"\bNACK\b", "NACK"),
    (r"\bNAK\b", "NAK"),
    (r"\bI\s+disagree\b", "disagreement"),
    (r"\bstrongly\s+disagree\b", "strong disagreement"),
    (r"\bthis\s+is\s+wrong\b", "this is wrong"),
    (r"\bfundamentally\s+(broken|flawed|wrong)\b", "fundamentally flawed"),
    (r"\bI\s+don'?t\s+think\s+this\s+(is\s+)?correct\b", "not correct"),
]

_NEEDS_WORK_SIGNALS = [
    (r"\bv\d+\s+needed\b", "new version needed"),
    (r"\bplease\s+(fix|rework|update|revise|resend)\b", "please fix"),
    (r"\bneeds?\s+(rework|changes?|fixing|update)\b", "needs rework"),
    (r"\bsome\s+concerns?\b", "some concerns"),
    (r"\bnit[s:,]\s", "nits"),
    (r"\bminor\s+(issue|comment|nit|suggestion)\b", "minor issues"),
    (r"\bcould\s+you\s+(please\s+)?(change|fix|update)\b", "change requested"),
    (r"\bI\s+would\s+prefer\b", "preference expressed"),
    (r"\bnot\s+quite\s+right\b", "not quite right"),
]

_POSITIVE_SIGNALS = [
    (r"\bapplied.*to\s+\w+", "applied"),
    (r"\bmerged\b", "merged"),
    (r"\bqueued\b", "queued"),
    (r"\blooks\s+good\s+to\s+me\b", "LGTM"),
    (r"\bLGTM\b", "LGTM"),
    (r"\bthanks,?\s*applied\b", "applied"),
    (r"\bpulled\s+into\b", "pulled"),
]


def _determine_sentiment(messages: List[Dict]) -> Tuple[Sentiment, List[str]]:
    """Heuristic sentiment analysis based on keywords across all thread messages.

    Returns:
        (Sentiment enum, list of human-readable signal descriptions).
    """
    all_bodies = " ".join(m.get("body", "") for m in messages)
    signals: List[str] = []

    # Check contentious first (strongest negative)
    for pattern, label in _CONTENTIOUS_SIGNALS:
        if re.search(pattern, all_bodies, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.CONTENTIOUS, signals

    # Check needs-work
    for pattern, label in _NEEDS_WORK_SIGNALS:
        if re.search(pattern, all_bodies, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.NEEDS_WORK, signals

    # Check positive
    for pattern, label in _POSITIVE_SIGNALS:
        if re.search(pattern, all_bodies, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.POSITIVE, signals

    return Sentiment.NEUTRAL, []


def _decode_mime_header(text: str) -> str:
    """Decode MIME-encoded header values like =?UTF-8?B?...?= or =?UTF-8?Q?...?=."""
    from email.header import decode_header
    try:
        parts = decode_header(text)
        decoded = []
        for data, charset in parts:
            if isinstance(data, bytes):
                decoded.append(data.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(data)
        return " ".join(decoded)
    except Exception:
        return text


def _extract_author_short(from_field: str) -> str:
    """Extract a short name from a From header like 'Linus Torvalds <torvalds@...>'."""
    # Decode MIME-encoded names first
    if "=?" in from_field:
        from_field = _decode_mime_header(from_field)

    # Try to get the name portion before <email>
    match = re.match(r"^(.+?)\s*<", from_field)
    if match:
        name = match.group(1).strip().strip('"')
        # Return first + last name if available
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[-1]}"
        return name
    # Fallback: use the email local part
    match = re.match(r"<?([^@>]+)", from_field)
    if match:
        return match.group(1)
    return from_field[:20]


def _is_trivial_message(body: str) -> bool:
    """Check if a message is just a tag line with no substantial content."""
    lines = [l.strip() for l in body.split("\n") if l.strip() and not l.strip().startswith(">")]
    # Filter out signature lines
    content_lines = []
    for line in lines:
        if line == "--" or line == "-- ":
            break
        content_lines.append(line)

    # If the only non-empty content is tag lines, it's trivial
    tag_pattern = re.compile(
        r"^(Acked-by|Reviewed-by|Tested-by|Signed-off-by|Cc|Reported-by|Suggested-by|Fixes):\s*",
        re.IGNORECASE,
    )
    non_tag_lines = [l for l in content_lines if not tag_pattern.match(l)]
    return len(non_tag_lines) <= 1


def _extract_first_sentence(body: str) -> str:
    """Extract the first meaningful sentence from non-quoted body text.

    Skips attribution lines, greetings, tag lines, and quoted text.
    """
    lines = body.split("\n")
    content: List[str] = []
    in_attribution = False

    for line in lines:
        stripped = line.strip()
        if stripped == "--" or stripped == "-- ":
            break
        if stripped.startswith(">"):
            continue
        # Skip tag lines
        if re.match(
            r"^(Acked-by|Reviewed-by|Tested-by|Signed-off-by|Cc|Reported-by|Suggested-by|Fixes):\s*",
            stripped,
            re.IGNORECASE,
        ):
            continue
        # Skip attribution lines
        if _is_attribution_line(stripped):
            in_attribution = True
            continue
        if in_attribution:
            if re.search(r"wrote:\s*$", stripped):
                in_attribution = False
                continue
            if not stripped:
                in_attribution = False
            continue
        # Skip greetings / valedictions
        if _is_greeting_or_valediction(stripped):
            continue
        if stripped:
            content.append(stripped)

    if not content:
        return ""

    # Join and take first sentence
    text = " ".join(content[:3])
    # Split on sentence boundaries
    match = re.match(r"^(.+?[.!?])\s", text)
    if match:
        return match.group(1)
    # If no sentence boundary, truncate
    if len(text) > 150:
        return text[:147] + "..."
    return text


def _extract_key_points(messages: List[Dict], max_points: int = 5) -> List[str]:
    """Extract key discussion points from thread messages.

    Prioritizes decision messages (applied, merged, queued) and
    non-trivial discussion contributions.
    """
    # Decision patterns to prioritize
    decision_patterns = [
        r"\bapplied\b",
        r"\bmerged\b",
        r"\bqueued\b",
        r"\bwill\s+fix\s+in\s+v\d+",
        r"\bwill\s+resend\b",
        r"\bdropped\b",
    ]

    decision_points: List[str] = []
    discussion_points: List[str] = []

    for msg in messages:
        body = msg.get("body", "")
        from_field = msg.get("from", "")

        if _is_trivial_message(body):
            continue

        author = _extract_author_short(from_field)
        sentence = _extract_first_sentence(body)
        if not sentence:
            continue

        point = f"{author}: {sentence}"

        # Check if this is a decision message
        is_decision = any(
            re.search(p, body, re.IGNORECASE) for p in decision_patterns
        )
        if is_decision:
            decision_points.append(point)
        else:
            discussion_points.append(point)

    # Combine: decisions first, then discussion, up to max_points
    result = decision_points[:max_points]
    remaining = max_points - len(result)
    if remaining > 0:
        result.extend(discussion_points[:remaining])

    return result


def _count_participants(messages: List[Dict]) -> int:
    """Count unique participants in a thread by email address."""
    participants = set()
    for msg in messages:
        from_field = msg.get("from", "").lower()
        # Extract email
        match = re.search(r"<([^>]+)>", from_field)
        if match:
            participants.add(match.group(1))
        else:
            participants.add(from_field.strip())
    return len(participants)


def _extract_description_lines(body: str) -> List[str]:
    """Extract description lines from a kernel patch/cover-letter body.

    Stops at diff separator (---), diffstat, actual diff, or signature.
    Skips tag lines, quoted text, and mail headers.
    """
    lines = body.split("\n")
    description_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Stop at diff separator
        if stripped == "---":
            break
        # Stop at diffstat
        if re.match(r"^\s*\d+\s+files?\s+changed", stripped):
            break
        # Stop at actual diff lines
        if stripped.startswith("diff --git"):
            break
        # Stop at signature
        if stripped in ("--", "-- "):
            break

        # Skip tag lines (Signed-off-by, Cc, etc.)
        if re.match(
            r"^(Acked-by|Reviewed-by|Tested-by|Signed-off-by|Cc|Reported-by"
            r"|Suggested-by|Fixes|Link|Message-Id|Precedence|List-Id"
            r"|X-Mailer|MIME-Version|Content-Type|Content-Transfer):",
            stripped,
            re.IGNORECASE,
        ):
            continue

        # Skip quoted text
        if stripped.startswith(">"):
            continue

        # Preserve blank lines as paragraph separators
        if not stripped:
            if description_lines and description_lines[-1] != "":
                description_lines.append("")
            continue

        description_lines.append(stripped)

    # Trim trailing blank entries
    while description_lines and description_lines[-1] == "":
        description_lines.pop()

    return description_lines


def _extract_patch_summary(messages: List[Dict], subject: str) -> str:
    """Extract a detailed summary of what the patch does.

    Looks at the cover letter (0/N) first; if the series has a cover letter
    with a full description, it is preserved at length (up to ~1000 chars).
    For single patches, extracts the commit message description (up to ~600 chars).

    Returns:
        A multi-sentence summary of the patch purpose, or "" if nothing found.
    """
    if not messages:
        return ""

    # Find the root message — prefer cover letter (0/N), then first message
    # without in-reply-to
    root_msg = messages[0]
    is_cover_letter = False

    for msg in messages:
        msg_subject = msg.get("subject", "")
        if re.search(r"\b0/\d+\]", msg_subject):
            root_msg = msg
            is_cover_letter = True
            break
        if not msg.get("in-reply-to"):
            root_msg = msg
            break

    body = root_msg.get("body", "")
    if not body:
        return ""

    description_lines = _extract_description_lines(body)
    if not description_lines:
        return ""

    # Reconstruct paragraphs (blank line = paragraph break)
    paragraphs: List[str] = []
    current: List[str] = []
    for line in description_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    if not paragraphs:
        return ""

    # Strip leading subject duplication from the first paragraph
    clean_subject = re.sub(r"\[.*?\]\s*", "", subject).strip()
    if clean_subject and paragraphs[0].lower().startswith(clean_subject.lower()):
        paragraphs[0] = paragraphs[0][len(clean_subject):].strip()
        paragraphs[0] = re.sub(r"^[\s.,:;-]+", "", paragraphs[0]).strip()
        if not paragraphs[0]:
            paragraphs.pop(0)

    if not paragraphs:
        return ""

    # Summary length scales with input size — larger descriptions get
    # longer summaries.  Cover letters get a higher floor/ceiling since
    # they're written specifically to explain the whole series.
    body_len = len(body)
    if is_cover_letter:
        max_chars = max(500, min(2000, body_len // 2))
    else:
        max_chars = max(300, min(1200, body_len // 2))

    result_parts: List[str] = []
    char_count = 0
    for para in paragraphs:
        if char_count + len(para) > max_chars and result_parts:
            break
        result_parts.append(para)
        char_count += len(para) + 1

    result = "\n\n".join(result_parts)

    # Hard cap with clean truncation
    hard_limit = max_chars + 100  # small grace
    if len(result) > hard_limit:
        result = result[:hard_limit].rsplit(" ", 1)[0] + "..."

    return result


# --- Discussion Progress Detection ---

# Patterns that indicate the patch has been accepted / applied
_ACCEPTED_PATTERNS = [
    r"\bapplied\s+to\b",
    r"\bthanks,?\s*applied\b",
    r"\bmerged\b",
    r"\bqueued\s+(for|in)\b",
    r"\bpulled\s+into\b",
    r"\bpushed\s+to\b",
]

# Patterns that indicate changes are requested
_CHANGES_REQUESTED_PATTERNS = [
    r"\bplease\s+(fix|rework|update|revise|resend|change)\b",
    r"\bneeds?\s+(rework|changes?|fixing|update|revision)\b",
    r"\bnot\s+quite\s+right\b",
    r"\bcould\s+you\s+(please\s+)?(change|fix|update|use|move|rename)\b",
    r"\bI\s+would\s+prefer\b",
    r"\bplease\s+address\b",
]

# Patterns that indicate a new version is expected
_NEW_VERSION_PATTERNS = [
    r"\bwill\s+(send|resend|submit|post)\s+(a\s+)?(v\d+|new\s+version|updated)",
    r"\bwill\s+fix\s+(this\s+)?in\s+v\d+",
    r"\bv\d+\s+(coming|incoming|on\s+the\s+way)",
    r"\bwill\s+rework\b",
    r"\bwill\s+address\b",
    r"\bwill\s+update\b",
]

# Patterns indicating the series is superseded
_SUPERSEDED_PATTERNS = [
    r"\bsuperseded\s+by\b",
    r"\breplaced\s+by\b",
    r"\bobsoleted?\s+by\b",
    r"\bplease\s+(ignore|disregard)\s+(this|the\s+previous)\b",
]


def _determine_discussion_progress(
    messages: List[Dict], subject: str
) -> Tuple[Optional[DiscussionProgress], str]:
    """Determine where the discussion stands.

    Examines thread messages in reverse chronological order to find the
    most recent state-indicating signal.

    Returns:
        (DiscussionProgress or None, human-readable progress detail string)
    """
    if not messages:
        return None, ""

    # Detect RFC patches: [RFC PATCH ...] or plain [RFC] in subject.
    # RFC status is preserved unless the thread shows a terminal outcome
    # (ACCEPTED or SUPERSEDED). Receiving review feedback is expected for RFCs
    # and should NOT demote them to UNDER_REVIEW/CHANGES_REQUESTED.
    is_rfc_patch = bool(re.search(r"\[RFC(?:\s+PATCH)?\b", subject, re.IGNORECASE))

    # Count substantive replies (not the author's own patch messages)
    root_from = messages[0].get("from", "").lower() if messages else ""
    root_email = ""
    email_match = re.search(r"<([^>]+)>", root_from)
    if email_match:
        root_email = email_match.group(1).lower()

    reply_count = 0
    has_ack_or_review_tag = False
    ack_tags: List[str] = []

    for msg in messages[1:]:  # Skip root message
        msg_from = msg.get("from", "").lower()
        msg_email_match = re.search(r"<([^>]+)>", msg_from)
        msg_email = msg_email_match.group(1).lower() if msg_email_match else msg_from

        # Count replies from others
        if msg_email != root_email:
            reply_count += 1

        body = msg.get("body", "")
        # Check for formal review tags
        for line in body.split("\n"):
            stripped = line.strip()
            tag_match = re.match(
                r"^(Acked-by|Reviewed-by|Tested-by):\s*(.+)",
                stripped,
                re.IGNORECASE,
            )
            if tag_match:
                has_ack_or_review_tag = True
                tag_type = tag_match.group(1)
                tag_person = _extract_author_short(tag_match.group(2))
                ack_tags.append(f"{tag_type} from {tag_person}")

    # Analyze messages in reverse order for most recent state signals
    # (latest messages carry the most weight for progress)
    reversed_msgs = list(reversed(messages))

    for msg in reversed_msgs:
        body = msg.get("body", "")
        from_field = msg.get("from", "")
        author = _extract_author_short(from_field)

        # Check accepted — terminal: overrides RFC status
        for pattern in _ACCEPTED_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                return (
                    DiscussionProgress.ACCEPTED,
                    f"{author} applied/merged the patch",
                )

        # Check superseded — terminal: overrides RFC status
        for pattern in _SUPERSEDED_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                return (
                    DiscussionProgress.SUPERSEDED,
                    f"Superseded by a newer version",
                )

        # Check new version expected (author saying they'll resend)
        msg_from = msg.get("from", "").lower()
        msg_email_match = re.search(r"<([^>]+)>", msg_from)
        msg_email = msg_email_match.group(1).lower() if msg_email_match else msg_from
        is_author = msg_email == root_email

        if is_author:
            for pattern in _NEW_VERSION_PATTERNS:
                if re.search(pattern, body, re.IGNORECASE):
                    return (
                        DiscussionProgress.NEW_VERSION_EXPECTED,
                        f"Author will post an updated version",
                    )

        # Check changes requested (from reviewers).
        # For RFC patches, reviewer feedback is part of the RFC process —
        # the RFC label takes precedence and will be applied below.
        if not is_author and not is_rfc_patch:
            for pattern in _CHANGES_REQUESTED_PATTERNS:
                if re.search(pattern, body, re.IGNORECASE):
                    return (
                        DiscussionProgress.CHANGES_REQUESTED,
                        f"{author} requested changes",
                    )

    # No terminal signal found — use structural heuristics.
    # RFC patches always retain RFC status at this point.
    if is_rfc_patch:
        if reply_count == 0:
            return DiscussionProgress.RFC, "RFC posted, awaiting feedback"
        if has_ack_or_review_tag:
            tags_str = "; ".join(ack_tags[:2])
            return DiscussionProgress.RFC, f"RFC receiving feedback: {tags_str}"
        return (
            DiscussionProgress.RFC,
            f"RFC under discussion ({reply_count} "
            f"{'reply' if reply_count == 1 else 'replies'})",
        )

    if reply_count == 0:
        return (
            DiscussionProgress.WAITING_FOR_REVIEW,
            "Posted, no replies yet",
        )

    if has_ack_or_review_tag:
        tags_str = "; ".join(ack_tags[:3])
        if len(ack_tags) > 3:
            tags_str += f" (+{len(ack_tags) - 3} more)"
        return (
            DiscussionProgress.UNDER_REVIEW,
            f"Under review with tags: {tags_str}",
        )

    return (
        DiscussionProgress.UNDER_REVIEW,
        f"Under review ({reply_count} {'reply' if reply_count == 1 else 'replies'} from "
        f"{_count_participants(messages) - 1} reviewer{'s' if _count_participants(messages) > 2 else ''})",
    )


def _is_attribution_line(line: str) -> bool:
    """Check if a line is an email attribution like 'On Mon, ... wrote:'."""
    # "On <date>, <name> wrote:" — often spans one or two lines
    if re.match(r"^On\s+\w{3},?\s+\d", line, re.IGNORECASE):
        return True
    # Continuation of attribution ending with "wrote:"
    if re.search(r"wrote:\s*$", line):
        return True
    # Some mailers: "John Doe <email> writes:"
    if re.search(r"<[^>]+>\s+writes?:\s*$", line):
        return True
    return False


def _is_greeting_or_valediction(line: str) -> bool:
    """Check if a line is a greeting, thanks, or sign-off."""
    lower = line.lower().strip()
    # Greetings
    if re.match(r"^(hi|hello|hey|dear)\b", lower):
        return True
    # Valedictions and short pleasantries
    if re.match(
        r"^(thanks|thank you|cheers|regards|best|sincerely|br|rgds|thx)\b",
        lower,
    ):
        return True
    # Single-word name sign-offs (e.g., "Miklos", "Josef")
    if re.match(r"^[A-Z][a-z]+$", line.strip()) and len(line.strip()) < 20:
        return True
    return False


def _extract_comment_body(body: str, max_chars: int = 500) -> str:
    """Extract the substantive content from a reply message.

    Strips attribution lines ('On ... wrote:'), greetings, valedictions,
    tag lines, and quoted text. Returns the reviewer's own commentary.
    """
    lines = body.split("\n")
    segments: List[str] = []
    current_segment: List[str] = []
    prev_was_quote = False
    in_attribution = False

    for line in lines:
        stripped = line.strip()

        # Stop at signature
        if stripped in ("--", "-- "):
            break

        # Skip tag lines
        if re.match(
            r"^(Acked-by|Reviewed-by|Tested-by|Signed-off-by|Cc|Reported-by"
            r"|Suggested-by|Fixes|Link|Message-Id):",
            stripped,
            re.IGNORECASE,
        ):
            continue

        # Skip attribution lines ("On Mon, 10 Feb 2025, ... wrote:")
        if _is_attribution_line(stripped):
            in_attribution = True
            continue
        if in_attribution:
            # Attribution can span multiple lines; ends at "wrote:" or a quote
            if stripped.startswith(">") or not stripped:
                in_attribution = False
            elif re.search(r"wrote:\s*$", stripped):
                in_attribution = False
                continue
            else:
                continue

        # Skip greetings / valedictions
        if _is_greeting_or_valediction(stripped):
            continue

        is_quote = stripped.startswith(">")

        if is_quote:
            # If we had non-quoted text, save it
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []
            prev_was_quote = True
        else:
            if stripped:
                current_segment.append(stripped)
            elif current_segment:
                # Blank line ends a segment
                segments.append(" ".join(current_segment))
                current_segment = []
            prev_was_quote = False

    if current_segment:
        segments.append(" ".join(current_segment))

    if not segments:
        return ""

    # Filter out very short fragments (< 10 chars) that are likely noise
    segments = [s for s in segments if len(s) >= 10]

    if not segments:
        return ""

    # Join segments with sentence separators
    result = " ".join(segments)
    if len(result) > max_chars:
        # Try to cut at sentence boundary
        truncated = result[:max_chars]
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("?")
        last_excl = truncated.rfind("!")
        best_break = max(last_period, last_question, last_excl)
        if best_break > max_chars * 0.5:
            result = truncated[:best_break + 1]
        else:
            result = truncated.rsplit(" ", 1)[0] + "..."

    return result


def _determine_message_sentiment(body: str) -> Tuple[Sentiment, List[str]]:
    """Determine sentiment for a single message body."""
    signals: List[str] = []

    for pattern, label in _CONTENTIOUS_SIGNALS:
        if re.search(pattern, body, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.CONTENTIOUS, signals

    for pattern, label in _NEEDS_WORK_SIGNALS:
        if re.search(pattern, body, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.NEEDS_WORK, signals

    for pattern, label in _POSITIVE_SIGNALS:
        if re.search(pattern, body, re.IGNORECASE):
            signals.append(label)
    if signals:
        return Sentiment.POSITIVE, signals

    return Sentiment.NEUTRAL, []


def _has_inline_review(body: str) -> bool:
    """Check if a message contains inline code review (quoted + commentary pattern)."""
    lines = body.split("\n")
    had_quote = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            had_quote = True
        elif had_quote and stripped and not stripped.startswith("--"):
            # Non-empty non-sig line right after a quote = inline review
            return True
    return False


def _extract_tags_from_body(body: str) -> List[str]:
    """Extract formal review tags (Acked-by, Reviewed-by, Tested-by) from body."""
    tags = []
    for line in body.split("\n"):
        stripped = line.strip()
        tag_match = re.match(
            r"^(Acked-by|Reviewed-by|Tested-by):\s*",
            stripped,
            re.IGNORECASE,
        )
        if tag_match:
            tags.append(tag_match.group(1))
    return tags


def _extract_review_comments(
    messages: List[Dict], max_comments: int = 10
) -> List[ReviewComment]:
    """Extract individual reviewer comments from thread messages.

    Groups messages by author, extracts a summary from each reviewer's
    messages, determines per-reviewer sentiment, and flags inline reviews.
    Only includes non-trivial replies (skips the root message and trivial
    tag-only messages).
    """
    if len(messages) <= 1:
        return []

    # Identify root author to separate author replies from reviewer comments
    root_from = messages[0].get("from", "").lower()
    root_email = ""
    email_match = re.search(r"<([^>]+)>", root_from)
    if email_match:
        root_email = email_match.group(1).lower()

    # Produce one ReviewComment per individual message (skipping root).
    # Each message carries its own date, author, and body — no grouping needed.
    from email.utils import parsedate as _parsedate
    import time as _time

    def _msg_ymd(msg: Dict) -> str:
        """Return YYYY-MM-DD for a message's Date header, or '' on failure."""
        raw = msg.get("date", "")
        if not raw:
            return ""
        try:
            pd = _parsedate(raw)
            return _time.strftime("%Y-%m-%d", pd) if pd else ""
        except Exception:
            return ""

    comments: List[ReviewComment] = []

    for msg in messages[1:]:  # Skip root message
        from_field = msg.get("from", "")
        msg_email_match = re.search(r"<([^>]+)>", from_field.lower())
        msg_email = msg_email_match.group(1) if msg_email_match else from_field.strip().lower()
        author_name = _extract_author_short(from_field)
        is_patch_author = (msg_email == root_email)
        ymd = _msg_ymd(msg)

        body = msg.get("body", "")
        tags = _extract_tags_from_body(body)

        if _is_trivial_message(body) and not tags:
            continue

        has_inline = _has_inline_review(body)

        # Build summary from this single message body
        if not _is_trivial_message(body):
            budget = max(200, min(800, len(body) // 3))
            summary = _extract_comment_body(body, max_chars=budget) or ""
        else:
            summary = ""

        if not summary:
            if tags:
                unique_tags = list(dict.fromkeys(tags))
                summary = f"Gave {', '.join(unique_tags)}"
            else:
                continue

        unique_tags = list(dict.fromkeys(tags))
        sentiment, signals = _determine_message_sentiment(body)

        if is_patch_author:
            author_name = f"{author_name} (author)"

        comments.append(ReviewComment(
            author=author_name,
            summary=summary,
            sentiment=sentiment,
            sentiment_signals=signals,
            has_inline_review=has_inline,
            tags_given=unique_tags,
            analysis_source="heuristic",
            raw_body=body,
            message_date=ymd,
        ))

    # Sort: by date ascending, patch author last within each date
    def sort_key(c: ReviewComment) -> tuple:
        is_author = c.author.endswith("(author)")
        return (c.message_date, is_author)

    comments.sort(key=sort_key)
    return comments[:max_comments]


def analyze_thread(
    thread_messages: List[Dict], activity_item: ActivityItem
) -> ConversationSummary:
    """Analyze a thread to produce a conversation summary, sentiment, patch summary,
    and discussion progress.

    Args:
        thread_messages: List of message dicts from LKMLClient.get_thread().
        activity_item: The activity item this thread relates to.

    Returns:
        ConversationSummary with all analysis fields populated.
    """
    if not thread_messages:
        return ConversationSummary(
            participant_count=0,
            key_points=["No thread data available"],
            sentiment=Sentiment.NEUTRAL,
            sentiment_signals=[],
        )

    participant_count = _count_participants(thread_messages)
    sentiment, signals = _determine_sentiment(thread_messages)
    key_points = _extract_key_points(thread_messages)

    if not key_points:
        key_points = [f"{len(thread_messages)} messages in thread"]

    # Extract patch summary (only for patch submissions)
    patch_summary = ""
    if activity_item.activity_type == ActivityType.PATCH_SUBMITTED:
        patch_summary = _extract_patch_summary(thread_messages, activity_item.subject)

    # Determine discussion progress
    progress, progress_detail = _determine_discussion_progress(
        thread_messages, activity_item.subject
    )

    # Extract individual review comments
    review_comments = _extract_review_comments(thread_messages)

    return ConversationSummary(
        participant_count=participant_count,
        key_points=key_points,
        sentiment=sentiment,
        sentiment_signals=signals,
        patch_summary=patch_summary,
        discussion_progress=progress,
        progress_detail=progress_detail,
        review_comments=review_comments,
        analysis_source="heuristic",
    )
