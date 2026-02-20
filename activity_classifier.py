"""Classify lore.kernel.org messages into patches submitted, reviewed, and acked."""

import logging
import re
from typing import Callable, List, Optional, Tuple

from models import ActivityItem, ActivityType, Developer

logger = logging.getLogger(__name__)

# Tag patterns where the developer is GIVING the tag (their name/email follows)
_ACK_PATTERNS = [
    (r"^Acked-by:\s*", "Acked-by"),
    (r"^Tested-by:\s*", "Tested-by"),
    (r"^Reviewed-by:\s*", "Reviewed-by"),
]

# Signals that a reply is a code review
_REVIEW_SIGNAL_PATTERNS = [
    r"\bnit[s:,]\s",
    r"\bminor[:\s]",
    r"\bsuggestion[:\s]",
    r"\bs/[^/]+/[^/]*/",  # sed-style substitution suggestion
    r"\bcould\s+you\s+(please\s+)?(change|fix|update|use|move|rename)",
    r"\bshould\s+we\b",
    r"\bwhat\s+about\b",
    r"\bI\s+think\b",
    r"\blooks\s+good\s+but\b",
    r"\bone\s+(more\s+)?(comment|thing|issue|question)",
    r"\bnot\s+sure\s+(about|if|why)",
    r"\bwhy\s+(not|do|is|are|did)\b",
    r"\bIMHO\b",
    r"\bIMO\b",
    r"\binstead\s+of\b",
    r"\bhave\s+you\s+considered\b",
]


def _is_ack(
    body: str, developer_name: str, developer_emails: List[str]
) -> Tuple[bool, Optional[str]]:
    """Check if the message body contains a tag given BY this developer.

    Returns:
        (is_ack, tag_type) where tag_type is e.g. "Acked-by", "Reviewed-by".
    """
    name_parts = [p.lower() for p in developer_name.split() if len(p) > 2]

    for line in body.split("\n"):
        stripped = line.strip()
        for pattern, tag_type in _ACK_PATTERNS:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                rest = stripped[match.end():].lower()
                # Check if developer's name appears after the tag
                if name_parts and all(part in rest for part in name_parts):
                    return True, tag_type
                # Check if developer's email appears after the tag
                for dev_email in developer_emails:
                    if dev_email.lower() in rest:
                        return True, tag_type
    return False, None


def _has_review_signals(body: str) -> bool:
    """Check if the message body contains code review signals."""
    # Check for quoted text followed by non-quoted commentary
    lines = body.split("\n")
    has_quoted = False
    has_reply_after_quote = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            has_quoted = True
        elif has_quoted and stripped and not stripped.startswith("--"):
            has_reply_after_quote = True
            break

    if not has_reply_after_quote:
        return False

    # Check for review-specific keywords
    for pattern in _REVIEW_SIGNAL_PATTERNS:
        if re.search(pattern, body, re.IGNORECASE):
            return True

    # Check for file path references (common in inline review)
    if re.search(r"\b\w+\.[ch]\b", body):
        return True

    return False


def _get_series_key(message_id: str) -> Optional[str]:
    """Extract a series grouping key from a message ID.

    Handles two common formats:
    1. git format-patch: '20250211.123456-1-user@domain' -> '20250211.123456'
    2. b4 tool: '20250209-name-v6-3-hash@domain' -> 'hash'
       (all patches in same series share the same hash before @)
    """
    # Format 1: timestamp.number prefix
    m = re.match(r"^(\d+\.\d+)", message_id)
    if m:
        return m.group(1)

    # Format 2: b4 style - extract hash before @
    local = message_id.split("@")[0]
    parts = local.rsplit("-", 1)
    if len(parts) == 2 and re.match(r"^[0-9a-f]{8,}$", parts[1]):
        return parts[1]  # The hash groups the series

    return None


def _normalize_series_title(title: str) -> str:
    """Normalize a patch title for cross-version deduplication.

    Strips version numbers and patch numbers so v5 and v6 of the same
    series can be grouped together.
    """
    normalized = re.sub(r"\[(?:RFC\s+)?PATCH[^\]]*\]", "[PATCH]", title)
    return normalized.strip()


def _deduplicate_patches(items: List[ActivityItem]) -> List[ActivityItem]:
    """Deduplicate patch series, keeping cover letter or first patch as representative.

    Handles both within-version dedup (same series submission) and
    cross-version dedup (v5 and v6 of the same series -> keep latest).
    """
    if not items:
        return items

    # Phase 1: Group by series key (within-version grouping)
    groups: dict[str, List[ActivityItem]] = {}
    ungrouped: List[ActivityItem] = []

    for item in items:
        key = _get_series_key(item.message_id)
        if key:
            groups.setdefault(key, []).append(item)
        else:
            ungrouped.append(item)

    # Pick best representative from each group
    phase1_results: List[ActivityItem] = []

    for key, group in groups.items():
        best = None
        total_patches = None
        for item in group:
            cover_match = re.search(r"\[(?:RFC\s+)?PATCH[^\]]*\s+0/(\d+)\]", item.subject)
            first_match = re.search(r"\[(?:RFC\s+)?PATCH[^\]]*\s+1/(\d+)\]", item.subject)
            if cover_match:
                best = item
                total_patches = int(cover_match.group(1))
                break
            elif first_match and best is None:
                best = item
                total_patches = int(first_match.group(1))

        if best is None:
            best = group[0]
            total_patches = len(group)

        best.series_patch_count = total_patches
        phase1_results.append(best)

    phase1_results.extend(ungrouped)

    # Phase 2: Cross-version dedup (keep only the latest version of each series)
    title_groups: dict[str, List[ActivityItem]] = {}
    for item in phase1_results:
        norm = _normalize_series_title(item.subject)
        title_groups.setdefault(norm, []).append(item)

    result: List[ActivityItem] = []
    for norm, group in title_groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Keep the one with the highest version number, or latest date
            best = max(group, key=lambda x: x.date or "")
            result.append(best)

    return result


def extract_patch_submissions(entries: List[dict]) -> List[dict]:
    """Filter Atom feed entries to only patch submissions (not replies).

    Returns the entries that represent new patch submissions (has [PATCH in
    title and is not a Re:).
    """
    patches = []
    for entry in entries:
        title = entry.get("title", "")
        is_reply = bool(re.match(r"^Re:\s*", title, re.IGNORECASE))
        has_patch_tag = bool(re.search(r"\[(?:RFC\s+)?PATCH", title))
        if not is_reply and has_patch_tag:
            patches.append(entry)
    return patches


def check_thread_activity_on_date(
    thread_messages: List[dict], target_date: str
) -> bool:
    """Check if a thread has any messages on the given date.

    Args:
        thread_messages: List of message dicts from get_thread().
        target_date: Date in YYYYMMDD format to check activity for.

    Returns:
        True if at least one message in the thread was sent on target_date.
    """
    from email.utils import parsedate_to_datetime

    for msg in thread_messages:
        date_str = msg.get("date", "")
        if not date_str:
            continue
        try:
            msg_dt = parsedate_to_datetime(date_str)
            msg_date = msg_dt.strftime("%Y%m%d")
            if msg_date == target_date:
                return True
        except Exception:
            # Fallback: check if the YYYYMMDD string appears in the date header
            if target_date[:4] in date_str:
                # Try a rough match: look for the month/day pattern
                try:
                    from datetime import datetime as dt
                    target_obj = dt.strptime(target_date, "%Y%m%d")
                    # Check for common date formats
                    day_str = str(target_obj.day)
                    month_abbr = target_obj.strftime("%b")
                    if day_str in date_str and month_abbr in date_str:
                        return True
                except Exception:
                    pass
    return False


# Patterns that identify a root (non-reply) message as an RFC or discussion thread.
# Matched against the message subject (case-insensitive).
_DISCUSSION_TAG_PATTERNS = [
    r"\[RFC\]",                  # plain [RFC] without PATCH
    r"\[LSF",                    # LSF/MM/BPF topics
    r"\[TOPIC\]",                # generic [TOPIC]
    r"\[DISCUSS",                # [DISCUSS] / [DISCUSSION]
    r"\[ANN\]",                  # announcements
    r"\[ANNOUNCE\]",
]

_DISCUSSION_TAG_RE = re.compile(
    "|".join(_DISCUSSION_TAG_PATTERNS), re.IGNORECASE
)


def _is_discussion(title: str) -> bool:
    """Return True if this non-reply, non-patch subject is a discussion/RFC thread."""
    return bool(_DISCUSSION_TAG_RE.search(title))


def classify_messages(
    entries: List[dict],
    developer: Developer,
    raw_fetcher: Callable[[str], str],
) -> Tuple[List[ActivityItem], List[ActivityItem], List[ActivityItem], List[ActivityItem]]:
    """Classify Atom feed entries into patches submitted, reviewed, acked, and discussions.

    Args:
        entries: List of dicts from get_user_messages_for_date.
        developer: The developer whose activity we're classifying.
        raw_fetcher: Callable that takes message_id and returns raw body text.

    Returns:
        Tuple of (patches_submitted, patches_reviewed, patches_acked, discussions_posted).
    """
    patches: List[ActivityItem] = []
    reviews: List[ActivityItem] = []
    acks: List[ActivityItem] = []
    discussions: List[ActivityItem] = []

    for entry in entries:
        title = entry.get("title", "")
        msg_id = entry.get("message_id", "")
        url = entry.get("url", "")
        updated = entry.get("updated", "")

        if not msg_id:
            continue

        # Ensure URL is a full lore.kernel.org link
        if url and not url.startswith("http"):
            url = f"https://lore.kernel.org{url}"
        if not url:
            url = f"https://lore.kernel.org/all/{msg_id}/"

        is_reply = bool(re.match(r"^Re:\s*", title, re.IGNORECASE))
        has_patch_tag = bool(re.search(r"\[(?:RFC\s+)?PATCH", title))

        if not is_reply and has_patch_tag:
            # Patch submission (includes [RFC PATCH ...])
            patches.append(ActivityItem(
                activity_type=ActivityType.PATCH_SUBMITTED,
                subject=title,
                message_id=msg_id,
                url=url,
                date=updated,
            ))
            logger.debug("PATCH: %s", title)

        elif not is_reply and _is_discussion(title):
            # Root message for a discussion/RFC/LSF topic thread
            discussions.append(ActivityItem(
                activity_type=ActivityType.DISCUSSION_POSTED,
                subject=title,
                message_id=msg_id,
                url=url,
                date=updated,
            ))
            logger.debug("DISCUSSION: %s", title)

        elif is_reply:
            # Need body to distinguish review from ack
            try:
                raw_body = raw_fetcher(msg_id)
            except Exception as e:
                logger.warning("Failed to fetch raw message %s: %s", msg_id, e)
                raw_body = ""

            # Check ack first (higher priority)
            is_ack_result, ack_type = _is_ack(
                raw_body, developer.name, developer.all_emails()
            )
            if is_ack_result:
                acks.append(ActivityItem(
                    activity_type=ActivityType.PATCH_ACKED,
                    subject=title,
                    message_id=msg_id,
                    url=url,
                    date=updated,
                    ack_type=ack_type,
                ))
                logger.debug("ACK (%s): %s", ack_type, title)

            elif has_patch_tag or _has_review_signals(raw_body):
                reviews.append(ActivityItem(
                    activity_type=ActivityType.PATCH_REVIEWED,
                    subject=title,
                    message_id=msg_id,
                    url=url,
                    date=updated,
                ))
                logger.debug("REVIEW: %s", title)
            else:
                logger.debug("SKIPPED (reply, no review signals): %s", title)
        else:
            logger.debug("SKIPPED (not patch, not reply, not discussion): %s", title)

    # Deduplicate patch series
    patches = _deduplicate_patches(patches)

    return patches, reviews, acks, discussions
