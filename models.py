"""Data models for LKML Daily Activity Tracker."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActivityType(Enum):
    PATCH_SUBMITTED = "patch_submitted"
    PATCH_REVIEWED = "patch_reviewed"
    PATCH_ACKED = "patch_acked"
    DISCUSSION_POSTED = "discussion_posted"  # RFC/LSF topic/plain discussion thread


class Sentiment(Enum):
    POSITIVE = "positive"
    NEEDS_WORK = "needs_work"
    CONTENTIOUS = "contentious"
    NEUTRAL = "neutral"


@dataclass
class Developer:
    name: str
    primary_email: str
    secondary_email: Optional[str] = None

    def all_emails(self) -> list[str]:
        emails = [self.primary_email]
        if self.secondary_email:
            emails.append(self.secondary_email)
        return emails


class DiscussionProgress(Enum):
    """Where the discussion currently stands."""
    ACCEPTED = "accepted"               # Applied / merged / queued
    CHANGES_REQUESTED = "changes_requested"  # Reviewer asked for changes
    UNDER_REVIEW = "under_review"       # Active review, no clear verdict yet
    NEW_VERSION_EXPECTED = "new_version_expected"  # Author will resend
    WAITING_FOR_REVIEW = "waiting_for_review"  # Submitted, no substantive replies
    SUPERSEDED = "superseded"           # A newer version exists
    RFC = "rfc"                         # RFC, not yet ready for merge


@dataclass
class ReviewComment:
    """An individual reviewer's comment within a thread."""
    author: str                         # Short name (e.g. "Chuck Lever")
    summary: str                        # Multi-sentence summary of what they said
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_signals: list[str] = field(default_factory=list)
    has_inline_review: bool = False     # True if they did inline code review
    tags_given: list[str] = field(default_factory=list)  # e.g. ["Reviewed-by", "Tested-by"]
    analysis_source: str = "heuristic"  # "heuristic" or "llm"
    raw_body: str = ""                  # Original comment text (quote-stripped, multi-msg joined)
    reply_to: str = ""                  # Short name of who this comment is replying to (if known)
    message_date: str = ""              # YYYY-MM-DD of earliest message from this reviewer


@dataclass
class ConversationSummary:
    participant_count: int = 0
    key_points: list[str] = field(default_factory=list)
    sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_signals: list[str] = field(default_factory=list)
    patch_summary: str = ""             # What the patch does (from cover letter / commit msg)
    discussion_progress: Optional['DiscussionProgress'] = None  # Where things stand
    progress_detail: str = ""           # Human-readable progress sentence
    review_comments: list[ReviewComment] = field(default_factory=list)  # Per-reviewer breakdowns
    analysis_source: str = "heuristic"  # "heuristic", "llm", or "llm-fallback-heuristic"


@dataclass
class LLMAnalysis:
    """A single LLM analysis result with backend/model attribution."""
    backend: str                        # e.g. "ollama", "anthropic"
    model: str                          # e.g. "llama3.1:8b", "claude-haiku-4-5"
    conversation: ConversationSummary = field(default_factory=ConversationSummary)

    @property
    def label(self) -> str:
        """Human-readable label like 'ollama/llama3.1:8b'."""
        return f"{self.backend}/{self.model}"


@dataclass
class ActivityItem:
    activity_type: ActivityType
    subject: str
    message_id: str
    url: str
    date: str
    in_reply_to: Optional[str] = None
    ack_type: Optional[str] = None
    series_patch_count: Optional[int] = None
    conversation: Optional[ConversationSummary] = None
    # For ongoing patches: submitted in last 14 days but with activity today
    is_ongoing: bool = False
    submitted_date: Optional[str] = None  # Original submission date (YYYY-MM-DD)
    # Multiple LLM analyses with attribution (populated when --llm-all is used)
    llm_analyses: list[LLMAnalysis] = field(default_factory=list)


@dataclass
class DeveloperReport:
    developer: Developer
    patches_submitted: list[ActivityItem] = field(default_factory=list)
    patches_reviewed: list[ActivityItem] = field(default_factory=list)
    patches_acked: list[ActivityItem] = field(default_factory=list)
    discussions_posted: list[ActivityItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DailyReport:
    date: str
    developer_reports: list[DeveloperReport] = field(default_factory=list)
    total_patches: int = 0
    total_reviews: int = 0
    total_acks: int = 0
    generation_time_seconds: float = 0.0
    # List of (backend, model) pairs used, e.g. [("ollama", "llama3.1:8b")]
    llm_backends: list[tuple[str, str]] = field(default_factory=list)
