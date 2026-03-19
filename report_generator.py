"""HTML report generator for LKML daily activity reports."""

import html
import re
from datetime import datetime
from typing import Optional

from models import (
    ActivityItem,
    ActivityType,
    ConversationSummary,
    DailyReport,
    DeveloperReport,
    DiscussionProgress,
    LLMAnalysis,
    ReviewComment,
    Sentiment,
)


def message_id_to_slug(message_id: str) -> str:
    """Convert a message-id to a filesystem-safe slug.

    Example: '<20250213.abc@kernel.org>' -> '20250213-abc-kernel-org'
    """
    # Strip angle brackets
    slug = message_id.strip("<>")
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug)
    # Collapse multiple hyphens, strip leading/trailing
    slug = re.sub(r"-+", "-", slug).strip("-")
    # Limit length to avoid overly long filenames
    return slug[:120]

_SENTIMENT_COLORS = {
    Sentiment.POSITIVE: ("#155724", "#d4edda"),
    Sentiment.NEEDS_WORK: ("#856404", "#fff3cd"),
    Sentiment.CONTENTIOUS: ("#721c24", "#f8d7da"),
    Sentiment.NEUTRAL: ("#383d41", "#e2e3e5"),
}

_SENTIMENT_LABELS = {
    Sentiment.POSITIVE: "Positive",
    Sentiment.NEEDS_WORK: "Needs Work",
    Sentiment.CONTENTIOUS: "Contentious",
    Sentiment.NEUTRAL: "Neutral",
}


_PROGRESS_STYLES = {
    DiscussionProgress.ACCEPTED: ("#155724", "#d4edda", "Accepted"),
    DiscussionProgress.CHANGES_REQUESTED: ("#856404", "#fff3cd", "Changes Requested"),
    DiscussionProgress.UNDER_REVIEW: ("#004085", "#cce5ff", "Under Review"),
    DiscussionProgress.NEW_VERSION_EXPECTED: ("#856404", "#fff3cd", "New Version Expected"),
    DiscussionProgress.WAITING_FOR_REVIEW: ("#383d41", "#e2e3e5", "Awaiting Review"),
    DiscussionProgress.SUPERSEDED: ("#383d41", "#e2e3e5", "Superseded"),
    DiscussionProgress.RFC: ("#0c5460", "#d1ecf1", "RFC"),
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _sentiment_badge(sentiment: Sentiment) -> str:
    color, bg = _SENTIMENT_COLORS[sentiment]
    label = _SENTIMENT_LABELS[sentiment]
    return (
        f'<span class="badge" style="color:{color};background:{bg}">'
        f"{label}</span>"
    )


def _progress_badge(progress: DiscussionProgress) -> str:
    color, bg, label = _PROGRESS_STYLES.get(
        progress, ("#383d41", "#e2e3e5", "Unknown")
    )
    return (
        f'<span class="progress-badge" style="color:{color};background:{bg}">'
        f"{label}</span>"
    )


def _render_review_comment(rc: ReviewComment) -> str:
    """Render an individual reviewer's comment block."""
    parts = []
    parts.append('<div class="review-comment">')

    # Author line with sentiment badge and tags
    parts.append('<div class="review-comment-header">')
    parts.append(f'<span class="review-author">{_esc(rc.author)}</span>')
    if rc.is_maintainer:
        parts.append('<span class="maintainer-badge">&#9733; Maintainer</span>')

    # Reply-to context
    if rc.reply_to:
        parts.append(f'<span class="reply-to-label">↳ replying to {_esc(rc.reply_to)}</span>')

    # Inline review indicator
    if rc.has_inline_review:
        parts.append('<span class="inline-review-badge">Inline Review</span>')

    # Tags given
    for tag in rc.tags_given:
        parts.append(f'<span class="review-tag-badge">{_esc(tag)}</span>')

    # Per-reviewer sentiment badge
    parts.append(_sentiment_badge(rc.sentiment))

    # Per-reviewer analysis source badge
    parts.append(_analysis_source_badge(rc.analysis_source))
    parts.append('</div>')

    # Comment summary text
    if rc.summary:
        parts.append(f'<div class="review-comment-text">{_esc(rc.summary)}</div>')

    # Collapsible raw body + lore link row
    if rc.raw_body or rc.message_id:
        parts.append('<div class="review-comment-footer">')
        if rc.raw_body:
            parts.append('<details class="raw-body-toggle">')
            parts.append('<summary>Show original comment</summary>')
            parts.append(f'<pre class="raw-body-text">{_esc(rc.raw_body)}</pre>')
            parts.append('</details>')
        if rc.message_id:
            lore_url = f"https://lore.kernel.org/r/{_esc(rc.message_id)}"
            parts.append(
                f'<a href="{lore_url}" target="_blank" rel="noopener" '
                f'class="lore-link">View on lore ↗</a>'
            )
        parts.append('</div>')

    # Sentiment signals
    if rc.sentiment_signals:
        signals_str = ", ".join(rc.sentiment_signals[:3])
        parts.append(f'<div class="review-comment-signals">Signals: {_esc(signals_str)}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _render_compact_reviews(conv: ConversationSummary, review_link: str) -> str:
    """Render a compact review summary line with a link to the detail page."""
    parts: list[str] = []
    parts.append('<div class="review-comments-compact">')

    # Build reviewer list, deduplicating by author and merging annotations
    author_annotations: dict[str, list[str]] = {}
    author_is_maintainer: dict[str, bool] = {}
    for rc in conv.review_comments:
        entry = author_annotations.setdefault(rc.author, [])
        if rc.tags_given:
            entry.extend(rc.tags_given)
        if rc.has_inline_review:
            entry.append("Inline Review")
        if rc.is_maintainer:
            author_is_maintainer[rc.author] = True

    reviewer_descs = []
    for author, annotations in author_annotations.items():
        desc = _esc(author)
        if author_is_maintainer.get(author):
            desc += ' <span class="maintainer-badge-inline">&#9733;</span>'
        # Deduplicate annotations (e.g. "Reviewed-by" from two segments)
        seen: set[str] = set()
        unique_ann = [a for a in annotations if not (a in seen or seen.add(a))]
        if unique_ann:
            desc += f' ({", ".join(_esc(a) for a in unique_ann)})'
        reviewer_descs.append(desc)

    parts.append(f'<span class="review-comments-header">'
                 f'{conv.participant_count} participants</span>')
    if reviewer_descs:
        parts.append(f'<span class="reviewer-list"> &mdash; '
                     f'{", ".join(reviewer_descs)}</span>')

    parts.append(f'<div class="review-detail-link">'
                 f'<a href="{_esc(review_link)}">View review comments &rarr;</a>'
                 f'</div>')
    parts.append("</div>")
    return "\n".join(parts)


_ANALYSIS_SOURCE_STYLES = {
    "heuristic": ("#6c4b00", "#ffeeba", "Heuristic"),
    "llm": ("#004085", "#cce5ff", "LLM"),
    "llm-per-reviewer": ("#004085", "#cce5ff", "LLM (per-reviewer)"),
    "llm-fallback-heuristic": ("#721c24", "#f8d7da", "LLM \u2192 Heuristic"),
}


def _analysis_source_badge(source: str) -> str:
    """Render a small badge indicating whether analysis came from LLM or heuristic."""
    color, bg, label = _ANALYSIS_SOURCE_STYLES.get(
        source, ("#383d41", "#e2e3e5", source)
    )
    return (
        f'<span class="analysis-source-badge" style="color:{color};background:{bg}"'
        f' title="Analysis source: {_esc(label)}">'
        f'{label}</span>'
    )


def _render_conversation_body(
    conv: ConversationSummary, review_link: Optional[str] = None
) -> str:
    """Render the body of a conversation summary (sentiment, progress, patch summary, reviews).

    Shared between single-analysis and multi-analysis card rendering.

    Args:
        conv: The conversation summary data.
        review_link: If provided, render compact review summary with link to detail page
                     instead of inline review comments.
    """
    parts: list[str] = []

    # Sentiment badge
    parts.append(_sentiment_badge(conv.sentiment))

    # Analysis source badge (heuristic / LLM / LLM → Heuristic fallback)
    parts.append(_analysis_source_badge(conv.analysis_source))

    # Discussion progress badge
    if conv.discussion_progress:
        parts.append(_progress_badge(conv.discussion_progress))

    # Patch summary (what the patch does) — supports multi-paragraph
    if conv.patch_summary:
        paras = [p.strip() for p in conv.patch_summary.split("\n\n") if p.strip()]
        parts.append('<div class="patch-summary">')
        for para in paras:
            parts.append(f"<p>{_esc(para)}</p>")
        parts.append("</div>")

    # Discussion progress detail
    if conv.progress_detail:
        parts.append(
            f'<div class="progress-detail">'
            f'<span class="progress-icon">&#9654;</span> '
            f'{_esc(conv.progress_detail)}'
            f'</div>'
        )

    # Maintainer review note (shown whenever ≥1 maintainer commented)
    maintainer_names = [rc.author for rc in conv.review_comments if rc.is_maintainer]
    if maintainer_names:
        seen: set[str] = set()
        unique_names = [n for n in maintainer_names if not (n in seen or seen.add(n))]
        names_str = ", ".join(_esc(n) for n in unique_names)
        parts.append(
            f'<div class="maintainer-review-note">'
            f'&#9733; Maintainer review: {names_str}'
            f'</div>'
        )

    # Individual review comments: compact with link, or inline (fallback)
    if conv.review_comments and review_link:
        parts.append(_render_compact_reviews(conv, review_link))
    elif conv.review_comments:
        parts.append('<div class="review-comments">')
        parts.append(f'<div class="review-comments-header">'
                     f'{conv.participant_count} participants</div>')
        for rc in conv.review_comments:
            parts.append(_render_review_comment(rc))
        parts.append("</div>")
    elif conv.key_points:
        parts.append('<div class="conversation-summary">')
        parts.append(f'<span class="participants">{conv.participant_count} participants</span>')
        if conv.sentiment_signals:
            signals = ", ".join(conv.sentiment_signals[:3])
            parts.append(f'<span class="signals">Signals: {_esc(signals)}</span>')
        parts.append("<ul>")
        for point in conv.key_points:
            parts.append(f"<li>{_esc(point)}</li>")
        parts.append("</ul>")
        parts.append("</div>")

    return "\n".join(parts)


def _render_llm_analysis_card(
    analysis: LLMAnalysis, review_link: Optional[str] = None
) -> str:
    """Render a single LLM analysis as an attributed card."""
    parts: list[str] = []
    parts.append('<div class="llm-analysis">')
    parts.append(f'<div class="llm-analysis-header">{_esc(analysis.label)}</div>')
    parts.append(_render_conversation_body(analysis.conversation, review_link=review_link))
    parts.append("</div>")
    return "\n".join(parts)


def _get_review_link(
    item: ActivityItem, review_links: Optional[dict[str, str]], report_date: str
) -> Optional[str]:
    """Look up the review detail page link for an activity item."""
    if not review_links:
        return None
    msg_id = item.message_id
    slug = review_links.get(msg_id)
    if slug:
        return f"reviews/{slug}.html#{report_date}"
    return None


def _last_comment_date(item: ActivityItem) -> str:
    """Return the most recent reviewer message_date across all review comments, or ''."""
    if not item.conversation:
        return ""
    dates = [rc.message_date for rc in item.conversation.review_comments if rc.message_date]
    return max(dates) if dates else ""


def _render_activity_item(
    item: ActivityItem, section_type: str,
    review_links: Optional[dict[str, str]] = None, report_date: str = ""
) -> str:
    parts = []
    css_class = "activity-item ongoing" if item.is_ongoing else "activity-item"
    parts.append(f'<div class="{css_class}">')

    # Ongoing badge
    if item.is_ongoing:
        parts.append(f'<span class="ongoing-badge">Ongoing</span>')
    # Submitted date (shown for all patches that have it, not just ongoing)
    if item.submitted_date:
        parts.append(f'<span class="submitted-date">Submitted {_esc(item.submitted_date)}</span>')
    # Last comment date — show eye-catching "TODAY" badge if activity is today,
    # or a "No recent activity" warning for ongoing patches idle for 2+ days.
    last_comment = _last_comment_date(item)
    if last_comment:
        if report_date and last_comment == report_date:
            parts.append('<span class="today-badge">&#128293; TODAY</span>')
        else:
            parts.append(f'<span class="last-comment-date">Last comment {_esc(last_comment)}</span>')
            # Flag stale ongoing patches: no activity for 2+ days
            if item.is_ongoing and report_date and last_comment:
                try:
                    from datetime import date as _date
                    d_report = _date.fromisoformat(report_date)
                    d_last = _date.fromisoformat(last_comment)
                    if (d_report - d_last).days >= 2:
                        parts.append('<span class="stale-badge">&#9201; No recent activity</span>')
                except ValueError:
                    pass
    elif item.is_ongoing:
        # Ongoing patch with no comments at all is also stale
        parts.append('<span class="stale-badge">&#9201; No comments yet</span>')

    # Title with link
    escaped_subject = _esc(item.subject)
    escaped_url = _esc(item.url)
    parts.append(f'<a href="{escaped_url}" target="_blank" class="item-link">{escaped_subject}</a>')

    # Ack type badge
    if item.ack_type:
        parts.append(f'<span class="ack-badge">{_esc(item.ack_type)}</span>')

    # Series patch count
    if item.series_patch_count and item.series_patch_count > 1:
        parts.append(f'<span class="patch-count">{item.series_patch_count} patches</span>')

    # Version chain: shown when multiple revisions of the same series were found
    if len(item.version_history) > 1:
        parts.append('<span class="version-chain">')
        for i, vh in enumerate(item.version_history):
            ver_label = f"v{vh['version']}"
            is_latest = (i == len(item.version_history) - 1)
            # Link older versions to their own detail page (if available) or lore URL
            ver_slug = review_links.get(vh["message_id"]) if review_links else None
            ver_href = _esc(f"reviews/{ver_slug}.html") if ver_slug else _esc(vh["url"])
            css = "version-badge latest" if is_latest else "version-badge"
            parts.append(f'<a href="{ver_href}" class="{css}" target="_blank">{ver_label}</a>')
            if not is_latest:
                parts.append('<span class="version-arrow">\u2192</span>')
        parts.append('</span>')

    # Individual patches sub-list (collapsible, cover letter representative only)
    if item.series_items:
        _SI_PATCH_NUM_RE = re.compile(
            r"\[(?:RFC\s+)?PATCH[^\]]*\s+(\d+)/\d+\]|\[RFC[^\]]*\b(\d+)/\d+\]",
            re.IGNORECASE,
        )
        total = item.series_patch_count or len(item.series_items)
        parts.append('<details class="series-patches">')
        parts.append(
            f'<summary class="series-patches-toggle">'
            f'Show {len(item.series_items)} individual patches</summary>'
        )
        parts.append('<ul class="series-patch-list">')
        for si in item.series_items:
            m = _SI_PATCH_NUM_RE.search(si.subject)
            num = (m.group(1) or m.group(2)) if m else "?"
            clean_title = re.sub(r"^\[.*?\]\s*", "", si.subject)
            # Always derive the local review page slug directly from the message-id.
            # review_links may not contain series items when the cover letter is
            # "ongoing" (the individual patches may not have been in the lookback
            # window), but the review page always exists on disk once it has been
            # generated for any prior day.
            si_slug = message_id_to_slug(si.message_id)
            si_href = _esc(f"reviews/{si_slug}.html")
            reviewer_badge = ""
            si_dates_html = ""
            if si.submitted_date:
                si_dates_html += f'<span class="si-date">Submitted {_esc(si.submitted_date)}</span>'
            if si.conversation and si.conversation.review_comments:
                n = len(si.conversation.review_comments)
                reviewer_badge = (
                    f' <span class="si-reviewers">'
                    f'{n} reviewer{"s" if n != 1 else ""}</span>'
                )
                rc_dates = [rc.message_date for rc in si.conversation.review_comments if rc.message_date]
                if rc_dates:
                    si_dates_html += f'<span class="si-date">Last comment {_esc(max(rc_dates))}</span>'
            parts.append(
                f'<li class="series-patch-item">'
                f'<span class="si-num">[{_esc(str(num))}/{total}]</span> '
                f'<a href="{si_href}" target="_blank" class="si-link">'
                f'{_esc(clean_title)}</a>'
                f'{reviewer_badge}'
                f'{(" " + si_dates_html) if si_dates_html else ""}'
                f'</li>'
            )
        parts.append('</ul>')
        parts.append('</details>')

    review_link = _get_review_link(item, review_links, report_date)

    # Multi-LLM analyses (when --llm-all produces multiple results)
    if len(item.llm_analyses) > 1:
        parts.append('<div class="llm-analyses">')
        for analysis in item.llm_analyses:
            parts.append(_render_llm_analysis_card(analysis, review_link=review_link))
        parts.append("</div>")
    elif item.conversation:
        # Single analysis (single backend or heuristic)
        parts.append(_render_conversation_body(item.conversation, review_link=review_link))

    parts.append("</div>")
    return "\n".join(parts)


def _series_key(message_id: str) -> str:
    """Extract the series identifier from a message-id.

    Kernel git-send-email format:
      <hash>.1770821420.git.user@domain>  →  '1770821420.git.user@domain'
      cover.1770821420.git.user@domain   →  '1770821420.git.user@domain'

    All patches in the same ``git format-patch`` run share the same Unix
    timestamp, so the suffix after the first component is the series key.
    For Gmail / non-kernel IDs the full message-id is returned so each item
    sorts independently.
    """
    mid = message_id.strip("<>")
    m = re.match(r"(?:[0-9a-f]{10,}|cover)\.([\d]+\.git\..+)$", mid, re.IGNORECASE)
    return m.group(1) if m else mid


def _patch_num(subject: str) -> int:
    """Return the X from '[PATCH X/Y]', or 0 for cover letters / standalone."""
    s = re.sub(r"^(?:Re:\s*)+", "", subject, flags=re.IGNORECASE).strip()
    m = re.search(r"\[(?:RFC\s+)?PATCH[^\]]*?\s+(\d+)/\d+\]", s, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _sort_activity_items(items: list[ActivityItem]) -> list[ActivityItem]:
    """Sort activity items with series-aware grouping.

    Patches that share a series key (the git-timestamp suffix common to all
    message-ids produced by a single ``git format-patch`` run) are kept
    together.  Groups themselves are sorted alphabetically by the subject of
    their lowest-numbered patch so related patches stay in one block and
    blocks appear in a predictable order.  Patches within a group are sorted
    by patch number (0 = cover letter first, then 1, 2, 3 …).

    Items whose message-ids do not follow the kernel format (Gmail, etc.)
    each form their own one-item group and are interleaved alphabetically.
    """
    # Pre-compute sort metadata once per item.
    def _meta(item: ActivityItem) -> tuple:
        sk = _series_key(item.message_id)
        num = _patch_num(item.subject)
        # Normalised subject for alphabetical comparisons (strip Re: + tag).
        s = re.sub(r"^(?:Re:\s*)+", "", item.subject, flags=re.IGNORECASE).strip()
        s = re.sub(r"^\[.*?\]\s*", "", s).strip().lower()
        return sk, num, s

    meta = {id(item): _meta(item) for item in items}

    # Group items by series key.
    groups: dict[str, list[ActivityItem]] = {}
    for item in items:
        sk = meta[id(item)][0]
        groups.setdefault(sk, []).append(item)

    # Within each group sort by (patch_num, normalised_subject).
    for grp in groups.values():
        grp.sort(key=lambda i: (meta[id(i)][1], meta[id(i)][2]))

    # Sort groups by the normalised subject of the first (lowest-numbered) item.
    sorted_groups = sorted(
        groups.values(),
        key=lambda grp: meta[id(grp[0])][2],
    )

    return [item for grp in sorted_groups for item in grp]


def _render_activity_section(
    items: list[ActivityItem], title: str, section_type: str,
    open_by_default: bool = False,
    review_links: Optional[dict[str, str]] = None, report_date: str = ""
) -> str:
    count = len(items)
    open_attr = " open" if open_by_default and count > 0 else ""

    parts = []
    parts.append(f"<details{open_attr}>")
    parts.append(f'<summary>{_esc(title)} <span class="count">({count})</span></summary>')

    if count == 0:
        parts.append('<div class="no-activity">No activity</div>')
    else:
        for item in _sort_activity_items(items):
            parts.append(_render_activity_item(
                item, section_type, review_links=review_links, report_date=report_date
            ))

    parts.append("</details>")
    return "\n".join(parts)


def _render_developer_section(
    dev_report: DeveloperReport,
    review_links: Optional[dict[str, str]] = None, report_date: str = ""
) -> str:
    total = (
        len(dev_report.patches_submitted)
        + len(dev_report.patches_reviewed)
        + len(dev_report.patches_acked)
        + len(dev_report.discussions_posted)
    )

    anchor = _name_to_anchor(dev_report.developer.name)
    parts = []
    parts.append(f'<div class="developer-section" id="{anchor}">')
    parts.append(f'<div class="developer-header">')
    parts.append(f'<h3>{_esc(dev_report.developer.name)}</h3>')
    if total == 0:
        parts.append('<span class="inactive-badge">No activity</span>')
    else:
        parts.append(f'<span class="active-badge">{total} items</span>')
    parts.append("</div>")

    # Errors
    if dev_report.errors:
        parts.append('<div class="errors">')
        for err in dev_report.errors:
            parts.append(f'<div class="error-msg">Error: {_esc(err)}</div>')
        parts.append("</div>")

    parts.append(_render_activity_section(
        dev_report.patches_submitted, "Patches Submitted", "patch",
        open_by_default=True, review_links=review_links, report_date=report_date
    ))
    parts.append(_render_activity_section(
        dev_report.discussions_posted, "Discussions / RFCs", "discussion",
        open_by_default=True, review_links=review_links, report_date=report_date
    ))
    parts.append(_render_activity_section(
        dev_report.patches_reviewed, "Reviews Given", "review",
        review_links=review_links, report_date=report_date
    ))
    parts.append(_render_activity_section(
        dev_report.patches_acked, "Acks / Tags Given", "ack",
        review_links=review_links, report_date=report_date
    ))

    parts.append("</div>")
    return "\n".join(parts)


def _name_to_anchor(name: str) -> str:
    """Convert a developer name to a section anchor id."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return "dev-" + slug.strip("-")



def _render_statistics(report: DailyReport) -> str:
    active_devs = sum(
        1
        for dr in report.developer_reports
        if dr.patches_submitted or dr.patches_reviewed or dr.patches_acked
            or dr.discussions_posted
    )
    total_devs = len(report.developer_reports)

    total_discussions = sum(len(dr.discussions_posted) for dr in report.developer_reports)

    discussion_card = ""
    if total_discussions:
        discussion_card = f"""
        <div class="stat-card">
            <div class="stat-number">{total_discussions}</div>
            <div class="stat-label">Discussions / RFCs</div>
        </div>"""

    # --- Permanent contributor table ---
    # Collect all active developers with their per-category counts
    has_discussions = total_discussions > 0
    contrib_rows = []
    for dr in sorted(report.developer_reports,
                     key=lambda r: r.developer.name.lower()):
        p = len(dr.patches_submitted)
        rv = len(dr.patches_reviewed)
        ack = len(dr.patches_acked)
        disc = len(dr.discussions_posted)
        if p == 0 and rv == 0 and ack == 0 and disc == 0:
            continue
        anchor = _name_to_anchor(dr.developer.name)

        def _cell(n: int) -> str:
            if n == 0:
                return '<td class="num zero">&mdash;</td>'
            return f'<td class="num">{n}</td>'

        disc_cell = _cell(disc) if has_discussions else ""
        contrib_rows.append(
            f'<tr>'
            f'<td><a href="#{anchor}">{_esc(dr.developer.name)}</a></td>'
            f'{_cell(p)}{disc_cell}{_cell(rv)}{_cell(ack)}'
            f'</tr>'
        )

    disc_th = '<th class="num">Discussions</th>' if has_discussions else ""
    contrib_table = ""
    if contrib_rows:
        contrib_table = f"""
    <div class="contributors-section">
        <h3>Contributors</h3>
        <table class="contributors-table">
            <thead><tr>
                <th>Developer</th>
                <th class="num">Patches</th>
                {disc_th}
                <th class="num">Reviews</th>
                <th class="num">Acks</th>
            </tr></thead>
            <tbody>{"".join(contrib_rows)}</tbody>
        </table>
    </div>"""

    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{report.total_patches}</div>
            <div class="stat-label">Patches Submitted</div>
        </div>{discussion_card}
        <div class="stat-card">
            <div class="stat-number">{report.total_reviews}</div>
            <div class="stat-label">Reviews Given</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{report.total_acks}</div>
            <div class="stat-label">Acks Given</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{active_devs}/{total_devs}</div>
            <div class="stat-label">Active Developers</div>
        </div>
    </div>
    {contrib_table}
    """


def _normalize_patch_subject(subject: str) -> str:
    """Strip 'Re:' prefix only, return lowercase subject.

    Used to match PATCH_REVIEWED/PATCH_ACKED items (whose subject starts with
    'Re: [PATCH...]') back to their originating PATCH_SUBMITTED item so their
    review data lands in the same per-patchset JSON file rather than a separate
    orphan file.

    The [PATCH vN X/Y] tag is intentionally preserved so that replies to
    individual patches within a series (e.g. 'Re: [PATCH 1/4] ...') do NOT
    match the cover letter ('[PATCH 0/4] ...').  Only a direct cover-letter
    reply ('Re: [PATCH 0/4] same title') matches the cover letter's slug.
    """
    # Strip one or more leading "Re:" prefixes; keep everything else intact
    s = re.sub(r"^(?:Re:\s*)+", "", subject, flags=re.IGNORECASE).strip()
    return s.lower()


def extract_reviews_data(daily_report: DailyReport, report_filename: str) -> list[dict]:
    """Extract review comment data from a DailyReport for JSON serialization.

    Returns a list of dicts, one per activity item that has review comments:
    [
        {
            "message_id": "<msg-id>",
            "slug": "sanitized-slug",
            "subject": "patch subject line",
            "url": "https://lore.kernel.org/...",
            "developer": "Developer Name",
            "date": "2026-02-15",
            "report_file": "2026-02-15_ollama_llama3.1-8b.html",
            "reviews": [ { "author", "summary", "sentiment", ... } ]
        }
    ]
    """
    results = []
    for dr in daily_report.developer_reports:
        # Build a normalized-title → PATCH_SUBMITTED item map so that
        # PATCH_REVIEWED/PATCH_ACKED replies with matching subjects can be
        # routed to the original patch's review JSON instead of a separate file.
        patch_by_norm_subject: dict[str, ActivityItem] = {}
        for p in dr.patches_submitted:
            norm = _normalize_patch_subject(p.subject)
            if norm:
                patch_by_norm_subject[norm] = p

        all_items = (
            dr.patches_submitted + dr.patches_reviewed + dr.patches_acked
            + dr.discussions_posted
        )
        for item in all_items:
            conv = item.conversation
            if not conv or not conv.review_comments:
                continue

            # For "Re: [PATCH ...] <title>" replies, try to find the originating
            # patch submission so the review data merges into that patch's JSON.
            effective_item = item
            if (
                item.activity_type in (ActivityType.PATCH_REVIEWED, ActivityType.PATCH_ACKED)
                and re.match(r"^Re:\s*", item.subject, re.IGNORECASE)
            ):
                norm = _normalize_patch_subject(item.subject)
                matched_patch = patch_by_norm_subject.get(norm)
                if matched_patch:
                    effective_item = matched_patch

            # Strip leading "Re:" for display when no match was found (reviewing
            # someone else's patch — keep the slug as-is but show a clean title).
            display_subject = effective_item.subject
            if effective_item is item and re.match(r"^Re:\s*", item.subject, re.IGNORECASE):
                display_subject = re.sub(
                    r"^(?:Re:\s*)+", "", item.subject, flags=re.IGNORECASE
                ).strip()

            reviews = []
            for rc in conv.review_comments:
                reviews.append({
                    "author": rc.author,
                    "summary": rc.summary,
                    "sentiment": rc.sentiment.value,
                    "sentiment_signals": rc.sentiment_signals,
                    "has_inline_review": rc.has_inline_review,
                    "tags_given": rc.tags_given,
                    "analysis_source": rc.analysis_source,
                    "raw_body": rc.raw_body,
                    "reply_to": rc.reply_to,
                    "message_date": rc.message_date,
                    "message_id": getattr(rc, "message_id", "") or "",
                    "email": getattr(rc, "email", "") or "",
                    "is_maintainer": bool(getattr(rc, "is_maintainer", False)),
                })
            results.append({
                "message_id": effective_item.message_id,
                "slug": message_id_to_slug(effective_item.message_id),
                "subject": display_subject,
                "url": effective_item.url,
                "developer": dr.developer.name,
                "date": daily_report.date,
                "report_file": report_filename,
                "analysis_source": conv.analysis_source,
                "patch_summary": conv.patch_summary or "",
                "reviews": reviews,
            })
    return results


def _is_contentious_item(item: ActivityItem) -> bool:
    """Return True if the item has strong pushback or NAK signals."""
    if not item.conversation:
        return False
    if item.conversation.sentiment == Sentiment.CONTENTIOUS:
        return True
    for rc in item.conversation.review_comments:
        combined = " ".join(rc.sentiment_signals + rc.tags_given).lower()
        if "nak" in combined or "nack" in combined:
            return True
        if rc.sentiment == Sentiment.CONTENTIOUS:
            return True
    return False


def _render_daily_summary(
    report: DailyReport,
    review_links: Optional[dict[str, str]] = None,
    report_date: str = "",
) -> str:
    """Render the top-level daily highlights summary for all engineers."""
    # Collect all submitted patches and discussions across all tracked devs
    all_items: list[tuple[str, ActivityItem]] = []
    for dr in report.developer_reports:
        for item in dr.patches_submitted + dr.discussions_posted:
            all_items.append((dr.developer.name, item))

    # 1. New patch series: submitted today (not ongoing), first version
    new_series = [
        (dev, item) for dev, item in all_items
        if (item.activity_type == ActivityType.PATCH_SUBMITTED
            and not item.is_ongoing
            and item.patch_version == 1)
    ]

    # 2. Strong pushback / NAKs
    pushed_back = [(dev, item) for dev, item in all_items if _is_contentious_item(item)]

    # 3. High activity: top 5 items with at least 2 participants, sorted desc
    def _participant_count(item: ActivityItem) -> int:
        return item.conversation.participant_count if item.conversation else 0

    high_activity = sorted(
        [(dev, item) for dev, item in all_items if _participant_count(item) >= 2],
        key=lambda x: _participant_count(x[1]),
        reverse=True,
    )[:5]

    # 4. Maintainer involvement
    def _maintainer_rcs(item: ActivityItem) -> list[ReviewComment]:
        if not item.conversation:
            return []
        return [rc for rc in item.conversation.review_comments if rc.is_maintainer]

    maintainer_items = [(dev, item) for dev, item in all_items if _maintainer_rcs(item)]

    # --- Rendering helpers ---

    _SENTIMENT_ICONS = {
        Sentiment.POSITIVE: "&#10003;",
        Sentiment.NEEDS_WORK: "&#9888;",
        Sentiment.CONTENTIOUS: "&#10007;",
        Sentiment.NEUTRAL: "",
    }

    def _summary_item(dev_name: str, item: ActivityItem, extra_html: str = "") -> str:
        anchor = _name_to_anchor(dev_name)
        dev_link = f'<a href="#{anchor}" class="summary-dev-link">{_esc(dev_name)}</a>'
        subject_html = (
            f'<a href="{_esc(item.url)}" target="_blank" rel="noopener" '
            f'class="item-link">{_esc(item.subject)}</a>'
        )
        badges = ""
        if item.series_patch_count and item.series_patch_count > 1:
            badges += f'<span class="patch-count">{item.series_patch_count} patches</span>'
        if item.patch_version > 1:
            badges += f'<span class="version-badge latest">v{item.patch_version}</span>'
        if item.conversation:
            badges += _sentiment_badge(item.conversation.sentiment)
        meta = f'<div class="summary-item-meta">by {dev_link}{(" &mdash; " + extra_html) if extra_html else ""}</div>'
        return f'<div class="summary-item">{subject_html}{badges}{meta}</div>'

    def _sub_section(title: str, icon: str, items: list[tuple], css_class: str, extra_fn=None) -> str:
        if not items:
            empty_html = '<div class="summary-empty">None today</div>'
            return (
                f'<div class="summary-section {css_class}">'
                f'<div class="summary-section-title">{icon} {_esc(title)}</div>'
                + empty_html
                + "</div>"
            )
        rows = []
        for dev, item in items:
            extra = extra_fn(item) if extra_fn else ""
            rows.append(_summary_item(dev, item, extra))
        count_badge = f'<span class="summary-count">{len(items)}</span>'
        return (
            f'<div class="summary-section {css_class}">'
            f'<div class="summary-section-title">{icon} {_esc(title)} {count_badge}</div>'
            + "".join(rows)
            + "</div>"
        )

    def _nak_extra(item: ActivityItem) -> str:
        if not item.conversation:
            return ""
        pushback = []
        for rc in item.conversation.review_comments:
            combined = " ".join(rc.sentiment_signals + rc.tags_given).lower()
            if "nak" in combined or "nack" in combined or rc.sentiment == Sentiment.CONTENTIOUS:
                pushback.append(_esc(rc.author))
        if pushback:
            return f'<span class="summary-pushback-names">Pushback: {", ".join(pushback)}</span>'
        return ""

    def _activity_extra(item: ActivityItem) -> str:
        if not item.conversation:
            return ""
        n = item.conversation.participant_count
        return f'<span class="summary-activity-count">{n} participants</span>'

    def _maintainer_extra(item: ActivityItem) -> str:
        rcs = _maintainer_rcs(item)
        if not rcs:
            return ""
        seen: set[str] = set()
        parts_m = []
        for rc in rcs:
            if rc.author in seen:
                continue
            seen.add(rc.author)
            icon = _SENTIMENT_ICONS.get(rc.sentiment, "")
            parts_m.append(f'{_esc(rc.author)}{(" " + icon) if icon else ""}')
        return f'<span class="summary-maintainer-names">&#9733; {", ".join(parts_m)}</span>'

    sections = "".join(filter(None, [
        _sub_section("New Patch Series", "&#128196;", new_series, "summary-new"),
        _sub_section("Strong Pushback / NAKs", "&#9940;", pushed_back, "summary-nak", _nak_extra),
        _sub_section("High Activity", "&#128293;", high_activity, "summary-active", _activity_extra),
        _sub_section("Maintainer Comments", "&#9733;", maintainer_items, "summary-maintainer", _maintainer_extra),
    ]))

    return (
        '<div class="daily-summary">'
        '<h3 class="daily-summary-title">&#9733; Today\'s Highlights</h3>'
        f'<div class="summary-grid">{sections}</div>'
        '</div>'
    )


def generate_html_report(
    daily_report: DailyReport,
    review_links: Optional[dict[str, str]] = None,
    log_filename: Optional[str] = None,
    progress_status: Optional[dict] = None,
) -> str:
    """Generate a complete self-contained HTML report.

    Args:
        daily_report: The DailyReport data structure.
        review_links: Optional mapping of message_id -> slug for review detail pages.
                      When provided, review comments are rendered as compact summaries
                      with links to per-patchset detail pages.
        log_filename: Optional log filename (e.g. "2026-02-15.log") for "View log" link.

    Returns:
        Complete HTML string ready to write to file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_date = daily_report.date

    # Build LLM info string for display in the report
    if daily_report.llm_backends:
        llm_label = " + ".join(
            f"{backend}/{model}" for backend, model in daily_report.llm_backends
        )
    else:
        llm_label = ""

    developer_sections = "\n".join(
        _render_developer_section(dr, review_links=review_links, report_date=report_date)
        for dr in sorted(daily_report.developer_reports, key=lambda r: r.developer.name.lower())
    )

    stats_section = _render_statistics(daily_report)
    summary_section = _render_daily_summary(daily_report, review_links=review_links, report_date=report_date)

    progress_html = ""
    refresh_script = ""
    if progress_status:
        done  = progress_status.get("done", 0)
        total = progress_status.get("total", 0)
        cur   = progress_status.get("current", "")
        last_updated = progress_status.get("last_updated", "")
        cur_line = (
            f' &mdash; <span class="progress-current">Processing: {_esc(cur)}</span>'
        ) if cur else ""
        updated_line = (
            f'<span class="progress-updated">Updated: {_esc(last_updated)}</span>'
        ) if last_updated else ""
        progress_html = (
            f'<div class="progress-banner">'
            f'<span class="progress-spinner">&#x27F3;</span>'
            f'<span class="progress-count">{done} / {total} developers complete</span>'
            f'{cur_line}'
            f'<span class="refresh-controls">'
            f'Auto-refresh:'
            f'<button class="refresh-btn" data-secs="60">1 min</button>'
            f'<button class="refresh-btn" data-secs="300">5 min</button>'
            f'<button class="refresh-btn" data-secs="600">10 min</button>'
            f'<span class="refresh-countdown-wrap">in <span id="refresh-countdown">60s</span></span>'
            f'</span>'
            f'{updated_line}'
            f'</div>'
        )
        refresh_script = """
<script>
(function () {
    var DEFAULT_SECS = 60;
    var remaining, timer;

    function stored() {
        try { return parseInt(localStorage.getItem('lkml_refresh_secs')) || DEFAULT_SECS; }
        catch (e) { return DEFAULT_SECS; }
    }
    function store(s) {
        try { localStorage.setItem('lkml_refresh_secs', s); } catch (e) {}
    }
    function tick() {
        remaining--;
        update();
        if (remaining <= 0) { clearInterval(timer); location.reload(); }
    }
    function start(secs) {
        clearInterval(timer);
        remaining = secs;
        update();
        timer = setInterval(tick, 1000);
    }
    function update() {
        var el = document.getElementById('refresh-countdown');
        if (el) el.textContent = remaining + 's';
    }
    function choose(secs) {
        store(secs);
        document.querySelectorAll('.refresh-btn').forEach(function (b) {
            b.classList.toggle('active', parseInt(b.dataset.secs) === secs);
        });
        start(secs);
    }
    document.addEventListener('DOMContentLoaded', function () {
        var secs = stored();
        document.querySelectorAll('.refresh-btn').forEach(function (b) {
            b.addEventListener('click', function () { choose(parseInt(b.dataset.secs)); });
            if (parseInt(b.dataset.secs) === secs) b.classList.add('active');
        });
        start(secs);
    });
})();
</script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LKML Activity Report - {_esc(daily_report.date)}{' [' + _esc(llm_label) + ']' if llm_label else ''}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 1.8em;
            margin-bottom: 4px;
            color: #1a1a1a;
        }}
        h2 {{
            font-size: 1.1em;
            color: #666;
            font-weight: normal;
            margin-bottom: 24px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: #fff;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .contributors-section {{
            margin-bottom: 32px;
        }}
        .contributors-section h3 {{
            font-size: 0.95em;
            color: #666;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
        }}
        .contributors-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88em;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .contributors-table th {{
            background: #f4f6f8;
            color: #555;
            font-weight: 600;
            text-align: left;
            padding: 8px 14px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .contributors-table th.num {{
            text-align: center;
        }}
        .contributors-table td {{
            padding: 7px 14px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: middle;
        }}
        .contributors-table td.num {{
            text-align: center;
            font-weight: 700;
            color: #2c3e50;
        }}
        .contributors-table td.zero {{
            color: #ccc;
            font-weight: normal;
        }}
        .contributors-table tr:last-child td {{
            border-bottom: none;
        }}
        .contributors-table tr:hover td {{
            background: #f9f9f9;
        }}
        .contributors-table a {{
            color: #2980b9;
            text-decoration: none;
            font-weight: 500;
        }}
        .contributors-table a:hover {{
            text-decoration: underline;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: 700;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 4px;
        }}
        .developer-section {{
            background: #fff;
            border-radius: 8px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .developer-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
            border-bottom: 1px solid #eee;
        }}
        .developer-header h3 {{
            font-size: 1.1em;
            margin: 0;
        }}
        .inactive-badge {{
            font-size: 0.75em;
            padding: 2px 10px;
            border-radius: 12px;
            background: #e2e3e5;
            color: #383d41;
        }}
        .active-badge {{
            font-size: 0.75em;
            padding: 2px 10px;
            border-radius: 12px;
            background: #cce5ff;
            color: #004085;
        }}
        details {{
            border-top: 1px solid #f0f0f0;
        }}
        summary {{
            cursor: pointer;
            padding: 12px 20px;
            font-weight: 600;
            font-size: 0.9em;
            color: #555;
            user-select: none;
        }}
        summary:hover {{ background: #fafafa; }}
        .count {{ color: #999; font-weight: normal; }}
        .activity-item {{
            padding: 10px 20px;
            border-bottom: 1px solid #f5f5f5;
        }}
        .activity-item:last-child {{ border-bottom: none; }}
        .item-link {{
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.9em;
        }}
        .item-link:hover {{ text-decoration: underline; }}
        .badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 8px;
            vertical-align: middle;
        }}
        .ack-badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 8px;
            background: #d1ecf1;
            color: #0c5460;
        }}
        .patch-count {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 8px;
            background: #e8daef;
            color: #6c3483;
        }}
        .version-chain {{
            display: inline-flex;
            align-items: center;
            gap: 3px;
            margin-left: 8px;
            vertical-align: middle;
        }}
        .version-badge {{
            display: inline-block;
            padding: 1px 7px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            background: #e8e8e8;
            color: #555;
            text-decoration: none;
        }}
        .version-badge:hover {{
            background: #d0d0d0;
        }}
        .version-badge.latest {{
            background: #d4edda;
            color: #155724;
        }}
        .version-badge.latest:hover {{
            background: #b8dfc5;
        }}
        .version-arrow {{
            font-size: 0.75em;
            color: #aaa;
            line-height: 1;
        }}
        .series-patches {{
            margin-top: 8px;
            margin-left: 4px;
        }}
        .series-patches-toggle {{
            cursor: pointer;
            font-size: 0.82em;
            color: #555;
            user-select: none;
            list-style: none;
        }}
        .series-patches-toggle::-webkit-details-marker {{ display: none; }}
        .series-patches-toggle::before {{ content: "\\25B6\\00A0"; font-size: 0.75em; }}
        details[open] .series-patches-toggle::before {{ content: "\\25BC\\00A0"; }}
        .series-patch-list {{
            list-style: none;
            margin: 6px 0 0 16px;
            padding: 0 0 0 10px;
            border-left: 2px solid #e0e0e0;
        }}
        .series-patch-item {{
            padding: 3px 0;
            font-size: 0.85em;
            line-height: 1.4;
        }}
        .si-num {{
            font-family: monospace;
            color: #888;
            font-size: 0.9em;
        }}
        .si-link {{ color: #0366d6; text-decoration: none; }}
        .si-link:hover {{ text-decoration: underline; }}
        .si-reviewers {{
            display: inline-block;
            font-size: 0.8em;
            background: #e8f4fd;
            color: #0366d6;
            border-radius: 8px;
            padding: 0 6px;
            margin-left: 4px;
        }}
        .conversation-summary {{
            margin-top: 6px;
            padding-left: 12px;
            border-left: 3px solid #eee;
            font-size: 0.82em;
            color: #666;
        }}
        .conversation-summary ul {{
            margin: 4px 0 4px 16px;
            padding: 0;
        }}
        .conversation-summary li {{
            margin-bottom: 2px;
        }}
        .participants {{
            margin-right: 12px;
        }}
        .signals {{
            color: #999;
            font-style: italic;
        }}
        .patch-summary {{
            margin-top: 6px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.82em;
            color: #444;
            line-height: 1.6;
        }}
        .patch-summary p {{
            margin: 0 0 6px 0;
        }}
        .patch-summary p:last-child {{
            margin-bottom: 0;
        }}
        .progress-badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-left: 8px;
            vertical-align: middle;
        }}
        .analysis-source-badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.65em;
            font-weight: 600;
            margin-left: 6px;
            vertical-align: middle;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        .progress-detail {{
            margin-top: 4px;
            font-size: 0.8em;
            color: #555;
            padding-left: 4px;
        }}
        .progress-icon {{
            font-size: 0.7em;
            color: #888;
        }}
        .review-comments {{
            margin-top: 8px;
            border-left: 3px solid #ddd;
            padding-left: 12px;
        }}
        .review-comments-header {{
            font-size: 0.78em;
            color: #888;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .review-comment {{
            margin-bottom: 8px;
            padding: 6px 10px;
            background: #fafbfc;
            border-radius: 4px;
            font-size: 0.82em;
        }}
        .review-comment:last-child {{
            margin-bottom: 0;
        }}
        .review-comment-header {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
        }}
        .review-author {{
            font-weight: 600;
            color: #333;
        }}
        .reply-to-label {{
            font-size: 0.78em;
            color: #888;
            font-style: italic;
        }}
        .inline-review-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 500;
            background: #e3f2fd;
            color: #1565c0;
        }}
        .maintainer-badge {{
            display: inline-block;
            padding: 0 7px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 600;
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
        }}
        .maintainer-badge-inline {{
            color: #b8860b;
            font-size: 0.9em;
        }}
        .maintainer-review-note {{
            display: inline-block;
            margin: 4px 0 6px 0;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: 600;
            background: #fff8e1;
            color: #795548;
            border-left: 3px solid #ffc107;
        }}
        .review-tag-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 500;
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .review-comment-text {{
            color: #555;
            line-height: 1.5;
        }}
        .review-comment-signals {{
            margin-top: 3px;
            font-size: 0.9em;
            color: #999;
            font-style: italic;
        }}
        .raw-body-toggle {{
            margin-top: 4px;
            font-size: 0.85em;
            border-top: none;
        }}
        .raw-body-toggle summary {{
            cursor: pointer;
            color: #666;
            padding: 2px 0;
            font-weight: 500;
            font-size: 0.9em;
        }}
        .raw-body-toggle summary:hover {{
            color: #333;
            background: transparent;
        }}
        .raw-body-text {{
            white-space: pre-wrap;
            font-size: 1em;
            background: #f8f8f8;
            padding: 8px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
            margin-top: 4px;
            line-height: 1.5;
            color: #444;
            border: 1px solid #e8e8e8;
        }}
        .review-comment-footer {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 4px;
        }}
        .lore-link {{
            display: inline-block;
            margin-top: 4px;
            font-size: 0.82em;
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
            white-space: nowrap;
        }}
        .lore-link:hover {{
            text-decoration: underline;
            color: #0056b3;
        }}
        .review-comments-compact {{
            margin-top: 8px;
            border-left: 3px solid #ddd;
            padding: 6px 12px;
            font-size: 0.82em;
            color: #666;
        }}
        .reviewer-list {{
            color: #555;
        }}
        .review-detail-link {{
            margin-top: 4px;
        }}
        .review-detail-link a {{
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }}
        .review-detail-link a:hover {{
            text-decoration: underline;
        }}
        .activity-item.ongoing {{
            border-left: 3px solid #6f42c1;
            background: #faf8ff;
        }}
        .ongoing-badge {{
            display: inline-block;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 0.7em;
            font-weight: 600;
            margin-right: 6px;
            background: #e8daef;
            color: #6f42c1;
            vertical-align: middle;
        }}
        .submitted-date {{
            font-size: 0.72em;
            color: #999;
            margin-right: 6px;
            vertical-align: middle;
        }}
        .last-comment-date {{
            font-size: 0.72em;
            color: #999;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .last-comment-date::before {{
            content: "·";
            margin-right: 6px;
            color: #ccc;
        }}
        .today-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #ff6b35, #f7c948);
            color: #fff;
            font-size: 0.72em;
            font-weight: 700;
            letter-spacing: 0.05em;
            border-radius: 8px;
            padding: 1px 8px;
            margin-right: 8px;
            vertical-align: middle;
            box-shadow: 0 1px 4px rgba(255,107,53,0.35);
        }}
        .stale-badge {{
            display: inline-block;
            background: #f8f0e0;
            color: #a07830;
            font-size: 0.72em;
            font-weight: 600;
            border-radius: 8px;
            padding: 1px 8px;
            margin-right: 8px;
            vertical-align: middle;
            border: 1px solid #e8d5b0;
        }}
        .si-date {{
            font-size: 0.68em;
            color: #aaa;
            margin-left: 6px;
            vertical-align: middle;
        }}
        .si-date + .si-date::before {{
            content: "·";
            margin-right: 5px;
            color: #ccc;
        }}
        .no-activity {{
            padding: 10px 20px;
            color: #aaa;
            font-size: 0.85em;
            font-style: italic;
        }}
        .errors {{
            padding: 8px 20px;
        }}
        .error-msg {{
            color: #721c24;
            background: #f8d7da;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.82em;
            margin-bottom: 4px;
        }}
        footer {{
            text-align: center;
            color: #aaa;
            font-size: 0.8em;
            margin-top: 32px;
            padding: 16px;
        }}
        footer a {{
            color: #999;
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}
        .llm-badge {{
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
            border-radius: 12px;
            padding: 2px 12px;
            font-size: 0.75em;
            font-weight: 600;
            vertical-align: middle;
            margin-left: 8px;
        }}
        .back-to-index {{
            margin-bottom: 16px;
        }}
        .back-to-index a {{
            color: #555;
            text-decoration: none;
            font-size: 0.85em;
        }}
        .back-to-index a:hover {{
            color: #1565c0;
            text-decoration: underline;
        }}
        .analysis-mode {{
            font-size: 0.85em;
            color: #888;
            margin-top: 4px;
        }}
        .log-link {{
            font-size: 0.85em;
            margin-top: 4px;
        }}
        .log-link a {{
            color: #0366d6;
            text-decoration: none;
        }}
        .log-link a:hover {{
            text-decoration: underline;
        }}
        .llm-analyses {{
            margin-top: 8px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .llm-analysis {{
            border: 1px solid #e0e0e0;
            border-left: 4px solid #90caf9;
            border-radius: 6px;
            padding: 10px 14px;
            background: #fafbfc;
        }}
        .llm-analysis:nth-child(2) {{
            border-left-color: #a5d6a7;
        }}
        .llm-analysis:nth-child(3) {{
            border-left-color: #ce93d8;
        }}
        .llm-analysis-header {{
            font-weight: 700;
            font-size: 0.78em;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            padding-bottom: 6px;
            border-bottom: 1px solid #eee;
        }}
        .progress-banner {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 6px;
            padding: 12px 20px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #856404;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .progress-spinner {{
            display: inline-block;
            animation: spin 1.2s linear infinite;
            font-style: normal;
        }}
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to   {{ transform: rotate(360deg); }}
        }}
        .progress-count {{
            font-weight: 700;
        }}
        .progress-current {{
            color: #0c5460;
            font-style: italic;
        }}
        .progress-updated {{
            font-size: 0.85em;
            color: #6c5500;
            opacity: 0.75;
        }}
        .refresh-controls {{
            margin-left: auto;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.82em;
            flex-shrink: 0;
        }}
        .refresh-btn {{
            background: #fff8e1;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 1px 9px;
            font-size: 0.9em;
            cursor: pointer;
            color: #856404;
            transition: background 0.15s;
        }}
        .refresh-btn:hover {{ background: #ffe082; }}
        .refresh-btn.active {{ background: #ffc107; color: #fff; font-weight: 700; }}
        .refresh-countdown-wrap {{
            color: #6c5500;
            opacity: 0.75;
            font-size: 0.9em;
            min-width: 32px;
        }}
        /* Daily summary / highlights */
        .daily-summary {{
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 32px;
        }}
        .daily-summary-title {{
            font-size: 1em;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}
        .summary-section {{
            border-radius: 6px;
            padding: 12px 14px;
            border-left: 4px solid #ccc;
            background: #fafafa;
        }}
        .summary-new {{
            border-left-color: #2980b9;
            background: #f0f7ff;
        }}
        .summary-nak {{
            border-left-color: #c0392b;
            background: #fff5f5;
        }}
        .summary-active {{
            border-left-color: #e67e22;
            background: #fff9f0;
        }}
        .summary-maintainer {{
            border-left-color: #f39c12;
            background: #fffbf0;
        }}
        .summary-section-title {{
            font-size: 0.82em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #555;
            margin-bottom: 10px;
        }}
        .summary-count {{
            display: inline-block;
            background: rgba(0,0,0,0.08);
            border-radius: 8px;
            padding: 0 6px;
            font-size: 0.85em;
            font-weight: 600;
            color: #555;
            margin-left: 4px;
        }}
        .summary-item {{
            padding: 6px 0;
            border-bottom: 1px solid rgba(0,0,0,0.06);
            font-size: 0.85em;
            line-height: 1.5;
        }}
        .summary-item:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}
        .summary-item-meta {{
            font-size: 0.82em;
            color: #888;
            margin-top: 2px;
        }}
        .summary-dev-link {{
            color: #2980b9;
            text-decoration: none;
            font-weight: 500;
        }}
        .summary-dev-link:hover {{
            text-decoration: underline;
        }}
        .summary-pushback-names {{
            display: inline-block;
            background: #fde8e8;
            color: #a93226;
            border-radius: 8px;
            padding: 0 7px;
            font-size: 0.88em;
            font-weight: 500;
        }}
        .summary-activity-count {{
            display: inline-block;
            background: #fdebd0;
            color: #935116;
            border-radius: 8px;
            padding: 0 7px;
            font-size: 0.88em;
            font-weight: 500;
        }}
        .summary-maintainer-names {{
            display: inline-block;
            background: #fef9e7;
            color: #9a7d0a;
            border-radius: 8px;
            padding: 0 7px;
            font-size: 0.88em;
            font-weight: 500;
            border: 1px solid #f9e79f;
        }}
        .summary-empty {{
            color: #aaa;
            font-style: italic;
            font-size: 0.9em;
            padding: 10px 4px;
        }}
    </style>
</head>
<body>
    <p class="back-to-index"><a href="index.html">&#8592; Back to Index</a></p>
    <h1>LKML Activity Report{' <span class="llm-badge">LLM: ' + _esc(llm_label) + '</span>' if llm_label else ''}</h1>
    <h2>{_esc(daily_report.date)} &mdash; Generated {_esc(now)}</h2>
    {'<p class="analysis-mode">Analysis: LLM-enriched (' + _esc(llm_label) + ')</p>' if llm_label else '<p class="analysis-mode">Analysis: Heuristic</p>'}
    {'<p class="log-link"><a href="logs/' + _esc(log_filename) + '">View generation log</a></p>' if log_filename else ''}
    {progress_html}

    {stats_section}

    {summary_section}

    {developer_sections}

    <footer>
        Generated in {daily_report.generation_time_seconds:.1f}s
        &bull; {len(daily_report.developer_reports)} developers tracked
        &bull; Data from lore.kernel.org
        {'&bull; LLM: ' + _esc(llm_label) if llm_label else '&bull; Heuristic analysis'}
        {'&bull; <a href="logs/' + _esc(log_filename) + '">Log</a>' if log_filename else ''}
    </footer>
{refresh_script}
</body>
</html>"""
