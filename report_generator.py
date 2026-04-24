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
    Developer,
    DeveloperReport,
    DiscussionProgress,
    LLMAnalysis,
    ReviewComment,
    Sentiment,
    WeeklyReport,
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


def _render_compact_reviews(
    conv: ConversationSummary,
    review_link: str,
    item: Optional[ActivityItem] = None,
) -> str:
    """Render review comments as a thread tree.

    For series cover letters (item.series_items populated): renders a lore-style
    2-level tree — cover letter row at top with its general discussion chips, then
    each individual patch indented below with its own per-patch reviewer chips.
    This mirrors the thread tree shown at the bottom of lore.kernel.org pages.

    For single patches: flat list of reviewer name chips with optional depth based
    on reply_to relationships.
    """
    parts: list[str] = []
    parts.append('<div class="review-comments-compact">')

    # ── Series cover letter: lore-style 2-level tree ─────────────────────────
    if item is not None and getattr(item, "series_items", None):
        _PATCH_NUM_RE_LOCAL = re.compile(
            r"\[(?:RFC\s+)?PATCH[^\]]*\s+(\d+)/\d+\]|\[RFC[^\]]*\b(\d+)/\d+\]",
            re.IGNORECASE,
        )
        total = item.series_patch_count or len(item.series_items)
        base_url = review_link.split("#")[0] if review_link else ""

        def _chip_date(msg_date: str) -> str:
            if not msg_date:
                return ""
            try:
                d = datetime.strptime(msg_date, "%Y-%m-%d")
                return d.strftime("%b") + " " + str(d.day)
            except (ValueError, TypeError):
                return msg_date

        def _reviewer_chips_html(rcs: list, link_base: str) -> str:
            chip_parts: list[str] = []
            seen_author_date: set[tuple] = set()
            for rc in rcs:
                dedup_key = (rc.author, rc.message_date)
                if dedup_key in seen_author_date:
                    continue
                seen_author_date.add(dedup_key)
                date_label = _chip_date(rc.message_date)
                date_anchor = rc.message_date or ""
                href = (
                    f"{_esc(link_base)}#{_esc(date_anchor)}"
                    if date_anchor and link_base
                    else _esc(link_base or item.url)
                )
                star = "&#9733;&nbsp;" if rc.is_maintainer else ""
                tags_html = ""
                for tag in rc.tags_given:
                    tags_html += f'<span class="rtree-tag">{_esc(tag)}</span>'
                if rc.has_inline_review:
                    tags_html += '<span class="rtree-tag rtree-tag-inline">Inline</span>'
                sent_class = f"rtree-sent-{rc.sentiment.value}" if rc.sentiment else ""
                chip_parts.append(
                    f'<span class="lore-reviewer-chip {sent_class}">'
                    f'<a href="{href}" class="rtree-link" title="{_esc(rc.author)}">'
                    f'<span class="rtree-author">{star}{_esc(rc.author)}</span>'
                    + (f'<span class="rtree-date">{_esc(date_label)}</span>' if date_label else "")
                    + f'</a>'
                    + tags_html
                    + f'</span>'
                )
            return "\n".join(chip_parts)

        parts.append('<div class="lore-tree">')

        # Build set of series message_ids to exclude from cover letter chips —
        # the author-batched grouping rolls in the patch-submission messages
        # themselves as "author" entries; those belong under each patch row.
        series_msg_ids: set[str] = {si.message_id for si in item.series_items}
        series_msg_ids.add(item.message_id)  # cover letter itself

        def _cover_chips(rcs: list) -> list:
            """Filter out entries whose message_id is a raw patch submission."""
            return [
                rc for rc in rcs
                if rc.message_id not in series_msg_ids
            ]

        # Cover letter row (00/N) with series-level discussion chips
        cl_title = re.sub(r"^\[.*?\]\s*", "", item.subject)
        cl_href = _esc(base_url) if base_url else _esc(item.url)
        parts.append('<div class="lore-tree-root">')
        parts.append(
            f'<a href="{cl_href}" class="lore-tree-link">'
            f'<span class="lore-patch-num">[00/{total}]</span> {_esc(cl_title)}'
            f'</a>'
        )
        cl_chips = _cover_chips(conv.review_comments) if conv.review_comments else []
        if cl_chips:
            parts.append('<div class="lore-tree-chips">')
            parts.append(_reviewer_chips_html(cl_chips, base_url))
            parts.append('</div>')
        parts.append('</div>')  # lore-tree-root

        # Individual patch rows as children
        for si in item.series_items:
            m = _PATCH_NUM_RE_LOCAL.search(si.subject)
            num = (m.group(1) or m.group(2)) if m else "?"
            clean_title = re.sub(r"^\[.*?\]\s*", "", si.subject)
            si_slug = message_id_to_slug(si.message_id)
            si_href_str = f"reviews/{si_slug}.html"
            si_href = _esc(si_href_str)
            si_conv = si.conversation
            has_reviews = si_conv and si_conv.review_comments

            parts.append('<div class="lore-tree-child">')
            parts.append(
                f'<span class="lore-tree-connector">&#9492;&#9472;</span>'
                f'<a href="{si_href}" class="lore-tree-link">'
                f'<span class="lore-patch-num">[{_esc(str(num))}/{total}]</span>'
                f' {_esc(clean_title)}'
                f'</a>'
            )
            if has_reviews:
                parts.append('<div class="lore-tree-chips lore-tree-chips-child">')
                parts.append(_reviewer_chips_html(si_conv.review_comments, si_href_str))
                parts.append('</div>')
            parts.append('</div>')  # lore-tree-child

        parts.append('</div>')  # lore-tree

        if review_link:
            parts.append(
                f'<div class="review-detail-link">'
                f'<a href="{_esc(review_link)}">View full discussion &rarr;</a>'
                f'</div>'
            )
        parts.append('</div>')  # review-comments-compact
        return "\n".join(parts)

    # ── Single patch (or series without series_items): reviewer chip tree ─────

    # ── Build the indentation tree from reply_to relationships ──────────────
    rcs = conv.review_comments
    if not rcs:
        parts.append("</div>")
        return "\n".join(parts)

    # Map lowercased author name → list of indices (same as _build_tree logic)
    name_idx: dict[str, list[int]] = {}
    for i, rc in enumerate(rcs):
        name = rc.author.lower().strip()
        name_idx.setdefault(name, []).append(i)

    # parent_of[i] = parent index, or None for roots; cycle-safe
    parent_of: dict[int, Optional[int]] = {}
    for i, rc in enumerate(rcs):
        rt = rc.reply_to.lower().strip() if rc.reply_to else ""
        parent_of[i] = None
        if rt:
            for pi in name_idx.get(rt, []):
                if pi != i:
                    # Cycle check: walk ancestors of pi
                    cur: Optional[int] = pi
                    seen_anc: set[int] = set()
                    is_cycle = False
                    while cur is not None:
                        if cur == i:
                            is_cycle = True
                            break
                        if cur in seen_anc:
                            break
                        seen_anc.add(cur)
                        cur = parent_of.get(cur)
                    if not is_cycle:
                        parent_of[i] = pi
                        break

    # Compute depth for each node by walking up parent chain
    def _depth(i: int) -> int:
        depth = 0
        cur = parent_of.get(i)
        visited: set[int] = set()
        while cur is not None and cur not in visited:
            depth += 1
            visited.add(cur)
            cur = parent_of.get(cur)
        return depth

    # Base URL for deep-links: strip existing fragment from review_link
    base_url = review_link.split("#")[0]

    # ── Render each comment as a tree row ───────────────────────────────────
    parts.append('<div class="review-tree">')

    _TREE_CONNECTORS = ["└─", "└─", "└─", "└─", "└─"]  # per-depth, uniform style

    for i, rc in enumerate(rcs):
        depth = _depth(i)

        # Deep-link to this comment's date anchor on the review page
        date_anchor = rc.message_date or ""
        href = f"{base_url}#{_esc(date_anchor)}" if date_anchor else _esc(review_link)

        # Format date as "Apr 20"
        date_label = ""
        if rc.message_date:
            try:
                d = datetime.strptime(rc.message_date, "%Y-%m-%d")
                date_label = d.strftime("%b") + " " + str(d.day)
            except (ValueError, TypeError):
                date_label = rc.message_date

        # Author with optional maintainer star
        star = '&#9733;&nbsp;' if rc.is_maintainer else ''
        author_html = f'{star}{_esc(rc.author)}'

        # Inline tag badges (Reviewed-by, NAK, etc.)
        tag_html = ""
        if rc.tags_given:
            seen_tags: set[str] = set()
            for tag in rc.tags_given:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tag_html += f'<span class="rtree-tag">{_esc(tag)}</span>'
        if rc.has_inline_review:
            tag_html += '<span class="rtree-tag rtree-tag-inline">Inline</span>'

        # Sentiment colour on the row
        sent_class = f"rtree-sent-{rc.sentiment.value}" if rc.sentiment else ""

        indent_style = f"padding-left:{depth * 18 + 4}px"
        parts.append(
            f'<div class="review-tree-row {sent_class}" style="{indent_style}">'
            f'{"<span class=\"rtree-connector\">\u2514\u2500</span>" if depth > 0 else ""}'
            f'<a href="{href}" class="rtree-link" title="Go to {_esc(rc.author)}\'s comment">'
            f'<span class="rtree-author">{author_html}</span>'
            f'{"<span class=\"rtree-date\">" + _esc(date_label) + "</span>" if date_label else ""}'
            f'</a>'
            f'{tag_html}'
            f'</div>'
        )

    parts.append('</div>')  # review-tree

    # Full-thread link at bottom
    parts.append(
        f'<div class="review-detail-link">'
        f'<a href="{_esc(review_link)}">View full discussion &rarr;</a>'
        f'</div>'
    )
    parts.append("</div>")  # review-comments-compact
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
    conv: ConversationSummary,
    review_link: Optional[str] = None,
    reviews_collapsed: bool = False,
    item: Optional[ActivityItem] = None,
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
        if reviews_collapsed:
            n = len(conv.review_comments)
            label = f"{n} reviewer comment{'s' if n != 1 else ''}"
            parts.append(f'<details class="reviews-collapsed"><summary class="reviews-collapsed-toggle">{label}</summary>')
            parts.append(_render_compact_reviews(conv, review_link, item=item))
            parts.append("</details>")
        else:
            parts.append(_render_compact_reviews(conv, review_link, item=item))
    elif conv.review_comments:
        n = len(conv.review_comments)
        label = f"{n} reviewer comment{'s' if n != 1 else ''} &mdash; {conv.participant_count} participants"
        if reviews_collapsed:
            parts.append(f'<details class="reviews-collapsed"><summary class="reviews-collapsed-toggle">{label}</summary>')
        parts.append('<div class="review-comments">')
        parts.append(f'<div class="review-comments-header">'
                     f'{conv.participant_count} participants</div>')
        for rc in conv.review_comments:
            parts.append(_render_review_comment(rc))
        parts.append("</div>")
        if reviews_collapsed:
            parts.append("</details>")
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
    analysis: LLMAnalysis,
    review_link: Optional[str] = None,
    reviews_collapsed: bool = False,
    item: Optional[ActivityItem] = None,
) -> str:
    """Render a single LLM analysis as an attributed card."""
    parts: list[str] = []
    parts.append('<div class="llm-analysis">')
    parts.append(f'<div class="llm-analysis-header">{_esc(analysis.label)}</div>')
    parts.append(_render_conversation_body(
        analysis.conversation, review_link=review_link,
        reviews_collapsed=reviews_collapsed, item=item,
    ))
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
    review_links: Optional[dict[str, str]] = None, report_date: str = "",
    reviews_collapsed: bool = False,
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
            si_dates_html = ""
            si_badges_html = ""
            si_contributors_html = ""
            if si.submitted_date:
                if report_date and si.submitted_date == report_date:
                    si_badges_html += '<span class="today-badge">&#128293; TODAY</span>'
                else:
                    si_dates_html += f'<span class="si-date">Submitted {_esc(si.submitted_date)}</span>'
            if si.conversation and si.conversation.review_comments:
                rc_dates = [rc.message_date for rc in si.conversation.review_comments if rc.message_date]
                if rc_dates:
                    last_rc = max(rc_dates)
                    if report_date and last_rc == report_date:
                        si_badges_html += '<span class="today-badge">&#128293; TODAY</span>'
                    else:
                        si_dates_html += f'<span class="si-date">Last comment {_esc(last_rc)}</span>'
                # Sentiment badge
                si_badges_html += _sentiment_badge(si.conversation.sentiment)
                # Discussion progress badge
                if si.conversation.discussion_progress:
                    si_badges_html += _progress_badge(si.conversation.discussion_progress)
                # Contributor list: unique reviewer names with maintainer star
                seen_contributors: set[str] = set()
                contributor_parts = []
                for rc in si.conversation.review_comments:
                    if rc.author in seen_contributors:
                        continue
                    seen_contributors.add(rc.author)
                    star = "&#9733;&nbsp;" if rc.is_maintainer else ""
                    contributor_parts.append(
                        f'<span class="si-contributor">{star}{_esc(rc.author)}</span>'
                    )
                if contributor_parts:
                    si_contributors_html = (
                        '<span class="si-contributors">'
                        + " ".join(contributor_parts)
                        + "</span>"
                    )
            parts.append(
                f'<li class="series-patch-item">'
                f'<span class="si-num">[{_esc(str(num))}/{total}]</span> '
                f'<a href="{si_href}" target="_blank" class="si-link">'
                f'{_esc(clean_title)}</a>'
                f'{(" " + si_badges_html) if si_badges_html else ""}'
                f'{(" " + si_dates_html) if si_dates_html else ""}'
                f'{(" " + si_contributors_html) if si_contributors_html else ""}'
                f'</li>'
            )
        parts.append('</ul>')
        parts.append('</details>')

    review_link = _get_review_link(item, review_links, report_date)

    # Multi-LLM analyses (when --llm-all produces multiple results)
    if len(item.llm_analyses) > 1:
        parts.append('<div class="llm-analyses">')
        for analysis in item.llm_analyses:
            parts.append(_render_llm_analysis_card(
                analysis, review_link=review_link,
                reviews_collapsed=reviews_collapsed, item=item,
            ))
        parts.append("</div>")
    elif item.conversation:
        # Single analysis (single backend or heuristic)
        parts.append(_render_conversation_body(
            item.conversation, review_link=review_link,
            reviews_collapsed=reviews_collapsed, item=item,
        ))

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
    review_links: Optional[dict[str, str]] = None, report_date: str = "",
    reviews_collapsed: bool = False,
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
                item, section_type, review_links=review_links, report_date=report_date,
                reviews_collapsed=reviews_collapsed,
            ))

    parts.append("</details>")
    return "\n".join(parts)


def _render_developer_section(
    dev_report: DeveloperReport,
    review_links: Optional[dict[str, str]] = None,
    report_date: str = "",
    collapsed: bool = False,
    reviews_collapsed: bool = False,
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

    if collapsed:
        # Entire body collapses under a <details> — header becomes the <summary>
        badge = (
            f'<span class="active-badge">{total} items</span>' if total
            else '<span class="inactive-badge">No activity</span>'
        )
        parts.append(f'<details class="dev-section-details">')
        parts.append(
            f'<summary class="developer-header">'
            f'<h3>{_esc(dev_report.developer.name)}</h3>'
            f'{badge}'
            f'</summary>'
        )
    else:
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

    # Activity sections: in weekly mode nothing is open by default
    open_subs = not collapsed
    parts.append(_render_activity_section(
        dev_report.patches_submitted, "Patches Submitted", "patch",
        open_by_default=open_subs, review_links=review_links, report_date=report_date,
        reviews_collapsed=reviews_collapsed,
    ))
    parts.append(_render_activity_section(
        dev_report.discussions_posted, "Discussions / RFCs", "discussion",
        open_by_default=open_subs, review_links=review_links, report_date=report_date,
        reviews_collapsed=reviews_collapsed,
    ))
    parts.append(_render_activity_section(
        dev_report.patches_reviewed, "Reviews Given", "review",
        review_links=review_links, report_date=report_date,
        reviews_collapsed=reviews_collapsed,
    ))
    parts.append(_render_activity_section(
        dev_report.patches_acked, "Acks / Tags Given", "ack",
        review_links=review_links, report_date=report_date,
        reviews_collapsed=reviews_collapsed,
    ))

    if collapsed:
        parts.append("</details>")

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
            f'<a href="#{anchor}" class="item-link">{_esc(item.subject)}</a>'
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


def _render_engineer_digest(
    report: DailyReport,
    review_links: Optional[dict[str, str]] = None,
    report_date: str = "",
) -> str:
    """Render the per-engineer activity digest card grid (Option A)."""

    def _linked_subject(item: ActivityItem, anchor: str = "") -> str:
        subj = _esc(item.subject)
        if anchor:
            return f'<a href="#{anchor}" class="item-link">{subj}</a>'
        url = item.url or ""
        if url:
            return f'<a href="{_esc(url)}" target="_blank" rel="noopener" class="item-link">{subj}</a>'
        return subj

    def _card(dr: "DeveloperReport") -> str:
        anchor = _name_to_anchor(dr.developer.name)
        ps = dr.patches_submitted
        pr = dr.patches_reviewed
        pa = dr.patches_acked
        disc = dr.discussions_posted
        is_active = bool(ps or pr or pa or disc)

        # ── badge row ──
        badges = []
        if ps:
            badges.append(f'<span class="digest-badge digest-badge-sub">{len(ps)} submitted</span>')
        if pr:
            badges.append(f'<span class="digest-badge digest-badge-rev">{len(pr)} reviewed</span>')
        if pa:
            badges.append(f'<span class="digest-badge digest-badge-ack">{len(pa)} acked</span>')
        if disc:
            badges.append(f'<span class="digest-badge digest-badge-disc">{len(disc)} discussed</span>')
        if not is_active:
            badges.append('<span class="digest-badge digest-badge-quiet">no activity</span>')

        badges_html = "".join(badges)

        # ── bullet list (max 5) ──
        bullets: list[str] = []

        # New submissions (non-ongoing)
        new_patches = [p for p in ps if not p.is_ongoing]
        if new_patches:
            if len(new_patches) == 1:
                p = new_patches[0]
                extra = ""
                if p.series_patch_count and p.series_patch_count > 1:
                    extra = f' <span class="digest-meta">({p.series_patch_count} patches)</span>'
                if p.patch_version > 1:
                    extra += f' <span class="digest-meta">(v{p.patch_version})</span>'
                bullets.append(f'<li data-icon="📝">{_linked_subject(p, anchor)}{extra}</li>')
            else:
                titles = ", ".join(
                    f'<a href="#{anchor}" class="item-link">{_esc(p.subject[:50])}</a>'
                    for p in new_patches[:3]
                )
                more = f" +{len(new_patches)-3} more" if len(new_patches) > 3 else ""
                bullets.append(f'<li data-icon="📝">Submitted {len(new_patches)} patches: {titles}{more}</li>')

        # Ongoing with activity
        ongoing = [p for p in ps if p.is_ongoing]
        for p in ongoing[:2]:
            conv = p.conversation
            meta_parts = []
            if conv:
                if conv.participant_count > 1:
                    meta_parts.append(f"{conv.participant_count} participants")
                if conv.sentiment and conv.sentiment != Sentiment.NEUTRAL:
                    meta_parts.append(f'<span class="digest-sent-{conv.sentiment.value}">{conv.sentiment.value.replace("_"," ")}</span>')
            meta = f' <span class="digest-meta">({", ".join(meta_parts)})</span>' if meta_parts else ""
            bullets.append(f'<li data-icon="🔄">{_linked_subject(p, anchor)}{meta}</li>')

        # Reviews given
        if pr:
            if len(pr) <= 2:
                for p in pr:
                    bullets.append(f'<li data-icon="👁">{_linked_subject(p, anchor)}</li>')
            else:
                bullets.append(
                    f'<li data-icon="👁">Reviewed {len(pr)} patches: '
                    + ", ".join(
                        f'<a href="#{anchor}" class="item-link">'
                        f'{_esc(p.subject[:45])}</a>'
                        for p in pr[:2]
                    )
                    + (f" +{len(pr)-2} more" if len(pr) > 2 else "")
                    + "</li>"
                )

        # Acks
        if pa:
            if len(pa) == 1:
                bullets.append(f'<li data-icon="✅">{_linked_subject(pa[0], anchor)} <span class="digest-meta">(acked)</span></li>')
            else:
                bullets.append(f'<li data-icon="✅">Acked {len(pa)} patches</li>')

        # Discussions
        if disc:
            if len(disc) == 1:
                bullets.append(f'<li data-icon="💬">{_linked_subject(disc[0], anchor)}</li>')
            else:
                bullets.append(f'<li data-icon="💬">Participated in {len(disc)} discussions</li>')

        # Cap at 5 bullets
        bullets = bullets[:5]

        if not is_active:
            body = '<p class="digest-quiet-msg">No upstream activity today.</p>'
        else:
            body = "<ul>" + "".join(bullets) + "</ul>"

        quiet_class = " digest-card--quiet" if not is_active else ""
        return (
            f'<div class="digest-card{quiet_class}">'
            f'<div class="digest-card-header">'
            f'<span class="digest-name">{_esc(dr.developer.name)}</span>'
            f'<div class="digest-badges">{badges_html}</div>'
            f'</div>'
            f'<div class="digest-card-body">{body}</div>'
            f'</div>'
        )

    # Sort: active first (by total activity desc), then quiet alphabetically
    active_drs = sorted(
        [dr for dr in report.developer_reports if dr.patches_submitted or dr.patches_reviewed or dr.patches_acked or dr.discussions_posted],
        key=lambda dr: len(dr.patches_submitted) + len(dr.patches_reviewed) + len(dr.patches_acked) + len(dr.discussions_posted),
        reverse=True,
    )
    quiet_drs = sorted(
        [dr for dr in report.developer_reports if not (dr.patches_submitted or dr.patches_reviewed or dr.patches_acked or dr.discussions_posted)],
        key=lambda dr: dr.developer.name.lower(),
    )

    cards = "".join(_card(dr) for dr in active_drs + quiet_drs)

    return (
        '<div class="engineer-digest">'
        '<h3 class="engineer-digest-title">&#128101; Engineer Activity Digest</h3>'
        f'<div class="digest-grid">{cards}</div>'
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
    digest_section = _render_engineer_digest(daily_report, review_links=review_links, report_date=report_date)

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
        .si-contributors {{
            display: inline-flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-left: 6px;
            vertical-align: middle;
        }}
        .si-contributor {{
            display: inline-block;
            font-size: 0.78em;
            background: #e8f4fd;
            color: #0366d6;
            border-radius: 8px;
            padding: 1px 7px;
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
        /* ── Compact review tree ── */
        .review-tree {{
            margin: 4px 0 2px 0;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .review-tree-row {{
            display: flex;
            align-items: center;
            gap: 4px;
            border-radius: 4px;
            padding: 2px 4px;
        }}
        .review-tree-row:hover {{
            background: rgba(0,0,0,0.04);
        }}
        .rtree-connector {{
            color: #bbb;
            font-size: 0.8em;
            flex-shrink: 0;
            margin-right: 2px;
        }}
        .rtree-link {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            text-decoration: none;
            color: inherit;
        }}
        .rtree-link:hover .rtree-author {{
            color: #0366d6;
            text-decoration: underline;
        }}
        .rtree-author {{
            font-weight: 600;
            font-size: 0.85em;
            color: #333;
        }}
        .rtree-date {{
            font-size: 0.75em;
            color: #999;
            white-space: nowrap;
        }}
        .rtree-tag {{
            display: inline-block;
            font-size: 0.7em;
            font-weight: 600;
            padding: 0 5px;
            border-radius: 8px;
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
            margin-left: 2px;
            white-space: nowrap;
        }}
        .rtree-tag-inline {{
            background: #e3f2fd;
            color: #1565c0;
            border-color: #bbdefb;
        }}
        .rtree-sent-contentious .rtree-author {{ color: #c62828; }}
        .rtree-sent-needs_work  .rtree-author {{ color: #e65100; }}
        .rtree-sent-positive    .rtree-author {{ color: #2e7d32; }}
        /* ── Lore-style series tree ── */
        .lore-tree {{
            margin: 6px 0 2px 0;
        }}
        .lore-tree-root {{
            margin-bottom: 4px;
        }}
        .lore-tree-child {{
            padding-left: 16px;
            margin-bottom: 3px;
        }}
        .lore-tree-connector {{
            color: #bbb;
            font-family: monospace;
            margin-right: 3px;
            font-size: 0.85em;
        }}
        .lore-tree-link {{
            color: #1a56db;
            text-decoration: none;
            font-size: 0.88em;
        }}
        .lore-tree-link:hover {{
            text-decoration: underline;
        }}
        .lore-patch-num {{
            color: #666;
            font-size: 0.82em;
            font-family: monospace;
            margin-right: 3px;
        }}
        .lore-tree-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 2px 0 1px 0;
            margin-top: 1px;
        }}
        .lore-tree-chips-child {{
            padding-left: 18px;
        }}
        .lore-reviewer-chip {{
            display: inline-flex;
            align-items: center;
            gap: 3px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 0 6px;
            font-size: 0.78em;
        }}
        .lore-reviewer-chip .rtree-author {{
            font-size: 0.9em;
        }}
        .review-detail-link {{
            margin-top: 5px;
        }}
        .review-detail-link a {{
            color: #0366d6;
            text-decoration: none;
            font-size: 0.8em;
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
        /* ── Engineer digest ── */
        .engineer-digest {{
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 32px;
        }}
        .engineer-digest-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 16px;
        }}
        .digest-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 12px;
        }}
        .digest-card {{
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 12px 14px;
            background: #fafbfc;
        }}
        .digest-card--quiet {{
            opacity: 0.45;
        }}
        .digest-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e1e4e8;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .digest-name {{
            font-weight: 700;
            font-size: 0.9rem;
            color: #1a1a2e;
        }}
        .digest-badges {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }}
        .digest-badge {{
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 0.68rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .digest-badge-sub  {{ background: #e6ffed; color: #22863a; border: 1px solid #bef5cb; }}
        .digest-badge-rev  {{ background: #ddf4ff; color: #0969da; border: 1px solid #b6e3ff; }}
        .digest-badge-ack  {{ background: #fff8c5; color: #9a6700; border: 1px solid #eac54f; }}
        .digest-badge-disc {{ background: #f3e8ff; color: #6e40c9; border: 1px solid #d2a8ff; }}
        .digest-badge-quiet {{ background: #f6f8fa; color: #8c959f; border: 1px solid #d0d7de; }}
        .digest-card-body ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .digest-card-body ul li {{
            font-size: 0.78rem;
            color: #57606a;
            padding: 3px 0 3px 20px;
            position: relative;
            line-height: 1.5;
            border-bottom: 1px solid #f0f2f4;
        }}
        .digest-card-body ul li:last-child {{ border-bottom: none; }}
        .digest-card-body ul li::before {{
            content: attr(data-icon);
            position: absolute;
            left: 0;
        }}
        .digest-card-body ul li a {{
            color: #24292f;
        }}
        .digest-card-body ul li a:hover {{
            color: #0969da;
            text-decoration: underline;
        }}
        .digest-meta {{
            color: #8c959f;
            font-size: 0.72rem;
        }}
        .digest-sent-needs_work {{ color: #d97706; }}
        .digest-sent-positive   {{ color: #16a34a; }}
        .digest-sent-contentious {{ color: #dc2626; }}
        .digest-quiet-msg {{
            font-size: 0.78rem;
            color: #8c959f;
            font-style: italic;
            padding: 2px 0;
        }}
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

    {digest_section}

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


# ---------------------------------------------------------------------------
# Weekly report rendering
# ---------------------------------------------------------------------------

def _merge_developer_reports(daily_reports: list[DailyReport]) -> list[DeveloperReport]:
    """Merge per-day DeveloperReports into one DeveloperReport per developer.

    Items are deduplicated by message_id: when the same patch appears on
    multiple days (ongoing activity) the most-recently-dated copy is kept.
    Each kept ActivityItem retains its original .date so callers can show
    which day an item originated from.
    """
    # group by developer name → per-list accumulator
    buckets: dict[str, dict] = {}   # name → {"dev": Developer, lists: {}}

    def _bucket(dev: Developer) -> dict:
        if dev.name not in buckets:
            buckets[dev.name] = {
                "dev": dev,
                "patches_submitted": {},   # message_id → ActivityItem
                "patches_reviewed": {},
                "patches_acked": {},
                "discussions_posted": {},
            }
        return buckets[dev.name]

    def _merge_list(target: dict, items: list[ActivityItem]) -> None:
        """Keep the most-recent ActivityItem for each message_id."""
        for item in items:
            mid = item.message_id or item.subject   # fallback to subject if no id
            existing = target.get(mid)
            if existing is None or item.date > existing.date:
                target[mid] = item

    for dr in daily_reports:
        for dev_report in dr.developer_reports:
            b = _bucket(dev_report.developer)
            _merge_list(b["patches_submitted"],  dev_report.patches_submitted)
            _merge_list(b["patches_reviewed"],   dev_report.patches_reviewed)
            _merge_list(b["patches_acked"],      dev_report.patches_acked)
            _merge_list(b["discussions_posted"], dev_report.discussions_posted)

    merged: list[DeveloperReport] = []
    for name in sorted(buckets.keys(), key=str.lower):
        b = buckets[name]
        merged.append(DeveloperReport(
            developer=b["dev"],
            patches_submitted=sorted(b["patches_submitted"].values(),  key=lambda x: x.date),
            patches_reviewed= sorted(b["patches_reviewed"].values(),   key=lambda x: x.date),
            patches_acked=    sorted(b["patches_acked"].values(),      key=lambda x: x.date),
            discussions_posted=sorted(b["discussions_posted"].values(), key=lambda x: x.date),
        ))
    return merged


def _render_weekly_summary(weekly_report: "WeeklyReport") -> str:
    """Render the top-level weekly highlights section (4 categories + LLM narrative)."""

    # Collect all activity items across every day
    all_items: list[tuple[str, ActivityItem]] = []
    for dr in weekly_report.daily_reports:
        for dev_report in dr.developer_reports:
            for item in dev_report.patches_submitted + dev_report.discussions_posted:
                all_items.append((dev_report.developer.name, item))

    # 1. New patch series: v1, not ongoing, across the whole week
    new_series = [
        (dev, item) for dev, item in all_items
        if (item.activity_type == ActivityType.PATCH_SUBMITTED
            and not item.is_ongoing
            and item.patch_version == 1)
    ]
    # Deduplicate by message_id — a v1 could appear in multiple days' data
    seen_new: set[str] = set()
    deduped_new: list[tuple[str, ActivityItem]] = []
    for dev, item in new_series:
        key = item.message_id or item.subject
        if key not in seen_new:
            seen_new.add(key)
            deduped_new.append((dev, item))

    # 2. Strong pushback / NAKs
    pushed_back: list[tuple[str, ActivityItem]] = []
    seen_nak: set[str] = set()
    for dev, item in all_items:
        if _is_contentious_item(item):
            key = item.message_id or item.subject
            if key not in seen_nak:
                seen_nak.add(key)
                pushed_back.append((dev, item))

    # 3. High activity — deduplicate by message_id, keep peak participant_count,
    #    count how many distinct days the thread appeared
    activity_map: dict[str, dict] = {}
    for dr in weekly_report.daily_reports:
        for dev_report in dr.developer_reports:
            for item in dev_report.patches_submitted + dev_report.discussions_posted:
                if not item.conversation:
                    continue
                cnt = item.conversation.participant_count
                if cnt < 2:
                    continue
                key = item.message_id or item.subject
                if key not in activity_map:
                    activity_map[key] = {"dev": dev_report.developer.name, "item": item, "peak": cnt, "days": 1}
                else:
                    activity_map[key]["days"] += 1
                    if cnt > activity_map[key]["peak"]:
                        activity_map[key]["peak"] = cnt
                        activity_map[key]["item"] = item

    high_activity = sorted(activity_map.values(), key=lambda x: x["peak"], reverse=True)[:8]

    # 4. Maintainer involvement
    def _maintainer_rcs(item: ActivityItem) -> list[ReviewComment]:
        if not item.conversation:
            return []
        return [rc for rc in item.conversation.review_comments if rc.is_maintainer]

    maintainer_items: list[tuple[str, ActivityItem]] = []
    seen_maint: set[str] = set()
    for dev, item in all_items:
        if _maintainer_rcs(item):
            key = item.message_id or item.subject
            if key not in seen_maint:
                seen_maint.add(key)
                maintainer_items.append((dev, item))

    # --- Rendering helpers (reuse daily-summary CSS classes) ---

    _SENTIMENT_ICONS = {
        Sentiment.POSITIVE: "&#10003;",
        Sentiment.NEEDS_WORK: "&#9888;",
        Sentiment.CONTENTIOUS: "&#10007;",
        Sentiment.NEUTRAL: "",
    }

    def _date_chip(item: ActivityItem) -> str:
        if not item.date:
            return ""
        try:
            d = datetime.strptime(item.date, "%Y-%m-%d")
            label = d.strftime("%b %-d")   # e.g. "Apr 7"
        except (ValueError, TypeError):
            label = item.date
        return f'<span class="date-chip">{_esc(label)}</span>'

    def _summary_item_weekly(dev_name: str, item: ActivityItem, extra_html: str = "") -> str:
        anchor = _name_to_anchor(dev_name)
        dev_link = f'<a href="#{anchor}" class="summary-dev-link">{_esc(dev_name)}</a>'
        subject_html = f'<a href="#{anchor}" class="item-link">{_esc(item.subject)}</a>'
        badges = _date_chip(item)
        if item.series_patch_count and item.series_patch_count > 1:
            badges += f'<span class="patch-count">{item.series_patch_count} patches</span>'
        if item.patch_version > 1:
            badges += f'<span class="version-badge latest">v{item.patch_version}</span>'
        if item.conversation:
            badges += _sentiment_badge(item.conversation.sentiment)
        meta = f'<div class="summary-item-meta">by {dev_link}{(" &mdash; " + extra_html) if extra_html else ""}</div>'
        return f'<div class="summary-item">{subject_html}{badges}{meta}</div>'

    def _sub_section_weekly(title: str, icon: str, items: list, css_class: str, extra_fn=None) -> str:
        if not items:
            return (
                f'<div class="summary-section {css_class}">'
                f'<div class="summary-section-title">{icon} {_esc(title)}</div>'
                '<div class="summary-empty">None this week</div>'
                '</div>'
            )
        rows = []
        for entry in items:
            if isinstance(entry, dict):
                dev, item = entry["dev"], entry["item"]
            else:
                dev, item = entry
            extra = extra_fn(item) if extra_fn else ""
            rows.append(_summary_item_weekly(dev, item, extra))
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

    def _activity_extra_weekly(entry: dict) -> str:
        peak = entry["peak"]
        days = entry["days"]
        days_str = f", {days} days" if days > 1 else ""
        return f'<span class="summary-activity-count">{peak} participants{days_str}</span>'

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

    # Narrative card (full-width, shown first if present)
    narrative_html = ""
    if weekly_report.narrative:
        narrative_html = (
            '<div class="summary-narrative">'
            f'<div class="summary-section-title">&#128196; Weekly Summary</div>'
            f'<div class="summary-narrative-text">{_esc(weekly_report.narrative)}</div>'
            '</div>'
        )

    # High-activity section needs special handling (entries are dicts)
    high_act_html = ""
    if not high_activity:
        high_act_html = (
            '<div class="summary-section summary-active">'
            '<div class="summary-section-title">&#128293; High Activity</div>'
            '<div class="summary-empty">None this week</div>'
            '</div>'
        )
    else:
        rows = []
        for entry in high_activity:
            extra = _activity_extra_weekly(entry)
            rows.append(_summary_item_weekly(entry["dev"], entry["item"], extra))
        count_badge = f'<span class="summary-count">{len(high_activity)}</span>'
        high_act_html = (
            f'<div class="summary-section summary-active">'
            f'<div class="summary-section-title">&#128293; High Activity {count_badge}</div>'
            + "".join(rows)
            + "</div>"
        )

    sections = narrative_html + "".join(filter(None, [
        _sub_section_weekly("New Patch Series", "&#128196;", deduped_new, "summary-new"),
        _sub_section_weekly("Strong Pushback / NAKs", "&#9940;", pushed_back, "summary-nak", _nak_extra),
        high_act_html,
        _sub_section_weekly("Maintainer Comments", "&#9733;", maintainer_items, "summary-maintainer", _maintainer_extra),
    ]))

    week_label = f"Week of {weekly_report.week_start} \u2013 {weekly_report.week_end}"
    return (
        '<div class="daily-summary">'
        f'<h3 class="daily-summary-title">&#9733; {_esc(week_label)} Highlights</h3>'
        f'<div class="summary-grid">{sections}</div>'
        '</div>'
    )


def generate_weekly_html_report(
    weekly_report: "WeeklyReport",
    review_links: Optional[dict[str, str]] = None,
) -> str:
    """Generate a complete self-contained HTML weekly report.

    Args:
        weekly_report: The WeeklyReport data structure.
        review_links: Optional mapping of message_id -> slug for review detail pages.

    Returns:
        Complete HTML string ready to write to file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # LLM label
    seen_backends: set[tuple] = set()
    unique_backends = []
    for pair in weekly_report.llm_backends:
        if pair not in seen_backends:
            seen_backends.add(pair)
            unique_backends.append(pair)
    llm_label = " + ".join(f"{b}/{m}" for b, m in unique_backends) if unique_backends else ""

    week_label = f"Week of {weekly_report.week_start} \u2013 {weekly_report.week_end}"
    days_covered = len(weekly_report.daily_reports)

    # Merge all developers across all days
    merged_devs = _merge_developer_reports(weekly_report.daily_reports)

    # Stats cards
    total_discussions = sum(
        len(dr.discussions_posted) for dr in merged_devs
    )
    discussion_card = ""
    if total_discussions:
        discussion_card = f"""
        <div class="stat-card">
            <div class="stat-number">{total_discussions}</div>
            <div class="stat-label">Discussions / RFCs</div>
        </div>"""

    stats_section = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{weekly_report.total_patches}</div>
            <div class="stat-label">Patches Submitted</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{weekly_report.total_reviews}</div>
            <div class="stat-label">Reviews Given</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{weekly_report.total_acks}</div>
            <div class="stat-label">Acks Given</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{days_covered}</div>
            <div class="stat-label">Days Covered</div>
        </div>
        {discussion_card}
    </div>"""

    # Weekly highlights
    highlights_section = _render_weekly_summary(weekly_report)

    # Per-developer sections — collapsed by default; reviewer feedback also collapsed
    developer_sections = "\n".join(
        _render_developer_section(
            dr,
            review_links=review_links,
            report_date=weekly_report.week_end,
            collapsed=True,
            reviews_collapsed=True,
        )
        for dr in merged_devs
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LKML Weekly Report - {_esc(weekly_report.iso_week)}{' [' + _esc(llm_label) + ']' if llm_label else ''}</title>
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
        h1 {{ font-size: 1.8em; margin-bottom: 4px; color: #1a1a1a; }}
        h2 {{ font-size: 1.1em; color: #666; font-weight: normal; margin-bottom: 24px; }}
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
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{ font-size: 2.5em; font-weight: 700; color: #2c3e50; }}
        .stat-label  {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
        .back-to-index {{ margin-bottom: 16px; }}
        .back-to-index a {{ color: #2980b9; text-decoration: none; }}
        .back-to-index a:hover {{ text-decoration: underline; }}
        .llm-badge {{
            font-size: 0.5em; vertical-align: middle;
            background: #e8f5e9; color: #2e7d32;
            border: 1px solid #a5d6a7; border-radius: 12px;
            padding: 2px 10px; font-weight: 600;
        }}
        .analysis-mode {{ font-size: 0.85em; color: #888; margin-bottom: 8px; }}
        /* ── Developer sections ── */
        .developer-section {{
            background: #fff; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 24px; margin-bottom: 24px;
        }}
        .developer-name {{ font-size: 1.4em; font-weight: 700; color: #2c3e50; margin-bottom: 16px; }}
        .activity-section {{ margin-bottom: 20px; }}
        .activity-title {{
            font-size: 0.9em; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.05em; color: #666;
            border-bottom: 2px solid #eee; padding-bottom: 6px; margin-bottom: 12px;
        }}
        .patch-item {{ margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #f0f0f0; }}
        .patch-item:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
        .patch-subject {{ font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }}
        .patch-subject a {{ color: #2980b9; text-decoration: none; }}
        .patch-subject a:hover {{ text-decoration: underline; }}
        .patch-meta {{ font-size: 0.82em; color: #888; margin-bottom: 6px; }}
        .patch-meta a {{ color: #2980b9; text-decoration: none; }}
        .patch-meta a:hover {{ text-decoration: underline; }}
        .patch-count {{
            display: inline-block; background: #e8f4fd; color: #2471a3;
            border-radius: 4px; padding: 1px 7px; font-size: 0.78em;
            font-weight: 600; margin-left: 6px; vertical-align: middle;
        }}
        .version-badge {{
            display: inline-block; border-radius: 4px; padding: 1px 7px;
            font-size: 0.78em; font-weight: 600; margin-left: 4px; vertical-align: middle;
        }}
        .version-badge.latest {{ background: #e8f5e9; color: #2e7d32; }}
        .sentiment-badge {{
            display: inline-block; border-radius: 4px; padding: 1px 8px;
            font-size: 0.78em; font-weight: 600; margin-left: 4px; vertical-align: middle;
        }}
        .ongoing-badge {{
            display: inline-block; background: #fff3e0; color: #ef6c00;
            border-radius: 4px; padding: 1px 7px; font-size: 0.75em;
            font-weight: 600; margin-left: 4px; vertical-align: middle;
        }}
        .today-badge {{
            display: inline-block; background: #ffebee; color: #c62828;
            border-radius: 4px; padding: 1px 7px; font-size: 0.75em;
            font-weight: 600; margin-left: 4px; vertical-align: middle;
        }}
        .stale-badge {{
            display: inline-block; background: #f3f3f3; color: #888;
            border-radius: 4px; padding: 1px 7px; font-size: 0.75em;
            font-weight: 600; margin-left: 4px; vertical-align: middle;
        }}
        .conversation-summary {{
            background: #f9f9f9; border-radius: 6px; padding: 12px;
            margin-top: 8px; font-size: 0.88em;
        }}
        .conversation-header {{
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
            margin-bottom: 6px;
        }}
        .key-point {{ margin: 2px 0; color: #555; }}
        .key-point::before {{ content: "• "; color: #999; }}
        .progress-label {{
            display: inline-block; border-radius: 4px; padding: 1px 8px;
            font-size: 0.78em; font-weight: 600; border: 1px solid;
        }}
        .review-comment-block {{
            background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
            padding: 10px 12px; margin-top: 8px; font-size: 0.85em;
        }}
        .reviewer-name {{ font-weight: 600; color: #2c3e50; }}
        .reviewer-sentiment {{ margin-left: 6px; }}
        .maintainer-star {{ color: #f39c12; margin-left: 4px; }}
        .reviewer-tags {{ margin-top: 4px; }}
        .tag-badge {{
            display: inline-block; border-radius: 3px; padding: 1px 5px;
            font-size: 0.75em; font-weight: 600; margin-right: 3px;
            background: #e8f5e9; color: #2e7d32;
        }}
        .reviewer-summary {{ margin-top: 4px; color: #555; line-height: 1.5; }}
        .item-link {{ color: #2980b9; text-decoration: none; }}
        .item-link:hover {{ text-decoration: underline; }}
        /* ── Engineer digest ── */
        .engineer-digest {{
            background: #fff; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px; margin-bottom: 32px;
        }}
        .engineer-digest-title {{ font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 16px; }}
        .digest-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
        .digest-card {{ border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 14px; background: #fafbfc; }}
        .digest-card--quiet {{ opacity: 0.45; }}
        .digest-card-header {{
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #e1e4e8;
            flex-wrap: wrap; gap: 6px;
        }}
        .digest-name {{ font-weight: 700; font-size: 0.9rem; color: #1a1a2e; }}
        .digest-badges {{ display: flex; gap: 4px; flex-wrap: wrap; }}
        .digest-badge {{ padding: 2px 7px; border-radius: 10px; font-size: 0.68rem; font-weight: 600; white-space: nowrap; }}
        .digest-badge-sub  {{ background: #e6ffed; color: #22863a; border: 1px solid #bef5cb; }}
        .digest-badge-rev  {{ background: #ddf4ff; color: #0969da; border: 1px solid #b6e3ff; }}
        .digest-badge-ack  {{ background: #fff8c5; color: #9a6700; border: 1px solid #eac54f; }}
        .digest-badge-disc {{ background: #f3e8ff; color: #6e40c9; border: 1px solid #d2a8ff; }}
        .digest-badge-quiet{{ background: #f6f8fa; color: #8c959f; border: 1px solid #d0d7de; }}
        .digest-card-body ul {{ list-style: none; padding: 0; margin: 0; }}
        .digest-card-body ul li {{
            font-size: 0.78rem; color: #57606a; padding: 3px 0 3px 20px;
            position: relative; line-height: 1.5; border-bottom: 1px solid #f0f2f4;
        }}
        .digest-card-body ul li:last-child {{ border-bottom: none; }}
        .digest-card-body ul li::before {{ content: attr(data-icon); position: absolute; left: 0; }}
        .digest-card-body ul li a {{ color: #24292f; }}
        .digest-card-body ul li a:hover {{ color: #0969da; text-decoration: underline; }}
        .digest-meta {{ color: #8c959f; font-size: 0.72rem; }}
        .digest-sent-needs_work  {{ color: #d97706; }}
        .digest-sent-positive    {{ color: #16a34a; }}
        .digest-sent-contentious {{ color: #dc2626; }}
        .digest-quiet-msg {{ font-size: 0.78rem; color: #8c959f; font-style: italic; padding: 2px 0; }}
        /* ── Daily/Weekly highlights ── */
        .daily-summary {{
            background: #fff; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px; margin-bottom: 32px;
        }}
        .daily-summary-title {{
            font-size: 1em; font-weight: 700; color: #1a1a1a;
            margin-bottom: 16px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0;
        }}
        .summary-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
        }}
        .summary-section {{ border-radius: 6px; padding: 12px 14px; border-left: 4px solid #ccc; background: #fafafa; }}
        .summary-new        {{ border-left-color: #2980b9; background: #f0f7ff; }}
        .summary-nak        {{ border-left-color: #c0392b; background: #fff5f5; }}
        .summary-active     {{ border-left-color: #e67e22; background: #fff9f0; }}
        .summary-maintainer {{ border-left-color: #f39c12; background: #fffbf0; }}
        .summary-section-title {{
            font-size: 0.82em; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: #555; margin-bottom: 10px;
        }}
        .summary-count {{
            display: inline-block; background: rgba(0,0,0,0.08); border-radius: 8px;
            padding: 0 6px; font-size: 0.85em; font-weight: 600; color: #555; margin-left: 4px;
        }}
        .summary-item {{
            padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06);
            font-size: 0.85em; line-height: 1.5;
        }}
        .summary-item:last-child {{ border-bottom: none; padding-bottom: 0; }}
        .summary-item-meta {{ font-size: 0.82em; color: #888; margin-top: 2px; }}
        .summary-dev-link {{ color: #2980b9; text-decoration: none; font-weight: 500; }}
        .summary-dev-link:hover {{ text-decoration: underline; }}
        .summary-pushback-names {{
            display: inline-block; background: #fde8e8; color: #a93226;
            border-radius: 8px; padding: 0 7px; font-size: 0.88em; font-weight: 500;
        }}
        .summary-activity-count {{
            display: inline-block; background: #fdebd0; color: #935116;
            border-radius: 8px; padding: 0 7px; font-size: 0.88em; font-weight: 500;
        }}
        .summary-maintainer-names {{
            display: inline-block; background: #fef9e7; color: #9a7d0a;
            border-radius: 8px; padding: 0 7px; font-size: 0.88em; font-weight: 500;
            border: 1px solid #f9e79f;
        }}
        .summary-empty {{ color: #aaa; font-style: italic; font-size: 0.9em; padding: 10px 4px; }}
        /* ── Weekly-specific ── */
        .summary-narrative {{
            grid-column: 1 / -1;
            background: #f0f4f8; border-left: 4px solid #4a90d9;
            border-radius: 6px; padding: 14px 16px;
        }}
        .summary-narrative-text {{ font-size: 0.95em; line-height: 1.7; color: #333; margin-top: 8px; }}
        .date-chip {{
            display: inline-block; font-size: 0.7em; background: #e8f0fe; color: #1a56db;
            border-radius: 4px; padding: 1px 5px; margin-right: 5px;
            font-weight: 600; vertical-align: middle;
        }}
        /* ── Collapsed developer sections ── */
        .dev-section-details > summary {{
            list-style: none;
            cursor: pointer;
        }}
        .dev-section-details > summary::-webkit-details-marker {{ display: none; }}
        .dev-section-details > summary::before {{
            content: "▶";
            font-size: 0.7em;
            color: #888;
            margin-right: 8px;
            display: inline-block;
            transition: transform 0.15s ease;
            vertical-align: middle;
        }}
        .dev-section-details[open] > summary::before {{
            transform: rotate(90deg);
        }}
        .dev-section-details > summary .developer-header,
        .dev-section-details > summary.developer-header {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }}
        /* ── Collapsed review feedback ── */
        .reviews-collapsed {{
            margin-top: 8px;
        }}
        .reviews-collapsed-toggle {{
            font-size: 0.82em;
            color: #2980b9;
            cursor: pointer;
            list-style: none;
            display: inline-block;
            padding: 2px 0;
        }}
        .reviews-collapsed-toggle::-webkit-details-marker {{ display: none; }}
        .reviews-collapsed-toggle::before {{
            content: "▶ ";
            font-size: 0.85em;
            color: #888;
        }}
        .reviews-collapsed[open] .reviews-collapsed-toggle::before {{
            content: "▼ ";
        }}
        /* ── Contributor table ── */
        .contributors-section {{ margin-bottom: 32px; }}
        .contributors-section h3 {{ font-size: 1em; font-weight: 700; color: #1a1a1a; margin-bottom: 12px; }}
        .contributors-table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .contributors-table th {{ background: #f6f8fa; padding: 8px 12px; text-align: left; font-weight: 600; color: #555; border-bottom: 2px solid #e1e4e8; }}
        .contributors-table td {{ padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }}
        .contributors-table tr:last-child td {{ border-bottom: none; }}
        .contributors-table th.num, .contributors-table td.num {{ text-align: right; }}
        .contributors-table td.zero {{ color: #ccc; }}
        .contributors-table a {{ color: #2980b9; text-decoration: none; }}
        .contributors-table a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <p class="back-to-index"><a href="index.html">&#8592; Back to Index</a></p>
    <h1>LKML Weekly Report{' <span class="llm-badge">LLM: ' + _esc(llm_label) + '</span>' if llm_label else ''}</h1>
    <h2>{_esc(week_label)} &mdash; {weekly_report.iso_week} &mdash; Generated {_esc(now)}</h2>
    {'<p class="analysis-mode">Analysis: LLM-enriched (' + _esc(llm_label) + ')</p>' if llm_label else '<p class="analysis-mode">Analysis: Heuristic</p>'}

    {stats_section}

    {highlights_section}

    {developer_sections}

    <footer>
        {weekly_report.iso_week} &bull; {days_covered} days covered
        &bull; {len(merged_devs)} developers tracked
        &bull; Data from lore.kernel.org
        {'&bull; LLM: ' + _esc(llm_label) if llm_label else '&bull; Heuristic analysis'}
    </footer>
</body>
</html>"""
