"""HTML report generator for LKML daily activity reports."""

import html
import re
from datetime import datetime
from typing import Optional

from models import (
    ActivityItem,
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

    # Inline review indicator
    if rc.has_inline_review:
        parts.append('<span class="inline-review-badge">Inline Review</span>')

    # Tags given
    for tag in rc.tags_given:
        parts.append(f'<span class="review-tag-badge">{_esc(tag)}</span>')

    # Per-reviewer sentiment badge
    parts.append(_sentiment_badge(rc.sentiment))
    parts.append('</div>')

    # Comment summary text
    if rc.summary:
        parts.append(f'<div class="review-comment-text">{_esc(rc.summary)}</div>')

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

    # Participant count
    reviewer_descs = []
    for rc in conv.review_comments:
        desc = _esc(rc.author)
        annotations = []
        if rc.tags_given:
            annotations.extend(rc.tags_given)
        if rc.has_inline_review:
            annotations.append("Inline Review")
        if annotations:
            desc += f' ({", ".join(_esc(a) for a in annotations)})'
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


def _render_activity_item(
    item: ActivityItem, section_type: str,
    review_links: Optional[dict[str, str]] = None, report_date: str = ""
) -> str:
    parts = []
    css_class = "activity-item ongoing" if item.is_ongoing else "activity-item"
    parts.append(f'<div class="{css_class}">')

    # Ongoing badge and submitted date
    if item.is_ongoing and item.submitted_date:
        parts.append(f'<span class="ongoing-badge">Ongoing</span>')
        parts.append(f'<span class="submitted-date">Submitted {_esc(item.submitted_date)}</span>')
    elif item.is_ongoing:
        parts.append(f'<span class="ongoing-badge">Ongoing</span>')

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
        for item in items:
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
    )

    parts = []
    parts.append('<div class="developer-section">')
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
        dev_report.patches_reviewed, "Reviews Given", "review",
        review_links=review_links, report_date=report_date
    ))
    parts.append(_render_activity_section(
        dev_report.patches_acked, "Acks / Tags Given", "ack",
        review_links=review_links, report_date=report_date
    ))

    parts.append("</div>")
    return "\n".join(parts)


def _build_contributor_tooltip(names_with_counts: list[tuple[str, int]]) -> str:
    """Build an HTML tooltip listing contributors and their counts."""
    if not names_with_counts:
        return ""
    lines = []
    for name, count in sorted(names_with_counts, key=lambda x: -x[1]):
        lines.append(f'<div class="tooltip-row"><span>{_esc(name)}</span><span class="tooltip-count">{count}</span></div>')
    return (
        '<div class="stat-tooltip">'
        + "\n".join(lines)
        + "</div>"
    )


def _render_statistics(report: DailyReport) -> str:
    active_devs = sum(
        1
        for dr in report.developer_reports
        if dr.patches_submitted or dr.patches_reviewed or dr.patches_acked
    )
    total_devs = len(report.developer_reports)

    # Build per-card contributor lists
    patch_contributors = [
        (dr.developer.name, len(dr.patches_submitted))
        for dr in report.developer_reports
        if dr.patches_submitted
    ]
    review_contributors = [
        (dr.developer.name, len(dr.patches_reviewed))
        for dr in report.developer_reports
        if dr.patches_reviewed
    ]
    ack_contributors = [
        (dr.developer.name, len(dr.patches_acked))
        for dr in report.developer_reports
        if dr.patches_acked
    ]
    active_names = [
        (dr.developer.name, len(dr.patches_submitted) + len(dr.patches_reviewed) + len(dr.patches_acked))
        for dr in report.developer_reports
        if dr.patches_submitted or dr.patches_reviewed or dr.patches_acked
    ]

    patch_tooltip = _build_contributor_tooltip(patch_contributors)
    review_tooltip = _build_contributor_tooltip(review_contributors)
    ack_tooltip = _build_contributor_tooltip(ack_contributors)
    active_tooltip = _build_contributor_tooltip(active_names)

    return f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{report.total_patches}</div>
            <div class="stat-label">Patches Submitted</div>
            {patch_tooltip}
        </div>
        <div class="stat-card">
            <div class="stat-number">{report.total_reviews}</div>
            <div class="stat-label">Reviews Given</div>
            {review_tooltip}
        </div>
        <div class="stat-card">
            <div class="stat-number">{report.total_acks}</div>
            <div class="stat-label">Acks Given</div>
            {ack_tooltip}
        </div>
        <div class="stat-card">
            <div class="stat-number">{active_devs}/{total_devs}</div>
            <div class="stat-label">Active Developers</div>
            {active_tooltip}
        </div>
    </div>
    """


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
        all_items = (
            dr.patches_submitted + dr.patches_reviewed + dr.patches_acked
        )
        for item in all_items:
            conv = item.conversation
            if not conv or not conv.review_comments:
                continue
            reviews = []
            for rc in conv.review_comments:
                reviews.append({
                    "author": rc.author,
                    "summary": rc.summary,
                    "sentiment": rc.sentiment.value,
                    "sentiment_signals": rc.sentiment_signals,
                    "has_inline_review": rc.has_inline_review,
                    "tags_given": rc.tags_given,
                })
            results.append({
                "message_id": item.message_id,
                "slug": message_id_to_slug(item.message_id),
                "subject": item.subject,
                "url": item.url,
                "developer": dr.developer.name,
                "date": daily_report.date,
                "report_file": report_filename,
                "reviews": reviews,
            })
    return results


def generate_html_report(
    daily_report: DailyReport,
    review_links: Optional[dict[str, str]] = None,
) -> str:
    """Generate a complete self-contained HTML report.

    Args:
        daily_report: The DailyReport data structure.
        review_links: Optional mapping of message_id -> slug for review detail pages.
                      When provided, review comments are rendered as compact summaries
                      with links to per-patchset detail pages.

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
        for dr in daily_report.developer_reports
    )

    stats_section = _render_statistics(daily_report)

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
            position: relative;
            cursor: default;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .stat-tooltip {{
            display: none;
            position: absolute;
            top: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: #1a1a1a;
            color: #f0f0f0;
            border-radius: 8px;
            padding: 10px 14px;
            min-width: 200px;
            max-width: 280px;
            z-index: 100;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            text-align: left;
            font-size: 0.82em;
            line-height: 1.5;
        }}
        .stat-tooltip::before {{
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-bottom-color: #1a1a1a;
        }}
        .stat-card:hover .stat-tooltip {{
            display: block;
        }}
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2px 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .tooltip-row:last-child {{
            border-bottom: none;
        }}
        .tooltip-count {{
            font-weight: 700;
            margin-left: 16px;
            color: #7ec8e3;
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
        .inline-review-badge {{
            display: inline-block;
            padding: 0 6px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 500;
            background: #e3f2fd;
            color: #1565c0;
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
            margin-right: 8px;
            vertical-align: middle;
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
        .analysis-mode {{
            font-size: 0.85em;
            color: #888;
            margin-top: 4px;
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
    </style>
</head>
<body>
    <h1>LKML Activity Report{' <span class="llm-badge">LLM: ' + _esc(llm_label) + '</span>' if llm_label else ''}</h1>
    <h2>{_esc(daily_report.date)} &mdash; Generated {_esc(now)}</h2>
    {'<p class="analysis-mode">Analysis: LLM-enriched (' + _esc(llm_label) + ')</p>' if llm_label else '<p class="analysis-mode">Analysis: Heuristic</p>'}

    {stats_section}

    {developer_sections}

    <footer>
        Generated in {daily_report.generation_time_seconds:.1f}s
        &bull; {len(daily_report.developer_reports)} developers tracked
        &bull; Data from lore.kernel.org
        {'&bull; LLM: ' + _esc(llm_label) if llm_label else '&bull; Heuristic analysis'}
    </footer>
</body>
</html>"""
