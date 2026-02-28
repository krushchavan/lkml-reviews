"""Generate a sample review HTML page for cover.1770821420.git.d@ilvokhin.com.

Demonstrates the generic approach used by the production pipeline:
  1. Fetch the full email thread from lore.kernel.org via LKMLClient.
  2. Call filter_subtree_messages() to keep only the cover-letter discussion
     (the 4 "Re: [PATCH 0/4] ..." messages), pruning individual patch
     sub-threads ([PATCH 1/4], [PATCH 2/4], etc.).
  3. Run heuristic analysis (analyze_thread) on the filtered messages.
  4. Convert the resulting ReviewComment objects into the build_review_html
     data dict and render the HTML.

This pattern works for any cover letter or patch — pass any message_id as
the root and filter_subtree_messages() extracts the right sub-thread.
"""
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from build_reviews import build_review_html
from lkml_client import LKMLClient
from models import ActivityItem, ActivityType
from thread_analyzer import analyze_thread, filter_subtree_messages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COVER_LETTER_ID = "cover.1770821420.git.d@ilvokhin.com"
COVER_SUBJECT   = "[PATCH 0/4] mm: zone lock tracepoint instrumentation"
COVER_URL       = f"https://lore.kernel.org/all/{COVER_LETTER_ID}/"

OUTPUT = pathlib.Path(
    r"C:\Users\krush\source\repos\lkml-reviews\reports\reviews"
    r"\cover.1770821420.git.d_ilvokhin.com.html"
)

# ---------------------------------------------------------------------------
# 1. Fetch the full thread
# ---------------------------------------------------------------------------
print(f"Fetching thread: {COVER_LETTER_ID} …")
client = LKMLClient()
result = client.get_thread(COVER_LETTER_ID)
all_messages = result.get("messages", [])
print(f"  Total messages in thread: {len(all_messages)}")

# ---------------------------------------------------------------------------
# 2. Filter to the cover-letter sub-thread
#    (prunes [PATCH 1/4] … [PATCH 4/4] branches and their review chains)
# ---------------------------------------------------------------------------
filtered = filter_subtree_messages(all_messages, COVER_LETTER_ID)
print(f"  Messages after subtree filter: {len(filtered)}")
for m in filtered:
    print(f"    [{m.get('message_id', '?')}]  {m.get('subject', '?')[:80]}")

# ---------------------------------------------------------------------------
# 3. Run heuristic analysis on the filtered sub-thread
# ---------------------------------------------------------------------------
activity_item = ActivityItem(
    activity_type=ActivityType.PATCH_SUBMITTED,
    subject=COVER_SUBJECT,
    message_id=COVER_LETTER_ID,
    url=COVER_URL,
    date="2026-02-11",
)
conv = analyze_thread(filtered, activity_item)

print(f"\nAnalysis: sentiment={conv.sentiment.value}, "
      f"{len(conv.review_comments)} review comment(s)")

# ---------------------------------------------------------------------------
# 4. Convert ReviewComment objects → build_review_html data dict
# ---------------------------------------------------------------------------
date_buckets: dict[str, list[dict]] = defaultdict(list)
for rc in conv.review_comments:
    date = rc.message_date or "unknown"
    date_buckets[date].append({
        "author":            rc.author,
        "summary":           rc.summary,
        "sentiment":         rc.sentiment.value,
        "sentiment_signals": rc.sentiment_signals,
        "has_inline_review": rc.has_inline_review,
        "tags_given":        rc.tags_given,
        "analysis_source":   rc.analysis_source,
        "raw_body":          rc.raw_body,
        "reply_to":          rc.reply_to,
        "message_date":      date,
        "message_id":        getattr(rc, "message_id", "") or "",
    })

dates: dict = {}
for i, date_str in enumerate(sorted(date_buckets)):
    entry: dict = {
        "report_file":    "",
        "analysis_source": "heuristic",
        "reviews":        date_buckets[date_str],
    }
    if i == 0:
        entry["patch_summary"] = conv.patch_summary or ""
    dates[date_str] = entry

data = {
    "subject": COVER_SUBJECT,
    "url":     COVER_URL,
    "dates":   dates,
}

# ---------------------------------------------------------------------------
# 5. Render and write
# ---------------------------------------------------------------------------
html = build_review_html(data)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(html, encoding="utf-8")
print(f"\nWritten to: {OUTPUT}")
print(f"Size: {len(html):,} bytes")
