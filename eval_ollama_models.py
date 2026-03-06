"""Comparative evaluation of local Ollama models for LKML thread analysis.

Tests all installed Ollama models against a representative LKML thread using
the project's actual prompt pipeline.  Measures:
  - JSON validity (can the model produce valid JSON at all?)
  - Schema compliance (are all 6 required fields present and correctly typed?)
  - Semantic quality (patch_summary non-trivial, review_comments populated, etc.)
  - Speed (tokens/sec, wall-clock time)

Usage:
    python eval_ollama_models.py
    python eval_ollama_models.py --verbose
    python eval_ollama_models.py --no-cache
"""

import argparse
import io
import json
import logging
import re
import sys
import time
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import re as _re_top
import subprocess

import requests

# ---------------------------------------------------------------------------
# Representative LKML thread (synthetic but realistic — avoids network calls)
# ---------------------------------------------------------------------------

SAMPLE_THREAD = [
    {
        "from": "Vlastimil Babka <vbabka@suse.cz>",
        "date": "2025-02-11",
        "subject": "[PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab",
        "body": """When a cpu partial slab is deactivated and put back onto the node partial
list, __slab_free() may be called concurrently and attempt to free the
slab a second time.  The slab allocator tracks this via a per-slab
'frozen' bit, but the bit is cleared before the cmpxchg loop that
re-adds the slab to the partial list.

Fix this by keeping frozen=1 until the slab is safely on the partial
list, then clearing it atomically.

This was found via slub_debug=F,Z on a 64-cpu machine running fio.

Signed-off-by: Vlastimil Babka <vbabka@suse.cz>
---
 mm/slub.c | 24 ++++++++++++++----------
 1 file changed, 14 insertions(+), 10 deletions(-)

diff --git a/mm/slub.c b/mm/slub.c
index 3a2e9f1..7d8cf22 100644
--- a/mm/slub.c
+++ b/mm/slub.c
@@ -2401,10 +2401,14 @@ static void deactivate_slab(struct kmem_cache *s,
 	if (m == M_PARTIAL) {
-		add_partial(n, page, DEACTIVATE_TO_TAIL);
-		page->frozen = 0;
+		/* Keep frozen=1 while adding to avoid concurrent double-free */
+		add_partial(n, page, DEACTIVATE_TO_TAIL);
+		smp_wmb();
+		WRITE_ONCE(page->frozen, 0);
 	}
""",
    },
    {
        "from": "Matthew Wilcox <willy@infradead.org>",
        "date": "2025-02-11",
        "subject": "Re: [PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab",
        "body": """On Tue, Feb 11, 2025 at 10:15:43AM +0100, Vlastimil Babka wrote:
> Fix this by keeping frozen=1 until the slab is safely on the partial
> list, then clearing it atomically.

Reviewed-by: Matthew Wilcox (Oracle) <willy@infradead.org>

One minor concern: should we use smp_store_release() here instead of
the smp_wmb() + WRITE_ONCE() pair?  The former is slightly more idiomatic
in current kernel code and avoids the two-instruction overhead on TSO
architectures.

Other than that the fix looks correct to me.  The race window you
described is real and I've seen similar issues in slub before.
""",
    },
    {
        "from": "David Rientjes <rientjes@google.com>",
        "date": "2025-02-12",
        "subject": "Re: [PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab",
        "body": """On Tue, 11 Feb 2025, Matthew Wilcox wrote:
> One minor concern: should we use smp_store_release() here instead of
> the smp_wmb() + WRITE_ONCE() pair?

I agree with Willy's suggestion.  smp_store_release() is the right
primitive here — it encodes both the ordering and the store semantics
in a single call.

Also, shouldn't the patch include a comment explaining WHY frozen must
stay set during the add_partial() call?  The current comment just says
"to avoid concurrent double-free" which is correct but leaves out the
mechanism (i.e., that __slab_free() checks frozen before deciding
whether to add_partial itself).

Reviewed-by: David Rientjes <rientjes@google.com>
""",
    },
    {
        "from": "Vlastimil Babka <vbabka@suse.cz>",
        "date": "2025-02-12",
        "subject": "Re: [PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab",
        "body": """On Wed, 12 Feb 2025, David Rientjes wrote:
> Also, shouldn't the patch include a comment explaining WHY frozen must
> stay set during the add_partial() call?

Good point, I'll add a more detailed comment in v4 explaining the
__slab_free() interaction.  I'll also switch to smp_store_release()
as both of you suggested.

Thanks for the reviews!
""",
    },
    {
        "from": "Christoph Lameter <cl@linux.com>",
        "date": "2025-02-13",
        "subject": "Re: [PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab",
        "body": """On Tue, 11 Feb 2025, Vlastimil Babka wrote:
> This was found via slub_debug=F,Z on a 64-cpu machine running fio.

Thanks for the detailed analysis and the fix.  The frozen bit logic
is notoriously subtle.

Acked-by: Christoph Lameter <cl@linux.com>
""",
    },
]

SAMPLE_SUBJECT = "[PATCH v3 1/2] mm/slub: fix double-free when deactivating partial slab"

# Required top-level JSON keys per the project's prompt spec
REQUIRED_KEYS = {
    "patch_summary",
    "overall_sentiment",
    "overall_sentiment_signals",
    "discussion_progress",
    "progress_detail",
    "review_comments",
}

VALID_SENTIMENTS = {"POSITIVE", "NEEDS_WORK", "CONTENTIOUS", "NEUTRAL"}
VALID_PROGRESS = {
    "ACCEPTED", "CHANGES_REQUESTED", "UNDER_REVIEW",
    "NEW_VERSION_EXPECTED", "WAITING_FOR_REVIEW", "SUPERSEDED", "RFC",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"

def _c(text, code):
    return f"{code}{text}{_RESET}"


def _score_result(parsed: dict | None) -> dict:
    """Return a dict of scored metrics (0–1 floats or counts)."""
    if parsed is None:
        return {
            "json_valid": False,
            "schema_score": 0.0,
            "semantic_score": 0.0,
            "issues": ["No valid JSON returned"],
        }

    issues = []
    schema_points = 0
    schema_total = len(REQUIRED_KEYS)

    for key in REQUIRED_KEYS:
        if key in parsed:
            schema_points += 1
        else:
            issues.append(f"Missing key: {key}")

    # Type checks
    extra_penalties = 0
    if not isinstance(parsed.get("patch_summary"), str) or not parsed.get("patch_summary"):
        issues.append("patch_summary empty or not a string")
        extra_penalties += 1
    if parsed.get("overall_sentiment") not in VALID_SENTIMENTS:
        issues.append(f"invalid overall_sentiment: {parsed.get('overall_sentiment')!r}")
        extra_penalties += 1
    if not isinstance(parsed.get("overall_sentiment_signals"), list):
        issues.append("overall_sentiment_signals not a list")
        extra_penalties += 1
    if parsed.get("discussion_progress") not in VALID_PROGRESS:
        issues.append(f"invalid discussion_progress: {parsed.get('discussion_progress')!r}")
        extra_penalties += 1
    if not isinstance(parsed.get("progress_detail"), str) or not parsed.get("progress_detail"):
        issues.append("progress_detail empty or not a string")
        extra_penalties += 1
    if not isinstance(parsed.get("review_comments"), list):
        issues.append("review_comments not a list")
        extra_penalties += 1

    schema_score = max(0.0, (schema_points - extra_penalties) / schema_total)

    # Semantic quality heuristics
    semantic_points = 0
    semantic_total = 7

    patch_summary = parsed.get("patch_summary", "")
    if isinstance(patch_summary, str) and len(patch_summary) > 50:
        semantic_points += 1  # non-trivial patch summary
    if isinstance(patch_summary, str) and len(patch_summary) > 150:
        semantic_points += 1  # detailed patch summary

    review_comments = parsed.get("review_comments", [])
    if isinstance(review_comments, list) and len(review_comments) >= 2:
        semantic_points += 1  # found multiple reviewers
    if isinstance(review_comments, list) and len(review_comments) >= 4:
        semantic_points += 1  # found most reviewers (thread has 3 replies + author follow-up)

    # Check review_comments have summaries
    if isinstance(review_comments, list) and review_comments:
        summaries_ok = sum(
            1 for rc in review_comments
            if isinstance(rc, dict) and isinstance(rc.get("summary"), str) and len(rc.get("summary", "")) > 30
        )
        if summaries_ok == len(review_comments):
            semantic_points += 1
        elif summaries_ok > 0:
            semantic_points += 0.5

    # Check for Reviewed-by / Acked-by tags (should have been detected)
    tags_found = []
    if isinstance(review_comments, list):
        for rc in review_comments:
            if isinstance(rc, dict):
                tags_found.extend(rc.get("tags_given", []))
    if "Reviewed-by" in tags_found or "Acked-by" in tags_found:
        semantic_points += 1

    # Progress should be CHANGES_REQUESTED or NEW_VERSION_EXPECTED (author said v4 coming)
    dp = parsed.get("discussion_progress", "")
    if dp in ("CHANGES_REQUESTED", "NEW_VERSION_EXPECTED", "UNDER_REVIEW"):
        semantic_points += 1

    semantic_score = semantic_points / semantic_total

    return {
        "json_valid": True,
        "schema_score": schema_score,
        "semantic_score": semantic_score,
        "issues": issues,
        "review_comments_count": len(review_comments) if isinstance(review_comments, list) else 0,
        "discussion_progress": parsed.get("discussion_progress", "?"),
        "tags_found": list(set(tags_found)),
        "patch_summary_len": len(patch_summary) if isinstance(patch_summary, str) else 0,
    }


# ---------------------------------------------------------------------------
# Build prompt (mirrors llm_summarizer logic without importing from it)
# ---------------------------------------------------------------------------

def build_thread_text(messages, max_chars=20000):
    parts = []
    for i, msg in enumerate(messages):
        from_field = msg.get("from", "unknown")
        date = msg.get("date", "")
        subject = msg.get("subject", "")
        body = msg.get("body", "")

        lines = body.split("\n")
        filtered = []
        for line in lines:
            if line.strip().startswith(">"):
                if not filtered or filtered[-1] != "[quoted text omitted]":
                    filtered.append("[quoted text omitted]")
            else:
                filtered.append(line)
        clean_body = "\n".join(filtered)
        if len(clean_body) > 3000:
            clean_body = clean_body[:2800] + "\n[... message truncated ...]"

        header = f"=== Message {i+1} | From: {from_field} | Date: {date} ==="
        if subject:
            header += f"\nSubject: {subject}"
        parts.append(f"{header}\n{clean_body}")

    full = "\n\n".join(parts)
    if len(full) > max_chars:
        first = parts[0]
        budget = max_chars - len(first) - 200
        tail_parts = []
        tl = 0
        for p in reversed(parts[1:]):
            if tl + len(p) > budget:
                break
            tail_parts.insert(0, p)
            tl += len(p)
        full = first + "\n\n[... middle messages omitted ...]\n\n" + "\n\n".join(tail_parts)
    return full


def build_prompt(thread_text, subject):
    return f"""You are a senior Linux kernel developer analyzing an LKML email thread.
Your job is to produce a concise analytical summary — NOT to quote or copy text from the emails.

Thread subject: {subject}

Return a JSON object with exactly these fields:

{{
  "patch_summary": "...",
  "overall_sentiment": "one of: POSITIVE, NEEDS_WORK, CONTENTIOUS, NEUTRAL",
  "overall_sentiment_signals": ["signal1", "signal2"],
  "discussion_progress": "one of: ACCEPTED, CHANGES_REQUESTED, UNDER_REVIEW, NEW_VERSION_EXPECTED, WAITING_FOR_REVIEW, SUPERSEDED, RFC",
  "progress_detail": "one sentence describing where things stand",
  "review_comments": [
    {{
      "author": "First Last",
      "reply_to": "Name or empty string",
      "summary": "your own 3-5 sentence analytical summary",
      "sentiment": "one of: POSITIVE, NEEDS_WORK, CONTENTIOUS, NEUTRAL",
      "sentiment_signals": ["signal1"],
      "has_inline_review": true,
      "tags_given": ["Reviewed-by"]
    }}
  ]
}}

IMPORTANT:
- Return ONLY valid JSON. No markdown fences, no explanation before or after.
- Use EXACTLY the field names shown above.
- All sentiment/progress values must be UPPERCASE.
- has_inline_review must be a JSON boolean (true/false), not a string.
- Your response MUST contain these top-level keys: "patch_summary", "overall_sentiment",
  "overall_sentiment_signals", "discussion_progress", "progress_detail", "review_comments".

--- THREAD MESSAGES ---
{thread_text}

--- END OF THREAD ---
Produce your analytical JSON now. Start with {{ end with }}."""


def parse_response(raw: str) -> dict | None:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and terminal control codes from text."""
    # Remove standard ANSI escape sequences
    ansi_escape = _re_top.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*\x07)')
    text = ansi_escape.sub('', text)
    # Remove any remaining control characters (except newlines/tabs)
    text = _re_top.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    return text


def run_model_api(model_name: str, prompt: str, base_url: str = "http://localhost:11434",
                  timeout: int = 600) -> tuple[str | None, float, float, str | None]:
    """Try the Ollama HTTP API.  Returns (raw, wall_secs, tps, error)."""
    start = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "format": "json",
                "stream": True,
                "options": {"num_predict": 2048, "temperature": 0.2},
            },
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()

        full = []
        eval_count = 0
        total_duration_ns = 0

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            if token:
                full.append(token)
            if chunk.get("done"):
                eval_count = chunk.get("eval_count", 0)
                total_duration_ns = chunk.get("total_duration", 0)
                break

        wall = time.monotonic() - start
        raw = "".join(full)
        tps = eval_count / (total_duration_ns / 1e9) if total_duration_ns > 0 else 0.0
        return raw, wall, tps, None

    except requests.exceptions.Timeout:
        wall = time.monotonic() - start
        return None, wall, 0.0, f"Timeout after {wall:.0f}s"
    except Exception as e:
        wall = time.monotonic() - start
        err_str = str(e)
        # Detect "model not found" to trigger CLI fallback
        if "not found" in err_str or "404" in err_str:
            return None, wall, 0.0, f"api_not_found:{err_str}"
        return None, wall, 0.0, err_str


def run_model_cli(model_name: str, prompt: str, timeout: int = 600) -> tuple[str | None, float, float, str | None]:
    """Fallback: run via `ollama run` CLI, stripping terminal control codes."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "--nowordwrap"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        wall = time.monotonic() - start
        if result.returncode != 0:
            stderr = _strip_ansi(result.stderr or "").strip()
            return None, wall, 0.0, f"CLI exit {result.returncode}: {stderr[:200]}"

        raw = _strip_ansi(result.stdout or "")
        # Rough tokens/sec: estimate ~4 chars/token
        tps = (len(raw) / 4) / wall if wall > 0 else 0.0
        return raw.strip(), wall, tps, None

    except subprocess.TimeoutExpired:
        wall = time.monotonic() - start
        return None, wall, 0.0, f"CLI timeout after {wall:.0f}s"
    except FileNotFoundError:
        wall = time.monotonic() - start
        return None, wall, 0.0, "ollama CLI not found in PATH"
    except Exception as e:
        wall = time.monotonic() - start
        return None, wall, 0.0, str(e)


def run_model(model_name: str, prompt: str, base_url: str = "http://localhost:11434",
              timeout: int = 600) -> tuple[str | None, float, float, str | None, str]:
    """
    Run model via API, falling back to CLI if the API reports model not found.
    Returns (raw_response, wall_time_secs, tokens_per_sec, error_message, method_used).
    """
    raw, wall, tps, err = run_model_api(model_name, prompt, base_url, timeout)
    if err and err.startswith("api_not_found:"):
        # Server doesn't have this model registered — try the CLI
        raw, wall, tps, err = run_model_cli(model_name, prompt, timeout)
        method = "cli"
    else:
        method = "api"
    return raw, wall, tps, err, method


# ---------------------------------------------------------------------------
# Recommend models not yet installed that fit in ~28GB RAM
# ---------------------------------------------------------------------------

RECOMMENDED_UNINSTALLED = [
    {
        "name": "phi4:14b",
        "size_gb": 8.9,
        "notes": "Microsoft Phi-4 — strong reasoning, instruction-following; good JSON compliance",
    },
    {
        "name": "mistral:7b",
        "size_gb": 4.1,
        "notes": "Mistral 7B v0.3 — fast on CPU, good structured output; smaller than llama3.1:8b",
    },
    {
        "name": "deepseek-r1:14b",
        "size_gb": 8.9,
        "notes": "DeepSeek-R1 14B — chain-of-thought reasoning; may wrap JSON in <think> tags (needs stripping)",
    },
    {
        "name": "qwen2.5:32b",
        "size_gb": 19.8,
        "notes": "Qwen 2.5 32B — largest practical model for 32GB RAM; may be slow on CPU",
    },
    {
        "name": "llama3.3:70b",
        "size_gb": 43.0,
        "notes": "Too large for 32GB RAM — skip unless you have more memory",
    },
]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate local Ollama models for LKML analysis.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="(reserved, not used here)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-model timeout in seconds (default 600)")
    parser.add_argument("--models", nargs="*", help="Override model list (default: all installed)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base_url = "http://localhost:11434"

    # Discover installed models
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        installed = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"{_c('ERROR', _RED)}: Cannot reach Ollama at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.models:
        models_to_test = args.models
    else:
        models_to_test = installed

    if not models_to_test:
        print("No models found. Run `ollama pull <model>` first.")
        sys.exit(1)

    # Build prompt
    thread_text = build_thread_text(SAMPLE_THREAD)
    prompt = build_prompt(thread_text, SAMPLE_SUBJECT)
    prompt_chars = len(prompt)

    print(f"\n{_c('LKML Model Evaluation', _BOLD)}")
    print(f"{'─' * 70}")
    print(f"  Test thread:  {SAMPLE_SUBJECT[:65]}")
    print(f"  Messages:     {len(SAMPLE_THREAD)}")
    print(f"  Prompt size:  {prompt_chars:,} chars (~{prompt_chars//4:,} tokens)")
    print(f"  Models:       {len(models_to_test)}")
    print(f"  Timeout:      {args.timeout}s per model")
    print(f"{'─' * 70}\n")

    results = []

    for model in models_to_test:
        print(f"  {_c('Running', _CYAN)} {_c(model, _BOLD)} ...", end="", flush=True)
        raw, wall, tps, error, method = run_model(model, prompt, base_url, timeout=args.timeout)

        if error:
            print(f"  {_c('FAILED', _RED)} ({error})")
            results.append({
                "model": model,
                "method": method,
                "error": error,
                "wall_secs": wall,
                "tps": 0,
                "json_valid": False,
                "schema_score": 0.0,
                "semantic_score": 0.0,
                "issues": [error],
                "raw": None,
                "parsed": None,
            })
            continue

        parsed = parse_response(raw or "")
        scores = _score_result(parsed)

        method_badge = _c(f"[{method}]", _DIM)
        status = _c("OK", _GREEN) if scores["json_valid"] and scores["schema_score"] >= 0.8 else \
                 _c("PARTIAL", _YELLOW) if scores["json_valid"] else _c("INVALID JSON", _RED)
        tps_str = f"{tps:.1f} tok/s" if tps > 0 else f"{wall:.0f}s"
        print(f"  {status}  [{tps_str}] {method_badge}")

        if scores.get("issues") and args.verbose:
            for issue in scores["issues"]:
                print(f"      {_c('!', _YELLOW)} {issue}")

        results.append({
            "model": model,
            "method": method,
            "error": None,
            "wall_secs": wall,
            "tps": tps,
            **scores,
            "raw": raw,
            "parsed": parsed,
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------

    print(f"\n{_c('Results Summary', _BOLD)}")
    print(f"{'─' * 100}")
    header = (
        f"  {'Model':<22} {'Via':>5} {'JSON':>5} {'Schema':>8} {'Semantic':>10} "
        f"{'Speed':>10} {'Time':>7} {'Progress':<22} {'Issues'}"
    )
    print(_c(header, _DIM))
    print(f"{'─' * 100}")

    for r in sorted(results, key=lambda x: -(x["schema_score"] + x["semantic_score"])):
        model = r["model"]
        via = r.get("method", "api")
        json_ok = _c("YES", _GREEN) if r["json_valid"] else _c("NO", _RED)
        schema = f"{r['schema_score']*100:.0f}%"
        semantic = f"{r['semantic_score']*100:.0f}%"
        speed = f"{r['tps']:.1f} t/s" if r["tps"] > 0 else "—"
        elapsed = f"{r['wall_secs']:.0f}s"
        progress = r.get("discussion_progress", r.get("error", "—") or "—")
        progress = str(progress)[:20]
        issue_count = len(r.get("issues", []))
        issue_str = f"{issue_count} issue(s)" if issue_count else _c("none", _GREEN)

        schema_colored = _c(schema, _GREEN if r["schema_score"] >= 0.9 else
                             _YELLOW if r["schema_score"] >= 0.6 else _RED)
        sem_colored = _c(semantic, _GREEN if r["semantic_score"] >= 0.7 else
                          _YELLOW if r["semantic_score"] >= 0.4 else _RED)

        print(
            f"  {model:<22} {via:>5}  {json_ok:>5}  {schema_colored:>8}  {sem_colored:>10}  "
            f"{speed:>10} {elapsed:>7}  {progress:<22} {issue_str}"
        )

    print(f"{'─' * 100}")

    # ---------------------------------------------------------------------------
    # Detailed issues
    # ---------------------------------------------------------------------------
    any_issues = any(r.get("issues") for r in results)
    if any_issues:
        print(f"\n{_c('Issues Detail', _BOLD)}")
        for r in results:
            if r.get("issues"):
                print(f"\n  {_c(r['model'], _BOLD)}:")
                for issue in r["issues"]:
                    print(f"    {_c('•', _YELLOW)} {issue}")

    # ---------------------------------------------------------------------------
    # Best model recommendation
    # ---------------------------------------------------------------------------
    valid_results = [r for r in results if r["json_valid"]]
    if valid_results:
        best = max(valid_results, key=lambda r: (r["schema_score"] + r["semantic_score"], r["tps"]))
        fastest_valid = max(valid_results, key=lambda r: r["tps"] if r["tps"] > 0 else 0)

        print(f"\n{_c('Recommendation (installed models)', _BOLD)}")
        print(f"{'─' * 70}")
        print(f"  {_c('Best overall quality:', _GREEN)}  {_c(best['model'], _BOLD)}")
        print(f"    Schema {best['schema_score']*100:.0f}%  |  Semantic {best['semantic_score']*100:.0f}%  |  {best['tps']:.1f} tok/s")
        if fastest_valid["model"] != best["model"]:
            print(f"  {_c('Fastest valid:', _CYAN)}         {_c(fastest_valid['model'], _BOLD)}")
            print(f"    {fastest_valid['tps']:.1f} tok/s  |  {fastest_valid['wall_secs']:.0f}s total")
    else:
        print(f"\n{_c('WARNING', _RED)}: No model produced valid JSON!")

    # ---------------------------------------------------------------------------
    # Models not yet installed — fit recommendation for 32GB RAM / CPU
    # ---------------------------------------------------------------------------
    print(f"\n{_c('Models worth pulling (for your 32 GB RAM, CPU-only setup)', _BOLD)}")
    print(f"{'─' * 70}")
    installed_bases = {m.split(":")[0] for m in installed}
    for rec in RECOMMENDED_UNINSTALLED:
        base = rec["name"].split(":")[0]
        if base in installed_bases:
            continue  # already have a variant
        fits = rec["size_gb"] <= 22  # leave ~10GB headroom for OS + data
        badge = _c("fits", _GREEN) if fits else _c("tight/skip", _RED)
        print(f"  [{badge}] {_c(rec['name'], _BOLD)} ({rec['size_gb']:.1f} GB)")
        print(f"          {rec['notes']}")

    print(f"\n  Pull with:  {_c('ollama pull <model-name>', _DIM)}")

    # ---------------------------------------------------------------------------
    # Raw JSON previews (verbose mode)
    # ---------------------------------------------------------------------------
    if args.verbose:
        print(f"\n{_c('Raw Parsed Outputs', _BOLD)}")
        for r in results:
            print(f"\n  {'━' * 60}")
            print(f"  {_c(r['model'], _BOLD)}")
            print(f"  {'━' * 60}")
            if r.get("parsed"):
                print(json.dumps(r["parsed"], indent=4, ensure_ascii=False)[:3000])
            elif r.get("raw"):
                print(f"  (raw, first 500 chars): {r['raw'][:500]}")
            else:
                print(f"  Error: {r.get('error')}")

    print()


if __name__ == "__main__":
    main()
