"""Clean up all generated report files and sync the deletion to GitHub.

Deletes:
  - reports/*.html          (dated daily reports + index)
  - reports/reviews/*.json  (patch review source data)
  - reports/reviews/*.html  (generated review detail pages)
  - reports/reviews/        (the directory itself, if empty after deletion)

The reports/.git directory is left untouched so the existing git repo can
be reused by generate_report.py --publish-github after cleanup.

After deletion the script optionally pushes the empty state to GitHub so
the Pages site is also cleared.

Usage:
    python clean_reports.py                    # dry-run (shows what would be deleted)
    python clean_reports.py --yes              # actually delete + push to GitHub
    python clean_reports.py --yes --no-push    # delete locally only, skip GitHub push
    python clean_reports.py --reports-dir PATH # custom reports directory
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader (same logic as generate_report.py so GitHub creds work)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """Load key=value pairs from a .env file into os.environ (non-destructive)."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path, redact: str = "") -> subprocess.CompletedProcess:
    display = " ".join(cmd)
    if redact:
        display = display.replace(redact, "***")
    logger.debug("git: %s (cwd=%s)", display, cwd)
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


_EMPTY_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LKML Activity Reports</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               background: #0d1117; color: #c9d1d9;
               display: flex; align-items: center; justify-content: center;
               min-height: 100vh; margin: 0; }
        .msg { text-align: center; }
        h1 { font-size: 1.5rem; margin-bottom: .5rem; }
        p  { color: #8b949e; font-size: .95rem; }
    </style>
</head>
<body>
    <div class="msg">
        <h1>LKML Activity Reports</h1>
        <p>No reports yet. Run <code>generate_report.py</code> to generate the first report.</p>
    </div>
</body>
</html>
"""


def _collect_targets(reports_dir: Path) -> list[Path]:
    """Return all files/directories that will be deleted (excluding .git and index.html)."""
    targets: list[Path] = []

    # Daily HTML reports (e.g. 2026-02-19_ollama_llama3.1-8b.html)
    # index.html is intentionally excluded — it will be replaced with an empty placeholder.
    for p in sorted(reports_dir.glob("*.html")):
        if p.name == "index.html":
            continue
        targets.append(p)

    # reviews/ subdirectory contents
    reviews_dir = reports_dir / "reviews"
    if reviews_dir.exists():
        for p in sorted(reviews_dir.glob("*.json")):
            targets.append(p)
        for p in sorted(reviews_dir.glob("*.html")):
            targets.append(p)
        # We'll also record the directory itself so we can rmdir it if empty
        targets.append(reviews_dir)

    return targets


def _delete_targets(targets: list[Path], dry_run: bool) -> int:
    """Delete the listed paths.  Returns count of items actually removed."""
    removed = 0
    for p in targets:
        if not p.exists():
            continue
        if dry_run:
            kind = "DIR " if p.is_dir() else "FILE"
            logger.info("  [dry-run] would delete %s %s", kind, p)
            removed += 1
            continue
        try:
            if p.is_dir():
                # Only remove if empty (files inside were already deleted above)
                try:
                    p.rmdir()
                    logger.info("  - removed dir  %s", p)
                    removed += 1
                except OSError:
                    logger.debug("  skip non-empty dir %s", p)
            else:
                p.unlink()
                logger.info("  - removed file %s", p)
                removed += 1
        except OSError as exc:
            logger.warning("  ! could not delete %s: %s", p, exc)
    return removed


# ---------------------------------------------------------------------------
# GitHub push
# ---------------------------------------------------------------------------

def push_cleanup_to_github(
    reports_dir: Path,
    repo: str,
    branch: str = "main",
    token: str = "",
) -> bool:
    """Stage the deletions and push to GitHub so Pages is also cleared."""

    # Normalise repo slug
    slug = repo.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", slug)
    if m:
        slug = m.group(1)

    public_url = f"https://github.com/{slug}.git"
    push_url = f"https://x-access-token:{token}@github.com/{slug}.git" if token else public_url

    git_dir = reports_dir / ".git"
    if not git_dir.exists():
        logger.error(
            "GitHub push: %s/.git not found. "
            "Run generate_report.py --publish-github at least once first.",
            reports_dir,
        )
        return False

    # Update remote URL (handles token rotation)
    _run(["git", "remote", "set-url", "origin", push_url], reports_dir, redact=token)

    # Stage all deletions (git add -A picks up removed files too)
    r = _run(["git", "add", "-A"], reports_dir)
    if r.returncode != 0:
        logger.error("GitHub push: git add failed: %s", r.stderr.strip())
        return False

    # Check status
    r_status = _run(["git", "status", "--porcelain"], reports_dir)
    if not r_status.stdout.strip():
        # Check for unpushed commits
        r_ahead = _run(["git", "rev-list", "--count", "@{u}..HEAD"], reports_dir)
        if r_ahead.returncode != 0 or r_ahead.stdout.strip() == "0":
            logger.info("GitHub push: nothing to push — remote already up to date.")
            return True
        logger.info(
            "GitHub push: %s unpushed commit(s) found — pushing now",
            r_ahead.stdout.strip(),
        )
    else:
        # Log what was staged
        deleted, modified = [], []
        for line in r_status.stdout.splitlines():
            if len(line) < 3:
                continue
            xy = line[:2]
            path = line[3:]
            if "D" in xy:
                deleted.append(path)
            else:
                modified.append(path)
        logger.info("GitHub push: %d deleted, %d other changes", len(deleted), len(modified))

        commit_msg = f"LKML reports cleanup {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        r = _run(["git", "commit", "-m", commit_msg], reports_dir)
        if r.returncode != 0:
            logger.error("GitHub push: git commit failed: %s", r.stderr.strip())
            return False
        logger.info("GitHub push: committed — %s", commit_msg)

    # Push
    r_upstream = _run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        reports_dir,
    )
    has_upstream = r_upstream.returncode == 0 and r_upstream.stdout.strip()
    push_cmd = (
        ["git", "push", "-u", "origin", branch, "--force-with-lease"]
        if has_upstream
        else ["git", "push", "-u", "origin", branch, "--force"]
    )

    logger.info("GitHub push: pushing to %s (branch: %s)…", slug, branch)
    r = _run(push_cmd, reports_dir, redact=token)
    if r.returncode != 0:
        logger.error(
            "GitHub push: git push failed:\n%s\n%s",
            r.stdout.strip(), r.stderr.strip(),
        )
        return False

    logger.info("GitHub push: pushed successfully to %s/%s", slug, branch)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Delete all generated LKML reports and review files, "
            "then push the cleanup to GitHub Pages."
        ),
        epilog=(
            "By default this is a DRY RUN — pass --yes to actually delete files. "
            "GitHub credentials are read from .env (GITHUB_REPO / GITHUB_TOKEN) "
            "or can be supplied via CLI flags."
        ),
    )
    p.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Path to the reports directory. Default: reports/",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag the script is a dry run.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing deletions to GitHub (local cleanup only).",
    )
    p.add_argument(
        "--github-repo",
        type=str,
        default=None,
        help=(
            "GitHub repository to push the cleanup to ('owner/repo' or full URL). "
            "Falls back to GITHUB_REPO env var."
        ),
    )
    p.add_argument(
        "--github-branch",
        type=str,
        default=None,
        help="Branch to push to. Falls back to GITHUB_BRANCH env var, then 'main'.",
    )
    p.add_argument(
        "--github-token",
        type=str,
        default=None,
        help=(
            "GitHub personal access token. Falls back to GITHUB_TOKEN env var."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    reports_dir = Path(args.reports_dir).resolve()
    if not reports_dir.exists():
        logger.error("Reports directory not found: %s", reports_dir)
        sys.exit(1)

    dry_run = not args.yes

    if dry_run:
        logger.info("=== DRY RUN — pass --yes to actually delete files ===")

    # Collect and display targets
    targets = _collect_targets(reports_dir)
    if not targets:
        logger.info("Nothing to delete — reports directory is already clean.")
        sys.exit(0)

    logger.info(
        "Found %d item(s) to delete in %s:",
        len(targets),
        reports_dir,
    )

    removed = _delete_targets(targets, dry_run=dry_run)

    if dry_run:
        logger.info(
            "Dry run complete: %d item(s) would be deleted "
            "and index.html would be replaced with an empty placeholder. "
            "Re-run with --yes to apply.",
            removed,
        )
        sys.exit(0)

    logger.info("Deleted %d item(s).", removed)

    # Write empty placeholder index.html
    index_path = reports_dir / "index.html"
    try:
        index_path.write_text(_EMPTY_INDEX_HTML, encoding="utf-8")
        logger.info("Written empty placeholder: %s", index_path)
    except OSError as exc:
        logger.warning("Could not write placeholder index.html: %s", exc)

    # GitHub push
    if args.no_push:
        logger.info("Skipping GitHub push (--no-push).")
        return

    repo = args.github_repo or os.environ.get("GITHUB_REPO", "")
    if not repo:
        logger.warning(
            "No GitHub repo configured — skipping push. "
            "Set GITHUB_REPO in .env or pass --github-repo."
        )
        return

    token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
    branch = (
        args.github_branch
        or os.environ.get("GITHUB_BRANCH", "main")
    )

    ok = push_cleanup_to_github(reports_dir, repo, branch=branch, token=token)
    if not ok:
        logger.error("GitHub push failed — see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
