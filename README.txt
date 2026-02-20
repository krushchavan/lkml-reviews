LKML Daily Activity Tracker
===========================

Tracks daily Linux Kernel Mailing List activity for a configurable list of
kernel developers. Fetches data from lore.kernel.org, classifies patches,
reviews, acks, and discussion threads, and produces styled HTML reports.
Optionally enriches reports with LLM-generated summaries (Ollama or Anthropic
Claude) and publishes the output to GitHub Pages.


Prerequisites
-------------

Python 3.11+ with packages: requests, anthropic (see requirements.txt).

For Docker: Docker and Docker Compose.

For LLM summaries: a running Ollama instance, or an Anthropic API key.

For GitHub publishing: git must be installed and a GitHub personal access
token with "Contents: write" permission is required.


Files
-----

  generate_report.py      Main report generator (fetch, classify, render, publish)
  report_generator.py     HTML report rendering
  build_index.py          Builds the reports/index.html page
  build_reviews.py        Builds per-patchset review HTML pages from JSON data
  clean_reports.py        Deletes all generated reports and syncs cleanup to GitHub
  activity_classifier.py  Classifies LKML messages (patches, reviews, acks, discussions)
  lkml_client.py          lore.kernel.org API client
  llm_summarizer.py       LLM backend abstraction (Ollama / Anthropic)
  llm_cache.py            Disk cache for LLM results
  thread_analyzer.py      Heuristic thread analysis
  models.py               Data models (ActivityType, DeveloperReport, etc.)
  kernel_developers_emails_1.csv   Developer list (name, primary + secondary emails)

Configuration:
  .env                    Optional environment variable file (see Environment Variables)


Environment Variables (.env)
----------------------------

Create a .env file in the project root to configure GitHub publishing and
other settings without using CLI flags. Real environment variables always
take precedence over .env values.

  # Timezone for report timestamps
  TZ=America/New_York

  # Anthropic API key (required for --llm --llm-backend anthropic or --llm-all)
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx

  # GitHub Pages publishing (used by --publish-github and clean_reports.py)
  GITHUB_REPO=https://github.com/owner/repo.git   # or owner/repo slug
  GITHUB_BRANCH=main
  GITHUB_TOKEN=github_pat_...                      # needs Contents: write


Usage Without Docker
--------------------

Install dependencies:

  pip install -r requirements.txt

Generate yesterday's report (heuristic analysis, no LLM):

  python generate_report.py

Generate a report for a specific date:

  python generate_report.py --date 2026-02-15

Generate reports for the last 7 days:

  python generate_report.py --days 7

Generate 7 days ending on a specific date:

  python generate_report.py --date 2026-02-15 --days 7

With LLM summaries via Ollama (requires Ollama running on localhost:11434):

  python generate_report.py --date 2026-02-15 --llm

With LLM summaries via Anthropic Claude:

  export ANTHROPIC_API_KEY=sk-ant-...
  python generate_report.py --date 2026-02-15 --llm --llm-backend anthropic

Compare both backends side-by-side:

  export ANTHROPIC_API_KEY=sk-ant-...
  python generate_report.py --date 2026-02-15 --llm-all

Custom developer list and output directory:

  python generate_report.py --people my_team.csv --output-dir /tmp/reports

Rebuild the index page (after generating reports):

  python build_index.py --reports-dir reports --logs-dir logs

Rebuild per-patchset review pages:

  python build_reviews.py --reports-dir reports

Per-report log files are written to the --logs-dir directory (default: logs/).
Each report gets its own log file with matching name:
  reports/2026-02-15.html  <-->  logs/2026-02-15.log
  reports/2026-02-15_ollama_llama3.1-8b.html  <-->  logs/2026-02-15_ollama_llama3.1-8b.log


GitHub Pages Publishing
-----------------------

Reports can be automatically pushed to a GitHub repository after generation.
The reports/ directory is treated as a self-contained git repo. On first use
it is initialised with git init; on subsequent runs it reuses the existing repo.

The GitHub token is embedded in the remote URL (no credential helper needed):

  # Publish after generating a report (reads GITHUB_REPO / GITHUB_TOKEN from .env)
  python generate_report.py --date 2026-02-15 --llm --publish-github

  # Re-publish existing reports without regenerating (useful after manual edits)
  python generate_report.py --publish-only

  # Override repo / branch / token on the CLI
  python generate_report.py --publish-github \
      --github-repo owner/repo \
      --github-branch gh-pages \
      --github-token ghp_...

GitHub Pages setup:
  1. Create the repository on GitHub (can be empty).
  2. In repo Settings → Pages, set the source to "Deploy from branch" → main (or
     whichever branch you push to), folder "/".
  3. Generate a fine-grained personal access token with "Contents: write"
     permission scoped to that repository.
  4. Set GITHUB_REPO, GITHUB_BRANCH, and GITHUB_TOKEN in your .env file.


Cleaning Up Reports
-------------------

clean_reports.py deletes all generated report files and syncs the deletion to
GitHub so the Pages site is also cleared. It always runs as a dry-run unless
--yes is passed.

Dry-run (shows what would be deleted, makes no changes):

  python clean_reports.py

Actually delete everything and push the cleanup to GitHub:

  python clean_reports.py --yes

Delete locally only, skip GitHub push:

  python clean_reports.py --yes --no-push

What is deleted:
  - reports/*.html          Dated daily reports
  - reports/reviews/*.json  Patch review source data
  - reports/reviews/*.html  Generated review detail pages
  - reports/reviews/        The directory itself (once empty)

What is preserved:
  - reports/.git            Git repo (reused by the next publish run)
  - reports/index.html      Replaced with a "No reports yet" placeholder page

After cleaning, the next run of generate_report.py will regenerate everything
from scratch and overwrite the placeholder index.html.


Usage With Docker
-----------------

The Docker setup includes three services:
  - lkml-tracker: the report generator (cron or on-demand)
  - web: nginx serving reports on port 8081
  - ollama: local LLM inference

1. CRON MODE (automated daily reports)

   Start the stack. Reports run daily at 2:00 AM UTC by default:

     docker compose up -d

   View reports at http://localhost:8081

   The cron schedule, LLM settings, and other options are configured via
   environment variables in docker-compose.yml or a .env file:

     CRON_SCHEDULE=0 2 * * *          # When to run (cron syntax)
     REPORT_ARGS=--llm --verbose      # Flags for generate_report.py
     RUN_ON_STARTUP=false             # Run immediately on container start
     ANTHROPIC_API_KEY=               # Required for Anthropic backend
     RETENTION_DAYS=30                # Auto-purge reports/logs older than N days
     PUID=1000                        # File ownership (Linux)
     PGID=1000
     GITHUB_REPO=owner/repo           # Push reports to GitHub Pages after each run
     GITHUB_BRANCH=main               # Branch to push to (default: main)
     GITHUB_TOKEN=github_pat_...      # Fine-grained token with Contents: write

   To use Anthropic instead of Ollama in cron mode, create a .env file:

     ANTHROPIC_API_KEY=sk-ant-...
     REPORT_ARGS=--llm --llm-backend anthropic --verbose

   Then restart:

     docker compose down
     docker compose up -d --build

2. ON-DEMAND MODE (one-off report generation)

   Generate a single report without cron:

     docker compose run --rm -e CRON_SCHEDULE= lkml-tracker --date 2026-02-15

   With Ollama LLM summaries:

     docker compose run --rm -e CRON_SCHEDULE= lkml-tracker --date 2026-02-15 --llm

   With Anthropic LLM summaries:

     docker compose run --rm -e CRON_SCHEDULE= -e ANTHROPIC_API_KEY=sk-ant-... \
       lkml-tracker --date 2026-02-15 --llm --llm-backend anthropic

   Generate reports for the past week:

     docker compose run --rm -e CRON_SCHEDULE= lkml-tracker --days 7 --llm

   Generate 7 days ending on a specific date:

     docker compose run --rm -e CRON_SCHEDULE= lkml-tracker --date 2026-02-15 --days 7 --llm


Output Structure
----------------

  reports/
    index.html                              Main index page (or placeholder after clean)
    2026-02-15.html                         Heuristic report
    2026-02-15_ollama_llama3.1-8b.html      LLM report (Ollama)
    2026-02-15_anthropic_claude-haiku-4-5.html  LLM report (Anthropic)
    reviews/
      <patchset-slug>.html                  Per-patchset review comments
      <patchset-slug>.json                  Review data (accumulates over time)

  logs/
    2026-02-15.log                          Log for 2026-02-15.html
    2026-02-15_ollama_llama3.1-8b.log       Log for 2026-02-15_ollama_llama3.1-8b.html

Each report has a 1:1 associated log file with the same filename stem.
Reports link to their log file in both the header and footer.
The index page shows each report alongside its matched log.

Reports, logs, and review data older than 30 days are automatically purged
after each report generation run. Configure with --retention-days or the
RETENTION_DAYS environment variable in Docker.


Web Interface
-------------

When the web service is running (docker compose up -d), browse to:

  http://localhost:8081            Index page with all reports
  http://localhost:8081/reviews/   Per-patchset review comments
  http://localhost:8081/logs/      Log files

When published to GitHub Pages, replace localhost:8081 with your Pages URL,
e.g.: https://owner.github.io/repo/


Command-Line Reference
----------------------

generate_report.py

  --date YYYY-MM-DD     Target date (default: yesterday)
  --days N              Number of days to generate (default: 1)
  --people FILE.csv     Developer list CSV (default: kernel_developers_emails_1.csv)
  --output-dir DIR      Report output directory (default: reports)
  --logs-dir DIR        Log file directory (default: logs)
  --rate-limit SECS     HTTP request delay (default: 1.0)
  --skip-threads        Skip thread fetching (faster, no summaries)
  --verbose, -v         Debug logging

  LLM options:
  --llm                 Enable LLM summaries (default backend: ollama)
  --llm-backend NAME    Backend: ollama or anthropic
  --llm-model MODEL     Override default model (single backend mode)
  --llm-all             Run both Ollama and Anthropic backends side-by-side
  --ollama-model MODEL  Ollama model for --llm-all (default: llama3.1:8b)
  --anthropic-model MODEL  Anthropic model for --llm-all (default: claude-haiku-4-5)
  --llm-no-cache        Disable LLM result caching
  --llm-dump DIR        Dump raw LLM responses to DIR for debugging
  --llm-monolithic      Force monolithic prompts (disable per-reviewer decomposition)

  Purge / retention:
  --retention-days N    Delete reports/logs/reviews older than N days (default: 30)
  --no-purge            Skip automatic purge after report generation
  --purge-only          Only run the purge (no report generation)

  GitHub publishing:
  --publish-github      Push reports/ to GitHub after generation
  --publish-only        Skip generation, push existing reports/ to GitHub immediately
  --github-repo REPO    Repository: 'owner/repo' slug or full GitHub URL
                        (falls back to GITHUB_REPO env var / .env)
  --github-branch NAME  Branch to push to (default: main;
                        falls back to GITHUB_BRANCH env var)
  --github-token TOKEN  GitHub personal access token with Contents: write permission
                        (falls back to GITHUB_TOKEN env var / .env)

----

build_index.py

  Scans reports/ and logs/, groups by date, and regenerates index.html.
  Called automatically by generate_report.py; run manually after edits.

  --reports-dir DIR     Reports directory (default: reports)
  --logs-dir DIR        Logs directory (default: logs)

----

build_reviews.py

  Scans reports/reviews/*.json and regenerates all per-patchset HTML pages.
  Called automatically by generate_report.py; run manually if JSON files were
  added or changed without running the full generator.

  --reports-dir DIR     Reports directory containing reviews/ subdir (default: reports)

----

clean_reports.py

  Deletes all generated report and review files, writes a placeholder index.html,
  and pushes the cleanup to GitHub. Dry-run by default — pass --yes to apply.

  --yes                 Actually delete files (without this flag: dry run only)
  --no-push             Delete locally only, skip GitHub push
  --reports-dir DIR     Reports directory (default: reports)
  --github-repo REPO    Repository to push the cleanup to
                        (falls back to GITHUB_REPO env var / .env)
  --github-branch NAME  Branch to push to (falls back to GITHUB_BRANCH / main)
  --github-token TOKEN  GitHub personal access token
                        (falls back to GITHUB_TOKEN env var / .env)
  --verbose, -v         Debug logging
