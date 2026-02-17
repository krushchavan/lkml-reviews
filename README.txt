LKML Daily Activity Tracker
===========================

Tracks daily Linux Kernel Mailing List activity for a configurable list of
kernel developers. Fetches data from lore.kernel.org, classifies patches,
reviews, and acks, and produces styled HTML reports. Optionally enriches
reports with LLM-generated summaries (Ollama or Anthropic Claude).


Prerequisites
-------------

Python 3.11+ with packages: requests, anthropic (see requirements.txt).

For Docker: Docker and Docker Compose.

For LLM summaries: a running Ollama instance, or an Anthropic API key.


Files
-----

  generate_report.py      Main report generator
  report_generator.py     HTML report rendering
  build_index.py          Builds the index.html page
  build_reviews.py        Builds per-patchset review HTML pages
  activity_classifier.py  Classifies LKML messages
  lkml_client.py          lore.kernel.org API client
  llm_summarizer.py       LLM backend (Ollama / Anthropic)
  llm_cache.py            Disk cache for LLM results
  thread_analyzer.py      Heuristic thread analysis
  models.py               Data models
  kernel_developers_emails_1.csv   Developer list (name, emails)


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
    index.html                              Main index page
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
    --llm-model MODEL     Override default model
    --llm-all             Run both Ollama and Anthropic
    --ollama-model MODEL  Ollama model for --llm-all (default: llama3.1:8b)
    --anthropic-model MODEL  Anthropic model for --llm-all (default: claude-haiku-4-5)
    --llm-no-cache        Disable LLM result caching

    Purge / retention:
    --retention-days N    Delete reports/logs/reviews older than N days (default: 30)
    --no-purge            Skip automatic purge after report generation
    --purge-only          Only run the purge (no report generation)
