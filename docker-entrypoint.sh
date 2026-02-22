#!/bin/bash
set -e

# docker-entrypoint.sh
# Handles two modes:
#   1. Cron mode: If CRON_SCHEDULE is set, installs a cron job and runs cron in foreground
#   2. On-demand mode: Otherwise, passes all arguments through to generate_report.py
#
# Per-report log files are created by Python (generate_report.py --logs-dir).
# Reports/logs older than RETENTION_DAYS (default: 30) are automatically purged.
# On Linux, set PUID/PGID to match your host user so files are created with
# the correct ownership (e.g., PUID=1000 PGID=1000).

LOG_DIR="/app/logs"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# --- Fix volume permissions first (as root, before anything else) ---
# On Linux, bind-mounted dirs may be owned by root with restricted permissions.
# Ensure they exist and are accessible.
for dir in /app/reports /app/reports/reviews /app/.llm_cache "$LOG_DIR"; do
    mkdir -p "$dir"
    chmod 777 "$dir" 2>/dev/null || true
    # Also fix any existing files inside
    find "$dir" -maxdepth 1 -type f -exec chmod 666 {} \; 2>/dev/null || true
done

# --- Set up non-root user if PUID/PGID are provided ---
RUN_AS=""
if [ -n "$PUID" ]; then
    PGID=${PGID:-$PUID}
    # Create group/user if they don't already exist
    getent group "$PGID" >/dev/null 2>&1 || groupadd -g "$PGID" appgroup
    id -u "$PUID" >/dev/null 2>&1 || useradd -u "$PUID" -g "$PGID" -M -s /bin/bash appuser
    USER_NAME=$(id -nu "$PUID")
    # Own the writable directories
    chown -R "$PUID:$PGID" /app/reports /app/.llm_cache "$LOG_DIR" 2>/dev/null || true
    RUN_AS="$USER_NAME"
    echo "[entrypoint] Running as $USER_NAME (uid=$PUID gid=$PGID)"
fi

# Helper: run a command as the app user (or root if PUID not set)
run_cmd() {
    if [ -n "$RUN_AS" ]; then
        su -s /bin/bash "$RUN_AS" -c "$*"
    else
        eval "$@"
    fi
}

if [ -n "$CRON_SCHEDULE" ]; then
    echo "[entrypoint] Cron mode: schedule='$CRON_SCHEDULE'"

    # Build the command line from REPORT_ARGS env var (space-separated flags)
    REPORT_CMD="cd /app && python generate_report.py --logs-dir ${LOG_DIR} --retention-days ${RETENTION_DAYS} ${REPORT_ARGS:-}"

    # Wrap in su if running as non-root
    if [ -n "$RUN_AS" ]; then
        REPORT_CMD="su -s /bin/bash $RUN_AS -c '$REPORT_CMD'"
    fi

    # Propagate environment variables into the cron environment.
    # Cron runs in a minimal environment, so we must pass through vars explicitly.
    ENV_FILE="/app/.env.cron"
    : > "$ENV_FILE"
    [ -n "$ANTHROPIC_API_KEY" ] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> "$ENV_FILE"
    [ -n "$OLLAMA_URL" ] && echo "OLLAMA_URL=$OLLAMA_URL" >> "$ENV_FILE"
    [ -n "$OUTPUT_DIR" ] && echo "OUTPUT_DIR=$OUTPUT_DIR" >> "$ENV_FILE"
    [ -n "$LLM_CACHE_DIR" ] && echo "LLM_CACHE_DIR=$LLM_CACHE_DIR" >> "$ENV_FILE"
    [ -n "$RETENTION_DAYS" ] && echo "RETENTION_DAYS=$RETENTION_DAYS" >> "$ENV_FILE"
    [ -n "$GITHUB_REPO" ] && echo "GITHUB_REPO=$GITHUB_REPO" >> "$ENV_FILE"
    [ -n "$GITHUB_BRANCH" ] && echo "GITHUB_BRANCH=$GITHUB_BRANCH" >> "$ENV_FILE"
    [ -n "$GITHUB_TOKEN" ] && echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> "$ENV_FILE"
    # Add PATH so python is found
    echo "PATH=$PATH" >> "$ENV_FILE"

    # Write crontab entry
    # Per-report logs are created by Python; stdout goes to Docker logs via PID 1
    # generate_report.py now rebuilds review pages, index, and publishes to GitHub
    # after every day processed, so no separate post-run steps are needed here.
    CRON_LINE="$CRON_SCHEDULE set -a && . /app/.env.cron && $REPORT_CMD >> /proc/1/fd/1 2>&1"
    echo "$CRON_LINE" | crontab -

    echo "[entrypoint] Cron job installed. Waiting for schedule..."
    echo "[entrypoint] Next run determined by: $CRON_SCHEDULE"
    echo "[entrypoint] Per-report logs written to: $LOG_DIR"

    # Run an initial report immediately if requested
    # generate_report.py handles review pages, index rebuild, and GitHub publish per-day.
    if [ "$RUN_ON_STARTUP" = "true" ]; then
        echo "[entrypoint] RUN_ON_STARTUP=true, running initial report..."
        run_cmd "cd /app && python generate_report.py --logs-dir ${LOG_DIR} --retention-days ${RETENTION_DAYS} ${REPORT_ARGS:-}" 2>&1
    fi

    # Start cron in foreground (cron itself must run as root)
    exec cron -f
else
    echo "[entrypoint] On-demand mode"
    echo "[entrypoint] Per-report logs written to: $LOG_DIR"
    # Pass all arguments through to generate_report.py (Python creates per-report logs)
    # generate_report.py handles review pages, index rebuild, and GitHub publish per-day.
    run_cmd "cd /app && python generate_report.py --logs-dir ${LOG_DIR} --retention-days ${RETENTION_DAYS} $*" 2>&1
fi
