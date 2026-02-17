#!/bin/bash
set -e

# docker-entrypoint.sh
# Handles two modes:
#   1. Cron mode: If CRON_SCHEDULE is set, installs a cron job and runs cron in foreground
#   2. On-demand mode: Otherwise, passes all arguments through to generate_report.py
#
# On Linux, set PUID/PGID to match your host user so files are created with
# the correct ownership (e.g., PUID=1000 PGID=1000).

LOG_DIR="/app/logs"

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

# Extract --date and --days from CLI arguments.
# Returns "END_DATE DAYS" (space-separated).
# Defaults: date=yesterday, days=1.
parse_report_args() {
    local date_val=""
    local days_val=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --date)   date_val="$2"; shift 2 ;;
            --date=*) date_val="${1#--date=}"; shift ;;
            --days)   days_val="$2"; shift 2 ;;
            --days=*) days_val="${1#--days=}"; shift ;;
            *)        shift ;;
        esac
    done
    if [ -z "$date_val" ]; then
        date_val=$(date -d "yesterday" '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d')
    fi
    days_val=${days_val:-1}
    echo "$date_val $days_val"
}

# Generate log filename.
# Single day:  lkml_2025-02-15.log
# Multi-day:   lkml_2025-02-09_to_2025-02-15.log
log_file() {
    local end_date="$1"
    local num_days="$2"
    if [ "$num_days" -gt 1 ] 2>/dev/null; then
        # Compute start date = end_date - (days - 1)
        local start_date
        start_date=$(date -d "$end_date - $((num_days - 1)) days" '+%Y-%m-%d' 2>/dev/null \
            || date -j -v-"$((num_days - 1))"d -f '%Y-%m-%d' "$end_date" '+%Y-%m-%d')
        echo "${LOG_DIR}/lkml_${start_date}_to_${end_date}.log"
    else
        echo "${LOG_DIR}/lkml_${end_date}.log"
    fi
}

if [ -n "$CRON_SCHEDULE" ]; then
    echo "[entrypoint] Cron mode: schedule='$CRON_SCHEDULE'"

    # Build the command line from REPORT_ARGS env var (space-separated flags)
    REPORT_CMD="cd /app && python generate_report.py ${REPORT_ARGS:-}"

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
    # Add PATH so python is found
    echo "PATH=$PATH" >> "$ENV_FILE"

    # Build the post-report commands (index + review pages)
    INDEX_CMD="cd /app && python build_index.py --reports-dir /app/reports --logs-dir /app/logs"
    REVIEWS_CMD="cd /app && python build_reviews.py --reports-dir /app/reports"
    if [ -n "$RUN_AS" ]; then
        INDEX_CMD="su -s /bin/bash $RUN_AS -c '$INDEX_CMD'"
        REVIEWS_CMD="su -s /bin/bash $RUN_AS -c '$REVIEWS_CMD'"
    fi

    # Write crontab entry
    # Logs to file named after the report date AND docker logs (via tee to PID 1 stdout)
    # Cron runs for yesterday's activity (generate_report.py default), so log uses yesterday's date
    # After report generation, rebuild review pages and index page
    CRON_LINE="$CRON_SCHEDULE set -a && . /app/.env.cron && REPORT_DATE=\$(date -d 'yesterday' '+\\%Y-\\%m-\\%d') && LOGFILE=${LOG_DIR}/lkml_\${REPORT_DATE}.log && $REPORT_CMD 2>&1 | tee \"\$LOGFILE\" >> /proc/1/fd/1 && $REVIEWS_CMD >> /proc/1/fd/1 2>&1 && $INDEX_CMD >> /proc/1/fd/1 2>&1"
    echo "$CRON_LINE" | crontab -

    echo "[entrypoint] Cron job installed. Waiting for schedule..."
    echo "[entrypoint] Next run determined by: $CRON_SCHEDULE"
    echo "[entrypoint] Logs written to: $LOG_DIR"

    # Run an initial report immediately if requested
    if [ "$RUN_ON_STARTUP" = "true" ]; then
        read -r REPORT_DATE REPORT_DAYS <<< "$(parse_report_args ${REPORT_ARGS:-})"
        LOGFILE=$(log_file "$REPORT_DATE" "$REPORT_DAYS")
        echo "[entrypoint] RUN_ON_STARTUP=true, running initial report for $REPORT_DATE (days=$REPORT_DAYS)..."
        echo "[entrypoint] Log file: $LOGFILE"
        run_cmd "cd /app && python generate_report.py ${REPORT_ARGS:-}" 2>&1 | tee "$LOGFILE"
        echo "[entrypoint] Building review pages..."
        run_cmd "cd /app && python build_reviews.py --reports-dir /app/reports"
        echo "[entrypoint] Rebuilding index page..."
        run_cmd "cd /app && python build_index.py --reports-dir /app/reports --logs-dir /app/logs"
    fi

    # Start cron in foreground (cron itself must run as root)
    exec cron -f
else
    read -r REPORT_DATE REPORT_DAYS <<< "$(parse_report_args "$@")"
    LOGFILE=$(log_file "$REPORT_DATE" "$REPORT_DAYS")
    if [ "$REPORT_DAYS" -gt 1 ] 2>/dev/null; then
        echo "[entrypoint] On-demand mode ($REPORT_DAYS days ending $REPORT_DATE)"
    else
        echo "[entrypoint] On-demand mode (report date: $REPORT_DATE)"
    fi
    echo "[entrypoint] Log file: $LOGFILE"
    # Pass all arguments through to generate_report.py, log to file and stdout
    run_cmd "cd /app && python generate_report.py $*" 2>&1 | tee "$LOGFILE"
    # Build review pages and rebuild the index page
    echo "[entrypoint] Building review pages..."
    run_cmd "cd /app && python build_reviews.py --reports-dir /app/reports"
    echo "[entrypoint] Rebuilding index page..."
    run_cmd "cd /app && python build_index.py --reports-dir /app/reports --logs-dir /app/logs"
fi
