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
mkdir -p "$LOG_DIR"

# --- Set up non-root user if PUID/PGID are provided ---
RUN_AS=""
if [ -n "$PUID" ]; then
    PGID=${PGID:-$PUID}
    # Create group/user if they don't already exist
    getent group "$PGID" >/dev/null 2>&1 || groupadd -g "$PGID" appgroup
    GROUP_NAME=$(getent group "$PGID" | cut -d: -f1)
    id -u "$PUID" >/dev/null 2>&1 || useradd -u "$PUID" -g "$PGID" -M -s /bin/bash appuser
    USER_NAME=$(id -nu "$PUID")
    # Own the writable directories
    chown -R "$PUID:$PGID" /app/reports /app/.llm_cache /app/logs
    RUN_AS="$USER_NAME"
    echo "[entrypoint] Running as $USER_NAME (uid=$PUID gid=$PGID)"
fi

# Ensure mounted volumes are writable
chmod -R a+rw /app/reports /app/.llm_cache /app/logs 2>/dev/null || true

# Helper: run a command as the app user (or root if PUID not set)
run_cmd() {
    if [ -n "$RUN_AS" ]; then
        su -s /bin/bash "$RUN_AS" -c "$*"
    else
        eval "$@"
    fi
}

# Generate a timestamped log filename: lkml_2025-02-15_023000.log
log_file() {
    echo "${LOG_DIR}/lkml_$(date '+%Y-%m-%d_%H%M%S').log"
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

    # Write crontab entry
    # Logs to timestamped file AND docker logs (via tee to PID 1 stdout)
    CRON_LINE="$CRON_SCHEDULE set -a && . /app/.env.cron && LOGFILE=${LOG_DIR}/lkml_\$(date '+\\%Y-\\%m-\\%d_\\%H\\%M\\%S').log && $REPORT_CMD 2>&1 | tee \"\$LOGFILE\" >> /proc/1/fd/1"
    echo "$CRON_LINE" | crontab -

    echo "[entrypoint] Cron job installed. Waiting for schedule..."
    echo "[entrypoint] Next run determined by: $CRON_SCHEDULE"
    echo "[entrypoint] Logs written to: $LOG_DIR"

    # Run an initial report immediately if requested
    if [ "$RUN_ON_STARTUP" = "true" ]; then
        LOGFILE=$(log_file)
        echo "[entrypoint] RUN_ON_STARTUP=true, running initial report..."
        echo "[entrypoint] Log file: $LOGFILE"
        run_cmd "cd /app && python generate_report.py ${REPORT_ARGS:-}" 2>&1 | tee "$LOGFILE"
    fi

    # Start cron in foreground (cron itself must run as root)
    exec cron -f
else
    LOGFILE=$(log_file)
    echo "[entrypoint] On-demand mode"
    echo "[entrypoint] Log file: $LOGFILE"
    # Pass all arguments through to generate_report.py, log to file and stdout
    run_cmd "cd /app && python generate_report.py $*" 2>&1 | tee "$LOGFILE"
fi
