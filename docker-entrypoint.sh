#!/bin/bash
set -e

# docker-entrypoint.sh
# Handles two modes:
#   1. Cron mode: If CRON_SCHEDULE is set, installs a cron job and runs cron in foreground
#   2. On-demand mode: Otherwise, passes all arguments through to generate_report.py

if [ -n "$CRON_SCHEDULE" ]; then
    echo "[entrypoint] Cron mode: schedule='$CRON_SCHEDULE'"

    # Build the command line from REPORT_ARGS env var (space-separated flags)
    REPORT_CMD="cd /app && python generate_report.py ${REPORT_ARGS:-}"

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
    # stdout/stderr redirected to PID 1 so `docker logs` captures output
    CRON_LINE="$CRON_SCHEDULE set -a && . /app/.env.cron && $REPORT_CMD >> /proc/1/fd/1 2>> /proc/1/fd/2"
    echo "$CRON_LINE" | crontab -

    echo "[entrypoint] Cron job installed. Waiting for schedule..."
    echo "[entrypoint] Next run determined by: $CRON_SCHEDULE"

    # Run an initial report immediately if requested
    if [ "$RUN_ON_STARTUP" = "true" ]; then
        echo "[entrypoint] RUN_ON_STARTUP=true, running initial report..."
        cd /app && python generate_report.py ${REPORT_ARGS:-}
    fi

    # Start cron in foreground
    exec cron -f
else
    echo "[entrypoint] On-demand mode"
    # Pass all arguments through to generate_report.py
    exec python /app/generate_report.py "$@"
fi
