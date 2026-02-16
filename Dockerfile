FROM python:3.12-slim

LABEL maintainer="krush"
LABEL description="LKML Daily Activity Tracker - generates HTML reports of Linux kernel developer activity"

# Install cron (needed for scheduled mode) and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY generate_report.py .
COPY test_thread.py .
COPY llm_summarizer.py .
COPY llm_cache.py .
COPY lkml_client.py .
COPY models.py .
COPY activity_classifier.py .
COPY thread_analyzer.py .
COPY report_generator.py .
COPY kernel_developers_emails_1.csv .

# Copy entrypoint script and fix Windows line endings
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN sed -i 's/\r$//' /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

# Create volume mount points
RUN mkdir -p /app/reports /app/.llm_cache

ENTRYPOINT ["/docker-entrypoint.sh"]
