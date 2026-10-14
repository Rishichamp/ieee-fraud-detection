FROM python:3.11-slim

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

# ── working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── python deps (layer-cached) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── create directories that must exist at runtime ────────────────────────────
RUN mkdir -p data models logs

# ── non-root user for security ───────────────────────────────────────────────
RUN useradd -m -u 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
