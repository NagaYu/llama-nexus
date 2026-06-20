# ---------------------------------------------------------------------------
# Llama-Nexus — self-hosted, $0.00 multimodal inference.
#
# This image ships ONLY the static dashboard + a tiny FastAPI control plane.
# There is no server-side model: every byte of inference runs in the visitor's
# browser on WebGPU. The container therefore stays small and needs no GPU.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Faster, quieter, reproducible Python.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXUS_HOST=0.0.0.0 \
    NEXUS_PORT=8000 \
    NEXUS_DB_PATH=/data/nexus.db

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source.
COPY app/ ./app/

# Persistent SQLite volume (conversation logs + prompt templates).
RUN mkdir -p /data
VOLUME ["/data"]

# Drop root.
RUN useradd --create-home --uid 10001 nexus && chown -R nexus:nexus /app /data
USER nexus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
