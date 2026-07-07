FROM python:3.10-slim

WORKDIR /app

ARG TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url "${TORCH_EXTRA_INDEX_URL}" torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY src/ src/
COPY scripts/ scripts/

RUN mkdir -p /app/src/indexing/indexes \
    /app/src/indexing/data \
    /app/src/database \
    /app/uploads \
    /app/.cache/huggingface \
    && chmod +x scripts/docker-entrypoint.sh \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["api"]
