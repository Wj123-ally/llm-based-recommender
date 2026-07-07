#!/usr/bin/env bash
set -euo pipefail

mode="${1:-api}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_api_key() {
  if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    log "DASHSCOPE_API_KEY is required for API recommendations."
    exit 1
  fi
}

ensure_database() {
  local db_path="/app/src/database/enriched_products.db"
  if [ -f "$db_path" ]; then
    log "SQLite database found: $db_path"
    return
  fi

  if [ "${BOOTSTRAP:-false}" != "true" ]; then
    log "SQLite database missing: $db_path"
    log "Mount ./src/database or set BOOTSTRAP=true with processed JSONL data mounted."
    exit 1
  fi

  log "Seeding SQLite database from processed JSONL..."
  python -m src.database.seed_jsonl_data
}

ensure_indexes() {
  local bm25_path="/app/src/indexing/indexes/bm25.pkl"
  if [ -f "$bm25_path" ]; then
    log "BM25 index found: $bm25_path"
    return
  fi

  if [ "${BOOTSTRAP:-false}" != "true" ]; then
    log "BM25 index missing: $bm25_path"
    log "Mount ./src/indexing/indexes or set BOOTSTRAP=true to rebuild indexes."
    exit 1
  fi

  log "Building product retrieval indexes..."
  python -m src.indexing.embedding
}

wait_for_api() {
  local base_url="${1%/}"
  local health_url="${base_url}/health"
  local attempts="${API_WAIT_ATTEMPTS:-90}"

  log "Waiting for API: $health_url"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$health_url" >/dev/null 2>&1; then
      log "API is ready."
      return
    fi
    sleep 2
  done

  log "API did not become ready in time."
  exit 1
}

start_api() {
  require_api_key
  ensure_database
  ensure_indexes

  log "Starting FastAPI on port 8000..."
  exec python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS:-1}" \
    --log-level "${LOG_LEVEL:-info}"
}

start_ui() {
  export API_URL="${API_URL:-http://api:8000}"
  wait_for_api "$API_URL"

  log "Starting Streamlit UI on port 8501..."
  exec python -m streamlit run src/ui/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true
}

start_admin() {
  export API_URL="${API_URL:-http://api:8000}"
  wait_for_api "$API_URL"

  log "Starting Streamlit admin on port 8502..."
  exec python -m streamlit run src/ui/admin_app.py \
    --server.address 0.0.0.0 \
    --server.port 8502 \
    --server.headless true
}

case "$mode" in
  api) start_api ;;
  ui) start_ui ;;
  admin) start_admin ;;
  *)
    log "Unknown mode: $mode"
    log "Usage: docker-entrypoint.sh [api|ui|admin]"
    exit 1
    ;;
esac
