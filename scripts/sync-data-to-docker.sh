#!/usr/bin/env bash
# =============================================================================
# 数据同步脚本 — 将本地预构建数据导入 Docker 命名卷
#
# 适用场景:
#   - 本地已构建好索引和数据库，想用 Docker 部署
#   - 不想使用 bind mount（Windows 下 bind mount 性能较差）
#
# 用法:
#   # 1. 先启动一次 API 以创建空的命名卷
#   docker compose up api -d
#   docker compose stop api
#
#   # 2. 运行此脚本将本地数据复制到卷中
#   bash scripts/sync-data-to-docker.sh
#
#   # 3. 重新启动全部服务
#   docker compose up -d
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────
# 检查本地数据是否存在
# ─────────────────────────────────────────────────────────────
check_local_data() {
    local has_data=false

    if [ -f "src/database/enriched_products.db" ]; then
        log_info "✓ 本地数据库存在 (enriched_products.db)"
        has_data=true
    else
        log_warn "✗ 本地数据库不存在"
    fi

    if [ -f "src/indexing/indexes/bm25.pkl" ]; then
        log_info "✓ 本地 BM25 索引存在"
        has_data=true
    else
        log_warn "✗ 本地 BM25 索引不存在"
    fi

    if [ -d "uploads/raw" ]; then
        log_info "✓ 本地上传目录存在"
    fi

    if [ "$has_data" = false ]; then
        log_error "本地没有任何可同步的数据。请先运行: python src/indexing/embedding.py"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────
# 同步到 Docker 卷
# ─────────────────────────────────────────────────────────────
sync_to_volume() {
    local volume_name="$1"
    local local_path="$2"
    local container_path="$3"

    if [ ! -e "$local_path" ]; then
        log_warn "跳过 $volume_name（本地路径 $local_path 不存在）"
        return 0
    fi

    log_info "同步 $local_path → $volume_name:$container_path"

    # 使用临时容器复制文件到卷
    docker run --rm \
        -v "${volume_name}:${container_path}" \
        -v "$(pwd)/${local_path}:/tmp/source:ro" \
        alpine:latest \
        sh -c "cp -a /tmp/source/. ${container_path}/ 2>/dev/null || cp -a /tmp/source ${container_path}"

    log_info "完成: $volume_name"
}

# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
main() {
    log_info "============================================"
    log_info "  数据同步：本地 → Docker 卷"
    log_info "============================================"

    check_local_data

    echo ""
    log_warn "注意: 需要先运行 'docker compose up api -d' 创建卷,"
    log_warn "      然后 'docker compose stop api' 停止容器。"
    echo ""
    read -rp "是否继续? [y/N] " confirm
    if [ "${confirm,,}" != "y" ]; then
        log_info "已取消"
        exit 0
    fi

    sync_to_volume "llm-based-recommender_database_data"  "src/database"               "/app/src/database"
    sync_to_volume "llm-based-recommender_indexes_data"    "src/indexing/indexes"       "/app/src/indexing/indexes"
    sync_to_volume "llm-based-recommender_raw_data"        "src/indexing/data"          "/app/src/indexing/data"
    sync_to_volume "llm-based-recommender_uploads_data"    "uploads"                    "/app/uploads"

    log_info "============================================"
    log_info "  同步完成！运行以下命令启动："
    log_info "  docker compose up -d"
    log_info "============================================"
}

main
