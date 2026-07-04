#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — one-command deployment for the Fraud Detection System
# Usage: bash scripts/deploy.sh [--with-training]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

WITH_TRAINING=false
[[ "${1:-}" == "--with-training" ]] && WITH_TRAINING=true

# ── 0. Prerequisites ──────────────────────────────────────────────────────────
command -v docker        >/dev/null || error "Docker not installed"
command -v docker compose >/dev/null || error "docker compose (v2) not installed"

# ── 1. .env ───────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    warn ".env created from template — edit values before production use"
fi

# ── 2. Self-signed SSL cert (dev) ─────────────────────────────────────────────
mkdir -p nginx/ssl
if [[ ! -f nginx/ssl/server.crt ]]; then
    info "Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key \
        -out    nginx/ssl/server.crt \
        -subj   "/C=US/ST=State/L=City/O=FraudDet/CN=localhost" \
        2>/dev/null
    info "SSL cert generated at nginx/ssl/"
fi

# ── 3. Directories ────────────────────────────────────────────────────────────
mkdir -p data models logs

# ── 4. Dataset check ─────────────────────────────────────────────────────────
if [[ ! -f data/creditcard.csv ]]; then
    warn "data/creditcard.csv not found."
    if command -v kaggle >/dev/null; then
        info "Downloading via Kaggle CLI..."
        kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
    else
        warn "Install kaggle CLI and run:"
        warn "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip"
        warn "Or download manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    fi
fi

# ── 5. Build images ───────────────────────────────────────────────────────────
info "Building Docker images..."
docker compose build --parallel

# ── 6. Train models (optional) ───────────────────────────────────────────────
if $WITH_TRAINING; then
    if [[ -f data/creditcard.csv ]]; then
        info "Running model training..."
        docker compose --profile train run --rm trainer
        info "Training complete. Models saved to model_store volume."
    else
        warn "Skipping training — dataset not present"
    fi
fi

# ── 7. Start stack ────────────────────────────────────────────────────────────
info "Starting services..."
docker compose up -d

# ── 8. Wait for health ────────────────────────────────────────────────────────
info "Waiting for services to become healthy..."
for i in $(seq 1 30); do
    sleep 2
    API_OK=$(docker inspect --format='{{.State.Health.Status}}' fraud_api    2>/dev/null || echo "missing")
    DASH_OK=$(docker inspect --format='{{.State.Health.Status}}' fraud_dashboard 2>/dev/null || echo "missing")
    if [[ "$API_OK" == "healthy" && "$DASH_OK" == "healthy" ]]; then
        info "All services healthy!"
        break
    fi
    echo -n "."
    if [[ $i -eq 30 ]]; then
        warn "Services slow to start — check: docker compose logs"
    fi
done

# ── 9. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Fraud Detection System — Deployed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Dashboard  →  https://localhost"
echo "  API        →  https://localhost/api/docs"
echo "  API direct →  http://localhost:8000/docs"
echo "  Health     →  http://localhost:8000/health"
echo "  Metrics    →  http://localhost:8000/metrics"
echo ""
echo "  Logs:  docker compose logs -f"
echo "  Stop:  docker compose down"
echo ""
