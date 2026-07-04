.PHONY: help build up down train logs shell test clean ssl

# ── default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Fraud Detection System — Commands"
	@echo "  ──────────────────────────────────"
	@echo "  make build    Build all Docker images"
	@echo "  make up       Start the full stack"
	@echo "  make down     Stop the stack"
	@echo "  make train    Run model training"
	@echo "  make logs     Tail all logs"
	@echo "  make shell    Shell into API container"
	@echo "  make test     Run API smoke tests"
	@echo "  make clean    Remove containers + volumes"
	@echo "  make ssl      Generate self-signed SSL cert"
	@echo "  make deploy   Full deploy (build + ssl + up)"
	@echo ""

# ── docker compose wrappers ───────────────────────────────────────────────────
build:
	docker compose build --parallel

up:
	docker compose up -d

down:
	docker compose down

train:
	docker compose --profile train run --rm trainer

logs:
	docker compose logs -f

shell:
	docker compose exec api /bin/bash

# ── testing ───────────────────────────────────────────────────────────────────
test:
	@echo "Running API smoke tests..."
	@curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@curl -sf -X POST http://localhost:8000/predict \
	    -H "Content-Type: application/json" \
	    -d '{"features": [0,0,0,-18,0,0,0,0,0,0,-9,0,0,-14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4892,0]}' \
	    | python3 -m json.tool
	@echo "Smoke tests passed ✓"

# ── cleanup ───────────────────────────────────────────────────────────────────
clean:
	docker compose down -v --remove-orphans
	docker image rm fraud_detection-api fraud_detection-dashboard 2>/dev/null || true

# ── SSL cert ─────────────────────────────────────────────────────────────────
ssl:
	mkdir -p nginx/ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
	    -keyout nginx/ssl/server.key \
	    -out    nginx/ssl/server.crt \
	    -subj   "/C=US/ST=State/L=City/O=FraudDet/CN=localhost"
	@echo "SSL cert generated at nginx/ssl/"

# ── full deploy ───────────────────────────────────────────────────────────────
deploy: ssl build up
	@echo ""
	@echo "  Stack deployed!"
	@echo "  Dashboard → https://localhost"
	@echo "  API docs  → http://localhost:8000/docs"
