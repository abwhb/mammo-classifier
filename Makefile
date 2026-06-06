.PHONY: help install api web test lint clean

help:
	@echo "Targets:"
	@echo "  install   Sync api + web dependencies"
	@echo "  api       Run FastAPI on :8080"
	@echo "  web       Run Next.js on :3000"
	@echo "  test      Run api pytest"
	@echo "  lint      Ruff + Next lint"
	@echo "  clean     Remove caches and node_modules"

install:
	cd apps/api && uv sync
	cd apps/web && pnpm install

api:
	cd apps/api && uv run uvicorn app.main:app --reload --port 8080

web:
	cd apps/web && pnpm dev

test:
	cd apps/api && uv run pytest -q

lint:
	cd apps/api && uv run ruff check .
	cd apps/web && pnpm lint

clean:
	rm -rf apps/api/.venv apps/api/.pytest_cache apps/api/.ruff_cache
	rm -rf apps/web/node_modules apps/web/.next
