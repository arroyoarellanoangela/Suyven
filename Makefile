.DEFAULT_GOAL := help
SHELL := /bin/bash

# ──────────────────────────────────────────────
# Suyven RAG Engine — Makefile
# ──────────────────────────────────────────────

.PHONY: help install dev lint format test docker-build docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run linters (ruff + mypy)
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff check --fix src/ tests/
	ruff format src/ tests/

test: ## Run tests with coverage
	python -m pytest tests/ -v --tb=short

docker-build: ## Build Docker image (CPU)
	docker build -t suyven:latest .

docker-up: ## Start services
	docker compose up -d

docker-down: ## Stop services
	docker compose down

clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist htmlcov .coverage
