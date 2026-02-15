.PHONY: lint format check typecheck test test-unit test-integration test-fast clean

PYTHON := .venv/bin/python
RUFF := .venv/bin/ruff

## Linting & Formatting

format: ## Format code with ruff
	$(RUFF) format macaw/ tests/

lint: ## Run ruff check (lint)
	$(RUFF) check macaw/ tests/

typecheck: ## Run mypy (type checking)
	$(PYTHON) -m mypy macaw/

check: format lint typecheck ## Format + lint + typecheck (all)

## Tests

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -q

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -q

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -q

test-fast: ## Run tests excluding those marked as slow
	$(PYTHON) -m pytest tests/ -q -m "not slow"

## CI (simulate local pipeline)

ci: format lint typecheck test ## Full pipeline: format + lint + typecheck + tests

## Utilities

clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

proto: ## Generate protobuf stubs
	PYTHON_BIN=$(PYTHON) bash scripts/generate_proto.sh

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
