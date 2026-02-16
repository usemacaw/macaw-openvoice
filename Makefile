.PHONY: lint format check typecheck test test-unit test-integration test-fast test-cov security audit complexity clean ci

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

all: format lint typecheck test security audit ## Format + lint + typecheck (all)

## Tests

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -q

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -q

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -q

test-fast: ## Run tests excluding those marked as slow
	$(PYTHON) -m pytest tests/ -q -m "not slow"

test-cov: ## Run unit tests with coverage report
	$(PYTHON) -m pytest tests/unit/ -q --cov=macaw --cov-report=term-missing --cov-report=html

## Security

security: ## Run bandit SAST scanner
	$(PYTHON) -m bandit -c pyproject.toml -r macaw/

audit: ## Run pip-audit for dependency vulnerabilities
	$(PYTHON) -m pip freeze | grep -v -E "^(torch|torchaudio)==" > .audit-requirements.txt
	$(PYTHON) -m pip_audit --desc -r .audit-requirements.txt
	@rm -f .audit-requirements.txt

## Code Quality

complexity: ## Cyclomatic complexity report (diagnostic, not in CI)
	@echo "── Cyclomatic Complexity (functions CC ≥ 11) ──"
	@$(PYTHON) -m radon cc macaw/ -a -s -n C --exclude "macaw/proto/*"
	@echo ""
	@echo "── Maintainability Index (files below A) ──"
	@$(PYTHON) -m radon mi macaw/ -s --exclude "macaw/proto/*" | grep -v " - A " || echo "  All files rated A — excellent!"
	@echo ""
	@# Fail if any function has CC >= 20 (D rating or worse)
	@$(PYTHON) -m radon cc macaw/ -n D --exclude "macaw/proto/*" --json | $(PYTHON) -c "\
	import json, sys; \
	data = json.load(sys.stdin); \
	hotspots = [(b['complexity'], b['name'], f, b['lineno']) for f, blocks in data.items() for b in blocks if b['complexity'] >= 20]; \
	[print(f'  CC {cc}: {name} ({path}:{line})') for cc, name, path, line in sorted(hotspots, reverse=True)]; \
	sys.exit(1) if hotspots else print('  No functions with CC ≥ 20 — passed!')"

## CI (simulate local pipeline)

ci: check test-cov security audit ## Full pipeline: format + lint + typecheck + tests with coverage + security + audit

## Utilities

clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage .coverage.*

proto: ## Generate protobuf stubs
	PYTHON_BIN=$(PYTHON) bash scripts/generate_proto.sh

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
