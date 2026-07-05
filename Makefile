# Lithos developer tasks. `make help` lists them.
UV ?= uv

.PHONY: help install test lint format typecheck check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the dev environment (core + data + eval extras)
	$(UV) sync --extra data --extra eval

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Lint with ruff
	$(UV) run ruff check .

format:  ## Auto-format + apply safe lint fixes
	$(UV) run ruff format . && $(UV) run ruff check . --fix

typecheck:  ## Type-check the package with mypy
	$(UV) run mypy lithos

check: lint typecheck test  ## Run every gate (lint + types + tests) — mirrors CI

clean:  ## Remove tool caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache
