# Contributor task runner. Every target here wraps a command that CI runs
# (.github/workflows/ci.yml); keep the two in step. Run `make` for the list.

UV ?= uv
UVX ?= uvx
PYTEST_DEPS = --with pytest --with hypothesis --with pytest-cov --with cloudpickle
PYTEST = $(UV) run $(PYTEST_DEPS) pytest
LINT_PATHS = python examples scripts tests

.DEFAULT_GOAL := help
.PHONY: help test test-one e2e lint format types shell yaml import release-check check

help:  ## list targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

test:  ## unit + fuzz suite with the 100% coverage gate (no GPUs needed)
	$(PYTEST) tests/ -q --cov=ray --cov-report=term-missing --cov-fail-under=100

test-one:  ## one test or pattern: make test-one T=tests/test_cli.py::test_x
	$(PYTEST) $(T) -q

e2e:  ## control-plane harnesses: single-node, multi-node, edge, driver-on-worker
	bash test/run_e2e.sh
	bash test/run_multinode.sh
	bash test/run_edge.sh
	bash test/run_driver_on_worker.sh

lint:  ## ruff check + black --check (as CI runs them)
	$(UVX) ruff check $(LINT_PATHS)
	$(UVX) black --check $(LINT_PATHS)

format:  ## rewrite with black
	$(UVX) black $(LINT_PATHS)

types:  ## mypy strict over python/ray
	$(UVX) --with cloudpickle mypy --config-file pyproject.toml python/ray

shell:  ## shellcheck the harnesses
	shellcheck -x test/*.sh test/dgx/*.sh

yaml:  ## yamllint the CI workflows
	$(UVX) yamllint -c .yamllint.yml .github/workflows/ci.yml .github/workflows/release.yml

release-check:  ## version numbers, docs and CHANGELOG agree (add TAG=v0.3.0 to check a cut)
	python3 scripts/check_release.py $(if $(TAG),--tag $(TAG),)

import:  ## import-only smoke test of the whole shim surface
	PYTHONPATH=python $(UV) run --with cloudpickle python examples/import_check.py

check: lint types shell yaml test import release-check e2e  ## everything CI runs
