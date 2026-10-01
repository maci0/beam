# Contributor task runner. Every target here wraps a command that CI runs
# (.github/workflows/ci.yml); keep the two in step. Run `make` for the list.

UV ?= uv
UVX ?= uvx
# Lint/type/YAML tools are pinned, not floating: an unpinned `uvx ruff` makes
# every push a fresh build of the toolchain, so a new ruff release can fail CI on
# a commit that did not touch a lintable file. Bump a pin on purpose, in the same
# commit as the reformat it forces. These are the versions the tools were last
# green with; `uvx tool@ver` re-resolves every invocation, so raise them together.
RUFF = $(UVX) ruff@0.16.9
BLACK = $(UVX) black@26.5.1
MYPY = $(UVX) --with cloudpickle mypy@2.3.1
YAMLLINT = $(UVX) yamllint@1.38.0
PYTEST_DEPS = --with pytest --with hypothesis --with pytest-cov --with cloudpickle
PYTEST = $(UV) run $(PYTEST_DEPS) pytest
LINT_PATHS = python examples scripts tests
# Reproducible-builds.org: a fixed epoch (so archives and wheels do not carry the
# build time) plus C collation and UTC for anything that formats or sorts.
# SOURCE_DATE_EPOCH comes from the last commit, so it is stable per tree; the git
# fallback covers a build from an unpacked tarball with no history.
SOURCE_DATE_EPOCH ?= $(shell git log -1 --pretty=%ct 2>/dev/null || echo 0)
export SOURCE_DATE_EPOCH
REPRO_ENV = TZ=UTC LC_ALL=C PYTHONHASHSEED=0

.DEFAULT_GOAL := help
.PHONY: help test test-one e2e lint format types shell yaml import build bundle repro release-check check

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
	$(RUFF) check $(LINT_PATHS)
	$(BLACK) --check $(LINT_PATHS)

format:  ## rewrite with black
	$(BLACK) $(LINT_PATHS)

types:  ## mypy strict over python/ray
	$(MYPY) --config-file pyproject.toml python/ray

shell:  ## shellcheck the harnesses
	shellcheck -x test/*.sh test/dgx/*.sh

yaml:  ## yamllint the CI workflows
	$(YAMLLINT) -c .yamllint.yml .github/workflows/ci.yml .github/workflows/release.yml

release-check:  ## version numbers, docs and CHANGELOG agree (add TAG=v0.3.0 to check a cut)
	python3 scripts/check_release.py $(if $(TAG),--tag $(TAG),)

import:  ## import-only smoke test of the whole shim surface
	PYTHONPATH=python $(UV) run --with cloudpickle python examples/import_check.py

build:  ## wheel + sdist from python/, same command the release workflow runs
	cd python && $(UV) build --out-dir dist

# The bind-mount payload, byte-for-byte what .github/workflows/release.yml makes:
# sorted entries, 0:0 owners, mtimes clamped to SOURCE_DATE_EPOCH, no atime/ctime
# in the pax headers. gzip over a pipe stamps no filename and no time of its own,
# so the .gz is deterministic too.
BUNDLE = tar --sort=name --numeric-owner --owner=0 --group=0 \
          --mtime="@$$SOURCE_DATE_EPOCH" --pax-option=delete=atime,delete=ctime \
          --format=posix --exclude=*/dist --exclude=__pycache__ \
          -czf $(1) python examples

bundle:  ## the bind-mount tarball the release publishes
	$(REPRO_ENV) sh -c '$(call BUNDLE,beam-bindmount.tar.gz)'

repro:  ## build the artifacts twice, perturb the inputs, and require identical bytes
	@set -eu; \
	rm -rf .repro-a .repro-b; \
	mkdir -p .repro-a/python .repro-b/python; \
	cp -r python/. .repro-a/python/; cp -r python/. .repro-b/python/; \
	cp -r examples .repro-a/; cp -r examples .repro-b/; \
	find .repro-a .repro-b -name __pycache__ -type d -prune -exec rm -rf {} +; \
	find .repro-b -type f -exec touch -t 201905050505 {} +; \
	(cd .repro-a/python && $(REPRO_ENV) $(UV) build --out-dir ../dist >/dev/null); \
	(cd .repro-b/python && $(REPRO_ENV) $(UV) build --out-dir ../dist >/dev/null); \
	(cd .repro-a && $(REPRO_ENV) sh -c '$(call BUNDLE,beam-repro-a.tar.gz)'); \
	(cd .repro-b && $(REPRO_ENV) sh -c '$(call BUNDLE,beam-repro-b.tar.gz)'); \
	cmp .repro-a/beam-repro-a.tar.gz .repro-b/beam-repro-b.tar.gz && echo 'reproducible: bundle'; \
	cmp .repro-a/dist/*.whl .repro-b/dist/*.whl && echo 'reproducible: wheel'; \
	cmp .repro-a/dist/*.tar.gz .repro-b/dist/*.tar.gz && echo 'reproducible: sdist'; \
	rm -rf .repro-a .repro-b

check: lint types shell yaml test import release-check e2e  ## everything CI runs
