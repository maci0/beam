# Optional: bake beam into an image instead of bind-mounting it at runtime.
# The supported path is the bind mount (see README / test/dgx) which needs no
# rebuild; this is here for those who want an immutable image.
#
#   docker build -t vllm-beam .
#   docker build --build-arg BASE=vllm/vllm-openai:v0.23.0 -t vllm-beam .
#
# BASE defaults to a floating :latest, so the same source gives a different image
# whenever upstream moves. Pass the tag you actually tested (v0.23.0 is what
# docs/OPERATIONS.md records) and record the digest if you need to rebuild the
# same bytes later.
ARG BASE=vllm/vllm-openai:latest

# uv is the build tool, pinned by version and digest: the same version the
# Makefile and both workflows pin. Bump the tag and digest together.
FROM ghcr.io/astral-sh/uv:0.12.14@sha256:1946145b8706ad9e5c0e79a513f9e324b58d5e38126bb2c8b7dbfca61febeb45 AS uv

FROM ${BASE}

COPY python /opt/beam/python
COPY examples /opt/beam/examples

# Install the shim as the `ray` package (import ray + the `ray` command), then
# smoke-test that every symbol vLLM needs resolves. uv is mounted for this step
# only, so it never lands in an image layer.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache /opt/beam/python \
    && python3 /opt/beam/examples/import_check.py

LABEL org.opencontainers.image.title="beam" \
      org.opencontainers.image.source="https://github.com/maci0/beam" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

# vLLM's entrypoint is inherited from the base image; cluster nodes override it
# with `ray start ...` (see docker/run_cluster style usage in test/dgx).
