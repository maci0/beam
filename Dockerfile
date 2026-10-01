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
FROM ${BASE}

# uv is the build tool: pinned, not whatever pip resolves on the day. Same
# version the Makefile and both workflows pin.
ARG UV_VERSION=0.12.14

COPY python /opt/beam/python
COPY examples /opt/beam/examples

# Install the shim as the `ray` package (import ray + the `ray` command), then
# smoke-test that every symbol vLLM needs resolves.
RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv pip install --system --no-cache /opt/beam/python \
    && python3 /opt/beam/examples/import_check.py

# vLLM's entrypoint is inherited from the base image; cluster nodes override it
# with `ray start ...` (see docker/run_cluster style usage in test/dgx).
