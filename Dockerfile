# Optional: bake beam into an image instead of bind-mounting it at runtime.
# The supported path is the bind mount (see README / test/dgx) which needs no
# rebuild; this is here for those who want an immutable image.
#
#   docker build -t vllm-beam .
#   docker build --build-arg BASE=vllm/vllm-openai:v0.23.0 -t vllm-beam .

ARG BASE=vllm/vllm-openai:latest
FROM ${BASE}

COPY python /opt/beam/python
COPY examples /opt/beam/examples

# Install the shim as the `ray` package (import ray + the `ray` command), then
# smoke-test that every symbol vLLM needs resolves.
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache /opt/beam/python \
    && python3 /opt/beam/examples/import_check.py

# vLLM's entrypoint is inherited from the base image; cluster nodes override it
# with `ray start ...` (see docker/run_cluster style usage in test/dgx).
