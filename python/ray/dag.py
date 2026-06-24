"""``ray.dag``: present only so type annotations resolve. beam does not
implement Ray's accelerated/compiled DAG; vLLM gates that on the presence of
``ray.experimental.compiled_dag_ref`` (which beam intentionally omits), so this
module's classes are never actually instantiated on the supported path.
"""

_UNSUPPORTED = (
    "beam does not implement Ray compiled DAG (VLLM_USE_RAY_COMPILED_DAG); "
    "use the default per-worker RPC path."
)


class CompiledDAG:
    def __init__(self, *a, **k):
        raise NotImplementedError(_UNSUPPORTED)


class InputNode:
    def __init__(self, *a, **k):
        raise NotImplementedError(_UNSUPPORTED)


class MultiOutputNode:
    def __init__(self, *a, **k):
        raise NotImplementedError(_UNSUPPORTED)
