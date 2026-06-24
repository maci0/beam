"""Empty ``ray.experimental`` package.

beam does not implement Ray's compiled/accelerated DAG. vLLM probes for it with
``importlib.util.find_spec("ray.experimental.compiled_dag_ref")``; if the parent
package is entirely absent that call *raises* ModuleNotFoundError instead of
returning None, which crashes vLLM with a confusing error. Providing this empty
package makes the probe return None, so vLLM reports the clean "Ray Compiled
Graph is not installed" message and (on a recent enough vLLM) uses the
MessageQueue executor instead. Force that with VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1.
"""
