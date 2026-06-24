"""``ray.runtime_env.RuntimeEnv``.

beam ignores runtime-env contents (no per-actor env/dependency provisioning); it
just needs the type to exist and behave like the dict vLLM builds. Env vars vLLM
puts here are already inherited by the actor subprocess from the daemon.
"""


class RuntimeEnv(dict):
    # accept both RuntimeEnv(env_vars=...) and RuntimeEnv({...}), like real ray
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
