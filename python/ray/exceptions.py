"""Minimal ``ray.exceptions`` so vLLM's imports and except-clauses resolve.

vLLM catches a few of these around remote calls. beam raises plain
RuntimeError from failed calls; these subclass it so existing handlers still fire.
"""


class RayError(RuntimeError):
    pass


class RayActorError(RayError):
    pass


class RayTaskError(RayError):
    pass


class GetTimeoutError(RayError):
    pass


class ActorDiedError(RayActorError):
    pass


class RayChannelError(RayError):
    pass
