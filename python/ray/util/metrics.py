"""``ray.util.metrics``: no-op metrics so vLLM's ray metric wrappers import and
run. beam does not export to Ray's metric backend (there isn't one); vLLM's own
Prometheus path is unaffected.
"""

from __future__ import annotations  # keep `X | None` valid on py3.9

from typing import Any


class Metric:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        tag_keys: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.name = name

    def set_default_tags(self, tags: dict[str, str]) -> Metric:
        return self

    def record(self, value: float, tags: dict[str, str] | None = None) -> None:
        pass

    def inc(self, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        pass

    def observe(self, value: float, tags: dict[str, str] | None = None) -> None:
        pass


class Gauge(Metric):
    pass


class Counter(Metric):
    pass


class Histogram(Metric):
    def __init__(
        self,
        name: str = "",
        description: str = "",
        boundaries: Any = None,
        tag_keys: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, description, tag_keys)
