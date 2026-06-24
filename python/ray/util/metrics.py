"""``ray.util.metrics``: no-op metrics so vLLM's ray metric wrappers import and
run. beam does not export to Ray's metric backend (there isn't one); vLLM's own
Prometheus path is unaffected.
"""


class Metric:
    def __init__(self, name="", description="", tag_keys=None, *args, **kwargs):
        self.name = name

    def set_default_tags(self, tags):
        return self

    def record(self, value, tags=None):
        pass

    def inc(self, value=1.0, tags=None):
        pass

    def observe(self, value, tags=None):
        pass


class Gauge(Metric):
    pass


class Counter(Metric):
    pass


class Histogram(Metric):
    def __init__(self, name="", description="", boundaries=None, tag_keys=None, *args, **kwargs):
        super().__init__(name, description, tag_keys)
