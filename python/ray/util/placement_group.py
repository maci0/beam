"""``ray.util.placement_group`` module.

Real ray exposes both a *module* ``ray.util.placement_group`` (this file) and a
*function* of the same name re-exported on ``ray.util``. vLLM uses both:
``ray.util.placement_group(bundles)`` and
``from ray.util.placement_group import PlacementGroup``. The function binding in
``ray/util/__init__.py`` shadows this module on the ``ray.util`` namespace while
the module stays importable by path, matching ray's behavior.
"""


class _PGId(str):
    """Placement-group id that is a plain string on the wire but also answers
    ``.hex()`` (vLLM calls ``pg.id.hex()``)."""

    def hex(self):
        return str(self)


class PlacementGroup:
    def __init__(self, pg_id, bundle_specs, strategy="PACK"):
        self.id = _PGId(pg_id)
        self.bundle_specs = list(bundle_specs)
        self.strategy = strategy

    def ready(self):
        # Bundles are placed synchronously at creation, so the readiness ref is
        # already resolved: ray.get(pg.ready()) returns immediately.
        from .. import ObjectRef

        return ObjectRef(self.id + "-ready", value=None, has_value=True)

    def wait(self, timeout=None) -> bool:
        return True


def placement_group(bundles, strategy="PACK", *args, **kwargs) -> PlacementGroup:
    from .. import _need

    specs = [{k: float(v) for k, v in b.items()} for b in bundles]
    resp, _ = _need().request({"t": "create_pg", "specs": specs})
    return PlacementGroup(resp["pg"], bundles, strategy)


def get_current_placement_group():
    # The vLLM driver is not launched inside a placement group, so it creates
    # its own. Returning None reflects that and matches ray's behavior here.
    return None


def remove_placement_group(pg) -> None:
    from .. import _need

    _need().request({"t": "remove_pg", "pg": pg.id})
