"""Unit + fuzz tests for the `ray.util` surface: placement groups, scheduling
strategies, metrics no-ops, `ray.util.state.list_nodes`, and
`ray._private.state`. Driven through the FakeClient pattern (monkeypatch
`ray._need`), so no daemon or sockets."""

import os
import sys

from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ray  # noqa: E402
import ray._private.state as _private_state  # noqa: E402
import ray.util.placement_group  # noqa: E402,F401  (ensure the submodule is loaded)
import ray.util.state as util_state  # noqa: E402
from ray import util  # noqa: E402
from ray.util import metrics, scheduling_strategies  # noqa: E402
from ray.util.placement_group import PlacementGroup, _PGId  # noqa: E402

# `ray.util.placement_group` the *attribute* is the re-exported function (it
# shadows the submodule on the package namespace, matching real ray). Reach the
# actual module object through sys.modules so we can test its functions.
pg_mod = sys.modules["ray.util.placement_group"]


class FakeClient:
    def __init__(self, responses=None, body=b""):
        self.responses = responses or {}
        self.body = body
        self.sent = []

    def request(self, header, payload=b""):
        self.sent.append((header, payload))
        t = header["t"]
        canned = self.responses.get(t, {})
        if canned.get("err"):
            raise RuntimeError(canned["err"])
        resp = {"t": t + "_ok", "resp": True, **canned}
        return resp, canned.get("_body", self.body)


def use(monkeypatch, fc):
    monkeypatch.setattr(ray, "_need", lambda: fc)
    return fc


# ---- _PGId ------------------------------------------------------------------


def test_pgid_hex_is_str_self():
    pid = _PGId("n1-pg3")
    assert pid.hex() == "n1-pg3" and isinstance(pid, str)


# ---- PlacementGroup ---------------------------------------------------------


def test_placement_group_object_fields():
    p = PlacementGroup("n1-pg1", [{"GPU": 1}], strategy="SPREAD")
    assert p.id == "n1-pg1" and p.strategy == "SPREAD"
    assert p.bundle_specs == [{"GPU": 1}]


def test_placement_group_ready_is_resolved():
    p = PlacementGroup("n1-pg1", [])
    ref = p.ready()
    assert ref._has_value and ref._value is None
    assert ref.id == "n1-pg1-ready"


def test_placement_group_wait_true():
    assert PlacementGroup("n1-pg1", []).wait() is True
    assert PlacementGroup("n1-pg1", []).wait(timeout=5) is True


def test_placement_group_creates_via_client(monkeypatch):
    fc = use(monkeypatch, FakeClient({"create_pg": {"pg": "n1-pg7"}}))
    p = pg_mod.placement_group([{"GPU": 1}, {"CPU": 1}])
    assert p.id == "n1-pg7"
    # specs are coerced to floats on the wire
    assert fc.sent[0][0]["specs"] == [{"GPU": 1.0}, {"CPU": 1.0}]


def test_get_current_placement_group_is_none():
    assert pg_mod.get_current_placement_group() is None


def test_remove_placement_group(monkeypatch):
    fc = use(monkeypatch, FakeClient({"remove_pg": {}}))
    p = PlacementGroup("n1-pg1", [])
    pg_mod.remove_placement_group(p)
    assert fc.sent[0][0] == {"t": "remove_pg", "pg": "n1-pg1"}


# ---- placement_group_table (on ray.util) ------------------------------------


def test_placement_group_table_single(monkeypatch):
    data = {"bundles": [{"node": "n1", "spec": {"GPU": 1}}, {"node": "n2", "spec": {}}]}
    use(monkeypatch, FakeClient({"pg_table": {"data": data}}))
    p = PlacementGroup("n1-pg1", [])
    table = util.placement_group_table(p)
    assert table["bundles_to_node_id"] == {0: "n1", 1: "n2"}
    assert table["state"] == "CREATED"


def test_placement_group_table_all(monkeypatch):
    data = {"pgs": {"n1-pg1": [{"node": "n1", "spec": {}}]}}
    use(monkeypatch, FakeClient({"pg_table": {"data": data}}))
    tables = util.placement_group_table()
    assert "n1-pg1" in tables
    assert tables["n1-pg1"]["bundles_to_node_id"] == {0: "n1"}


def test_get_node_ip_address(monkeypatch):
    monkeypatch.setattr(ray, "_get_ip", lambda: "9.8.7.6")
    assert util.get_node_ip_address() == "9.8.7.6"


# ---- scheduling strategies --------------------------------------------------


def test_pg_scheduling_strategy_fields():
    p = PlacementGroup("n1-pg1", [])
    s = scheduling_strategies.PlacementGroupSchedulingStrategy(p, 2, True)
    assert s.placement_group is p
    assert s.placement_group_bundle_index == 2
    assert s.placement_group_capture_child_tasks is True


def test_node_affinity_strategy_fields():
    s = scheduling_strategies.NodeAffinitySchedulingStrategy("n1", soft=True, extra=1)
    assert s.node_id == "n1" and s.soft is True


# ---- metrics (no-ops) -------------------------------------------------------


def test_metric_record_inc_observe_are_noops():
    m = metrics.Metric("m", "desc")
    assert m.name == "m"
    assert m.record(1.0) is None
    assert m.inc() is None
    assert m.inc(3.0, {"a": "b"}) is None
    assert m.observe(2.0) is None
    assert m.set_default_tags({"x": "y"}) is m  # returns self for chaining


def test_gauge_counter_histogram():
    assert isinstance(metrics.Gauge("g"), metrics.Metric)
    assert isinstance(metrics.Counter("c"), metrics.Metric)
    h = metrics.Histogram("h", "d", boundaries=[1, 2, 3])
    assert h.name == "h" and h.record(1.0) is None


# ---- ray.util.state.list_nodes ----------------------------------------------


def test_list_nodes(monkeypatch):
    use(
        monkeypatch,
        FakeClient(
            {
                "status": {
                    "nodes": [
                        {"node": "n1", "ip": "1.1.1.1", "alive": True},
                        {"node": "n2", "ip": "2.2.2.2", "alive": False},
                    ]
                }
            }
        ),
    )
    nodes = util_state.list_nodes()
    assert [(n.node_id, n.state) for n in nodes] == [("n1", "ALIVE"), ("n2", "DEAD")]
    assert nodes[0].node_ip == "1.1.1.1"


def test_list_nodes_empty(monkeypatch):
    use(monkeypatch, FakeClient({"status": {"nodes": []}}))
    assert util_state.list_nodes() == []


# ---- ray._private.state -----------------------------------------------------


def test_available_resources_per_node(monkeypatch):
    use(monkeypatch, FakeClient({"resources": {"data": {"n1": {"GPU": 2.0, "CPU": 1.0}}}}))
    out = _private_state.available_resources_per_node()
    assert out == {"n1": {"GPU": 2.0, "CPU": 1.0}}


def test_total_resources_per_node(monkeypatch):
    use(
        monkeypatch,
        FakeClient({"status": {"nodes": [{"node": "n1", "ngpu": 4}, {"node": "n2", "ngpu": 0}]}}),
    )
    out = _private_state.total_resources_per_node()
    assert out == {"n1": {"GPU": 4.0, "CPU": 1.0}, "n2": {"GPU": 0.0, "CPU": 1.0}}


def test_private_state_object_delegates(monkeypatch):
    use(monkeypatch, FakeClient({"resources": {"data": {"n1": {"GPU": 1.0}}}}))
    assert _private_state.state._available_resources_per_node() == {"n1": {"GPU": 1.0}}


# ---- fuzz -------------------------------------------------------------------


@given(
    st.lists(
        st.dictionaries(
            st.sampled_from(["GPU", "CPU"]),
            st.integers(min_value=0, max_value=8),
            min_size=1,
            max_size=2,
        ),
        max_size=6,
    )
)
def test_fuzz_placement_group_specs_coerced(bundles):
    """Whatever bundle dicts come in, placement_group sends them as all-float
    specs and never crashes."""
    import unittest.mock as mock

    fc = FakeClient({"create_pg": {"pg": "n1-pg1"}})
    with mock.patch.object(ray, "_need", lambda: fc):
        pg_mod.placement_group(bundles)
    sent_specs = fc.sent[0][0]["specs"]
    assert len(sent_specs) == len(bundles)
    for spec in sent_specs:
        assert all(isinstance(v, float) for v in spec.values())


@given(st.text(min_size=1))
def test_fuzz_pgid_hex_roundtrip(s):
    assert _PGId(s).hex() == s


@given(st.integers(min_value=-1, max_value=10), st.booleans())
def test_fuzz_pg_scheduling_strategy(bundle_idx, capture):
    p = PlacementGroup("n1-pg1", [])
    s = scheduling_strategies.PlacementGroupSchedulingStrategy(p, bundle_idx, capture)
    assert s.placement_group_bundle_index == bundle_idx
