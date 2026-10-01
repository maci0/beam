"""Unit + fuzz tests for the `ray` shim's translation logic. A fake daemon
client records requests and returns canned responses, so these run with no
daemon, no sockets, no GPUs."""

import os
import sys
import time as _time

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ray
from ray import _proto
from ray.exceptions import GetTimeoutError
from ray.runtime_env import RuntimeEnv

ObjectRef = ray.ObjectRef  # noqa: E405  (bound at import time for the seam tests)


class FakeClient:
    """Mirrors DaemonClient.request: an "err" key in the canned response raises
    RuntimeError, exactly like the real client does for a daemon error frame."""

    def __init__(self, responses=None, body=b""):
        self.responses = responses or {}
        self.body = body
        self.sent = []

    def request(self, header, payload=b""):
        self.sent.append((header, payload))
        t = header["t"]
        canned = self.responses.get(t, {})
        if canned.get("err"):
            raise RuntimeError(canned["err"])  # same contract as DaemonClient
        resp = {"t": t + "_ok", "resp": True, **canned}
        return resp, canned.get("_body", self.body)


def use(monkeypatch, fc):
    monkeypatch.setattr(ray, "_need", lambda: fc)
    return fc


# ---- ObjectRef ----
def test_objectref_has_value_shortcuts_get(monkeypatch):
    fc = use(monkeypatch, FakeClient())
    ref = ray.ObjectRef("x", value=99, has_value=True)
    assert ray.get(ref) == 99
    assert fc.sent == []  # no daemon round-trip for a local value


def test_objectref_eq_hash():
    a, b = ray.ObjectRef("n1-o1"), ray.ObjectRef("n1-o1")
    assert a == b and hash(a) == hash(b)
    assert a != ray.ObjectRef("n1-o2")


@given(st.text(min_size=1))
def test_objectref_id_roundtrip(s):
    assert ray.ObjectRef(s).id == s  # never crashes, id preserved


# ---- put / get ----
def test_put_pickles_and_returns_ref(monkeypatch):
    fc = use(monkeypatch, FakeClient({"put": {"obj": "n1-o1"}}))
    ref = ray.put({"a": 1})
    assert ref.id == "n1-o1"
    assert _proto.loads(fc.sent[0][1]) == {"a": 1}


def test_get_unpickles_body(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"_body": _proto.dumps([1, 2, 3])}}))
    assert ray.get(ray.ObjectRef("n1-o5")) == [1, 2, 3]


def test_get_timeout_maps_to_GetTimeoutError(monkeypatch):
    # drive the real path: daemon returns an "err" with the exact on_get string,
    # the client raises RuntimeError, the shim re-raises GetTimeoutError
    use(
        monkeypatch,
        FakeClient({"get": {"err": "GetTimeoutError: object n1-o1 not ready in 0.01s"}}),
    )
    with pytest.raises(GetTimeoutError):
        ray.get(ray.ObjectRef("n1-o1"), timeout=0.01)


def test_get_other_error_reraises(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"err": "unknown object n1-o1"}}))
    with pytest.raises(RuntimeError):
        ray.get(ray.ObjectRef("n1-o1"))


def test_err_response_propagates_from_put(monkeypatch):
    use(monkeypatch, FakeClient({"put": {"err": "daemon exploded"}}))
    with pytest.raises(RuntimeError, match="daemon exploded"):
        ray.put(123)


def test_get_list_preserves_order(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"_body": _proto.dumps(7)}}))
    assert ray.get([ray.ObjectRef("a"), ray.ObjectRef("b")]) == [7, 7]


# ---- wait ----
def test_wait_num_returns_capped(monkeypatch):
    use(monkeypatch, FakeClient({"stat": {"ready": True}}))
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=99, timeout=1)
    assert len(ready) == 1 and not_ready == []  # cap at len(refs), no hang


def test_wait_timeout_returns_partial(monkeypatch):
    use(monkeypatch, FakeClient({"stat": {"ready": False}}))
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=0)
    assert ready == [] and len(not_ready) == 1


def test_wait_num_returns_zero_returns_immediately(monkeypatch):
    """num_returns=0 asks for nothing, so wait must not block: the caller wants
    an empty ready set back now, not a poll loop that never satisfies it."""
    use(monkeypatch, FakeClient({"stat": {"ready": False}}))
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=0, timeout=None)
    # zero wanted is met by the first pass, so nothing lands in ready
    assert ready == [] and not_ready == [ray.ObjectRef("a")]


def test_wait_empty_refs(monkeypatch):
    use(monkeypatch, FakeClient())
    assert ray.wait([], num_returns=1, timeout=1) == ([], [])  # no hang on empty input


def test_get_empty_body_yields_none(monkeypatch):
    """A get_ok with no payload is a None value, not a decode error."""
    use(monkeypatch, FakeClient({"get": {}}))
    assert ray.get(ray.ObjectRef("n1-o1")) is None


def test_get_empty_list(monkeypatch):
    fc = use(monkeypatch, FakeClient())
    assert ray.get([]) == []
    assert fc.sent == []  # nothing to fetch, no daemon round-trip


def test_get_timeout_ends_at_the_deadline_when_the_daemon_never_answers(monkeypatch):
    """A get issued with a timeout must end at that timeout even if the daemon
    is unreachable: without a bound on the socket read, the caller hangs."""

    class Unreachable:
        def request(self, header, payload=b"", timeout=None):
            if timeout is None:
                raise AssertionError("get sent no budget; the read cannot be bounded")
            raise TimeoutError("socket timed out")

    monkeypatch.setattr(ray, "_need", lambda: Unreachable())
    started = _time.monotonic()
    with pytest.raises(GetTimeoutError, match="n1-o1"):
        ray.get(ray.ObjectRef("n1-o1"), timeout=0.05)
    assert _time.monotonic() - started < 1.0


def test_get_passes_the_remaining_budget_to_the_daemon(monkeypatch):
    seen = []

    class Recorder:
        def request(self, header, payload=b"", timeout=None):
            seen.append((header.get("timeout"), timeout))
            return {"t": "get_ok"}, _proto.dumps(1)

    monkeypatch.setattr(ray, "_need", lambda: Recorder())
    ray.get(ray.ObjectRef("a"), timeout=1.0)
    assert seen and seen[0][0] is not None
    # the daemon's budget and the socket budget are the same remaining time
    assert seen[0][1] == pytest.approx(seen[0][0])


def test_get_without_a_timeout_sends_no_budget(monkeypatch):
    seen = []

    class Recorder:
        def request(self, header, payload=b"", timeout=None):
            seen.append((header.get("timeout"), timeout))
            return {"t": "get_ok"}, _proto.dumps(1)

    monkeypatch.setattr(ray, "_need", lambda: Recorder())
    assert ray.get(ray.ObjectRef("a")) == 1
    assert seen == [(None, None)]


def test_budgeted_falls_back_for_a_client_without_a_budget(monkeypatch):
    """Clients whose request() takes no timeout kwarg keep working."""

    class OldClient:
        def __init__(self):
            self.n = 0

        def request(self, header, payload=b""):
            self.n += 1
            return {"t": "get_ok"}, _proto.dumps(self.n)

    monkeypatch.setattr(ray, "_need", lambda: OldClient())
    assert ray.get(ray.ObjectRef("a"), timeout=1.0) == 1


# ---- deadlines run on the monotonic clock ----
class SteppingClock:
    """Stands in for the `time` module with a wall clock that jumps forward or
    backward mid-wait, exactly as an NTP step, a manual clock change, a
    leap-second smear, or a host suspend does. The monotonic clock is real, so
    any deadline built on `time.time()` is wrong here and one built on
    `time.monotonic()` is not."""

    def __init__(self, step_at_call, delta):
        self.step_at_call = step_at_call
        self.delta = delta
        self.calls = 0

    def time(self):
        return _time.time() + (self.delta if self.calls >= self.step_at_call else 0.0)

    def monotonic(self):
        return _time.monotonic()

    def sleep(self, seconds):
        _time.sleep(seconds)


@pytest.fixture
def stepping_clock(monkeypatch):
    def install(step_at_call, delta):
        clock = SteppingClock(step_at_call, delta)
        monkeypatch.setattr(ray, "time", clock)
        return clock

    return install


def test_wait_survives_forward_clock_step(stepping_clock, monkeypatch):
    """A +1h step mid-wait must not retire the caller's 0.2s timeout early."""
    clock = stepping_clock(1, +3600.0)  # step after the first stat poll

    class NeverReady:
        def request(self, header, payload=b""):
            clock.calls += 1
            return {"t": "stat_ok", "ready": False}, b""

    monkeypatch.setattr(ray, "_need", lambda: NeverReady())
    started = _time.monotonic()
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=0.2)
    elapsed = _time.monotonic() - started
    assert ready == [] and not_ready == [ray.ObjectRef("a")]
    assert elapsed >= 0.15, "wait gave up after %.3fs: the clock step ended the wait" % elapsed


def test_get_per_ref_timeout_ignores_backward_clock_step(stepping_clock, monkeypatch):
    """A -1h step must not inflate the per-ref timeout past the caller's budget."""
    clock = stepping_clock(1, -3600.0)  # step between the first and second get

    seen = []

    class Recorder:
        def request(self, header, payload=b""):
            clock.calls += 1
            seen.append(header.get("timeout"))
            return {"t": "get_ok"}, b""

    monkeypatch.setattr(ray, "_need", lambda: Recorder())
    ray.get([ray.ObjectRef("a"), ray.ObjectRef("b")], timeout=1.0)
    assert len(seen) == 2 and all(t is not None and t <= 1.0 for t in seen), seen


def test_remaining_never_negative():
    assert ray._remaining(None) is None  # no deadline: no remaining budget
    assert ray._deadline(None) is None
    assert ray._remaining(ray._deadline(5.0)) == pytest.approx(5.0, abs=0.5)
    assert ray._remaining(_time.monotonic() - 1.0) == 0.0  # expired, clamped


def test_wait_bounds_a_slow_stat_with_its_timeout(monkeypatch):
    """The wait budget must reach the stat round-trip, not just the sleep: a
    daemon that answers late (owner node dropped) must not push wait far past
    the timeout the caller was promised."""
    seen = []

    class SlowDaemon:
        def request(self, header, payload=b"", timeout=None):
            seen.append(timeout)
            raise TimeoutError("stat timed out")

    monkeypatch.setattr(ray, "_need", lambda: SlowDaemon())
    started = _time.monotonic()
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=0.05)
    elapsed = _time.monotonic() - started
    assert ready == [] and len(not_ready) == 1
    assert seen and all(t is not None and t <= 0.05 for t in seen), seen
    assert elapsed < 0.2, "wait ran %.3fs past a 0.05s budget" % elapsed


def test_wait_sends_no_budget_without_a_timeout(monkeypatch):
    seen = []

    class NeverReady:
        def __init__(self):
            self.n = 0

        def request(self, header, payload=b"", timeout=None):
            seen.append(timeout)
            self.n += 1
            if self.n >= 2:
                return {"t": "stat_ok", "ready": True}, b""
            return {"t": "stat_ok", "ready": False}, b""

    monkeypatch.setattr(ray, "_need", lambda: NeverReady())
    monkeypatch.setattr(ray.time, "sleep", lambda s: None)
    ready, _ = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=None)
    assert len(ready) == 1 and seen and all(t is None for t in seen), seen


def test_wait_falls_back_for_a_client_without_a_budget(monkeypatch):
    """A client whose request() takes no timeout kwarg is still polled."""

    class OldClient:
        def request(self, header, payload=b""):
            return {"t": "stat_ok", "ready": True}, b""

    monkeypatch.setattr(ray, "_need", lambda: OldClient())
    ready, _ = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=1)
    assert len(ready) == 1


def test_wait_does_not_oversleep_the_deadline(monkeypatch):
    """The poll sleep is clipped to the remaining budget, so a wait on a fixed
    cadence (vLLM's liveness thread) does not drift past its timeout."""
    slept = []

    class NeverReady:
        def request(self, header, payload=b"", timeout=None):
            return {"t": "stat_ok", "ready": False}, b""

    monkeypatch.setattr(ray, "_need", lambda: NeverReady())
    monkeypatch.setattr(ray.time, "sleep", lambda s: slept.append(s))
    ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=0.02)
    assert slept, "wait never polled"
    assert max(slept) <= 0.02, slept


def test_wait_returns_when_the_clock_steps_past_the_deadline(monkeypatch):
    """A forward clock step that lands between the deadline check and the sleep
    must end the wait, not clip the sleep to a negative value."""
    clock = _time.monotonic()

    class SteppingMonotonic:
        """Steps forward on the 4th reading: _deadline, the per-stat budget,
        the deadline check, then the pre-sleep budget."""

        def __init__(self):
            self.calls = 0

        def time(self):
            return _time.time()

        def monotonic(self):
            self.calls += 1
            return _time.monotonic() + (3600.0 if self.calls >= 4 else 0.0)

        def sleep(self, seconds):
            assert seconds >= 0, "wait asked to sleep %rs" % seconds
            _time.sleep(0)

    class NeverReady:
        def request(self, header, payload=b"", timeout=None):
            return {"t": "stat_ok", "ready": False}, b""

    monkeypatch.setattr(ray, "_need", lambda: NeverReady())
    monkeypatch.setattr(ray, "time", SteppingMonotonic())
    started = _time.monotonic()
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=10.0)
    assert _time.monotonic() - started < 1.0
    assert ready == [] and len(not_ready) == 1
    assert clock  # keep the name used


# ---- virtual-clock seams (BEAM_CLOCK / BEAM_SLEEP) ----
class VirtualClock:
    """The simulator's clock: a float the test advances, plus a sleep hook that
    jumps it forward instead of blocking. A shim run driven through this never
    waits real seconds, and its deadline arithmetic is fully reproducible."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def now_seconds(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds  # time passes instantly

    def install(self, monkeypatch):
        import types

        mod = types.ModuleType("simclock")
        mod.now_seconds = self.now_seconds
        mod.sleep = self.sleep
        monkeypatch.setitem(sys.modules, "simclock", mod)
        monkeypatch.setattr(ray, "_CLOCK_HOOK", "simclock:now_seconds")
        monkeypatch.setenv("BEAM_SLEEP", "simclock:sleep")
        return self


def test_deadlines_run_on_the_virtual_clock(monkeypatch):
    clock = VirtualClock().install(monkeypatch)
    deadline = ray._deadline(2.0)
    assert deadline == 1002.0
    assert ray._remaining(deadline) == 2.0
    clock.now += 0.5
    assert ray._remaining(deadline) == 1.5
    clock.now += 10.0
    assert ray._remaining(deadline) == 0.0  # clamped, never negative


def test_wait_polls_on_the_virtual_clock(monkeypatch):
    clock = VirtualClock().install(monkeypatch)
    polls = {"n": 0}

    class ReadyAtPoll3:
        def request(self, header, payload=b""):
            polls["n"] += 1
            return {"t": "stat_ok", "ready": polls["n"] >= 3}, b""

    monkeypatch.setattr(ray, "_need", lambda: ReadyAtPoll3())
    started = _time.monotonic()
    ref = ObjectRef("a")
    ready, not_ready = ray.wait([ref], num_returns=1, timeout=5.0)
    assert ready == [ref] and not_ready == []
    assert clock.slept == [ray._WAIT_POLL_INTERVAL] * 2  # two polls, no real time
    assert _time.monotonic() - started < 1.0


def test_wait_timeout_comes_from_the_virtual_clock(monkeypatch):
    clock = VirtualClock().install(monkeypatch)

    class NeverReady:
        def request(self, header, payload=b""):
            return {"t": "stat_ok", "ready": False}, b""

    monkeypatch.setattr(ray, "_need", lambda: NeverReady())
    ref = ObjectRef("a")
    ready, not_ready = ray.wait([ref], num_returns=1, timeout=0.01)
    assert ready == [] and not_ready == [ref]
    assert clock.now >= 1000.0 + 0.01  # the virtual clock, not the wall clock


def test_shim_sleep_without_hook_sleeps(monkeypatch):
    monkeypatch.delenv("BEAM_SLEEP", raising=False)
    started = _time.monotonic()
    ray._sleep(0.01)
    assert _time.monotonic() - started >= 0.005


def test_shim_sleep_rejects_an_async_hook(monkeypatch):
    import types

    mod = types.ModuleType("simclock")

    async def bad_sleep(seconds):
        return None

    mod.sleep = bad_sleep
    monkeypatch.setitem(sys.modules, "simclock", mod)
    monkeypatch.setenv("BEAM_SLEEP", "simclock:sleep")
    with pytest.raises(RuntimeError, match="synchronous"):
        ray._sleep(1.0)


# ---- remote / options ----
def test_remote_sends_float_num_gpus(monkeypatch):
    fc = use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))

    class W:
        def __init__(self, x=0):
            self.x = x

    h = ray.remote(num_gpus=0.5)(W).remote()
    assert h is not None
    assert fc.sent[0][0]["ngpu"] == 0.5  # fractional preserved, not truncated to 0


def test_options_merges(monkeypatch):
    use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))

    class W:
        pass

    rc = ray.remote(W).options(num_gpus=1)
    assert rc._options.get("num_gpus") == 1


# ---- RuntimeEnv ----
def test_runtime_env_kwargs_and_positional():
    assert RuntimeEnv(env_vars={"A": "1"})["env_vars"] == {"A": "1"}
    assert RuntimeEnv({"working_dir": "/x"})["working_dir"] == "/x"  # positional dict


@given(st.dictionaries(st.text(), st.text(), max_size=6))
def test_runtime_env_fuzz(d):
    assert dict(RuntimeEnv(d)) == d
