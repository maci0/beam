"""Edge-case suite for the beam ray shim. CPU-only (BEAM_NUM_GPUS fakes GPUs),
no torch. Exercises behaviors the happy-path demo doesn't:

  A error propagation     actor method raises -> ray.get raises with the message
  B put / get             round-trip an arbitrary object
  C wait                  num_returns / timeout, partial readiness
  D parallelism           calls to different actors run concurrently
  E large payload         multi-MB return value survives the frame protocol
  F pg exhaustion         asking for more GPUs than exist fails cleanly
  G leak fix              a second placement group reuses GPUs the first freed
"""

import time

import ray
from ray.util import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class W:
    def __init__(self, rank):
        self.rank = rank

    def boom(self):
        raise ValueError("kaboom rank %d" % self.rank)

    def slow(self, secs):
        time.sleep(secs)
        return self.rank

    def big(self, n):
        return b"x" * n

    def echo(self, x):
        return x

    def none(self):
        return None

    def gids(self):
        import ray as _r

        return _r.get_gpu_ids()


def mk(pg, i):
    strat = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)
    return ray.remote(num_gpus=1, scheduling_strategy=strat)(W).remote(i)


def main():
    ray.init()
    pg = placement_group([{"GPU": 1}] * 2)
    ray.get(pg.ready())
    w0, w1 = mk(pg, 0), mk(pg, 1)

    # A: error propagation
    try:
        ray.get(w0.boom.remote())
        raise AssertionError("expected an exception from boom()")
    except Exception as e:
        assert "kaboom rank 0" in str(e), str(e)
    print("A error-propagation OK")

    # B: put / get round-trip
    ref = ray.put({"a": [1, 2, 3], "b": ("x", 9)})
    assert ray.get(ref) == {"a": [1, 2, 3], "b": ("x", 9)}
    print("B put/get OK")

    # C: wait — fast ready, slow not, with timeout=0 (non-blocking poll)
    slow = w0.slow.remote(0.5)
    fast = w1.echo.remote(7)
    ray.get(fast)  # make sure fast has completed
    ready, not_ready = ray.wait([slow, fast], num_returns=1, timeout=0)
    assert len(ready) + len(not_ready) == 2, (ready, not_ready)
    assert fast in ready and slow in not_ready, (ready, not_ready)
    assert ray.get(slow) == 0
    print("C wait OK")

    # D: two actors run their calls concurrently, not serialized
    t = time.time()
    assert ray.get([w0.slow.remote(1.0), w1.slow.remote(1.0)]) == [0, 1]
    dt = time.time() - t
    assert dt < 1.8, "actors did not run in parallel: %.2fs" % dt
    print("D parallelism %.2fs OK" % dt)

    # E: multi-MB payload through the frame protocol
    data = ray.get(w0.big.remote(2_000_000))
    assert len(data) == 2_000_000
    print("E large-payload OK")

    # F: placement group bigger than the cluster fails cleanly
    try:
        placement_group([{"GPU": 1}] * 99)
        raise AssertionError("expected pg exhaustion error")
    except Exception as e:
        assert "more GPUs" in str(e), str(e)
    print("F pg-exhaustion OK")

    # H: CPU actor (no num_gpus, no placement group) lands on the head with no GPU
    cpu = ray.remote(W).remote(99)
    assert ray.get(cpu.echo.remote(5)) == 5
    assert ray.get(cpu.gids.remote()) == []  # CPU actor -> no GPU ids
    assert ray.get(w0.gids.remote()) == [0]  # GPU actor -> its bundle's GPU
    print("H cpu-vs-gpu-actor OK")

    # I: ray.get preserves submit order even when later refs finish first
    a = w0.slow.remote(0.4)
    b = w1.echo.remote(123)
    assert ray.get([a, b]) == [0, 123]
    print("I result-order OK")

    # J: wait(num_returns=2) blocks until both are ready
    pend = [w0.slow.remote(0.3), w1.slow.remote(0.3)]
    ready, not_ready = ray.wait(pend, num_returns=2, timeout=5)
    assert len(ready) == 2 and not not_ready, (ready, not_ready)
    print("J wait-num-returns OK")

    # K: None return survives the pickle round-trip
    assert ray.get(w0.none.remote()) is None
    print("K none-return OK")

    # L: 50 serialized calls on one actor, in order, no deadlock
    assert ray.get([w0.echo.remote(i) for i in range(50)]) == list(range(50))
    print("L throughput OK")

    # M: calling a killed actor errors cleanly
    victim = ray.remote(W).remote(7)
    assert ray.get(victim.echo.remote(1)) == 1
    ray.kill(victim)
    try:
        ray.get(victim.echo.remote(1))
        raise AssertionError("expected error calling a killed actor")
    except Exception as e:
        assert "unknown actor" in str(e), str(e)
    print("M kill-then-call OK")

    # G: free this run's actors+pg, then a fresh pg must reuse the GPUs
    for w in (w0, w1):
        ray.kill(w)
    from ray.util import remove_placement_group

    remove_placement_group(pg)
    pg2 = placement_group([{"GPU": 1}] * 2)  # would fail if GPUs leaked
    ray.get(pg2.ready())
    print("G leak-fix OK")

    ray.shutdown()
    print("ALL EDGE OK")


if __name__ == "__main__":
    main()
