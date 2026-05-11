"""Tests for pythia.api.reachability."""

from __future__ import annotations

import math

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.reachability import ReachabilityEstimator
from pythia.api.state import SimulationState


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------

class _AlwaysReachModel(BaseModel):
    """Sets state.data['done'] = True on the first step (dt=1.0)."""

    @property
    def name(self) -> str:
        return "always_reach"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        state.data["done"] = True
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class _NeverReachModel(BaseModel):
    """Terminates immediately (dt=0) without ever setting the flag."""

    @property
    def name(self) -> str:
        return "never_reach"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return 0.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class _CounterModel(BaseModel):
    """Increments state.data['count'] by 1 each step; dt=1.0."""

    @property
    def name(self) -> str:
        return "counter"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        state.data["count"] = state.data.get("count", 0) + 1
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class _ProbabilisticModel(BaseModel):
    """Flips a fair coin on each step; sets 'heads' on the first heads."""

    @property
    def name(self) -> str:
        return "probabilistic"

    def initialize(self, config: SimulationConfig) -> None:
        import random
        self._rng = random.Random(config.random_seed)

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        if self._rng.random() < 0.5:
            state.data["heads"] = True
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


def _cfg(**kwargs) -> SimulationConfig:
    return SimulationConfig(max_time=10.0, random_seed=42, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_stores_model_and_config(self) -> None:
        model = _AlwaysReachModel()
        config = _cfg()
        est = ReachabilityEstimator(model, config)
        assert est._model is model
        assert est._config is config


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_zero_runs_raises(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        with pytest.raises(ValueError):
            est.estimate(0, 5.0, lambda s: True)

    def test_negative_runs_raises(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        with pytest.raises(ValueError):
            est.estimate(-1, 5.0, lambda s: True)


# ---------------------------------------------------------------------------
# Extreme probabilities
# ---------------------------------------------------------------------------

class TestExtremeProbabilities:
    def test_always_reach_gives_probability_one(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        p, std = est.estimate(50, 5.0, lambda s: s.data.get("done", False))
        assert p == pytest.approx(1.0)
        assert std == pytest.approx(0.0)

    def test_never_reach_gives_probability_zero(self) -> None:
        est = ReachabilityEstimator(_NeverReachModel(), _cfg())
        p, std = est.estimate(50, 5.0, lambda s: s.data.get("done", False))
        assert p == pytest.approx(0.0)
        assert std == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Transient predicate
# ---------------------------------------------------------------------------

class TestTransientPredicate:
    def test_always_false_transient_gives_zero(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        p, _ = est.estimate(
            50, 5.0,
            final_predicate=lambda s: s.data.get("done", False),
            transient_predicate=lambda s: False,
        )
        assert p == pytest.approx(0.0)

    def test_transient_allows_reach(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        p, _ = est.estimate(
            50, 5.0,
            final_predicate=lambda s: s.data.get("done", False),
            transient_predicate=lambda s: True,
        )
        assert p == pytest.approx(1.0)

    def test_transient_violated_after_threshold(self) -> None:
        # Counter model: transient holds while count < 3;
        # final holds when count >= 5 → never reachable
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        p, _ = est.estimate(
            50, 10.0,
            final_predicate=lambda s: s.data.get("count", 0) >= 5,
            transient_predicate=lambda s: s.data.get("count", 0) < 3,
        )
        assert p == pytest.approx(0.0)

    def test_final_reached_before_transient_violated(self) -> None:
        # Final holds at count==3; transient holds while count < 5
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        p, _ = est.estimate(
            50, 10.0,
            final_predicate=lambda s: s.data.get("count", 0) >= 3,
            transient_predicate=lambda s: s.data.get("count", 0) < 5,
        )
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------

class TestDeadline:
    def test_deadline_too_short_gives_zero(self) -> None:
        # CounterModel increments count by 1 each step (dt=1.0);
        # final at count>=5 (t=5); deadline=3 → never reached
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        p, _ = est.estimate(50, 3.0, lambda s: s.data.get("count", 0) >= 5)
        assert p == pytest.approx(0.0)

    def test_deadline_sufficient_gives_one(self) -> None:
        # CounterModel reaches count>=3 at t=3; deadline=5 → always reached
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        p, _ = est.estimate(50, 5.0, lambda s: s.data.get("count", 0) >= 3)
        assert p == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Standard deviation formula
# ---------------------------------------------------------------------------

class TestStdDev:
    def test_std_formula(self) -> None:
        # With a 50/50 model and enough runs the std should be near 0.5/sqrt(n)
        est = ReachabilityEstimator(_ProbabilisticModel(), _cfg())
        p, std = est.estimate(10000, 1.0, lambda s: s.data.get("heads", False))
        expected_std = math.sqrt(p * (1.0 - p) / 10000)
        assert std == pytest.approx(expected_std, rel=1e-6)

    def test_std_is_zero_when_p_is_zero(self) -> None:
        est = ReachabilityEstimator(_NeverReachModel(), _cfg())
        _, std = est.estimate(20, 1.0, lambda s: False)
        assert std == pytest.approx(0.0)

    def test_std_is_zero_when_p_is_one(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        _, std = est.estimate(20, 5.0, lambda s: s.data.get("done", False))
        assert std == pytest.approx(0.0)

    def test_std_maximised_near_half(self) -> None:
        # For n=100, maximum std is 0.5/sqrt(100) = 0.05
        est = ReachabilityEstimator(_ProbabilisticModel(), _cfg())
        p, std = est.estimate(100, 1.0, lambda s: s.data.get("heads", False))
        assert std == pytest.approx(math.sqrt(p * (1.0 - p) / 100), rel=1e-6)
        assert std <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        config = SimulationConfig(max_time=10.0, random_seed=7)
        pred = lambda s: s.data.get("heads", False)
        est = ReachabilityEstimator(_ProbabilisticModel(), config)
        p1, std1 = est.estimate(200, 5.0, pred)
        p2, std2 = est.estimate(200, 5.0, pred)
        assert p1 == p2
        assert std1 == std2

    def test_result_is_float_pair(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        result = est.estimate(10, 5.0, lambda s: s.data.get("done", False))
        p, std = result
        assert isinstance(p, float)
        assert isinstance(std, float)

    def test_probability_in_unit_interval(self) -> None:
        est = ReachabilityEstimator(_ProbabilisticModel(), _cfg())
        p, _ = est.estimate(100, 5.0, lambda s: s.data.get("heads", False))
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# estimate_over_time
# ---------------------------------------------------------------------------

class TestEstimateOverTime:
    # -- error handling --

    def test_zero_runs_raises(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        with pytest.raises(ValueError):
            est.estimate_over_time(0, 3.0, 1.0, lambda s: True)

    def test_negative_dt_raises(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        with pytest.raises(ValueError):
            est.estimate_over_time(10, 3.0, -1.0, lambda s: True)

    def test_zero_dt_raises(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        with pytest.raises(ValueError):
            est.estimate_over_time(10, 3.0, 0.0, lambda s: True)

    # -- length of result --

    def test_k_equals_ceil_deadline_over_dt(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        import math
        for deadline, dt, expected_k in [
            (3.0, 1.0, 3),
            (3.0, 0.5, 6),
            (1.0, 0.3, 4),   # ceil(1.0/0.3) = ceil(3.333) = 4
            (2.0, 2.0, 1),
        ]:
            result = est.estimate_over_time(10, deadline, dt, lambda s: True)
            assert len(result) == math.ceil(deadline / dt), \
                f"deadline={deadline} dt={dt} expected {expected_k} got {len(result)}"

    def test_last_bucket_covers_deadline(self) -> None:
        import math
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        deadline, dt = 3.0, 1.0
        k = math.ceil(deadline / dt)
        assert k * dt >= deadline

    # -- monotonicity --

    def test_probabilities_non_decreasing(self) -> None:
        # CounterModel: goal reached at count>=2 (t=2); deadline=5, dt=1
        # P(reached by t=1)=0, P(reached by t=2)=1, P(reached by t=3..5)=1
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        results = est.estimate_over_time(
            50, 5.0, 1.0, lambda s: s.data.get("count", 0) >= 2
        )
        probs = [p for p, _ in results]
        assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1))

    # -- correctness --

    def test_never_reach_all_zero(self) -> None:
        est = ReachabilityEstimator(_NeverReachModel(), _cfg())
        results = est.estimate_over_time(50, 3.0, 1.0, lambda s: False)
        for p, std in results:
            assert p == pytest.approx(0.0)
            assert std == pytest.approx(0.0)

    def test_always_reach_first_step_all_one(self) -> None:
        # AlwaysReachModel sets done=True at t=1; deadline=3, dt=1
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        results = est.estimate_over_time(
            50, 3.0, 1.0, lambda s: s.data.get("done", False)
        )
        for p, std in results:
            assert p == pytest.approx(1.0)
            assert std == pytest.approx(0.0)

    def test_zero_before_goal_one_after(self) -> None:
        # CounterModel: goal at count>=3 (t=3); dt=1, deadline=5
        # P(by t=1)=0, P(by t=2)=0, P(by t=3..5)=1
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        results = est.estimate_over_time(
            50, 5.0, 1.0, lambda s: s.data.get("count", 0) >= 3
        )
        assert results[0][0] == pytest.approx(0.0)  # t=1
        assert results[1][0] == pytest.approx(0.0)  # t=2
        assert results[2][0] == pytest.approx(1.0)  # t=3

    def test_std_formula_each_bucket(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        results = est.estimate_over_time(
            50, 3.0, 1.0, lambda s: s.data.get("done", False)
        )
        for p, std in results:
            assert std == pytest.approx(math.sqrt(p * (1.0 - p) / 50), rel=1e-6)

    def test_last_bucket_matches_estimate(self) -> None:
        # The last bucket's probability should match estimate() at the deadline.
        config = SimulationConfig(max_time=5.0, random_seed=3)
        est = ReachabilityEstimator(_CounterModel(), config)
        pred = lambda s: s.data.get("count", 0) >= 3
        p_single, _ = est.estimate(200, 5.0, pred)
        results = est.estimate_over_time(200, 5.0, 1.0, pred)
        assert results[-1][0] == pytest.approx(p_single, abs=1e-9)

    # -- transient predicate --

    def test_transient_predicate_limits_reach(self) -> None:
        # Transient holds only while count < 2; goal at count>=3 → never reachable
        est = ReachabilityEstimator(_CounterModel(), _cfg())
        results = est.estimate_over_time(
            50, 5.0, 1.0,
            final_predicate=lambda s: s.data.get("count", 0) >= 3,
            transient_predicate=lambda s: s.data.get("count", 0) < 2,
        )
        for p, _ in results:
            assert p == pytest.approx(0.0)

    # -- reproducibility --

    def test_same_seed_same_result(self) -> None:
        config = SimulationConfig(max_time=5.0, random_seed=7)
        est = ReachabilityEstimator(_CounterModel(), config)
        pred = lambda s: s.data.get("count", 0) >= 2
        r1 = est.estimate_over_time(100, 5.0, 1.0, pred)
        r2 = est.estimate_over_time(100, 5.0, 1.0, pred)
        assert r1 == r2

    def test_returns_tuple_of_pairs(self) -> None:
        est = ReachabilityEstimator(_AlwaysReachModel(), _cfg())
        results = est.estimate_over_time(10, 2.0, 1.0, lambda s: True)
        assert isinstance(results, tuple)
        for item in results:
            p, std = item
            assert isinstance(p, float)
            assert isinstance(std, float)
