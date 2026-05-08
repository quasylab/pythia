"""Tests for pythia.api.collector and pythia.api.measure."""

from __future__ import annotations

import pytest

from pythia.api.collector import DataCollector
from pythia.api.config import SimulationConfig
from pythia.api.measure import Measure
from pythia.api.runner import SimulationRunner
from pythia.api.state import SimulationState


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

class TestMeasure:
    def test_name_stored(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=1.0)
        assert m.name == "x"

    def test_frequency_stored(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=0.5)
        assert m.frequency == 0.5

    def test_function_called(self) -> None:
        state = SimulationState(data={"v": 7.0})
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        assert m.function(state) == 7.0

    def test_zero_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency"):
            Measure("x", lambda s: 0.0, frequency=0.0)

    def test_negative_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="frequency"):
            Measure("x", lambda s: 0.0, frequency=-1.0)

    def test_frozen(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=1.0)
        with pytest.raises(AttributeError):
            m.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DataCollector construction
# ---------------------------------------------------------------------------

class TestDataCollectorConstruction:
    def test_empty_measures(self) -> None:
        collector = DataCollector([])
        assert collector.measures == ()

    def test_measures_stored_in_order(self) -> None:
        m1 = Measure("a", lambda s: 1.0, frequency=1.0)
        m2 = Measure("b", lambda s: 2.0, frequency=1.0)
        collector = DataCollector([m1, m2])
        assert collector.measures == (m1, m2)

    def test_duplicate_name_raises(self) -> None:
        m1 = Measure("x", lambda s: 1.0, frequency=1.0)
        m2 = Measure("x", lambda s: 2.0, frequency=0.5)
        with pytest.raises(ValueError, match="duplicate"):
            DataCollector([m1, m2])

    def test_records_initially_empty(self) -> None:
        m = Measure("x", lambda s: 1.0, frequency=1.0)
        collector = DataCollector([m])
        assert collector.records["x"] == []


# ---------------------------------------------------------------------------
# Zero-order hold semantics (the core collection logic)
# ---------------------------------------------------------------------------

class TestZeroOrderHoldSemantics:
    """Verify the example from the requirements doc exactly."""

    def test_example_from_requirements(self) -> None:
        """
        Simulation: s1@t=0, s2@t=1.5, s3@t=2.3 (done), max_time=3.0.
        Measure frequency=0.5.
        Expected records:
          (0.0, m(s1)), (0.5, m(s1)), (1.0, m(s1)),
          (1.5, m(s2)), (2.0, m(s2)),
          (2.5, m(s3)), (3.0, m(s3))
        """
        # Use elapsed_time itself as the measured value so we can distinguish
        # which state was used at each collection point.
        m = Measure("v", lambda s: s.data["tag"], frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=3.0)

        s1 = SimulationState(elapsed_time=0.0, data={"tag": 1.0})
        s2 = SimulationState(elapsed_time=1.5, data={"tag": 2.0})
        s3 = SimulationState(elapsed_time=2.3, data={"tag": 3.0})

        collector.collect(s1)  # t=0
        collector.collect(s2)  # t=1.5
        collector.collect(s3)  # t=2.3
        collector.finalize(s3)  # fill up to max_time=3.0

        records = collector.records["v"]
        expected_times  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        expected_values = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

        assert [t for t, _ in records] == pytest.approx(expected_times)
        assert [v for _, v in records] == pytest.approx(expected_values)

    def test_initial_collect_at_t0_records_one_point(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 7.0}))
        records = collector.records["x"]
        assert len(records) == 1
        assert records[0] == (0.0, 7.0)

    def test_step_at_non_grid_time_uses_previous_state_retroactively(self) -> None:
        # freq=1.0; steps at t=0 and t=2.7.  Scheduled point t=1.0 and t=2.0
        # should use the state from t=0.
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 10.0}))
        collector.collect(SimulationState(elapsed_time=2.7, data={"v": 20.0}))

        records = collector.records["x"]
        # t=0.0: 10 (current state)
        # t=1.0: 10 (retroactive, from t=0 state)
        # t=2.0: 10 (retroactive, from t=0 state)
        # t=2.7 is NOT a grid point → nothing recorded at 2.7
        assert [t for t, _ in records] == pytest.approx([0.0, 1.0, 2.0])
        assert [v for _, v in records] == pytest.approx([10.0, 10.0, 10.0])

    def test_step_at_grid_time_uses_new_state(self) -> None:
        # freq=0.5; step at t=0 then at t=1.5 (which is 3*0.5, a grid point).
        m = Measure("x", lambda s: s.data["v"], frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        collector.collect(SimulationState(elapsed_time=1.5, data={"v": 2.0}))

        records = collector.records["x"]
        # t=0.0: 1.0; t=0.5: 1.0; t=1.0: 1.0 — retroactive from t=0 state
        # t=1.5: 2.0 — on grid, uses new state
        times  = [t for t, _ in records]
        values = [v for _, v in records]
        assert times  == pytest.approx([0.0, 0.5, 1.0, 1.5])
        assert values == pytest.approx([1.0, 1.0, 1.0, 2.0])

    def test_finalize_fills_up_to_max_time(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        # Simulation ends early at t=2.3.
        collector.collect(SimulationState(elapsed_time=2.3, data={"v": 2.0}))
        collector.finalize(SimulationState(elapsed_time=2.3, data={"v": 2.0}))

        records = collector.records["x"]
        times = [t for t, _ in records]
        # Expected: 0, 1, 2 (from collect calls) then 3, 4, 5 (from finalize)
        assert times == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_finalize_includes_max_time_when_on_grid(self) -> None:
        # max_time=3.0, freq=0.5 → last point is 3.0
        m = Measure("x", lambda s: 0.0, frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=3.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        times = [t for t, _ in collector.records["x"]]
        assert times[-1] == pytest.approx(3.0)

    def test_finalize_stops_before_max_time_when_off_grid(self) -> None:
        # max_time=1.0, freq=0.3 → grid: 0, 0.3, 0.6, 0.9. 1.2 > 1.0 → stop at 0.9
        m = Measure("x", lambda s: 0.0, frequency=0.3)
        collector = DataCollector([m])
        collector.reset(max_time=1.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        times = [t for t, _ in collector.records["x"]]
        assert all(t <= 1.0 for t in times)
        assert max(times) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# DataCollector.reset
# ---------------------------------------------------------------------------

class TestDataCollectorReset:
    def test_reset_clears_records(self) -> None:
        m = Measure("x", lambda s: 1.0, frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.reset(max_time=5.0)
        assert collector.records["x"] == []

    def test_reset_restarts_from_t0(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        collector.collect(SimulationState(elapsed_time=3.5, data={"v": 2.0}))
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 9.0}))
        records = collector.records["x"]
        assert len(records) == 1
        assert records[0] == (0.0, 9.0)

    def test_records_are_copies(self) -> None:
        m = Measure("x", lambda s: 1.0, frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=2.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        snap = collector.records["x"]
        snap.append((99.0, 99.0))  # type: ignore[index]
        assert len(collector.records["x"]) == 1  # internal not affected

    def test_multiple_measures_independent(self) -> None:
        m1 = Measure("fast", lambda s: 1.0, frequency=0.5)
        m2 = Measure("slow", lambda s: 2.0, frequency=2.0)
        collector = DataCollector([m1, m2])
        collector.reset(max_time=4.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        # fast: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 → 9 points
        # slow: 0, 2.0, 4.0 → 3 points
        assert len(collector.records["fast"]) == 9
        assert len(collector.records["slow"]) == 3


# ---------------------------------------------------------------------------
# Integration with SimulationRunner
# ---------------------------------------------------------------------------

class TestDataCollectorWithRunner:
    def test_runner_resets_collector_at_start(
        self, minimal_config: SimulationConfig
    ) -> None:
        m = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=99.0)
        collector._records["t"].append((999.0, 999.0))  # stale record

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()
        assert all(t <= minimal_config.max_time for t, _ in collector.records["t"])

    def test_runner_collects_at_t0_and_after_each_step(
        self, minimal_config: SimulationConfig
    ) -> None:
        # NoOpModel returns dt=1.0; max_time=10.0, frequency=1.0.
        # Expected times: 0, 1, 2, ..., 10 → 11 points.
        m = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        collector = DataCollector([m])

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()

        times = [t for t, _ in collector.records["t"]]
        assert times == pytest.approx(list(range(0, 11)))  # 0 through 10 inclusive

    def test_runner_finalizes_after_loop(
        self, minimal_config: SimulationConfig
    ) -> None:
        # max_time=10.0, frequency=0.5 → 21 points (0, 0.5, ..., 10.0)
        m = Measure("t", lambda s: s.elapsed_time, frequency=0.5)
        collector = DataCollector([m])

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()

        times = [t for t, _ in collector.records["t"]]
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(10.0)
        assert len(times) == 21

    def test_independent_runs_do_not_accumulate(
        self, minimal_config: SimulationConfig
    ) -> None:
        m = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        collector = DataCollector([m])

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()
        first_count = len(collector.records["t"])
        runner.run()
        second_count = len(collector.records["t"])
        assert first_count == second_count

    def test_collector_none_does_not_raise(
        self, minimal_config: SimulationConfig
    ) -> None:
        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config)
        runner.run()

    def test_end_to_end_with_population_model(self) -> None:
        """Collect wolf counts every 0.25 time units over a stochastic run."""
        from pythia.api.config import SimulationConfig
        from pythia.api.runner import SimulationRunner
        from pythia.models.pm.counting_state import CountingState
        from pythia.models.pm.population import Population
        from pythia.models.pm.population_model import PopulationModel
        from pythia.models.pm.rule import Rule

        pop = Population()
        wolf = pop.register("wolf")

        def wolf_count(s: SimulationState) -> float:
            cs = s.data.get("pm.counting_state")
            if cs is None:
                return 0.0
            return float(cs.counting_vector.get(wolf.index, 0))

        model = PopulationModel(
            population=pop,
            initial_state={wolf: 20},
            rules=[
                Rule(
                    pre=CountingState({wolf: 1}),
                    post=CountingState({}),
                    rate_function=lambda s: float(
                        s.counting_vector.get(wolf.index, 0)
                    ),
                )
            ],
        )

        collector = DataCollector([
            Measure(name="wolves", function=wolf_count, frequency=0.25)
        ])
        config = SimulationConfig(max_time=5.0, random_seed=42)
        runner = SimulationRunner(model, config, data_collector=collector)
        runner.run()

        records = collector.records["wolves"]
        times = [t for t, _ in records]
        values = [v for _, v in records]

        # First point is t=0; last is <= max_time.
        assert times[0] == pytest.approx(0.0)
        assert times[-1] <= config.max_time + 1e-9
        # All values non-negative; count is non-increasing (only deaths).
        assert all(v >= 0.0 for v in values)
        assert values[0] >= values[-1]
        # Times are uniformly spaced by frequency.
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert all(abs(d - 0.25) < 1e-9 for d in diffs)
