"""Tests for pythia.api.collector and pythia.api.measure."""

from __future__ import annotations

import pytest

from pythia.api.collector import DataCollector
from pythia.api.config import SimulationConfig
from pythia.api.data_table import DataTable
from pythia.api.measure import Measure
from pythia.api.runner import SimulationRunner
from pythia.api.state import SimulationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _times(collector: DataCollector) -> list[float]:
    return list(collector.records.times)


def _values(collector: DataCollector, name: str) -> list[float]:
    t = collector.records
    return [t.get(ts)[name] for ts in t.times]


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

    def test_empty_measures_records_none(self) -> None:
        assert DataCollector([]).records is None

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
        assert len(collector.records) == 0

    def test_records_returns_data_table(self) -> None:
        m = Measure("x", lambda s: 1.0, frequency=1.0)
        collector = DataCollector([m])
        assert isinstance(collector.records, DataTable)

    def test_data_table_has_all_labels(self) -> None:
        m1 = Measure("wolves", lambda s: 0.0, frequency=1.0)
        m2 = Measure("rabbits", lambda s: 0.0, frequency=1.0)
        collector = DataCollector([m1, m2])
        assert set(collector.records.labels) == {"wolves", "rabbits"}


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------

class TestFrequency:
    def test_single_measure_uses_its_frequency(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=0.5)
        assert DataCollector([m]).frequency == pytest.approx(0.5)

    def test_mean_of_two_measures(self) -> None:
        m1 = Measure("a", lambda s: 0.0, frequency=0.5)
        m2 = Measure("b", lambda s: 0.0, frequency=1.5)
        assert DataCollector([m1, m2]).frequency == pytest.approx(1.0)

    def test_mean_of_three_measures(self) -> None:
        m1 = Measure("a", lambda s: 0.0, frequency=1.0)
        m2 = Measure("b", lambda s: 0.0, frequency=2.0)
        m3 = Measure("c", lambda s: 0.0, frequency=3.0)
        assert DataCollector([m1, m2, m3]).frequency == pytest.approx(2.0)

    def test_empty_measures_default_frequency(self) -> None:
        assert DataCollector([]).frequency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Zero-order hold semantics
# ---------------------------------------------------------------------------

class TestZeroOrderHoldSemantics:
    def test_example_from_requirements(self) -> None:
        """s1@t=0, s2@t=1.5, s3@t=2.3 (done), max_time=3.0, frequency=0.5."""
        m = Measure("v", lambda s: s.data["tag"], frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=3.0)

        s1 = SimulationState(elapsed_time=0.0, data={"tag": 1.0})
        s2 = SimulationState(elapsed_time=1.5, data={"tag": 2.0})
        s3 = SimulationState(elapsed_time=2.3, data={"tag": 3.0})

        collector.collect(s1)
        collector.collect(s2)
        collector.collect(s3)
        collector.finalize(s3)

        assert _times(collector) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        assert _values(collector, "v") == pytest.approx([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    def test_initial_collect_at_t0_records_one_point(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 7.0}))
        assert _times(collector) == pytest.approx([0.0])
        assert _values(collector, "x") == pytest.approx([7.0])

    def test_step_at_non_grid_time_uses_previous_state_retroactively(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 10.0}))
        collector.collect(SimulationState(elapsed_time=2.7, data={"v": 20.0}))
        assert _times(collector) == pytest.approx([0.0, 1.0, 2.0])
        assert _values(collector, "x") == pytest.approx([10.0, 10.0, 10.0])

    def test_step_at_grid_time_uses_new_state(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        collector.collect(SimulationState(elapsed_time=1.5, data={"v": 2.0}))
        assert _times(collector) == pytest.approx([0.0, 0.5, 1.0, 1.5])
        assert _values(collector, "x") == pytest.approx([1.0, 1.0, 1.0, 2.0])

    def test_finalize_fills_up_to_max_time(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        collector.collect(SimulationState(elapsed_time=2.3, data={"v": 2.0}))
        collector.finalize(SimulationState(elapsed_time=2.3, data={"v": 2.0}))
        assert _times(collector) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_finalize_includes_max_time_when_on_grid(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=0.5)
        collector = DataCollector([m])
        collector.reset(max_time=3.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        assert _times(collector)[-1] == pytest.approx(3.0)

    def test_finalize_stops_before_max_time_when_off_grid(self) -> None:
        m = Measure("x", lambda s: 0.0, frequency=0.3)
        collector = DataCollector([m])
        collector.reset(max_time=1.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        times = _times(collector)
        assert all(t <= 1.0 for t in times)
        assert max(times) == pytest.approx(0.9)

    def test_all_measures_in_every_row(self) -> None:
        m1 = Measure("a", lambda s: 1.0, frequency=1.0)
        m2 = Measure("b", lambda s: 2.0, frequency=1.0)
        collector = DataCollector([m1, m2])
        collector.reset(max_time=2.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        for t in collector.records.times:
            row = collector.records.get(t)
            assert "a" in row
            assert "b" in row


# ---------------------------------------------------------------------------
# Mean frequency drives shared schedule
# ---------------------------------------------------------------------------

class TestSharedSchedule:
    def test_mean_frequency_determines_schedule(self) -> None:
        # mean of 0.5 and 1.5 = 1.0
        m1 = Measure("a", lambda s: 1.0, frequency=0.5)
        m2 = Measure("b", lambda s: 2.0, frequency=1.5)
        collector = DataCollector([m1, m2])
        assert collector.frequency == pytest.approx(1.0)
        collector.reset(max_time=3.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        assert _times(collector) == pytest.approx([0.0, 1.0, 2.0, 3.0])


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
        assert len(collector.records) == 0

    def test_reset_restarts_from_t0(self) -> None:
        m = Measure("x", lambda s: s.data["v"], frequency=1.0)
        collector = DataCollector([m])
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 1.0}))
        collector.collect(SimulationState(elapsed_time=3.5, data={"v": 2.0}))
        collector.reset(max_time=5.0)
        collector.collect(SimulationState(elapsed_time=0.0, data={"v": 9.0}))
        assert len(collector.records) == 1
        assert collector.records.get(0.0)["x"] == pytest.approx(9.0)

    def test_multiple_measures_same_table(self) -> None:
        m1 = Measure("fast", lambda s: 1.0, frequency=0.5)
        m2 = Measure("slow", lambda s: 2.0, frequency=2.0)
        # mean frequency = 1.25
        collector = DataCollector([m1, m2])
        collector.reset(max_time=4.0)
        collector.collect(SimulationState(elapsed_time=0.0))
        collector.finalize(SimulationState(elapsed_time=0.0))
        assert "fast" in collector.records.labels
        assert "slow" in collector.records.labels
        # All rows have both columns
        for t in collector.records.times:
            row = collector.records.get(t)
            assert "fast" in row
            assert "slow" in row


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
        collector._table.add_row(999.0, {"t": 999.0})

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()
        assert all(t <= minimal_config.max_time for t in collector.records.times)

    def test_runner_collects_at_t0_and_after_each_step(
        self, minimal_config: SimulationConfig
    ) -> None:
        m = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        collector = DataCollector([m])

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()

        assert _times(collector) == pytest.approx(list(range(0, 11)))

    def test_runner_finalizes_after_loop(
        self, minimal_config: SimulationConfig
    ) -> None:
        m = Measure("t", lambda s: s.elapsed_time, frequency=0.5)
        collector = DataCollector([m])

        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config, data_collector=collector)
        runner.run()

        times = _times(collector)
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
        first_count = len(collector.records)
        runner.run()
        second_count = len(collector.records)
        assert first_count == second_count

    def test_collector_none_does_not_raise(
        self, minimal_config: SimulationConfig
    ) -> None:
        from tests.conftest import NoOpModel
        runner = SimulationRunner(NoOpModel(), minimal_config)
        runner.run()

    def test_end_to_end_with_population_model(self) -> None:
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

        times = _times(collector)
        values = _values(collector, "wolves")

        assert times[0] == pytest.approx(0.0)
        assert times[-1] <= config.max_time + 1e-9
        assert all(v >= 0.0 for v in values)
        assert values[0] >= values[-1]
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert all(abs(d - 0.25) < 1e-9 for d in diffs)
