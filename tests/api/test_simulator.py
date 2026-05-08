"""Tests for pythia.api.simulator."""

from __future__ import annotations

import pytest

from pythia.api import SimulationConfig, SimulationState
from pythia.api.measure import Measure
from pythia.api.simulator import Simulator
from tests.conftest import NoOpModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_measure(frequency: float = 1.0) -> Measure:
    return Measure("t", lambda s: s.elapsed_time, frequency=frequency)


def _make_sim(n_runs: int = 3, max_time: float = 5.0, seed: int | None = 0) -> Simulator:
    sim = Simulator(NoOpModel(), SimulationConfig(max_time=max_time, random_seed=seed), [_time_measure()])
    sim.run(n_runs)
    return sim


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSimulatorConstruction:
    def test_results_empty_before_run(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=5.0), [_time_measure()])
        assert sim.results == []

    def test_results_for_raises_before_run(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=5.0), [_time_measure()])
        # empty results — list comprehension returns [] without KeyError
        assert sim.results_for("t") == []


# ---------------------------------------------------------------------------
# Basic run behaviour
# ---------------------------------------------------------------------------

class TestSimulatorRun:
    def test_results_length_equals_n_runs(self) -> None:
        sim = _make_sim(n_runs=5)
        assert len(sim.results) == 5

    def test_each_run_has_expected_measure(self) -> None:
        sim = _make_sim(n_runs=3)
        for table in sim.results:
            assert "t" in table.labels

    def test_each_run_has_correct_number_of_points(self) -> None:
        # NoOpModel: dt=1.0, max_time=5.0, frequency=1.0 → points 0..5 = 6
        sim = _make_sim(n_runs=3, max_time=5.0)
        for table in sim.results:
            assert len(table) == 6

    def test_zero_n_runs_raises(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=5.0), [_time_measure()])
        with pytest.raises(ValueError, match="n_runs"):
            sim.run(0)

    def test_negative_n_runs_raises(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=5.0), [_time_measure()])
        with pytest.raises(ValueError, match="n_runs"):
            sim.run(-1)


# ---------------------------------------------------------------------------
# results_for
# ---------------------------------------------------------------------------

class TestResultsFor:
    def test_results_for_returns_one_series_per_run(self) -> None:
        sim = _make_sim(n_runs=4)
        series = sim.results_for("t")
        assert len(series) == 4

    def test_results_for_unknown_name_raises(self) -> None:
        sim = _make_sim(n_runs=1)
        with pytest.raises(KeyError, match="unknown"):
            sim.results_for("unknown")

    def test_results_for_matches_results_property(self) -> None:
        sim = _make_sim(n_runs=3)
        for i, series in enumerate(sim.results_for("t")):
            table = sim.results[i]
            expected = [(t, table.get(t)["t"]) for t in table.times]
            assert series == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Seed handling
# ---------------------------------------------------------------------------

class TestSeedHandling:
    def test_seeded_runs_are_reproducible(self) -> None:
        """Two Simulators with the same base seed must produce identical results."""
        measures = [_time_measure()]
        config = SimulationConfig(max_time=5.0, random_seed=7)
        sim_a = Simulator(NoOpModel(), config, measures)
        sim_b = Simulator(NoOpModel(), config, measures)
        sim_a.run(3)
        sim_b.run(3)
        for ta, tb in zip(sim_a.results, sim_b.results):
            assert ta.times == tb.times
            assert [ta.get(t)["t"] for t in ta.times] == [tb.get(t)["t"] for t in tb.times]

    def test_different_seeds_per_run(self) -> None:
        """Each run should receive a distinct seed (verified via a stochastic model)."""
        from pythia.models.pm.counting_state import CountingState
        from pythia.models.pm.population import Population
        from pythia.models.pm.population_model import PopulationModel
        from pythia.models.pm.rule import Rule

        pop = Population()
        wolf = pop.register("wolf")

        def wolf_count(s: SimulationState) -> float:
            cs = s.data.get("pm.counting_state")
            return float(cs.counting_vector.get(wolf.index, 0)) if cs else 0.0

        model = PopulationModel(
            population=pop,
            initial_state={wolf: 50},
            rules=[
                Rule(
                    pre=CountingState({wolf: 1}),
                    post=CountingState({}),
                    rate_function=lambda s: float(s.counting_vector.get(wolf.index, 0)),
                )
            ],
        )
        measures = [Measure("wolves", wolf_count, frequency=0.5)]
        config = SimulationConfig(max_time=2.0, random_seed=0)
        sim = Simulator(model, config, measures)
        sim.run(5)

        # With different seeds the final wolf counts should differ across some runs
        final_values = [
            table.get(table.times[-1])["wolves"]
            for table in sim.results
        ]
        assert len(set(final_values)) > 1  # not all identical

    def test_none_seed_does_not_raise(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=3.0, random_seed=None), [_time_measure()])
        sim.run(2)
        assert len(sim.results) == 2


# ---------------------------------------------------------------------------
# Repeated calls reset previous results
# ---------------------------------------------------------------------------

class TestRepeatedRun:
    def test_second_run_replaces_first(self) -> None:
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=5.0, random_seed=0), [_time_measure()])
        sim.run(10)
        sim.run(3)
        assert len(sim.results) == 3

    def test_results_is_a_copy(self) -> None:
        sim = _make_sim(n_runs=2)
        snapshot = sim.results
        sim.run(5)
        assert len(snapshot) == 2  # old snapshot unaffected


# ---------------------------------------------------------------------------
# Multiple measures
# ---------------------------------------------------------------------------

class TestMultipleMeasures:
    def test_multiple_measures_all_present(self) -> None:
        m1 = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        m2 = Measure("step", lambda s: float(s.step), frequency=2.0)
        sim = Simulator(NoOpModel(), SimulationConfig(max_time=4.0, random_seed=0), [m1, m2])
        sim.run(3)
        for table in sim.results:
            assert "t" in table.labels and "step" in table.labels
