"""Tests for pythia.api.tracer."""

from __future__ import annotations

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.data_tracer import DataTracer
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState
from pythia.api.tracer import Tracer


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------

class _TickModel(BaseModel):
    """Advances time by 1.0 on each step; stores elapsed_time in data['t']."""

    @property
    def name(self) -> str:
        return "tick"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        state.data["t"] = state.elapsed_time + 1.0
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class _ImmediateStopModel(BaseModel):
    """Returns dt=0 on the first step."""

    @property
    def name(self) -> str:
        return "stop"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return 0.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class _SeedCheckModel(BaseModel):
    """Records the random_seed it received during initialize."""

    seed_received: int | None = None

    @property
    def name(self) -> str:
        return "seed_check"

    def initialize(self, config: SimulationConfig) -> None:
        _SeedCheckModel.seed_received = config.random_seed

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return 0.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


def _tracer(labels: list[str] | None = None) -> DataTracer:
    lbls = labels or ["v"]
    return DataTracer(lbls, lambda s: {"obj": {l: s.elapsed_time for l in lbls}})


def _cfg(max_time: float = 3.0, seed: int = 0) -> SimulationConfig:
    return SimulationConfig(max_time=max_time, random_seed=seed)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_data_tracer_stored(self) -> None:
        dt = _tracer()
        t = Tracer(dt)
        assert t.data_tracer is dt


# ---------------------------------------------------------------------------
# run — lifecycle
# ---------------------------------------------------------------------------

class TestRunLifecycle:
    def test_initialize_called(self) -> None:
        _SeedCheckModel.seed_received = None
        model = _SeedCheckModel()
        dt = _tracer()
        Tracer(dt).run(model, SimulationConfig(max_time=1.0, random_seed=99))
        assert _SeedCheckModel.seed_received == 99

    def test_finalize_called(self) -> None:
        finalized = []

        class FinalizeModel(_TickModel):
            def finalize(self, state: SimulationState) -> None:
                finalized.append(True)

        Tracer(_tracer()).run(FinalizeModel(), _cfg(1.0))
        assert finalized == [True]

    def test_finalize_called_on_immediate_stop(self) -> None:
        finalized = []

        class EarlyStop(_ImmediateStopModel):
            def finalize(self, state: SimulationState) -> None:
                finalized.append(True)

        Tracer(_tracer()).run(EarlyStop(), _cfg(5.0))
        assert finalized == [True]


# ---------------------------------------------------------------------------
# run — initial trace at t=0
# ---------------------------------------------------------------------------

class TestInitialTrace:
    def test_initial_state_traced(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(3.0))
        assert 0.0 in dt.get_table("obj").times

    def test_initial_value_is_zero(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(3.0))
        assert dt.get_table("obj").get(0.0)["v"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run — step tracing
# ---------------------------------------------------------------------------

class TestStepTracing:
    def test_observations_at_each_step(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(3.0))
        assert dt.get_table("obj").times == (0.0, 1.0, 2.0, 3.0)

    def test_values_match_elapsed_time(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(3.0))
        for t in (0.0, 1.0, 2.0, 3.0):
            assert dt.get_table("obj").get(t)["v"] == pytest.approx(t)

    def test_stops_at_deadline(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(2.0))
        assert max(dt.get_table("obj").times) == pytest.approx(2.0)

    def test_stops_on_zero_dt(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_ImmediateStopModel(), _cfg(10.0))
        # Only the initial trace at t=0 should be recorded
        assert dt.get_table("obj").times == (0.0,)

    def test_no_step_past_deadline(self) -> None:
        dt = _tracer()
        Tracer(dt).run(_TickModel(), _cfg(2.0))
        assert all(t <= 2.0 for t in dt.get_table("obj").times)


# ---------------------------------------------------------------------------
# run — multiple calls accumulate data
# ---------------------------------------------------------------------------

class TestMultipleRuns:
    def test_second_run_adds_rows(self) -> None:
        dt = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
        tracer = Tracer(dt)
        tracer.run(_TickModel(), _cfg(1.0))
        tracer.run(_TickModel(), _cfg(1.0))
        # Same timestamps → second run overwrites first (DataTable semantics)
        assert len(dt.get_table("obj")) == 2  # t=0 and t=1

    def test_config_forwarded_each_run(self) -> None:
        seeds: list[int] = []

        class RecordSeed(_TickModel):
            def initialize(self, config: SimulationConfig) -> None:
                seeds.append(config.random_seed)

        tracer = Tracer(_tracer())
        tracer.run(RecordSeed(), SimulationConfig(max_time=1.0, random_seed=1))
        tracer.run(RecordSeed(), SimulationConfig(max_time=1.0, random_seed=2))
        assert seeds == [1, 2]
