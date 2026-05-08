"""Tests for pythia.api.runner."""

from __future__ import annotations

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.events import EventBus, EventKind, SimulationEvent
from pythia.api.exceptions import ModelError, RunnerError
from pythia.api.model import BaseModel
from pythia.api.runner import SimulationRunner
from pythia.api.state import SimulationState


class CounterModel(BaseModel):
    """Increments ``state.data['count']`` on every step."""

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


class EarlyStopModel(BaseModel):
    """Returns dt=0.0 after *stop_at* steps."""

    def __init__(self, stop_at: int) -> None:
        self._stop_at = stop_at

    @property
    def name(self) -> str:
        return "early_stop"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        if state.step + 1 >= self._stop_at:
            state.is_done = True
            return 0.0, state
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class ErrorInStepModel(BaseModel):
    """Raises on the first call to step."""

    @property
    def name(self) -> str:
        return "error_step"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        raise ValueError("intentional error")

    def finalize(self, state: SimulationState) -> None:
        pass


class ErrorInInitModel(BaseModel):
    """Raises in initialize."""

    @property
    def name(self) -> str:
        return "error_init"

    def initialize(self, config: SimulationConfig) -> None:
        raise RuntimeError("init failed")

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


class BadReturnModel(BaseModel):
    """Returns a non-tuple from step."""

    @property
    def name(self) -> str:
        return "bad_return"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return "not a tuple"  # type: ignore[return-value]

    def finalize(self, state: SimulationState) -> None:
        pass


class TestSimulationRunnerValidation:
    def test_non_model_raises(self, minimal_config: SimulationConfig) -> None:
        with pytest.raises(RunnerError):
            SimulationRunner("not_a_model", minimal_config)  # type: ignore[arg-type]

    def test_non_config_raises(self, no_op_model: BaseModel) -> None:
        with pytest.raises(RunnerError):
            SimulationRunner(no_op_model, {"max_time": 10.0})  # type: ignore[arg-type]


class TestSimulationRunnerRunLoop:
    def test_runs_correct_number_of_steps(self, minimal_config: SimulationConfig) -> None:
        # CounterModel returns dt=1.0; max_time=10.0 → exactly 10 steps
        runner = SimulationRunner(CounterModel(), minimal_config)
        state = runner.run()
        assert state.data["count"] == 10

    def test_elapsed_time_equals_max_time(self, minimal_config: SimulationConfig) -> None:
        runner = SimulationRunner(CounterModel(), minimal_config)
        state = runner.run()
        assert state.elapsed_time == pytest.approx(minimal_config.max_time)

    def test_early_termination_via_zero_dt(self, minimal_config: SimulationConfig) -> None:
        runner = SimulationRunner(EarlyStopModel(stop_at=3), minimal_config)
        state = runner.run()
        assert state.step == 3
        assert state.is_done is True

    def test_bad_step_return_raises_model_error(
        self, minimal_config: SimulationConfig
    ) -> None:
        with pytest.raises(ModelError):
            SimulationRunner(BadReturnModel(), minimal_config).run()

    def test_exception_in_step_wrapped_as_model_error(
        self, minimal_config: SimulationConfig
    ) -> None:
        with pytest.raises(ModelError):
            SimulationRunner(ErrorInStepModel(), minimal_config).run()

    def test_exception_in_initialize_wrapped_as_model_error(
        self, minimal_config: SimulationConfig
    ) -> None:
        with pytest.raises(ModelError):
            SimulationRunner(ErrorInInitModel(), minimal_config).run()


class TestSimulationRunnerHistory:
    def test_history_empty_by_default(self, minimal_config: SimulationConfig) -> None:
        runner = SimulationRunner(CounterModel(), minimal_config)
        runner.run()
        assert runner.history == []

    def test_history_length_matches_steps(self, minimal_config: SimulationConfig) -> None:
        runner = SimulationRunner(CounterModel(), minimal_config, record_history=True)
        runner.run()
        assert len(runner.history) == 10  # max_time=10.0, dt=1.0

    def test_history_snapshots_are_independent(
        self, minimal_config: SimulationConfig
    ) -> None:
        runner = SimulationRunner(CounterModel(), minimal_config, record_history=True)
        runner.run()
        counts = [s.data["count"] for s in runner.history]
        assert counts == list(range(1, 11))

    def test_history_cleared_between_runs(
        self, minimal_config: SimulationConfig
    ) -> None:
        runner = SimulationRunner(CounterModel(), minimal_config, record_history=True)
        runner.run()
        runner.run()
        assert len(runner.history) == 10  # max_time=10.0, dt=1.0


class TestSimulationRunnerEvents:
    def _collect_events(
        self, model: BaseModel, config: SimulationConfig
    ) -> list[SimulationEvent]:
        events: list[SimulationEvent] = []
        bus = EventBus()
        for kind in EventKind:
            bus.subscribe(kind, events.append)
        SimulationRunner(model, config, event_bus=bus).run()
        return events

    def test_run_started_emitted(self, minimal_config: SimulationConfig) -> None:
        events = self._collect_events(CounterModel(), minimal_config)
        kinds = [e.kind for e in events]
        assert EventKind.RUN_STARTED in kinds

    def test_run_finished_emitted(self, minimal_config: SimulationConfig) -> None:
        events = self._collect_events(CounterModel(), minimal_config)
        kinds = [e.kind for e in events]
        assert EventKind.RUN_FINISHED in kinds

    def test_step_completed_count_matches_steps(
        self, minimal_config: SimulationConfig
    ) -> None:
        events = self._collect_events(CounterModel(), minimal_config)
        step_events = [e for e in events if e.kind == EventKind.STEP_COMPLETED]
        assert len(step_events) == 10  # max_time=10.0, dt=1.0

    def test_run_failed_emitted_on_error(
        self, minimal_config: SimulationConfig
    ) -> None:
        events: list[SimulationEvent] = []
        bus = EventBus()
        for kind in EventKind:
            bus.subscribe(kind, events.append)
        with pytest.raises(ModelError):
            SimulationRunner(ErrorInStepModel(), minimal_config, event_bus=bus).run()
        kinds = [e.kind for e in events]
        assert EventKind.RUN_FAILED in kinds
        assert EventKind.RUN_FINISHED not in kinds
