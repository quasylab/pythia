"""Simulation runner — the orchestration engine.

:class:`SimulationRunner` owns the run loop and coordinates the lifecycle
calls to a :class:`~pythia.api.model.BaseModel`.  It emits
:class:`~pythia.api.events.SimulationEvent` objects via an optional
:class:`~pythia.api.events.EventBus` so that external observers (loggers,
visualisers, checkpointers) can react without modifying the runner.

Example::

    from pythia.api import (
        SimulationConfig,
        SimulationRunner,
        SimulationState,
        BaseModel,
    )

    class CounterModel(BaseModel):
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

    config = SimulationConfig(max_time=5.0)
    runner = SimulationRunner(CounterModel(), config)
    final_state = runner.run()
    assert final_state.data["count"] == 5
"""

from __future__ import annotations

import logging

from pythia.api.collector import DataCollector
from pythia.api.config import SimulationConfig
from pythia.api.events import EventBus, EventKind, SimulationEvent
from pythia.api.exceptions import ModelError, RunnerError
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Orchestrates a simulation run for a given model and configuration.

    Attributes:
        model: The model under simulation.
        config: The immutable run configuration.
        history: Per-step state snapshots.  Populated only when
            *record_history* is ``True`` at construction time; otherwise an
            empty list.

    Args:
        model: A concrete :class:`~pythia.api.model.BaseModel` instance.
        config: Configuration for the run.
        event_bus: Optional :class:`~pythia.api.events.EventBus`.  If
            ``None``, a fresh bus (with no subscribers) is used internally.
        record_history: When ``True``, a
            :meth:`~pythia.api.state.SimulationState.snapshot` is appended
            to :attr:`history` after each step.  Disabled by default to avoid
            unbounded memory growth.
        data_collector: Optional :class:`~pythia.api.collector.DataCollector`.
            When provided, :meth:`~pythia.api.collector.DataCollector.reset`
            is called before the run loop and
            :meth:`~pythia.api.collector.DataCollector.collect` is called
            after each step.

    Raises:
        RunnerError: If *model* or *config* are not valid objects.

    Example:
        >>> runner = SimulationRunner(my_model, config, record_history=True)
        >>> state = runner.run()
        >>> len(runner.history)  # one entry per step
        ...
    """

    def __init__(
        self,
        model: BaseModel,
        config: SimulationConfig,
        *,
        event_bus: EventBus | None = None,
        record_history: bool = False,
        data_collector: DataCollector | None = None,
    ) -> None:
        """Initialise the runner."""
        if not isinstance(model, BaseModel):
            raise RunnerError(
                f"model must be a BaseModel instance, got {type(model)!r}"
            )
        if not isinstance(config, SimulationConfig):
            raise RunnerError(
                f"config must be a SimulationConfig instance, got {type(config)!r}"
            )

        self.model = model
        self.config = config
        self._bus: EventBus = event_bus if event_bus is not None else EventBus()
        self._record_history = record_history
        self._data_collector: DataCollector | None = data_collector
        self.history: list[SimulationState] = []

    def run(self) -> SimulationState:
        """Execute the full simulation lifecycle and return the final state.

        Lifecycle order:

        1. Emit :attr:`~pythia.api.events.EventKind.RUN_STARTED`.
        2. Call :meth:`~pythia.api.model.BaseModel.initialize`.
        3. Repeat until ``elapsed_time >= max_time`` or the model returns
           ``dt = 0.0``:

           a. Call :meth:`~pythia.api.model.BaseModel.step`.
           b. Advance the state clock by the returned *dt*.
           c. Optionally append a snapshot to :attr:`history`.
           d. Emit :attr:`~pythia.api.events.EventKind.STEP_COMPLETED`.

        4. Call :meth:`~pythia.api.model.BaseModel.finalize`.
        5. Emit :attr:`~pythia.api.events.EventKind.RUN_FINISHED`.

        If any exception escapes from the model, the runner wraps it in a
        :class:`~pythia.api.exceptions.ModelError`, emits
        :attr:`~pythia.api.events.EventKind.RUN_FAILED`, and re-raises.

        Returns:
            The :class:`~pythia.api.state.SimulationState` at the end of the
            run (after :meth:`~pythia.api.model.BaseModel.finalize`).

        Raises:
            ModelError: If the model raises during any lifecycle method.
        """
        self.history.clear()
        state = SimulationState()

        self._emit(EventKind.RUN_STARTED, state=None)
        logger.info("Starting run: model=%r, max_time=%.4f", self.model.name, self.config.max_time)

        if self._data_collector is not None:
            self._data_collector.reset(self.config.max_time)

        try:
            self.model.initialize(self.config)
            self.model.setup(state)
        except Exception as exc:
            self._fail(exc, state)

        if self._data_collector is not None:
            self._data_collector.collect(state)  # capture initial state at t=0

        try:
            while state.elapsed_time < self.config.max_time:
                step_result = self.model.step(state)

                if (
                    not isinstance(step_result, tuple)
                    or len(step_result) != 2
                    or not isinstance(step_result[1], SimulationState)
                ):
                    raise ModelError(
                        f"{self.model.name!r}.step() must return a tuple "
                        f"(float, SimulationState), got {type(step_result)!r}"
                    )

                dt, state = step_result
                state.advance(dt)

                done = dt <= 0.0
                if done:
                    state.is_done = True

                if self._data_collector is not None:
                    self._data_collector.collect(state)

                if self._record_history:
                    self.history.append(state.snapshot())

                self._emit(EventKind.STEP_COMPLETED, state=state)
                logger.debug("Step %d completed (elapsed=%.4f)", state.step, state.elapsed_time)

                if done:
                    break

            if self._data_collector is not None:
                self._data_collector.finalize(state)

            self.model.finalize(state)

        except ModelError:
            self._emit(EventKind.RUN_FAILED, state=state)
            raise
        except Exception as exc:
            self._fail(exc, state)

        self._emit(EventKind.RUN_FINISHED, state=state)
        logger.info(
            "Run finished: model=%r, steps=%d, elapsed=%.4f",
            self.model.name,
            state.step,
            state.elapsed_time,
        )
        return state

    def _emit(self, kind: EventKind, *, state: SimulationState | None) -> None:
        """Construct and emit a :class:`~pythia.api.events.SimulationEvent`."""
        event = SimulationEvent(
            kind=kind,
            step=state.step if state is not None else 0,
            state=state,
            metadata={"model": self.model.name},
        )
        self._bus.emit(event)

    def _fail(self, exc: BaseException, state: SimulationState) -> None:
        """Wrap *exc* in :class:`~pythia.api.exceptions.ModelError`, emit, and raise."""
        model_error = ModelError(
            f"Model {self.model.name!r} raised an exception: {exc}"
        ) if not isinstance(exc, ModelError) else exc

        self._emit(EventKind.RUN_FAILED, state=state)
        raise model_error from exc
