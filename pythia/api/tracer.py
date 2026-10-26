"""Tracer — run a single simulation and collect data into a DataTracer.

A :class:`Tracer` drives a model through its full lifecycle, calling
:meth:`~pythia.api.data_tracer.DataTracer.trace` after the initial state and
after every step up to the deadline defined in the
:class:`~pythia.api.config.SimulationConfig`.

Example::

    from pythia.api import SimulationConfig
    from pythia.api.data_tracer import DataTracer
    from pythia.api.tracer import Tracer

    data_tracer = DataTracer(
        labels=["position"],
        trace_function=lambda s: {"agent": {"position": s.data.get("pos", 0.0)}},
    )

    tracer = Tracer(data_tracer)
    tracer.run(model, SimulationConfig(max_time=10.0, random_seed=0))

    table = data_tracer.get_table("agent")
    table.times    # (0.0, 1.0, 2.0, ...)
"""

from __future__ import annotations

from pythia.api.config import SimulationConfig
from pythia.api.data_tracer import DataTracer
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState


class Tracer:
    """Run one simulation and populate a :class:`~pythia.api.data_tracer.DataTracer`.

    The tracer owns a :class:`~pythia.api.data_tracer.DataTracer` that it
    feeds with observations.  Each call to :meth:`run` executes the full
    model lifecycle — ``initialize`` → step loop → ``finalize`` — and calls
    :meth:`~pythia.api.data_tracer.DataTracer.trace` on the
    :class:`~pythia.api.state.SimulationState` at time ``0`` and after every
    step until the deadline (*config.max_time*) is reached or the model
    signals termination (``dt ≤ 0``).

    Multiple calls to :meth:`run` accumulate additional rows into the same
    :class:`~pythia.api.data_tracer.DataTracer`.

    Args:
        data_tracer: The :class:`~pythia.api.data_tracer.DataTracer` that
            collects measurements during the run.

    Examples:
        >>> from pythia.api import BaseModel, SimulationConfig, SimulationState
        >>> class CountModel(BaseModel):
        ...     @property
        ...     def name(self): return "count"
        ...     def initialize(self, config): self._n = 0
        ...     def step(self, state):
        ...         self._n += 1
        ...         state.data["n"] = float(self._n)
        ...         return 1.0, state
        ...     def finalize(self, state): pass
        >>> from pythia.api.data_tracer import DataTracer
        >>> dt = DataTracer(["n"], lambda s: {"obj": {"n": s.data.get("n", 0.0)}})
        >>> Tracer(dt).run(CountModel(), SimulationConfig(max_time=3.0, random_seed=0))
        >>> dt.times("obj")
        (0.0, 1.0, 2.0, 3.0)
    """

    def __init__(self, data_tracer: DataTracer) -> None:
        self._data_tracer = data_tracer

    @property
    def data_tracer(self) -> DataTracer:
        """The :class:`~pythia.api.data_tracer.DataTracer` collecting data."""
        return self._data_tracer

    def run(self, model: BaseModel, config: SimulationConfig) -> None:
        """Execute one simulation run and record observations.

        Initialises *model* with *config*, traces the initial state at
        ``elapsed_time = 0``, then advances the simulation step by step.
        After each step the state is traced; the loop stops when
        ``state.elapsed_time >= config.max_time`` or the model returns
        ``dt ≤ 0``.  :meth:`~pythia.api.model.BaseModel.finalize` is always
        called before returning.

        Args:
            model: The simulation model to run.
            config: Run configuration.  ``config.max_time`` is the deadline.

        Examples:
            >>> from pythia.api import BaseModel, SimulationConfig, SimulationState
            >>> class TrivialModel(BaseModel):
            ...     @property
            ...     def name(self): return "trivial"
            ...     def initialize(self, config): pass
            ...     def step(self, state):
            ...         return 0.0, state
            ...     def finalize(self, state): pass
            >>> from pythia.api.data_tracer import DataTracer
            >>> dt = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
            >>> Tracer(dt).run(TrivialModel(), SimulationConfig(max_time=5.0))
            >>> dt.get_table("obj").get(0.0)
            {'v': 0.0}
        """
        model.initialize(config)
        state = SimulationState()
        self._data_tracer.trace(state)

        while state.elapsed_time < config.max_time:
            dt, state = model.step(state)
            if dt <= 0.0:
                break
            state.advance(dt)
            self._data_tracer.trace(state)

        model.finalize(state)
