"""Simulation state.

:class:`SimulationState` is a mutable dataclass that carries everything
that changes as the simulation progresses.  Models read and write their
domain-specific data through the open-ended :attr:`~SimulationState.data`
mapping.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any


@dataclasses.dataclass
class SimulationState:
    """Mutable carrier for simulation state at a single point in time.

    Attributes:
        step: Current iteration index, starting at ``0``.
        elapsed_time: Accumulated simulated time (``step * time_step``).
            Updated atomically by :meth:`advance`.
        data: Open-ended payload for model-specific state.  Models own their
            keys; use namespaced strings (e.g. ``"sir.infected"``) to avoid
            collisions between future modules.
        is_done: Set to ``True`` by a model to request early termination
            (e.g. on convergence or an unrecoverable error).

    Examples:
        >>> state = SimulationState()
        >>> state.advance(0.5)
        >>> state.step
        1
        >>> state.elapsed_time
        0.5
    """

    step: int = 0
    elapsed_time: float = 0.0
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    is_done: bool = False

    def advance(self, time_step: float) -> None:
        """Increment *step* and accumulate *elapsed_time* atomically.

        Advancing through this method rather than mutating fields directly
        prevents the two counters from drifting apart.

        Args:
            time_step: The duration of the tick just completed, as returned
                by :meth:`~pythia.api.model.BaseModel.step`.  A value of
                ``0.0`` increments the step counter without advancing time.
        """
        self.step += 1
        self.elapsed_time += time_step

    def snapshot(self) -> SimulationState:
        """Return a deep copy of this state.

        Useful when the runner records per-step history: each snapshot is
        independent and will not reflect subsequent mutations.

        Returns:
            A new :class:`SimulationState` with the same field values,
            including a deep copy of :attr:`data`.

        Examples:
            >>> state = SimulationState(step=3, data={"x": [1, 2, 3]})
            >>> snap = state.snapshot()
            >>> state.data["x"].append(4)
            >>> len(snap.data["x"])  # snapshot is unaffected by the append
            3
        """
        return copy.deepcopy(self)
