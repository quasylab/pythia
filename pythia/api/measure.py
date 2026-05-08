"""Measure — a named, time-triggered sampling function.

A :class:`Measure` binds a name, a sampling frequency (in simulation time
units), and a function that extracts a scalar value from a
:class:`~pythia.api.state.SimulationState`.  Measures are the building
blocks of :class:`~pythia.api.collector.DataCollector`.

Example::

    from pythia.api.measure import Measure

    wolf_count = Measure(
        name="wolves",
        function=lambda s: s.data["pm.counting_state"].counting_vector.get(0, 0),
        frequency=0.25,
    )
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from pythia.api.state import SimulationState


@dataclasses.dataclass(frozen=True)
class Measure:
    """A named scalar measurement sampled at a fixed time frequency.

    Attributes:
        name: Unique identifier used as the key in
            :attr:`~pythia.api.collector.DataCollector.records`.
        function: A callable that accepts a
            :class:`~pythia.api.state.SimulationState` and returns a
            ``float``.  It must not mutate the state.
        frequency: Minimum elapsed-simulation-time interval between
            successive samples.  Must be strictly positive.

    Raises:
        ValueError: If *frequency* is not strictly positive.

    Example:
        >>> state = SimulationState(data={"pm.population": 42.0})
        >>> m = Measure(name="pop", function=lambda s: s.data["pm.population"], frequency=1.0)
        >>> m.function(state)
        42.0
    """

    name: str
    function: Callable[[SimulationState], float]
    frequency: float

    def __post_init__(self) -> None:
        """Validate frequency after construction."""
        if self.frequency <= 0.0:
            raise ValueError(
                f"Measure '{self.name}': frequency must be strictly positive, "
                f"got {self.frequency!r}"
            )
