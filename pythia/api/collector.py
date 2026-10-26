"""DataCollector — time-triggered recording of scalar measures.

A :class:`DataCollector` holds a set of :class:`~pythia.api.measure.Measure`
objects and records all of them at the same uniformly-spaced simulation-time
points.  The shared sampling frequency is the arithmetic mean of the
individual measure frequencies.  All observations are stored in a single
:class:`~pythia.api.data_table.DataTable` whose column labels are the measure
names.

The collection follows a **zero-order hold** model: the value at any scheduled
time point is the measure applied to the most recent simulation state that
was available *at or before* that time.

For example, if a simulation produces states at t=0 (``s1``), t=1.5 (``s2``),
and t=2.3 (``s3``, done) with ``max_time=3.0`` and two measures whose mean
frequency is 0.5, the scheduled times are 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.
Each row of the resulting :class:`~pythia.api.data_table.DataTable` contains
the values of **all** measures at that scheduled time.

Timestamps are exact multiples of the computed frequency
(``k * frequency``); the final record is always at the scheduled
point ``<= max_time``.

Usage with :class:`~pythia.api.runner.SimulationRunner`::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.api.collector import DataCollector
    from pythia.api.measure import Measure

    collector = DataCollector([
        Measure("wolves", lambda s: ..., frequency=0.25),
        Measure("rabbits", lambda s: ..., frequency=0.75),
    ])
    # shared frequency = mean(0.25, 0.75) = 0.5

    runner = SimulationRunner(model, SimulationConfig(max_time=10.0), data_collector=collector)
    runner.run()

    table = collector.records          # single DataTable
    for t in table.times:
        row = table.get(t)
        print(f"t={t:.2f}  wolves={row['wolves']:.0f}  rabbits={row['rabbits']:.0f}")
"""

from __future__ import annotations

from collections.abc import Iterable

from pythia.api.data_table import DataTable
from pythia.api.measure import Measure
from pythia.api.state import SimulationState


class DataCollector:
    """Collects scalar measurements at a shared uniform simulation-time interval.

    The sampling frequency is the arithmetic mean of the registered measures'
    individual frequencies.  All measures are sampled at exactly the same
    scheduled times and stored together in a single
    :class:`~pythia.api.data_table.DataTable`.

    Args:
        measures: The :class:`~pythia.api.measure.Measure` objects to record.
            Every measure must have a unique :attr:`~pythia.api.measure.Measure.name`.

    Raises:
        ValueError: If two measures share the same name.

    Examples:
        >>> from pythia.api.state import SimulationState
        >>> from pythia.api.measure import Measure
        >>> m = Measure("x", lambda s: s.elapsed_time, frequency=1.0)
        >>> collector = DataCollector([m])
        >>> collector.reset(max_time=3.0)
        >>> collector.collect(SimulationState(elapsed_time=0.0))
        >>> collector.collect(SimulationState(elapsed_time=2.5))
        >>> collector.finalize(SimulationState(elapsed_time=2.5))
        >>> collector.records.times
        (0.0, 1.0, 2.0, 2.5, 3.0)
    """

    def __init__(self, measures: Iterable[Measure]) -> None:
        self._measures: tuple[Measure, ...] = tuple(measures)
        seen: set[str] = set()
        for m in self._measures:
            if m.name in seen:
                raise ValueError(
                    f"DataCollector: duplicate measure name {m.name!r}"
                )
            seen.add(m.name)

        if self._measures:
            self._frequency: float = (
                sum(m.frequency for m in self._measures) / len(self._measures)
            )
            self._table: DataTable | None = DataTable([m.name for m in self._measures])
        else:
            self._frequency = 1.0
            self._table = None

        self._next_index: int = 0
        self._pending_values: dict[str, float | None] = {
            m.name: None for m in self._measures
        }
        self._max_time: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The registered measures, in registration order."""
        return self._measures

    @property
    def frequency(self) -> float:
        """The shared sampling frequency (mean of individual measure frequencies).

        Examples:
            >>> m1 = Measure("a", lambda s: 0.0, frequency=0.5)
            >>> m2 = Measure("b", lambda s: 0.0, frequency=1.5)
            >>> DataCollector([m1, m2]).frequency
            1.0
        """
        return self._frequency

    @property
    def records(self) -> DataTable | None:
        """The single :class:`~pythia.api.data_table.DataTable` holding all
        collected observations, or ``None`` when no measures are registered.

        The table has one column per measure (labelled by measure name) and
        one row per scheduled time point.  Use
        :attr:`~pythia.api.data_table.DataTable.times` for the sorted
        timestamps and :meth:`~pythia.api.data_table.DataTable.get` to
        retrieve all measure values at a specific time.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> from pythia.api.measure import Measure
            >>> m = Measure("v", lambda s: 1.0, frequency=1.0)
            >>> c = DataCollector([m])
            >>> c.reset(2.0)
            >>> c.collect(SimulationState(elapsed_time=0.0))
            >>> c.finalize(SimulationState(elapsed_time=0.0))
            >>> c.records.times
            (0.0, 1.0, 2.0)
            >>> c.records.get(0.0)["v"]
            1.0
        """
        return self._table

    def reset(self, max_time: float) -> None:
        """Clear all records and prepare for a new run.

        Args:
            max_time: The simulation time horizon (``config.max_time``).
        """
        self._max_time = max_time
        if self._measures:
            self._table = DataTable([m.name for m in self._measures])
        self._next_index = 0
        self._pending_values = {m.name: None for m in self._measures}

    def collect(self, state: SimulationState) -> None:
        """Record all scheduled sample points up to the current elapsed time.

        Args:
            state: The current simulation state.
        """
        if not self._measures or self._table is None:
            return

        t = state.elapsed_time
        idx = self._next_index

        # Fill scheduled times strictly before t using cached pending values.
        while self._frequency * idx < t:
            sched_t = self._frequency * idx
            row = {
                m.name: (
                    self._pending_values[m.name]
                    if self._pending_values[m.name] is not None
                    else m.function(state)
                )
                for m in self._measures
            }
            self._table.add_row(sched_t, row)
            idx += 1

        # Evaluate all measures once at the current state.
        current_values = {m.name: m.function(state) for m in self._measures}

        # Fill the scheduled time that coincides with t (if any).
        if self._frequency * idx <= t:
            sched_t = self._frequency * idx
            self._table.add_row(sched_t, current_values)
            idx += 1

        self._next_index = idx
        for m in self._measures:
            self._pending_values[m.name] = current_values[m.name]

    def finalize(self, state: SimulationState) -> None:
        """Fill remaining scheduled points up to ``max_time`` with *state*.

        Args:
            state: The final simulation state.
        """
        if not self._measures or self._table is None:
            return

        idx = self._next_index
        sched_t = self._frequency * idx
        while sched_t <= self._max_time:
            row = {m.name: m.function(state) for m in self._measures}
            self._table.add_row(sched_t, row)
            idx += 1
            sched_t = self._frequency * idx
        self._next_index = idx
