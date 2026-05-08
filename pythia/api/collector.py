"""DataCollector — time-triggered recording of scalar measures.

A :class:`DataCollector` holds a set of :class:`~pythia.api.measure.Measure`
objects and records them at uniformly-spaced simulation-time points.  The
collection follows a **zero-order hold** model: the value at any scheduled
time point is the measure applied to the most recent simulation state that
was available *at or before* that time.

For example, if a simulation produces states at t=0 (``s1``), t=1.5 (``s2``),
and t=2.3 (``s3``, done) with ``max_time=3.0``, and a measure is collected
every 0.5 time units, the resulting records are::

    (0.0, m(s1)), (0.5, m(s1)), (1.0, m(s1)),
    (1.5, m(s2)), (2.0, m(s2)),
    (2.5, m(s3)), (3.0, m(s3))

Timestamps are exact multiples of :attr:`~pythia.api.measure.Measure.frequency`
computed as ``k * frequency`` (integer arithmetic avoids floating-point
drift).  The final record is always at the scheduled point ``<=  max_time``.

Usage with :class:`~pythia.api.runner.SimulationRunner`::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.api.collector import DataCollector
    from pythia.api.measure import Measure

    collector = DataCollector([
        Measure("wolves",
                lambda s: s.data.get("pm.counting_state",
                                     CountingState({})).counting_vector.get(0, 0),
                frequency=0.25),
    ])

    config = SimulationConfig(max_time=10.0, random_seed=0)
    runner = SimulationRunner(model, config, data_collector=collector)
    runner.run()

    for t, v in collector.records["wolves"]:
        print(f"t={t:.2f}  wolves={v}")
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping, Sequence

from pythia.api.measure import Measure
from pythia.api.state import SimulationState


class DataCollector:
    """Collects scalar measurements at uniform simulation-time intervals.

    After each simulation step the runner calls :meth:`collect`; the
    collector fills in all scheduled sample points up to the current
    elapsed time using a zero-order hold — the state that was current
    *at or before* each sample point.  Retroactive sample points (between
    the previous and current step) use the snapshot of the previous state;
    a sample point that coincides exactly with the current step time uses
    the current (post-transition) state.

    The runner also calls :meth:`collect` once before the loop with the
    initial state at ``t=0``, and :meth:`finalize` after the loop to fill
    any remaining scheduled points up to ``max_time``.

    Args:
        measures: The :class:`~pythia.api.measure.Measure` objects to record.
            Every measure must have a unique :attr:`~pythia.api.measure.Measure.name`.

    Raises:
        ValueError: If two measures share the same name.

    Example:
        >>> from pythia.api.state import SimulationState
        >>> from pythia.api.measure import Measure
        >>> m = Measure("x", lambda s: s.elapsed_time, frequency=1.0)
        >>> collector = DataCollector([m])
        >>> collector.reset(max_time=3.0)
        >>> collector.collect(SimulationState(elapsed_time=0.0))
        >>> collector.collect(SimulationState(elapsed_time=2.5))
        >>> collector.finalize(SimulationState(elapsed_time=2.5))
        >>> [t for t, _ in collector.records["x"]]
        [0.0, 1.0, 2.0, 2.5, 3.0]
    """

    def __init__(self, measures: Iterable[Measure]) -> None:
        """Initialise the collector with the given measures."""
        self._measures: tuple[Measure, ...] = tuple(measures)
        seen: set[str] = set()
        for m in self._measures:
            if m.name in seen:
                raise ValueError(
                    f"DataCollector: duplicate measure name {m.name!r}"
                )
            seen.add(m.name)
        self._records: dict[str, list[tuple[float, float]]] = {
            m.name: [] for m in self._measures
        }
        self._next_index: dict[str, int] = {m.name: 0 for m in self._measures}
        # Cached float value from the previous collect() call, used for
        # retroactive filling; avoids deep-copying the state object.
        self._pending_value: dict[str, float | None] = {m.name: None for m in self._measures}
        self._max_time: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def measures(self) -> tuple[Measure, ...]:
        """The registered measures, in registration order."""
        return self._measures

    @property
    def records(self) -> Mapping[str, Sequence[tuple[float, float]]]:
        """Immutable view of all collected ``(scheduled_time, value)`` pairs.

        The outer mapping is keyed by measure name.  Each value is a sequence
        of ``(scheduled_time, value)`` tuples in chronological order.  The
        *scheduled_time* is an exact multiple of the measure's
        :attr:`~pythia.api.measure.Measure.frequency`, not the actual step
        time.  The returned sequences are copies; mutating them does not
        affect the collector's internal state.
        """
        return types.MappingProxyType(
            {name: list(samples) for name, samples in self._records.items()}
        )

    def reset(self, max_time: float) -> None:
        """Clear all records and prepare for a new run.

        Called automatically by :class:`~pythia.api.runner.SimulationRunner`
        at the start of each run.  *max_time* is used by :meth:`finalize` to
        determine how far to extend records after the simulation ends.

        Args:
            max_time: The simulation time horizon (``config.max_time``).
        """
        self._max_time = max_time
        self._records = {m.name: [] for m in self._measures}
        self._next_index = {m.name: 0 for m in self._measures}
        self._pending_value = {m.name: None for m in self._measures}

    def collect(self, state: SimulationState) -> None:
        """Record all scheduled sample points up to the current elapsed time.

        For each measure, fills every scheduled point ``k * frequency`` where
        ``k * frequency <= state.elapsed_time``:

        - Points strictly before ``state.elapsed_time`` use the value cached
          from the *previous* call to :meth:`collect` (zero-order hold from
          the last step).  If no previous value exists (first call at t=0),
          the current state is evaluated instead.
        - The point that coincides exactly with ``state.elapsed_time`` uses
          the current (post-transition) state.

        The measure function is evaluated at most **once** per call (for the
        current state); retroactive filling reuses the cached float from the
        prior call, avoiding any need to copy or re-evaluate past state.

        Args:
            state: The current simulation state.
        """
        t = state.elapsed_time
        for measure in self._measures:
            idx = self._next_index[measure.name]
            pending = self._pending_value[measure.name]

            # Fill scheduled times strictly before t using the cached value
            # from the previous step (zero-order hold).
            while measure.frequency * idx < t:
                sched_t = measure.frequency * idx
                # On the very first call pending is None; fall back to current.
                value = pending if pending is not None else measure.function(state)
                self._records[measure.name].append((sched_t, value))
                idx += 1

            # Fill the scheduled time at t (if t falls on a grid point) using
            # the current (new) state.
            current_value = measure.function(state)
            if measure.frequency * idx <= t:
                sched_t = measure.frequency * idx
                self._records[measure.name].append((sched_t, current_value))
                idx += 1

            self._next_index[measure.name] = idx
            # Cache the current value for retroactive use in the next call.
            self._pending_value[measure.name] = current_value

    def finalize(self, state: SimulationState) -> None:
        """Fill remaining scheduled points up to ``max_time`` with *state*.

        Called by the runner after the loop ends (whether because ``dt = 0``
        or ``elapsed_time >= max_time``).  Any scheduled points between the
        last collected time and ``max_time`` (inclusive) are recorded using
        the final simulation state.

        Args:
            state: The final simulation state.
        """
        for measure in self._measures:
            idx = self._next_index[measure.name]
            sched_t = measure.frequency * idx
            while sched_t <= self._max_time:
                self._records[measure.name].append((sched_t, measure.function(state)))
                idx += 1
                sched_t = measure.frequency * idx
            self._next_index[measure.name] = idx
