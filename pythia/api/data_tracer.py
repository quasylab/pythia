"""DataTracer — collect labelled float data from a running simulation.

A :class:`DataTracer` wraps a user-supplied *trace function* that extracts
named, labelled measurements from a :class:`~pythia.api.state.SimulationState`
at each call to :meth:`~DataTracer.trace`.  The measurements are stored in one
:class:`~pythia.api.data_table.DataTable` per named object, indexed by the
elapsed simulation time.

Example::

    from pythia.api.data_tracer import DataTracer
    from pythia.api.state import SimulationState

    tracer = DataTracer(
        labels=["position", "velocity"],
        trace_function=lambda state: {
            "agent_0": {
                "position": state.data["pos"],
                "velocity": state.data["vel"],
            }
        },
    )

    # Inside a simulation loop:
    tracer.trace(state)

    table = tracer.get_table("agent_0")
    table.times          # (0.0, 1.0, ...)
    table.get(0.0)       # {"position": ..., "velocity": ...}
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pythia.api.data_table import DataTable
from pythia.api.state import SimulationState


class DataTracer:
    """Collect labelled float measurements from a simulation into :class:`DataTable` objects.

    One :class:`~pythia.api.data_table.DataTable` is maintained per named
    object returned by the trace function.  Tables are created lazily on first
    encounter and share the same set of column labels.

    The two configuration fields — :attr:`labels` and :attr:`trace_function` —
    are set at construction time and cannot be reassigned.

    Args:
        labels: Column labels shared by every :class:`~pythia.api.data_table.DataTable`
            produced by this tracer.  Must be non-empty and contain no
            duplicates (constraints enforced by :class:`~pythia.api.data_table.DataTable`).
        trace_function: A callable ``SimulationState -> dict[str, dict[str, float]]``
            that maps object names to label-value dictionaries.  An object
            may be absent from the returned map on any given call (its table
            simply receives no row for that timestamp).

    Raises:
        AttributeError: On any attempt to reassign :attr:`labels` or
            :attr:`trace_function` after construction.

    Examples:
        >>> from pythia.api.state import SimulationState
        >>> tracer = DataTracer(
        ...     labels=["x"],
        ...     trace_function=lambda s: {"obj": {"x": float(s.step)}},
        ... )
        >>> state = SimulationState(step=0, elapsed_time=0.0)
        >>> tracer.trace(state)
        >>> tracer.get_table("obj").get(0.0)
        {'x': 0.0}
    """

    def __init__(
        self,
        labels: Iterable[str],
        trace_function: Callable[[SimulationState], dict[str, dict[str, float]]],
    ) -> None:
        object.__setattr__(self, "_labels", tuple(labels))
        object.__setattr__(self, "_trace_function", trace_function)
        object.__setattr__(self, "_tables", {})

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DataTracer is immutable")

    # ------------------------------------------------------------------
    # Immutable configuration properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        """The column labels shared by all data tables.

        Examples:
            >>> DataTracer(["a", "b"], lambda s: {}).labels
            ('a', 'b')
        """
        return self._labels  # type: ignore[return-value]

    @property
    def trace_function(
        self,
    ) -> Callable[[SimulationState], dict[str, dict[str, float]]]:
        """The function used to extract measurements from a simulation state."""
        return self._trace_function  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def trace(self, state: SimulationState) -> None:
        """Record one observation from *state* into the relevant data tables.

        Calls :attr:`trace_function` with *state*, then for each
        ``(name, values)`` pair in the result, appends a row at
        ``state.elapsed_time`` to the :class:`~pythia.api.data_table.DataTable`
        for *name* (creating it if necessary).

        Args:
            state: The current simulation state.  ``state.elapsed_time`` is
                used as the row timestamp.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> tracer = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
            >>> s = SimulationState(elapsed_time=2.5)
            >>> tracer.trace(s)
            >>> tracer.get_table("obj").get(2.5)
            {'v': 2.5}
        """
        tables: dict[str, DataTable] = self._tables  # type: ignore[assignment]
        result = self._trace_function(state)  # type: ignore[misc]
        for name, values in result.items():
            if name not in tables:
                tables[name] = DataTable(self._labels)  # type: ignore[arg-type]
            tables[name].add_row(state.elapsed_time, values)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def names(self) -> frozenset[str]:
        """The names of all objects for which data has been collected.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> tracer = DataTracer(["x"], lambda s: {"a": {"x": 1.0}, "b": {"x": 2.0}})
            >>> tracer.trace(SimulationState())
            >>> tracer.names == frozenset({"a", "b"})
            True
        """
        return frozenset(self._tables)  # type: ignore[arg-type]

    def get_table(self, name: str) -> DataTable:
        """Return the :class:`~pythia.api.data_table.DataTable` for *name*.

        Args:
            name: The object name whose table is requested.

        Returns:
            The :class:`~pythia.api.data_table.DataTable` holding all
            observations recorded for *name*.

        Raises:
            KeyError: If no data has been collected for *name*.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> tracer = DataTracer(["v"], lambda s: {"obj": {"v": 9.8}})
            >>> tracer.trace(SimulationState())
            >>> tracer.get_table("obj").get(0.0)
            {'v': 9.8}
        """
        tables: dict[str, DataTable] = self._tables  # type: ignore[assignment]
        if name not in tables:
            raise KeyError(name)
        return tables[name]

    @property
    def tables(self) -> dict[str, DataTable]:
        """A shallow copy of the internal ``name -> DataTable`` mapping.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> tracer = DataTracer(["x"], lambda s: {"p": {"x": 0.0}})
            >>> tracer.trace(SimulationState())
            >>> "p" in tracer.tables
            True
        """
        return dict(self._tables)  # type: ignore[arg-type]

    def times(self, name: str) -> tuple[float, ...]:
        """Return the observation timestamps recorded for *name* in ascending order.

        Args:
            name: The object name whose timestamps are requested.

        Returns:
            A sorted tuple of floats.

        Raises:
            KeyError: If no data has been collected for *name*.

        Examples:
            >>> from pythia.api.state import SimulationState
            >>> tracer = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
            >>> tracer.trace(SimulationState(elapsed_time=1.0))
            >>> tracer.trace(SimulationState(elapsed_time=3.0))
            >>> tracer.times("obj")
            (1.0, 3.0)
        """
        return self.get_table(name).times

    def __repr__(self) -> str:
        tables: dict = self._tables  # type: ignore[assignment]
        return (
            f"DataTracer(labels={self._labels!r}, "  # type: ignore[attr-defined]
            f"names={sorted(tables.keys())!r})"
        )
