"""DataTable — a time-indexed labelled float data store.

A :class:`DataTable` associates an ordered sequence of time observations with
labelled float data.  The set of labels is fixed at construction time; new
rows are added one at a time by supplying a timestamp and a complete mapping
from every label to a float value.

Example::

    from pythia.api.data_table import DataTable

    table = DataTable(["x", "y"])
    table.add_row(0.0, {"x": 1.0, "y": 2.0})
    table.add_row(1.0, {"x": 3.0, "y": 4.0})

    table.times           # (0.0, 1.0)
    table.get(0.0)        # {"x": 1.0, "y": 2.0}
    table.get(1.0)["y"]   # 4.0
"""

from __future__ import annotations

from collections.abc import Iterable


class DataTable:
    """A time-indexed table with a fixed set of column labels storing float values.

    Every value stored in the table must be a :class:`float`.  Integer values
    are automatically promoted to float; booleans and all other types are
    rejected.

    Rows are stored internally and returned in ascending time order.  Adding a
    row at a time that already exists overwrites the previous entry for that
    timestamp.

    Args:
        labels: The column labels for this table.  Must be non-empty and
            contain no duplicates.

    Raises:
        ValueError: If *labels* is empty or contains duplicate entries.

    Example:
        >>> table = DataTable(["a", "b"])
        >>> table.add_row(0.0, {"a": 1.0, "b": 2.0})
        >>> table.add_row(1.0, {"a": 3.0, "b": 4.0})
        >>> table.times
        (0.0, 1.0)
        >>> table.get(0.0)["b"]
        2.0
    """

    def __init__(self, labels: Iterable[str]) -> None:
        _labels: tuple[str, ...] = tuple(labels)
        if not _labels:
            raise ValueError("DataTable must have at least one label")
        if len(set(_labels)) != len(_labels):
            duplicates = [l for l in _labels if _labels.count(l) > 1]
            raise ValueError(
                f"Labels must be unique; duplicates found: {sorted(set(duplicates))!r}"
            )
        self._labels: tuple[str, ...] = _labels
        self._rows: dict[float, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        """The column labels in the order they were supplied.

        Example:
            >>> DataTable(["x", "y"]).labels
            ('x', 'y')
        """
        return self._labels

    @property
    def times(self) -> tuple[float, ...]:
        """The observation timestamps in ascending order.

        Example:
            >>> table = DataTable(["v"])
            >>> table.add_row(2.0, {"v": 0.0})
            >>> table.add_row(0.5, {"v": 1.0})
            >>> table.times
            (0.5, 2.0)
        """
        return tuple(sorted(self._rows))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_row(self, time: float, values: dict[str, float]) -> None:
        """Add or replace the observation at *time*.

        *values* must contain exactly one float entry for every label in this
        table — no more, no less.  Integer values are automatically promoted
        to float; booleans and all other types raise :exc:`TypeError`.

        Args:
            time: The timestamp for this observation.
            values: A mapping from every label to its float value.

        Raises:
            ValueError: If *values* does not contain exactly the set of labels
                defined for this table.
            TypeError: If any value is not a :class:`float` or :class:`int`
                (booleans are rejected).

        Example:
            >>> table = DataTable(["x"])
            >>> table.add_row(1.0, {"x": 42.0})
            >>> table.get(1.0)["x"]
            42.0
        """
        given = set(values)
        expected = set(self._labels)
        if given != expected:
            extra = sorted(given - expected)
            missing = sorted(expected - given)
            parts: list[str] = []
            if extra:
                parts.append(f"unexpected labels: {extra!r}")
            if missing:
                parts.append(f"missing labels: {missing!r}")
            raise ValueError("; ".join(parts))

        row: dict[str, float] = {}
        for label, val in values.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"Values must be float; got {type(val).__name__!r}"
                    f" for label {label!r}"
                )
            row[label] = float(val)

        self._rows[time] = row

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, time: float) -> dict[str, float]:
        """Return a copy of the row at *time*.

        Args:
            time: The timestamp to look up.

        Returns:
            A dictionary mapping every label to its float value at *time*.

        Raises:
            KeyError: If no row has been recorded at *time*.

        Example:
            >>> table = DataTable(["a", "b"])
            >>> table.add_row(0.0, {"a": 1.0, "b": 2.0})
            >>> table.get(0.0) == {"a": 1.0, "b": 2.0}
            True
        """
        if time not in self._rows:
            raise KeyError(time)
        return dict(self._rows[time])

    def __len__(self) -> int:
        """Return the number of recorded observations.

        Example:
            >>> table = DataTable(["v"])
            >>> len(table)
            0
            >>> table.add_row(0.0, {"v": 1.0})
            >>> len(table)
            1
        """
        return len(self._rows)

    def __repr__(self) -> str:
        return f"DataTable(labels={self._labels!r}, rows={len(self._rows)})"
