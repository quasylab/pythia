"""DataStatistics — per-label, per-time-point statistical summaries over many runs.

A :class:`DataStatistics` object accepts a collection of
:class:`~pythia.api.data_table.DataTable` objects — one per simulation run —
that share the same column labels and the same observation times.  It uses
NumPy to compute seven statistics for each ``(label, time)`` pair across all
runs and exposes the results as dedicated
:class:`~pythia.api.data_table.DataTable` objects (one per statistic).

Computed statistics:

* **mean** — arithmetic mean
* **std** — sample standard deviation (ddof = 1; 0 when only one run)
* **q1** — 25th percentile (first quartile)
* **q2** — 50th percentile (median / second quartile)
* **q3** — 75th percentile (third quartile)
* **min** — minimum value
* **max** — maximum value

Requires NumPy (``pip install 'pythia[visualization]'``).

Example::

    from pythia.api.data_table import DataTable
    from pythia.util.data_statistics import DataStatistics

    run1 = DataTable(["x"])
    run1.add_row(0.0, {"x": 1.0})
    run1.add_row(1.0, {"x": 3.0})

    run2 = DataTable(["x"])
    run2.add_row(0.0, {"x": 2.0})
    run2.add_row(1.0, {"x": 5.0})

    stats = DataStatistics([run1, run2])
    stats.mean.get(0.0)["x"]   # 1.5
    stats.min.get(1.0)["x"]    # 3.0
    stats.q2.get(0.0)["x"]     # 1.5
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DataStatistics requires numpy.  "
        "Install it with: pip install 'pythia[visualization]'"
    ) from exc

from pythia.api.data_table import DataTable


class DataStatistics:
    """Compute per-label, per-time-point statistics across a collection of runs.

    All :class:`~pythia.api.data_table.DataTable` objects in *data* must share
    identical column labels and identical observation times.

    Args:
        data: A non-empty list of :class:`~pythia.api.data_table.DataTable`
            objects with the same labels and time points.

    Raises:
        ValueError: If *data* is empty, or if the tables have different labels
            or different time points.

    Examples:
        >>> from pythia.api.data_table import DataTable
        >>> t1 = DataTable(["v"])
        >>> t1.add_row(0.0, {"v": 2.0})
        >>> t2 = DataTable(["v"])
        >>> t2.add_row(0.0, {"v": 4.0})
        >>> stats = DataStatistics([t1, t2])
        >>> stats.mean.get(0.0)["v"]
        3.0
        >>> stats.min.get(0.0)["v"]
        2.0
        >>> stats.max.get(0.0)["v"]
        4.0
    """

    def __init__(self, data: list[DataTable]) -> None:
        if not data:
            raise ValueError("data must contain at least one DataTable")

        ref_labels = data[0].labels
        ref_times = data[0].times
        for i, table in enumerate(data[1:], start=1):
            if table.labels != ref_labels:
                raise ValueError(
                    f"DataTable at index {i} has labels {table.labels!r}; "
                    f"expected {ref_labels!r}"
                )
            if table.times != ref_times:
                raise ValueError(
                    f"DataTable at index {i} has times {table.times!r}; "
                    f"expected {ref_times!r}"
                )

        self._labels: tuple[str, ...] = ref_labels
        self._times: tuple[float, ...] = ref_times
        self._n: int = len(data)

        # Allocate one output DataTable per statistic.
        labels_list = list(ref_labels)
        self._mean = DataTable(labels_list)
        self._std  = DataTable(labels_list)
        self._q1   = DataTable(labels_list)
        self._q2   = DataTable(labels_list)
        self._q3   = DataTable(labels_list)
        self._min  = DataTable(labels_list)
        self._max  = DataTable(labels_list)

        ddof = 1 if self._n > 1 else 0

        for t in ref_times:
            mean_row: dict[str, float] = {}
            std_row:  dict[str, float] = {}
            q1_row:   dict[str, float] = {}
            q2_row:   dict[str, float] = {}
            q3_row:   dict[str, float] = {}
            min_row:  dict[str, float] = {}
            max_row:  dict[str, float] = {}

            for label in ref_labels:
                values = np.array([table.get(t)[label] for table in data], dtype=float)
                q1, q2, q3 = np.percentile(values, [25.0, 50.0, 75.0])
                mean_row[label] = float(np.mean(values))
                std_row[label]  = float(np.std(values, ddof=ddof))
                q1_row[label]   = float(q1)
                q2_row[label]   = float(q2)
                q3_row[label]   = float(q3)
                min_row[label]  = float(np.min(values))
                max_row[label]  = float(np.max(values))

            self._mean.add_row(t, mean_row)
            self._std.add_row(t, std_row)
            self._q1.add_row(t, q1_row)
            self._q2.add_row(t, q2_row)
            self._q3.add_row(t, q3_row)
            self._min.add_row(t, min_row)
            self._max.add_row(t, max_row)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        """The column labels shared by all input tables.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["a", "b"])
            >>> t.add_row(0.0, {"a": 1.0, "b": 2.0})
            >>> DataStatistics([t]).labels
            ('a', 'b')
        """
        return self._labels

    @property
    def times(self) -> tuple[float, ...]:
        """The observation timestamps shared by all input tables.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 1.0})
            >>> t.add_row(1.0, {"v": 2.0})
            >>> DataStatistics([t]).times
            (0.0, 1.0)
        """
        return self._times

    @property
    def n(self) -> int:
        """Number of runs (input tables).

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 1.0})
            >>> DataStatistics([t, t]).n
            2
        """
        return self._n

    # ------------------------------------------------------------------
    # Statistics (each exposed as a DataTable)
    # ------------------------------------------------------------------

    @property
    def mean(self) -> DataTable:
        """Arithmetic mean across runs.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> DataStatistics([t1, t2]).mean.get(0.0)["v"]
            2.0
        """
        return self._mean

    @property
    def std(self) -> DataTable:
        """Sample standard deviation (ddof=1) across runs; 0.0 for a single run.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> round(DataStatistics([t1, t2]).std.get(0.0)["v"], 10)
            1.4142135624
        """
        return self._std

    @property
    def q1(self) -> DataTable:
        """25th percentile (first quartile) across runs."""
        return self._q1

    @property
    def q2(self) -> DataTable:
        """50th percentile (median / second quartile) across runs.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2, t3 = DataTable(["v"]), DataTable(["v"]), DataTable(["v"])
            >>> for t, v in [(t1, 1.0), (t2, 2.0), (t3, 3.0)]:
            ...     t.add_row(0.0, {"v": v})
            >>> DataStatistics([t1, t2, t3]).q2.get(0.0)["v"]
            2.0
        """
        return self._q2

    @property
    def q3(self) -> DataTable:
        """75th percentile (third quartile) across runs."""
        return self._q3

    @property
    def min(self) -> DataTable:
        """Minimum value across runs.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> DataStatistics([t1, t2]).min.get(0.0)["v"]
            1.0
        """
        return self._min

    @property
    def max(self) -> DataTable:
        """Maximum value across runs.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> DataStatistics([t1, t2]).max.get(0.0)["v"]
            3.0
        """
        return self._max

    def __repr__(self) -> str:
        return (
            f"DataStatistics(n={self._n}, labels={self._labels!r}, "
            f"times={len(self._times)} points)"
        )
