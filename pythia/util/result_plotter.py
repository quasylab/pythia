"""ResultPlotter — visualise multi-run simulation measurements with Matplotlib.

A :class:`ResultPlotter` accepts the :class:`~pythia.api.data_table.DataTable`
objects produced by one or more simulation runs and generates one Matplotlib
figure per distinct column label.

Two families of plots are available:

* **Raw runs** (:meth:`plot`, :meth:`plot_measure`) — one line per run.
* **Statistical summary** (:meth:`plot_statistics`,
  :meth:`plot_measure_statistics`) — mean line with a shaded interquartile
  range (Q1–Q3) band and a lighter shaded min–max band.  Statistics are
  computed by :class:`~pythia.util.data_statistics.DataStatistics` and cached
  on first access via the :attr:`statistics` property.

The expected data format is a :class:`list` of
:class:`~pythia.api.data_table.DataTable` objects — one per simulation run —
where the column labels are the measure names and each row is one time
observation.

Requires the ``matplotlib`` and ``numpy`` packages
(``pip install 'pythia[visualization]'``).

Example::

    from pythia.api.data_table import DataTable
    from pythia.util.result_plotter import ResultPlotter

    run1 = DataTable(["population"])
    run1.add_row(0.0, {"population": 100.0})
    run1.add_row(1.0, {"population": 110.0})

    run2 = DataTable(["population"])
    run2.add_row(0.0, {"population": 100.0})
    run2.add_row(1.0, {"population": 108.0})

    plotter = ResultPlotter([run1, run2])
    figs  = plotter.plot()                    # raw lines
    sfigs = plotter.plot_statistics()         # mean + IQR + min/max bands
"""

from __future__ import annotations

from pythia.api.data_table import DataTable

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ResultPlotter requires matplotlib.  "
        "Install it with: pip install 'pythia[visualization]'"
    ) from exc


class ResultPlotter:
    """Generate Matplotlib figures from multi-run simulation data.

    Args:
        data: A list of :class:`~pythia.api.data_table.DataTable` objects, one
            per simulation run.  Column labels are the measure names; rows are
            time observations.

    Raises:
        ValueError: If *data* is empty.

    Example:
        >>> from pythia.api.data_table import DataTable
        >>> t = DataTable(["x"])
        >>> t.add_row(0.0, {"x": 0.0})
        >>> t.add_row(1.0, {"x": 1.0})
        >>> plotter = ResultPlotter([t])
        >>> plotter.measures == frozenset({"x"})
        True
    """

    def __init__(self, data: list[DataTable]) -> None:
        if not data:
            raise ValueError("data must contain at least one run")
        self._data: list[DataTable] = list(data)
        self._statistics = None   # lazily computed by the statistics property

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def measures(self) -> frozenset[str]:
        """All measure names (column labels) present across every run.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["a", "b"])
            >>> t.add_row(0.0, {"a": 1.0, "b": 2.0})
            >>> ResultPlotter([t]).measures == frozenset({"a", "b"})
            True
        """
        result: set[str] = set()
        for table in self._data:
            result.update(table.labels)
        return frozenset(result)

    @property
    def statistics(self):
        """A :class:`~pythia.util.data_statistics.DataStatistics` built from
        the run data, computed on first access and cached.

        All tables must share the same labels and time points; otherwise
        :class:`~pythia.util.data_statistics.DataStatistics` raises a
        :exc:`ValueError`.

        Returns:
            :class:`~pythia.util.data_statistics.DataStatistics`
        """
        if self._statistics is None:
            from pythia.util.data_statistics import DataStatistics
            self._statistics = DataStatistics(self._data)
        return self._statistics

    # ------------------------------------------------------------------
    # Raw-run plotting
    # ------------------------------------------------------------------

    def plot(self) -> list[Figure]:
        """Generate one raw-run figure per measure.

        Each figure contains one labelled line per run.  Figures are returned
        in alphabetical order of measure name.  Each figure is displayed
        immediately via :func:`matplotlib.pyplot.show`.

        Returns:
            A list of :class:`matplotlib.figure.Figure` objects, one per
            measure, in alphabetical order.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 1.0})
            >>> t.add_row(1.0, {"v": 2.0})
            >>> figs = ResultPlotter([t]).plot()
            >>> len(figs)
            1
        """
        return [self.plot_measure(m) for m in sorted(self.measures)]

    def plot_measure(self, measure: str) -> Figure:
        """Generate a raw-run figure for a single *measure*.

        Args:
            measure: The column label to plot.

        Returns:
            A :class:`matplotlib.figure.Figure` with one line per run.

        Raises:
            KeyError: If *measure* is not present in any run.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["speed"])
            >>> t.add_row(0.0, {"speed": 0.0})
            >>> t.add_row(1.0, {"speed": 9.8})
            >>> fig = ResultPlotter([t]).plot_measure("speed")
            >>> fig.axes[0].get_title()
            'speed'
        """
        if measure not in self.measures:
            raise KeyError(measure)

        fig, ax = plt.subplots()
        n_runs = len(self._data)
        label_width = len(str(n_runs))

        for idx, table in enumerate(self._data):
            if measure not in table.labels:
                continue
            times = list(table.times)
            values = [table.get(t)[measure] for t in times]
            ax.plot(times, values, label=f"run {idx + 1:0{label_width}d}")

        ax.set_title(measure)
        ax.set_xlabel("time")
        ax.set_ylabel(measure)
        if n_runs > 1:
            ax.legend()

        plt.show()
        return fig

    # ------------------------------------------------------------------
    # Statistical plotting
    # ------------------------------------------------------------------

    def plot_statistics(self) -> list[Figure]:
        """Generate one statistical-summary figure per measure.

        Each figure shows the **mean** as a solid line, the interquartile
        range (Q1–Q3) as a shaded band, and the full min–max range as a
        lighter shaded band.  Figures are returned in alphabetical order of
        measure name.  Each figure is displayed immediately via
        :func:`matplotlib.pyplot.show`.

        All tables must share the same labels and time points so that
        :attr:`statistics` can be computed.

        Returns:
            A list of :class:`matplotlib.figure.Figure` objects, one per
            measure, in alphabetical order.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> figs = ResultPlotter([t1, t2]).plot_statistics()
            >>> len(figs)
            1
        """
        return [self.plot_measure_statistics(m) for m in sorted(self.measures)]

    def plot_measure_statistics(self, measure: str) -> Figure:
        """Generate a statistical-summary figure for a single *measure*.

        The figure contains:

        * **Mean** — solid line.
        * **Q1–Q3 band** — shaded area between the first and third quartiles
          (interquartile range), opacity 0.35.
        * **Min–Max band** — shaded area between the overall minimum and
          maximum, opacity 0.15.

        Args:
            measure: The column label to plot.

        Returns:
            A :class:`matplotlib.figure.Figure` summarising the distribution
            of *measure* across all runs.

        Raises:
            KeyError: If *measure* is not present in the statistics.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> t1, t2 = DataTable(["v"]), DataTable(["v"])
            >>> t1.add_row(0.0, {"v": 1.0}); t2.add_row(0.0, {"v": 3.0})
            >>> fig = ResultPlotter([t1, t2]).plot_measure_statistics("v")
            >>> fig.axes[0].get_title()
            'v'
        """
        stats = self.statistics
        if measure not in frozenset(stats.labels):
            raise KeyError(measure)

        times = list(stats.times)
        mean_vals = [stats.mean.get(t)[measure] for t in times]
        q1_vals   = [stats.q1.get(t)[measure]   for t in times]
        q3_vals   = [stats.q3.get(t)[measure]   for t in times]
        min_vals  = [stats.min.get(t)[measure]  for t in times]
        max_vals  = [stats.max.get(t)[measure]  for t in times]

        fig, ax = plt.subplots()

        ax.fill_between(times, min_vals, max_vals,
                        alpha=0.15, label="min–max")
        ax.fill_between(times, q1_vals, q3_vals,
                        alpha=0.35, label="Q1–Q3")
        ax.plot(times, mean_vals, label="mean")

        ax.set_title(measure)
        ax.set_xlabel("time")
        ax.set_ylabel(measure)
        ax.legend()

        plt.show()
        return fig
