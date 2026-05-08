"""ResultPlotter — visualise multi-run simulation measurements with Matplotlib.

A :class:`ResultPlotter` accepts the :class:`~pythia.api.data_table.DataTable`
objects produced by one or more simulation runs and generates one Matplotlib
figure per distinct column label.  Each run contributes one labelled line to
the figure for that measure.

The expected data format is a :class:`list` of
:class:`~pythia.api.data_table.DataTable` objects — one per simulation run —
where the column labels are the measure names and each row is one time
observation.

Requires the ``matplotlib`` package (install with
``pip install pythia[visualization]``).

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
    figs = plotter.plot()          # one Figure per measure
    figs[0].savefig("population.png")
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
    """Generate one Matplotlib figure per measure from multi-run simulation data.

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

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self) -> list[Figure]:
        """Generate one figure per measure and return the list of figures.

        Each figure contains one labelled line per run.  Figures are returned
        in alphabetical order of measure name and are *not* shown
        automatically; call ``plt.show()`` or save them as needed.

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
        """Generate a figure for a single *measure*.

        Args:
            measure: The column label to plot.

        Returns:
            A :class:`matplotlib.figure.Figure` with one line per run whose
            table contains *measure* as a label.  Times come from
            :attr:`~pythia.api.data_table.DataTable.times` and are already
            in ascending order.

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
            label = f"run {idx + 1:0{label_width}d}"
            ax.plot(times, values, label=label)

        ax.set_title(measure)
        ax.set_xlabel("time")
        ax.set_ylabel(measure)
        if n_runs > 1:
            ax.legend()

        return fig
