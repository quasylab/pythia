"""Tests for pythia.util.result_plotter."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import pytest
from matplotlib.figure import Figure

from pythia.api.data_table import DataTable
from pythia.util.result_plotter import ResultPlotter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _table(labels: list[str], rows: dict[float, dict[str, float]]) -> DataTable:
    t = DataTable(labels)
    for time, values in rows.items():
        t.add_row(time, values)
    return t


ONE_RUN: list[DataTable] = [
    _table(["speed"], {0.0: {"speed": 0.0}, 1.0: {"speed": 9.8}, 2.0: {"speed": 19.6}}),
]

TWO_RUNS: list[DataTable] = [
    _table(["speed", "height"], {
        0.0: {"speed": 0.0, "height": 100.0},
        1.0: {"speed": 9.8, "height": 95.1},
    }),
    _table(["speed", "height"], {
        0.0: {"speed": 0.0, "height": 100.0},
        1.0: {"speed": 8.9, "height": 96.0},
    }),
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_data_accepted(self) -> None:
        ResultPlotter(ONE_RUN)  # should not raise

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError):
            ResultPlotter([])

    def test_data_stored(self) -> None:
        p = ResultPlotter(ONE_RUN)
        assert len(p._data) == 1

    def test_accepts_list_of_data_tables(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 1.0})
        ResultPlotter([t])  # should not raise


# ---------------------------------------------------------------------------
# measures property
# ---------------------------------------------------------------------------

class TestMeasures:
    def test_single_measure(self) -> None:
        p = ResultPlotter(ONE_RUN)
        assert p.measures == frozenset({"speed"})

    def test_multiple_measures_from_one_run(self) -> None:
        p = ResultPlotter(TWO_RUNS)
        assert p.measures == frozenset({"speed", "height"})

    def test_measures_union_across_runs(self) -> None:
        t1 = _table(["a"], {0.0: {"a": 1.0}})
        t2 = _table(["b"], {0.0: {"b": 2.0}})
        p = ResultPlotter([t1, t2])
        assert p.measures == frozenset({"a", "b"})

    def test_returns_frozenset(self) -> None:
        assert isinstance(ResultPlotter(ONE_RUN).measures, frozenset)


# ---------------------------------------------------------------------------
# plot_measure
# ---------------------------------------------------------------------------

class TestPlotMeasure:
    def test_returns_figure(self) -> None:
        fig = ResultPlotter(ONE_RUN).plot_measure("speed")
        assert isinstance(fig, Figure)

    def test_figure_has_one_axes(self) -> None:
        fig = ResultPlotter(ONE_RUN).plot_measure("speed")
        assert len(fig.axes) == 1

    def test_title_is_measure_name(self) -> None:
        fig = ResultPlotter(ONE_RUN).plot_measure("speed")
        assert fig.axes[0].get_title() == "speed"

    def test_xlabel_is_time(self) -> None:
        fig = ResultPlotter(ONE_RUN).plot_measure("speed")
        assert fig.axes[0].get_xlabel() == "time"

    def test_ylabel_is_measure_name(self) -> None:
        fig = ResultPlotter(ONE_RUN).plot_measure("speed")
        assert fig.axes[0].get_ylabel() == "speed"

    def test_one_line_per_run(self) -> None:
        fig = ResultPlotter(TWO_RUNS).plot_measure("speed")
        assert len(fig.axes[0].lines) == 2

    def test_times_in_ascending_order(self) -> None:
        t = _table(["v"], {2.0: {"v": 20.0}, 0.0: {"v": 0.0}, 1.0: {"v": 10.0}})
        fig = ResultPlotter([t]).plot_measure("v")
        x_data = list(fig.axes[0].lines[0].get_xdata())
        assert x_data == [0.0, 1.0, 2.0]

    def test_unknown_measure_raises(self) -> None:
        with pytest.raises(KeyError):
            ResultPlotter(ONE_RUN).plot_measure("missing")

    def test_runs_missing_measure_skipped(self) -> None:
        t1 = _table(["a"], {0.0: {"a": 1.0}})
        t2 = _table(["b"], {0.0: {"b": 2.0}})
        fig = ResultPlotter([t1, t2]).plot_measure("a")
        assert len(fig.axes[0].lines) == 1

    def test_values_read_correctly(self) -> None:
        t = _table(["x"], {0.0: {"x": 5.0}, 1.0: {"x": 10.0}})
        fig = ResultPlotter([t]).plot_measure("x")
        y_data = list(fig.axes[0].lines[0].get_ydata())
        assert y_data == pytest.approx([5.0, 10.0])


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

class TestPlot:
    def test_returns_list_of_figures(self) -> None:
        figs = ResultPlotter(ONE_RUN).plot()
        assert isinstance(figs, list)
        assert all(isinstance(f, Figure) for f in figs)

    def test_one_figure_per_measure(self) -> None:
        figs = ResultPlotter(TWO_RUNS).plot()
        assert len(figs) == len(ResultPlotter(TWO_RUNS).measures)

    def test_figures_in_alphabetical_order(self) -> None:
        t = _table(["z", "a", "m"], {
            0.0: {"z": 1.0, "a": 2.0, "m": 3.0}
        })
        figs = ResultPlotter([t]).plot()
        titles = [f.axes[0].get_title() for f in figs]
        assert titles == sorted(titles)

    def test_single_measure_single_figure(self) -> None:
        figs = ResultPlotter(ONE_RUN).plot()
        assert len(figs) == 1
        assert figs[0].axes[0].get_title() == "speed"
