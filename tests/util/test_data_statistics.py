"""Tests for pythia.util.data_statistics."""

from __future__ import annotations

import math

import pytest

from pythia.api.data_table import DataTable
from pythia.util.data_statistics import DataStatistics


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _table(labels: list[str], rows: dict[float, dict[str, float]]) -> DataTable:
    t = DataTable(labels)
    for time, values in rows.items():
        t.add_row(time, values)
    return t


def _single(label: str, t: float, v: float) -> DataTable:
    return _table([label], {t: {label: v}})


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_table_accepted(self) -> None:
        t = _single("x", 0.0, 1.0)
        DataStatistics([t])  # no raise

    def test_multiple_tables_accepted(self) -> None:
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 0.0, 2.0)
        DataStatistics([t1, t2])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DataStatistics([])

    def test_mismatched_labels_raises(self) -> None:
        t1 = _table(["a"], {0.0: {"a": 1.0}})
        t2 = _table(["b"], {0.0: {"b": 1.0}})
        with pytest.raises(ValueError, match="labels"):
            DataStatistics([t1, t2])

    def test_mismatched_times_raises(self) -> None:
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 1.0, 2.0)
        with pytest.raises(ValueError, match="times"):
            DataStatistics([t1, t2])


# ---------------------------------------------------------------------------
# Metadata properties
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_labels(self) -> None:
        t = _table(["a", "b"], {0.0: {"a": 1.0, "b": 2.0}})
        assert DataStatistics([t]).labels == ("a", "b")

    def test_times(self) -> None:
        t = _table(["v"], {0.0: {"v": 1.0}, 1.0: {"v": 2.0}})
        assert DataStatistics([t]).times == (0.0, 1.0)

    def test_n_single(self) -> None:
        t = _single("x", 0.0, 1.0)
        assert DataStatistics([t]).n == 1

    def test_n_multiple(self) -> None:
        t = _single("x", 0.0, 1.0)
        assert DataStatistics([t, t, t]).n == 3


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_single_run(self) -> None:
        t = _single("x", 0.0, 5.0)
        assert DataStatistics([t]).mean.get(0.0)["x"] == pytest.approx(5.0)

    def test_two_runs(self) -> None:
        t1 = _single("x", 0.0, 2.0)
        t2 = _single("x", 0.0, 4.0)
        assert DataStatistics([t1, t2]).mean.get(0.0)["x"] == pytest.approx(3.0)

    def test_multiple_times(self) -> None:
        t1 = _table(["v"], {0.0: {"v": 1.0}, 1.0: {"v": 3.0}})
        t2 = _table(["v"], {0.0: {"v": 3.0}, 1.0: {"v": 5.0}})
        stats = DataStatistics([t1, t2])
        assert stats.mean.get(0.0)["v"] == pytest.approx(2.0)
        assert stats.mean.get(1.0)["v"] == pytest.approx(4.0)

    def test_multiple_labels(self) -> None:
        t1 = _table(["a", "b"], {0.0: {"a": 1.0, "b": 10.0}})
        t2 = _table(["a", "b"], {0.0: {"a": 3.0, "b": 20.0}})
        stats = DataStatistics([t1, t2])
        assert stats.mean.get(0.0)["a"] == pytest.approx(2.0)
        assert stats.mean.get(0.0)["b"] == pytest.approx(15.0)

    def test_returns_data_table(self) -> None:
        assert isinstance(DataStatistics([_single("x", 0.0, 1.0)]).mean, DataTable)


# ---------------------------------------------------------------------------
# std
# ---------------------------------------------------------------------------

class TestStd:
    def test_single_run_is_zero(self) -> None:
        t = _single("x", 0.0, 5.0)
        assert DataStatistics([t]).std.get(0.0)["x"] == pytest.approx(0.0)

    def test_two_identical_runs_is_zero(self) -> None:
        t = _single("x", 0.0, 3.0)
        assert DataStatistics([t, t]).std.get(0.0)["x"] == pytest.approx(0.0)

    def test_known_std(self) -> None:
        # values [1, 2, 3] → sample std = 1.0
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 0.0, 2.0)
        t3 = _single("x", 0.0, 3.0)
        assert DataStatistics([t1, t2, t3]).std.get(0.0)["x"] == pytest.approx(1.0)

    def test_returns_data_table(self) -> None:
        assert isinstance(DataStatistics([_single("x", 0.0, 1.0)]).std, DataTable)


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------

class TestMinMax:
    def test_min_single(self) -> None:
        t = _single("x", 0.0, 7.0)
        assert DataStatistics([t]).min.get(0.0)["x"] == pytest.approx(7.0)

    def test_max_single(self) -> None:
        t = _single("x", 0.0, 7.0)
        assert DataStatistics([t]).max.get(0.0)["x"] == pytest.approx(7.0)

    def test_min_multiple(self) -> None:
        t1 = _single("x", 0.0, 3.0)
        t2 = _single("x", 0.0, 1.0)
        t3 = _single("x", 0.0, 2.0)
        assert DataStatistics([t1, t2, t3]).min.get(0.0)["x"] == pytest.approx(1.0)

    def test_max_multiple(self) -> None:
        t1 = _single("x", 0.0, 3.0)
        t2 = _single("x", 0.0, 1.0)
        t3 = _single("x", 0.0, 2.0)
        assert DataStatistics([t1, t2, t3]).max.get(0.0)["x"] == pytest.approx(3.0)

    def test_min_leq_mean_leq_max(self) -> None:
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 0.0, 5.0)
        t3 = _single("x", 0.0, 9.0)
        stats = DataStatistics([t1, t2, t3])
        mn = stats.min.get(0.0)["x"]
        me = stats.mean.get(0.0)["x"]
        mx = stats.max.get(0.0)["x"]
        assert mn <= me <= mx


# ---------------------------------------------------------------------------
# quartiles
# ---------------------------------------------------------------------------

class TestQuartiles:
    def test_q2_is_median_odd(self) -> None:
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 0.0, 2.0)
        t3 = _single("x", 0.0, 3.0)
        assert DataStatistics([t1, t2, t3]).q2.get(0.0)["x"] == pytest.approx(2.0)

    def test_q2_is_median_even(self) -> None:
        t1 = _single("x", 0.0, 1.0)
        t2 = _single("x", 0.0, 3.0)
        assert DataStatistics([t1, t2]).q2.get(0.0)["x"] == pytest.approx(2.0)

    def test_q1_leq_q2_leq_q3(self) -> None:
        runs = [_single("x", 0.0, float(v)) for v in range(1, 9)]
        stats = DataStatistics(runs)
        q1 = stats.q1.get(0.0)["x"]
        q2 = stats.q2.get(0.0)["x"]
        q3 = stats.q3.get(0.0)["x"]
        assert q1 <= q2 <= q3

    def test_single_run_quartiles_equal_value(self) -> None:
        t = _single("x", 0.0, 4.0)
        stats = DataStatistics([t])
        assert stats.q1.get(0.0)["x"] == pytest.approx(4.0)
        assert stats.q2.get(0.0)["x"] == pytest.approx(4.0)
        assert stats.q3.get(0.0)["x"] == pytest.approx(4.0)

    def test_all_stats_return_data_table(self) -> None:
        t = _single("x", 0.0, 1.0)
        stats = DataStatistics([t])
        assert isinstance(stats.q1, DataTable)
        assert isinstance(stats.q2, DataTable)
        assert isinstance(stats.q3, DataTable)


# ---------------------------------------------------------------------------
# Output DataTable structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_output_tables_have_same_labels(self) -> None:
        t = _table(["a", "b"], {0.0: {"a": 1.0, "b": 2.0}})
        stats = DataStatistics([t])
        for prop in (stats.mean, stats.std, stats.q1, stats.q2, stats.q3, stats.min, stats.max):
            assert set(prop.labels) == {"a", "b"}

    def test_output_tables_have_same_times(self) -> None:
        t = _table(["v"], {0.0: {"v": 1.0}, 1.0: {"v": 2.0}, 2.0: {"v": 3.0}})
        stats = DataStatistics([t])
        for prop in (stats.mean, stats.std, stats.q1, stats.q2, stats.q3, stats.min, stats.max):
            assert prop.times == (0.0, 1.0, 2.0)

    def test_repr(self) -> None:
        t = _single("x", 0.0, 1.0)
        r = repr(DataStatistics([t]))
        assert "DataStatistics" in r
        assert "n=1" in r
