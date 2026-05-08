"""Tests for pythia.api.data_table."""

from __future__ import annotations

import pytest

from pythia.api.data_table import DataTable


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_label(self) -> None:
        t = DataTable(["x"])
        assert t.labels == ("x",)

    def test_multiple_labels(self) -> None:
        t = DataTable(["a", "b", "c"])
        assert t.labels == ("a", "b", "c")

    def test_label_order_preserved(self) -> None:
        t = DataTable(["z", "a", "m"])
        assert t.labels == ("z", "a", "m")

    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one label"):
            DataTable([])

    def test_duplicate_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            DataTable(["x", "y", "x"])

    def test_starts_empty(self) -> None:
        assert len(DataTable(["x"])) == 0

    def test_times_empty_initially(self) -> None:
        assert DataTable(["x"]).times == ()


# ---------------------------------------------------------------------------
# add_row — label validation
# ---------------------------------------------------------------------------

class TestAddRowLabels:
    def test_add_single_row(self) -> None:
        t = DataTable(["x"])
        t.add_row(1.0, {"x": 1.0})
        assert len(t) == 1

    def test_add_multiple_rows(self) -> None:
        t = DataTable(["x", "y"])
        t.add_row(0.0, {"x": 1.0, "y": 2.0})
        t.add_row(1.0, {"x": 3.0, "y": 4.0})
        assert len(t) == 2

    def test_overwrite_existing_time(self) -> None:
        t = DataTable(["x"])
        t.add_row(1.0, {"x": 1.0})
        t.add_row(1.0, {"x": 99.0})
        assert len(t) == 1
        assert t.get(1.0)["x"] == pytest.approx(99.0)

    def test_extra_label_raises(self) -> None:
        t = DataTable(["x"])
        with pytest.raises(ValueError, match="unexpected"):
            t.add_row(1.0, {"x": 1.0, "z": 2.0})

    def test_missing_label_raises(self) -> None:
        t = DataTable(["x", "y"])
        with pytest.raises(ValueError, match="missing"):
            t.add_row(1.0, {"x": 1.0})

    def test_wrong_labels_raises(self) -> None:
        t = DataTable(["x", "y"])
        with pytest.raises(ValueError):
            t.add_row(1.0, {"a": 1.0, "b": 2.0})

    def test_input_dict_not_mutated_by_caller(self) -> None:
        t = DataTable(["x"])
        d = {"x": 1.0}
        t.add_row(0.0, d)
        d["x"] = 99.0
        assert t.get(0.0)["x"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# add_row — float type enforcement
# ---------------------------------------------------------------------------

class TestAddRowTypeEnforcement:
    def test_float_accepted(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 3.14})
        assert t.get(0.0)["x"] == pytest.approx(3.14)

    def test_int_promoted_to_float(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 7})
        val = t.get(0.0)["x"]
        assert isinstance(val, float)
        assert val == pytest.approx(7.0)

    def test_bool_rejected(self) -> None:
        t = DataTable(["x"])
        with pytest.raises(TypeError):
            t.add_row(0.0, {"x": True})

    def test_string_rejected(self) -> None:
        t = DataTable(["x"])
        with pytest.raises(TypeError):
            t.add_row(0.0, {"x": "hello"})

    def test_none_rejected(self) -> None:
        t = DataTable(["x"])
        with pytest.raises(TypeError):
            t.add_row(0.0, {"x": None})  # type: ignore[dict-item]

    def test_all_values_must_be_float(self) -> None:
        t = DataTable(["x", "y"])
        with pytest.raises(TypeError):
            t.add_row(0.0, {"x": 1.0, "y": "bad"})  # type: ignore[dict-item]

    def test_stored_as_float(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 5})
        assert type(t.get(0.0)["x"]) is float


# ---------------------------------------------------------------------------
# times property
# ---------------------------------------------------------------------------

class TestTimes:
    def test_returns_sorted_order(self) -> None:
        t = DataTable(["v"])
        t.add_row(3.0, {"v": 0.0})
        t.add_row(1.0, {"v": 0.0})
        t.add_row(2.0, {"v": 0.0})
        assert t.times == (1.0, 2.0, 3.0)

    def test_insertion_order_does_not_matter(self) -> None:
        t = DataTable(["v"])
        t.add_row(5.0, {"v": 0.0})
        t.add_row(0.0, {"v": 0.0})
        assert t.times == (0.0, 5.0)

    def test_returns_tuple(self) -> None:
        t = DataTable(["v"])
        t.add_row(1.0, {"v": 0.0})
        assert isinstance(t.times, tuple)

    def test_duplicate_time_not_duplicated(self) -> None:
        t = DataTable(["v"])
        t.add_row(1.0, {"v": 0.0})
        t.add_row(1.0, {"v": 1.0})
        assert t.times == (1.0,)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_correct_values(self) -> None:
        t = DataTable(["a", "b"])
        t.add_row(0.0, {"a": 10.0, "b": 20.0})
        assert t.get(0.0) == {"a": 10.0, "b": 20.0}

    def test_returns_copy(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 1.0})
        row = t.get(0.0)
        row["x"] = 99.0
        assert t.get(0.0)["x"] == pytest.approx(1.0)

    def test_missing_time_raises_key_error(self) -> None:
        t = DataTable(["x"])
        with pytest.raises(KeyError):
            t.get(99.0)

    def test_multiple_rows_independent(self) -> None:
        t = DataTable(["v"])
        t.add_row(0.0, {"v": 1.0})
        t.add_row(1.0, {"v": 2.0})
        assert t.get(0.0)["v"] == pytest.approx(1.0)
        assert t.get(1.0)["v"] == pytest.approx(2.0)

    def test_all_values_are_float(self) -> None:
        t = DataTable(["x", "y"])
        t.add_row(0.0, {"x": 1, "y": 2})
        row = t.get(0.0)
        assert type(row["x"]) is float
        assert type(row["y"]) is float


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_empty(self) -> None:
        assert len(DataTable(["x"])) == 0

    def test_after_add(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 1.0})
        t.add_row(1.0, {"x": 2.0})
        assert len(t) == 2

    def test_overwrite_does_not_increase_len(self) -> None:
        t = DataTable(["x"])
        t.add_row(0.0, {"x": 1.0})
        t.add_row(0.0, {"x": 2.0})
        assert len(t) == 1


# ---------------------------------------------------------------------------
# labels property
# ---------------------------------------------------------------------------

class TestLabels:
    def test_returns_tuple(self) -> None:
        assert isinstance(DataTable(["x"]).labels, tuple)

    def test_immutable_against_original_list(self) -> None:
        lst = ["a", "b"]
        t = DataTable(lst)
        lst.append("c")
        assert t.labels == ("a", "b")
