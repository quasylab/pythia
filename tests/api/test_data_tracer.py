"""Tests for pythia.api.data_tracer."""

from __future__ import annotations

import pytest

from pythia.api.data_table import DataTable
from pythia.api.data_tracer import DataTracer
from pythia.api.state import SimulationState


def _state(elapsed: float = 0.0, step: int = 0) -> SimulationState:
    return SimulationState(step=step, elapsed_time=elapsed)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_labels_stored(self) -> None:
        t = DataTracer(["x", "y"], lambda s: {})
        assert t.labels == ("x", "y")

    def test_label_order_preserved(self) -> None:
        t = DataTracer(["z", "a"], lambda s: {})
        assert t.labels == ("z", "a")

    def test_trace_function_stored(self) -> None:
        f = lambda s: {}
        t = DataTracer(["x"], f)
        assert t.trace_function is f

    def test_starts_with_no_names(self) -> None:
        t = DataTracer(["x"], lambda s: {})
        assert t.names == frozenset()


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_labels_not_reassignable(self) -> None:
        t = DataTracer(["x"], lambda s: {})
        with pytest.raises(AttributeError):
            t.labels = ("y",)  # type: ignore[misc]

    def test_trace_function_not_reassignable(self) -> None:
        t = DataTracer(["x"], lambda s: {})
        with pytest.raises(AttributeError):
            t.trace_function = lambda s: {}  # type: ignore[misc]

    def test_arbitrary_attr_not_assignable(self) -> None:
        t = DataTracer(["x"], lambda s: {})
        with pytest.raises(AttributeError):
            t.foo = "bar"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# trace — basic recording
# ---------------------------------------------------------------------------

class TestTrace:
    def test_creates_table_on_first_encounter(self) -> None:
        t = DataTracer(["v"], lambda s: {"obj": {"v": 1.0}})
        t.trace(_state(0.0))
        assert "obj" in t.names

    def test_records_at_elapsed_time(self) -> None:
        t = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
        t.trace(_state(2.5))
        assert t.get_table("obj").get(2.5) == {"v": 2.5}

    def test_multiple_calls_accumulate_rows(self) -> None:
        t = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
        t.trace(_state(0.0))
        t.trace(_state(1.0))
        t.trace(_state(2.0))
        assert t.times("obj") == (0.0, 1.0, 2.0)

    def test_empty_result_is_noop(self) -> None:
        t = DataTracer(["v"], lambda s: {})
        t.trace(_state(0.0))
        assert t.names == frozenset()

    def test_multiple_objects_recorded(self) -> None:
        t = DataTracer(["v"], lambda s: {"a": {"v": 1.0}, "b": {"v": 2.0}})
        t.trace(_state(0.0))
        assert t.names == frozenset({"a", "b"})
        assert t.get_table("a").get(0.0)["v"] == pytest.approx(1.0)
        assert t.get_table("b").get(0.0)["v"] == pytest.approx(2.0)

    def test_objects_appear_incrementally(self) -> None:
        call_count = 0

        def fn(s: SimulationState) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"a": {"v": 1.0}}
            return {"a": {"v": 2.0}, "b": {"v": 3.0}}

        t = DataTracer(["v"], fn)
        t.trace(_state(0.0))
        assert t.names == frozenset({"a"})
        t.trace(_state(1.0))
        assert t.names == frozenset({"a", "b"})

    def test_state_passed_to_trace_function(self) -> None:
        received: list[SimulationState] = []
        t = DataTracer(["v"], lambda s: received.append(s) or {})
        state = _state(3.0)
        t.trace(state)
        assert received[0] is state


# ---------------------------------------------------------------------------
# get_table
# ---------------------------------------------------------------------------

class TestGetTable:
    def test_returns_data_table(self) -> None:
        t = DataTracer(["x"], lambda s: {"obj": {"x": 0.0}})
        t.trace(_state())
        assert isinstance(t.get_table("obj"), DataTable)

    def test_missing_name_raises(self) -> None:
        t = DataTracer(["x"], lambda s: {})
        with pytest.raises(KeyError):
            t.get_table("missing")

    def test_table_has_correct_labels(self) -> None:
        t = DataTracer(["a", "b"], lambda s: {"obj": {"a": 1.0, "b": 2.0}})
        t.trace(_state())
        assert t.get_table("obj").labels == ("a", "b")


# ---------------------------------------------------------------------------
# names property
# ---------------------------------------------------------------------------

class TestNames:
    def test_returns_frozenset(self) -> None:
        t = DataTracer(["x"], lambda s: {"p": {"x": 0.0}})
        t.trace(_state())
        assert isinstance(t.names, frozenset)

    def test_reflects_all_seen_names(self) -> None:
        t = DataTracer(["x"], lambda s: {"p": {"x": 1.0}, "q": {"x": 2.0}})
        t.trace(_state())
        assert t.names == frozenset({"p", "q"})


# ---------------------------------------------------------------------------
# tables property
# ---------------------------------------------------------------------------

class TestTablesProperty:
    def test_returns_dict(self) -> None:
        t = DataTracer(["x"], lambda s: {"obj": {"x": 0.0}})
        t.trace(_state())
        assert isinstance(t.tables, dict)

    def test_returns_copy(self) -> None:
        t = DataTracer(["x"], lambda s: {"obj": {"x": 0.0}})
        t.trace(_state())
        snapshot = t.tables
        del snapshot["obj"]
        assert "obj" in t.tables


# ---------------------------------------------------------------------------
# times method
# ---------------------------------------------------------------------------

class TestTimes:
    def test_returns_sorted_times(self) -> None:
        t = DataTracer(["v"], lambda s: {"obj": {"v": s.elapsed_time}})
        t.trace(_state(3.0))
        t.trace(_state(1.0))
        t.trace(_state(2.0))
        assert t.times("obj") == (1.0, 2.0, 3.0)

    def test_missing_name_raises(self) -> None:
        t = DataTracer(["v"], lambda s: {})
        with pytest.raises(KeyError):
            t.times("missing")
