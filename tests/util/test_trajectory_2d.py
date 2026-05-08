"""Tests for pythia.util.trajectory_2d."""

from __future__ import annotations

import pytest

from pythia.api.data_table import DataTable
from pythia.util.trajectory_2d import Trajectory2D


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _table(*rows: tuple[float, float, float]) -> DataTable:
    """Build a DataTable(["px", "py"]) from (t, px, py) triples."""
    t = DataTable(["px", "py"])
    for time, px, py in rows:
        t.add_row(time, {"px": px, "py": py})
    return t


def _ftrace(m: dict) -> tuple[float, float, str]:
    return m["px"], m["py"], "ro"


def _traj(*rows: tuple[float, float, float]) -> Trajectory2D:
    return Trajectory2D(_table(*rows), _ftrace)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_point(self) -> None:
        traj = _traj((0.0, 1.0, 2.0))
        assert len(traj) == 1

    def test_multiple_points(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert len(traj) == 3

    def test_empty_table_raises(self) -> None:
        t = DataTable(["px", "py"])
        with pytest.raises(ValueError, match="non-empty"):
            Trajectory2D(t, _ftrace)

    def test_ftrace_called_for_each_row(self) -> None:
        calls: list[dict] = []
        t = _table((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        Trajectory2D(t, lambda m: calls.append(dict(m)) or (0.0, 0.0, "."))
        assert len(calls) == 2

    def test_ftrace_receives_correct_row(self) -> None:
        received: list[dict] = []
        t = _table((0.5, 7.0, 8.0))
        Trajectory2D(t, lambda m: received.append(dict(m)) or (0.0, 0.0, "."))
        assert received[0]["px"] == pytest.approx(7.0)
        assert received[0]["py"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_setattr_raises(self) -> None:
        traj = _traj((0.0, 1.0, 2.0))
        with pytest.raises(AttributeError):
            traj._points = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_first(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        assert traj.get(0) == (0.0, 1.0, 2.0, "ro")

    def test_get_last(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        assert traj.get(1) == (1.0, 3.0, 4.0, "ro")

    def test_get_negative_index(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        assert traj.get(-1) == (1.0, 3.0, 4.0, "ro")

    def test_get_out_of_range_raises(self) -> None:
        traj = _traj((0.0, 1.0, 2.0))
        with pytest.raises(IndexError):
            traj.get(1)

    def test_record_contains_time(self) -> None:
        traj = _traj((2.5, 0.0, 0.0))
        assert traj.get(0)[0] == pytest.approx(2.5)

    def test_record_contains_fmt(self) -> None:
        t = DataTable(["px", "py"])
        t.add_row(0.0, {"px": 0.0, "py": 0.0})
        traj = Trajectory2D(t, lambda m: (0.0, 0.0, "b^"))
        assert traj.get(0)[3] == "b^"


# ---------------------------------------------------------------------------
# times / xs / ys / formats
# ---------------------------------------------------------------------------

class TestProperties:
    def test_times_in_order(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert traj.times == (0.0, 1.0, 2.0)

    def test_xs(self) -> None:
        traj = _traj((0.0, 3.0, 0.0), (1.0, 7.0, 0.0))
        assert traj.xs == pytest.approx((3.0, 7.0))

    def test_ys(self) -> None:
        traj = _traj((0.0, 0.0, 4.0), (1.0, 0.0, 9.0))
        assert traj.ys == pytest.approx((4.0, 9.0))

    def test_formats(self) -> None:
        t = DataTable(["v"])
        t.add_row(0.0, {"v": 0.0})
        t.add_row(1.0, {"v": 1.0})
        fmts = ["r.", "b^"]
        idx = 0

        def ftrace(m: dict) -> tuple[float, float, str]:
            nonlocal idx
            fmt = fmts[idx]
            idx += 1
            return 0.0, 0.0, fmt

        traj = Trajectory2D(t, ftrace)
        assert traj.formats == ("r.", "b^")

    def test_all_properties_same_length(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        assert len(traj.times) == len(traj.xs) == len(traj.ys) == len(traj.formats) == 2


# ---------------------------------------------------------------------------
# normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_returns_trajectory_2d(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (2.0, 2.0, 4.0))
        assert isinstance(traj.normalise(0.0, 2.0, 1.0), Trajectory2D)

    def test_correct_number_of_points(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (4.0, 4.0, 8.0))
        n = traj.normalise(0.0, 4.0, 1.0)
        assert len(n) == 5   # t = 0, 1, 2, 3, 4

    def test_times_are_multiples_of_dt(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
        n = traj.normalise(0.0, 3.0, 1.0)
        for t in n.times:
            assert abs(t % 1.0) < 1e-9

    def test_exact_match_uses_trajectory_point(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 5.0, 7.0))
        n = traj.normalise(0.0, 1.0, 1.0)
        assert n.get(1) == (1.0, pytest.approx(5.0), pytest.approx(7.0), "ro")

    def test_interpolation_midpoint(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (2.0, 2.0, 4.0))
        n = traj.normalise(0.0, 2.0, 1.0)
        # t=1 is midpoint: x=1, y=2
        rec = n.get(1)
        assert rec[0] == pytest.approx(1.0)
        assert rec[1] == pytest.approx(1.0)
        assert rec[2] == pytest.approx(2.0)

    def test_interpolation_fmt_from_earlier_point(self) -> None:
        t = DataTable(["px", "py"])
        t.add_row(0.0, {"px": 0.0, "py": 0.0})
        t.add_row(2.0, {"px": 2.0, "py": 0.0})
        calls = iter(["r.", "b^"])
        traj = Trajectory2D(t, lambda m: (m["px"], m["py"], next(calls)))
        # fmt at t=1 (interpolated) should be fmt of the t=0 point
        n = traj.normalise(0.0, 2.0, 1.0)
        assert n.get(1)[3] == "r."

    def test_past_end_uses_last_point(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        n = traj.normalise(0.0, 3.0, 1.0)
        last = traj.get(-1)
        rec = n.get(-1)
        assert rec[1] == pytest.approx(last[1])
        assert rec[2] == pytest.approx(last[2])
        assert rec[3] == last[3]

    def test_from_greater_than_zero(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
        n = traj.normalise(2.0, 4.0, 1.0)
        assert n.times[0] == pytest.approx(2.0)
        assert len(n) == 3   # t = 2, 3, 4

    def test_single_point_result(self) -> None:
        traj = _traj((0.0, 5.0, 6.0), (10.0, 7.0, 8.0))
        n = traj.normalise(0.0, 0.0, 1.0)
        assert len(n) == 1
        assert n.get(0)[0] == pytest.approx(0.0)

    def test_no_points_in_range_raises(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            traj.normalise(0.1, 0.4, 0.5)   # k*0.5: 0.0, 0.5, ... none in [0.1, 0.4]

    def test_result_is_immutable(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        n = traj.normalise(0.0, 1.0, 0.5)
        with pytest.raises(AttributeError):
            n._points = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# min_step
# ---------------------------------------------------------------------------

class TestMinStep:
    def test_uniform_spacing(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert traj.min_step() == pytest.approx(1.0)

    def test_non_uniform_spacing(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3.0, 2.0, 2.0))
        assert traj.min_step() == pytest.approx(1.0)

    def test_minimum_selected(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert traj.min_step() == pytest.approx(0.5)

    def test_two_points(self) -> None:
        traj = _traj((1.0, 0.0, 0.0), (4.0, 1.0, 1.0))
        assert traj.min_step() == pytest.approx(3.0)

    def test_single_point_raises(self) -> None:
        traj = _traj((0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="at least two"):
            traj.min_step()


# ---------------------------------------------------------------------------
# Sequence protocol
# ---------------------------------------------------------------------------

class TestSequenceProtocol:
    def test_len(self) -> None:
        assert len(_traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) == 2

    def test_iter_yields_records(self) -> None:
        traj = _traj((0.0, 1.0, 2.0), (1.0, 3.0, 4.0))
        records = list(traj)
        assert records[0] == (0.0, 1.0, 2.0, "ro")
        assert records[1] == (1.0, 3.0, 4.0, "ro")

    def test_iter_count(self) -> None:
        traj = _traj((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert sum(1 for _ in traj) == 3

    def test_repr(self) -> None:
        assert "Trajectory2D" in repr(_traj((0.0, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# ftrace integration
# ---------------------------------------------------------------------------

class TestFtraceIntegration:
    def test_fmt_varies_per_point(self) -> None:
        t = DataTable(["v"])
        t.add_row(0.0, {"v": 0.0})
        t.add_row(1.0, {"v": 1.0})
        traj = Trajectory2D(
            t,
            lambda m: (m["v"], m["v"] * 2, "ro" if m["v"] < 0.5 else "b^"),
        )
        assert traj.get(0)[3] == "ro"
        assert traj.get(1)[3] == "b^"

    def test_position_derived_from_row(self) -> None:
        t = DataTable(["angle"])
        import math
        t.add_row(0.0, {"angle": 0.0})
        t.add_row(1.0, {"angle": math.pi / 2})
        traj = Trajectory2D(t, lambda m: (math.cos(m["angle"]), math.sin(m["angle"]), "."))
        assert traj.get(0)[1] == pytest.approx(1.0)   # cos(0)
        assert traj.get(1)[2] == pytest.approx(1.0)   # sin(π/2)
