"""Tests for pythia.util.trace_viewer."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any pyplot import

import pytest
from matplotlib.animation import FuncAnimation

from pythia.api.data_table import DataTable
from pythia.api.data_tracer import DataTracer
from pythia.api.state import SimulationState
from pythia.util.trace_viewer import TraceViewer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracer_with(*objects: tuple[str, list[tuple[float, float, float]]]) -> DataTracer:
    """Build a DataTracer pre-populated with (name, [(t, px, py), ...]) data."""
    tables: dict[str, DataTable] = {}
    for name, rows in objects:
        table = DataTable(["px", "py"])
        for t, px, py in rows:
            table.add_row(t, {"px": px, "py": py})
        tables[name] = table

    def trace_fn(s: SimulationState) -> dict:
        return {}  # not used; we inject tables directly below

    tracer = DataTracer(["px", "py"], trace_fn)
    # Inject pre-built tables via the internal mapping (bypassing trace)
    for name, table in tables.items():
        tracer._tables[name] = table  # type: ignore[attr-defined]
    return tracer


_FTRACE = lambda m: (m["px"], m["py"], "ro")

ONE_OBJ = _tracer_with(
    ("agent", [(0.0, 0.0, 0.0), (2.0, 2.0, 4.0)])
)

TWO_OBJ = _tracer_with(
    ("a", [(0.0, 0.0, 0.0), (2.0, 1.0, 2.0)]),
    ("b", [(0.0, 3.0, 3.0), (2.0, 4.0, 5.0)]),
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_tracer_stored(self) -> None:
        tv = TraceViewer(ONE_OBJ, _FTRACE)
        assert tv.tracer is ONE_OBJ

    def test_ftrace_stored(self) -> None:
        tv = TraceViewer(ONE_OBJ, _FTRACE)
        assert tv.ftrace is _FTRACE


# ---------------------------------------------------------------------------
# create_animation — return type and structure
# ---------------------------------------------------------------------------

class TestCreateAnimation:
    def test_returns_func_animation(self) -> None:
        anim = TraceViewer(ONE_OBJ, _FTRACE).create_animation(0.0, 2.0, 1.0)
        assert isinstance(anim, FuncAnimation)

    def test_correct_frame_count(self) -> None:
        # start=0, end=2, dt=0.5 → frames at 0.0, 0.5, 1.0, 1.5, 2.0 → 5 frames
        anim = TraceViewer(ONE_OBJ, _FTRACE).create_animation(0.0, 2.0, 0.5)
        assert anim._interval == 100  # default interval is 100 ms

    def test_two_objects(self) -> None:
        anim = TraceViewer(TWO_OBJ, _FTRACE).create_animation(0.0, 2.0, 1.0)
        assert isinstance(anim, FuncAnimation)

    def test_empty_tracer_raises(self) -> None:
        empty = DataTracer(["px", "py"], lambda s: {})
        with pytest.raises(ValueError, match="no named objects"):
            TraceViewer(empty, _FTRACE).create_animation(0.0, 2.0, 1.0)

    def test_figure_axes_limits_set(self) -> None:
        anim = TraceViewer(ONE_OBJ, _FTRACE).create_animation(0.0, 2.0, 1.0)
        # Trigger first frame to verify axes are set up
        anim._draw_next_frame(0, False)
        ax = anim._fig.axes[0]
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        # Limits should encompass trajectory xs=[0,2] and ys=[0,4]
        assert x_lo < 0.0 and x_hi > 2.0
        assert y_lo < 0.0 and y_hi > 4.0

    def test_each_frame_plots_one_artist_per_object(self) -> None:
        anim = TraceViewer(TWO_OBJ, _FTRACE).create_animation(0.0, 2.0, 1.0)
        anim._draw_next_frame(0, False)
        ax = anim._fig.axes[0]
        assert len(ax.lines) == 2

    def test_repeat_enabled(self) -> None:
        anim = TraceViewer(ONE_OBJ, _FTRACE).create_animation(0.0, 2.0, 1.0)
        assert anim._repeat is True
