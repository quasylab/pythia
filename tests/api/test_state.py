"""Tests for pythia.api.state."""

from __future__ import annotations

import pytest

from pythia.api.state import SimulationState


class TestSimulationStateAdvance:
    def test_step_increments(self) -> None:
        state = SimulationState()
        state.advance(1.0)
        assert state.step == 1

    def test_elapsed_time_accumulates(self) -> None:
        state = SimulationState()
        state.advance(0.5)
        state.advance(0.5)
        assert state.elapsed_time == pytest.approx(1.0)

    def test_multiple_advances(self) -> None:
        state = SimulationState()
        for _ in range(5):
            state.advance(2.0)
        assert state.step == 5
        assert state.elapsed_time == pytest.approx(10.0)


class TestSimulationStateSnapshot:
    def test_snapshot_independence(self) -> None:
        state = SimulationState(data={"x": [1, 2, 3]})
        snap = state.snapshot()
        state.data["x"].append(4)
        assert snap.data["x"] == [1, 2, 3]

    def test_snapshot_values_match(self) -> None:
        state = SimulationState(step=3, elapsed_time=3.0, is_done=True)
        snap = state.snapshot()
        assert snap.step == 3
        assert snap.elapsed_time == 3.0
        assert snap.is_done is True

    def test_snapshot_is_different_object(self) -> None:
        state = SimulationState()
        snap = state.snapshot()
        assert snap is not state


class TestSimulationStateDefaults:
    def test_initial_values(self) -> None:
        state = SimulationState()
        assert state.step == 0
        assert state.elapsed_time == 0.0
        assert state.data == {}
        assert state.is_done is False


