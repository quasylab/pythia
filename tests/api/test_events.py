"""Tests for pythia.api.events."""

from __future__ import annotations

import logging

import pytest

from pythia.api.events import EventBus, EventKind, SimulationEvent
from pythia.api.state import SimulationState


def _make_event(kind: EventKind, step: int = 0) -> SimulationEvent:
    return SimulationEvent(kind=kind, step=step, state=SimulationState())


class TestEventKind:
    def test_all_kinds_defined(self) -> None:
        kinds = {e.value for e in EventKind}
        assert "run_started" in kinds
        assert "step_completed" in kinds
        assert "run_finished" in kinds
        assert "run_failed" in kinds


class TestSimulationEvent:
    def test_frozen(self) -> None:
        event = _make_event(EventKind.RUN_STARTED)
        with pytest.raises((AttributeError, TypeError)):
            event.step = 99  # type: ignore[misc]

    def test_metadata_defaults_empty(self) -> None:
        event = _make_event(EventKind.RUN_STARTED)
        assert event.metadata == {}


class TestEventBusSubscribeEmit:
    def test_handler_called_on_emit(self, event_bus: EventBus) -> None:
        received: list[SimulationEvent] = []
        event_bus.subscribe(EventKind.RUN_STARTED, received.append)
        event = _make_event(EventKind.RUN_STARTED)
        event_bus.emit(event)
        assert received == [event]

    def test_handler_not_called_for_different_kind(self, event_bus: EventBus) -> None:
        received: list[SimulationEvent] = []
        event_bus.subscribe(EventKind.RUN_FINISHED, received.append)
        event_bus.emit(_make_event(EventKind.RUN_STARTED))
        assert received == []

    def test_multiple_handlers_called_in_order(self, event_bus: EventBus) -> None:
        order: list[int] = []
        event_bus.subscribe(EventKind.STEP_COMPLETED, lambda _: order.append(1))
        event_bus.subscribe(EventKind.STEP_COMPLETED, lambda _: order.append(2))
        event_bus.emit(_make_event(EventKind.STEP_COMPLETED))
        assert order == [1, 2]

    def test_emit_multiple_events(self, event_bus: EventBus) -> None:
        count = [0]
        event_bus.subscribe(EventKind.STEP_COMPLETED, lambda _: count.__setitem__(0, count[0] + 1))
        for i in range(5):
            event_bus.emit(_make_event(EventKind.STEP_COMPLETED, step=i))
        assert count[0] == 5


class TestEventBusUnsubscribe:
    def test_unsubscribed_handler_not_called(self, event_bus: EventBus) -> None:
        received: list[SimulationEvent] = []
        event_bus.subscribe(EventKind.RUN_STARTED, received.append)
        event_bus.unsubscribe(EventKind.RUN_STARTED, received.append)
        event_bus.emit(_make_event(EventKind.RUN_STARTED))
        assert received == []

    def test_unsubscribe_unknown_handler_is_silent(self, event_bus: EventBus) -> None:
        # Should not raise
        event_bus.unsubscribe(EventKind.RUN_STARTED, lambda e: None)


class TestEventBusExceptionIsolation:
    def test_broken_handler_does_not_propagate(self, event_bus: EventBus) -> None:
        def bad_handler(_: SimulationEvent) -> None:
            raise RuntimeError("handler error")

        event_bus.subscribe(EventKind.RUN_STARTED, bad_handler)
        # Must not raise
        event_bus.emit(_make_event(EventKind.RUN_STARTED))

    def test_subsequent_handler_still_called_after_broken_one(
        self, event_bus: EventBus
    ) -> None:
        received: list[SimulationEvent] = []

        def bad_handler(_: SimulationEvent) -> None:
            raise RuntimeError("error")

        event_bus.subscribe(EventKind.RUN_STARTED, bad_handler)
        event_bus.subscribe(EventKind.RUN_STARTED, received.append)
        event_bus.emit(_make_event(EventKind.RUN_STARTED))
        assert len(received) == 1

    def test_broken_handler_logged(
        self, event_bus: EventBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        def bad_handler(_: SimulationEvent) -> None:
            raise ValueError("oops")

        event_bus.subscribe(EventKind.RUN_STARTED, bad_handler)
        with caplog.at_level(logging.WARNING, logger="pythia.api.events"):
            event_bus.emit(_make_event(EventKind.RUN_STARTED))
        assert any("raised an exception" in r.message for r in caplog.records)
