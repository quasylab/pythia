"""Lightweight synchronous event bus for simulation lifecycle hooks.

The :class:`EventBus` is the primary extension seam in Pythia.  The
:class:`~pythia.api.runner.SimulationRunner` emits events at well-defined
points in the run lifecycle; external components (loggers, visualisers,
checkpointers) subscribe to those events without importing anything from the
runner.

Examples:
    >>> bus = EventBus()
    >>> received = []
    >>> bus.subscribe(EventKind.STEP_COMPLETED, received.append)
    >>> from pythia.api.state import SimulationState
    >>> bus.emit(SimulationEvent(EventKind.STEP_COMPLETED, step=1, state=SimulationState()))
    >>> len(received)
    1
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pythia.api.state import SimulationState

logger = logging.getLogger(__name__)

Handler = "Callable[[SimulationEvent], None]"


class EventKind(enum.Enum):
    """Enumeration of lifecycle events emitted by the runner.

    Attributes:
        RUN_STARTED: Fired once, immediately before the first call to
            :meth:`~pythia.api.model.BaseModel.initialize`.
        STEP_COMPLETED: Fired after each successful call to
            :meth:`~pythia.api.model.BaseModel.step`.
        RUN_FINISHED: Fired once, after
            :meth:`~pythia.api.model.BaseModel.finalize` returns normally.
        RUN_FAILED: Fired once, when the run terminates due to an exception.
    """

    RUN_STARTED = "run_started"
    STEP_COMPLETED = "step_completed"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


@dataclasses.dataclass(frozen=True)
class SimulationEvent:
    """Immutable payload delivered to event handlers.

    Attributes:
        kind: The lifecycle point at which this event was emitted.
        step: The iteration index at the time of emission.
        state: A reference to the current simulation state.  May be ``None``
            for :attr:`EventKind.RUN_STARTED` before the state is created.
        metadata: Additional context provided by the emitter.
    """

    kind: EventKind
    step: int
    state: SimulationState | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class EventBus:
    """Synchronous publish-subscribe event bus.

    Handlers are called in subscription order.  Exceptions raised by a
    handler are *caught and logged* — they are never propagated — so a broken
    plugin cannot abort a simulation run.

    Examples:
        >>> bus = EventBus()
        >>> log = []
        >>> bus.subscribe(EventKind.RUN_STARTED, log.append)
        >>> from pythia.api.state import SimulationState
        >>> bus.emit(SimulationEvent(EventKind.RUN_STARTED, step=0))
        >>> len(log)
        1
    """

    def __init__(self) -> None:
        """Initialise an empty event bus."""
        self._handlers: dict[EventKind, list[Callable[[SimulationEvent], None]]] = (
            defaultdict(list)
        )

    def subscribe(
        self,
        kind: EventKind,
        handler: Callable[[SimulationEvent], None],
    ) -> None:
        """Register *handler* to be called whenever *kind* is emitted.

        Args:
            kind: The event kind to subscribe to.
            handler: A callable that accepts a single :class:`SimulationEvent`
                argument.  It must not raise; any exceptions are caught by the
                bus.
        """
        self._handlers[kind].append(handler)

    def unsubscribe(
        self,
        kind: EventKind,
        handler: Callable[[SimulationEvent], None],
    ) -> None:
        """Remove a previously registered *handler* for *kind*.

        Silently does nothing if *handler* was not registered.

        Args:
            kind: The event kind from which to unsubscribe.
            handler: The exact callable object that was passed to
                :meth:`subscribe`.
        """
        try:
            self._handlers[kind].remove(handler)
        except ValueError:
            pass

    def emit(self, event: SimulationEvent) -> None:
        """Deliver *event* to all handlers registered for its kind.

        Handler exceptions are caught and logged at ``WARNING`` level.

        Args:
            event: The event to deliver.
        """
        for handler in list(self._handlers[event.kind]):
            try:
                handler(event)
            except Exception:
                logger.warning(
                    "Event handler %r raised an exception for event %r",
                    handler,
                    event.kind,
                    exc_info=True,
                )
