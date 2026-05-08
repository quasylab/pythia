"""Agent — a named behavioral state in a Network of Broadcasting Agents.

An :class:`Agent` represents one behavioral state that a participant in an
NBA system can occupy.  Two instances with the same name are equal and hash
to the same value.

Transitions between agent states are stored and managed externally by
:class:`~pythia.models.nba.agent_definitions.AgentDefinitions`, which also
guarantees that each name maps to a single :class:`Agent` instance.

Example::

    from pythia.models.nba.agent import Agent

    idle = Agent("idle")
    busy = Agent("busy")

    assert idle == Agent("idle")
    assert idle != busy
    assert {idle: 1}[Agent("idle")] == 1
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class Agent:
    """A named behavioral state in an NBA system.

    Two :class:`Agent` instances with the same name are equal and
    interchangeable as dictionary keys or set members.

    Transitions are not stored on the agent itself; they are managed by
    :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`.

    Attributes:
        name: The unique string identifier for this agent state.

    Raises:
        ValueError: If *name* is empty.

    Example:
        >>> a1 = Agent("idle")
        >>> a2 = Agent("idle")
        >>> a1 == a2
        True
        >>> {a1: 1}[a2]
        1
    """

    name: str

    def __post_init__(self) -> None:
        """Validate that name is non-empty."""
        if not self.name:
            raise ValueError("Agent name must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Agent({self.name!r})"
