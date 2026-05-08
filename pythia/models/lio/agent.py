"""Agent — a named participant in the Language of Interacting Objects.

An :class:`Agent` represents a participant in a LIO system, identified by
its name.  Two instances with the same name are considered equal and hash
to the same value.

Transitions between agents are stored and managed externally by
:class:`~pythia.models.lio.agent_definitions.AgentDefinitions`, which also
guarantees that each name maps to a single :class:`Agent` instance.

Example::

    from pythia.models.lio.agent import Agent

    a = Agent("a")
    b = Agent("b")

    assert a == Agent("a")
    assert a != b
    assert {a: 1}[Agent("a")] == 1
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class Agent:
    """A named participant in a LIO system.

    Two :class:`Agent` instances with the same name are equal and
    interchangeable as dictionary keys or set members.

    Transitions are not stored on the agent itself; they are managed by
    :class:`~pythia.models.lio.agent_definitions.AgentDefinitions`.

    Attributes:
        name: The unique string identifier for this agent.

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
