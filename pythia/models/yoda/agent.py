"""Agent — a named participant in a YODA system.

An :class:`Agent` identifies one participant in a YODA system.  Two agents
with the same name are considered equal and hash to the same value, making
them safe to use as dictionary keys or set members.

Example::

    from pythia.models.yoda.agent import Agent

    a = Agent("alice")
    b = Agent("bob")

    assert a == Agent("alice")
    assert a != b
    assert {a: 1}[Agent("alice")] == 1
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class Agent:
    """A named participant in a YODA system.

    Two :class:`Agent` instances with the same name are equal and
    interchangeable as dictionary keys or set members.

    Attributes:
        name: The unique string identifier for this agent.

    Raises:
        ValueError: If *name* is empty.

    Examples:
        >>> a1 = Agent("alice")
        >>> a2 = Agent("alice")
        >>> a1 == a2
        True
        >>> {a1: 1}[a2]
        1
        >>> Agent("alice") != Agent("bob")
        True
    """

    _name: str = dataclasses.field(repr=False)

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Agent name must be a non-empty string")
        object.__setattr__(self, "_name", name)

    @property
    def name(self) -> str:
        """The unique string identifier for this agent.

        Examples:
            >>> Agent("alice").name
            'alice'
        """
        return self._name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Agent({self._name!r})"
