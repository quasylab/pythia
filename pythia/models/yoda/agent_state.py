"""AgentState — a named behavioural state of an agent in a YODA system.

An :class:`AgentState` represents one of the discrete behavioural states
that an agent can occupy.  States are identified solely by their name: two
instances with the same name are equal and hash to the same value.

Example::

    from pythia.models.yoda.agent_state import AgentState

    idle = AgentState("idle")
    busy = AgentState("busy")

    assert idle == AgentState("idle")
    assert idle != busy
    assert {idle: 1}[AgentState("idle")] == 1
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class AgentState:
    """A named behavioural state of a YODA agent.

    Two :class:`AgentState` instances with the same name are equal and
    interchangeable as dictionary keys or set members.

    Attributes:
        name: The unique string identifier for this behavioural state.

    Raises:
        ValueError: If *name* is empty.

    Examples:
        >>> s1 = AgentState("idle")
        >>> s2 = AgentState("idle")
        >>> s1 == s2
        True
        >>> {s1: 1}[s2]
        1
        >>> AgentState("idle") != AgentState("busy")
        True
    """

    _name: str = dataclasses.field(repr=False)

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("AgentState name must be a non-empty string")
        object.__setattr__(self, "_name", name)

    @property
    def name(self) -> str:
        """The unique string identifier for this behavioural state.

        Examples:
            >>> AgentState("idle").name
            'idle'
        """
        return self._name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentState):
            return NotImplemented
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"AgentState({self._name!r})"
