"""Action — atomic interaction unit in the Language of Interacting Objects.

An :class:`Action` identifies a single interaction that an agent can execute.
Actions are value objects: two :class:`Action` instances with the same name
are considered equal and hash to the same value, making them safe to use as
dictionary keys or set members.

Example::

    from pythia.models.lio.action import Action

    send = Action("send")
    recv = Action("recv")

    assert send == Action("send")
    assert send != recv
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class Action:
    """An atomic action that an agent in a LIO system can execute.

    Actions are identified solely by their :attr:`name`.  Two :class:`Action`
    instances with the same name are equal and interchangeable as dict keys or
    set members.

    Attributes:
        name: The unique string identifier for this action.

    Raises:
        ValueError: If *name* is empty.

    Examples:
        >>> a = Action("move")
        >>> b = Action("move")
        >>> a == b
        True
        >>> a == Action("stop")
        False
        >>> {a: 1}[b]
        1
    """

    name: str

    def __post_init__(self) -> None:
        """Validate that name is non-empty."""
        if not self.name:
            raise ValueError("Action name must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Action({self.name!r})"
