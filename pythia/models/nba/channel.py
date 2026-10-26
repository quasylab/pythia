"""Channel — named communication channel in a Network of Broadcasting Agents.

A :class:`Channel` identifies a communication medium over which agents
exchange messages.  Channels are value objects: two instances with the same
name are equal and hash-equivalent, making them safe to use as dictionary
keys or set members.

Example::

    from pythia.models.nba.channel import Channel

    c = Channel("data")
    assert c == Channel("data")
    assert c != Channel("control")
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class Channel:
    """A named communication channel.

    Two :class:`Channel` instances with the same name are equal and
    interchangeable as dictionary keys or set members.

    Attributes:
        name: The unique string identifier for this channel.

    Raises:
        ValueError: If *name* is empty.

    Examples:
        >>> c1 = Channel("alpha")
        >>> c2 = Channel("alpha")
        >>> c1 == c2
        True
        >>> c1 == Channel("beta")
        False
        >>> {c1: 1}[c2]
        1
    """

    name: str

    def __post_init__(self) -> None:
        """Validate that name is non-empty."""
        if not self.name:
            raise ValueError("Channel name must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Channel):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Channel({self.name!r})"
