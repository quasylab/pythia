"""NBA actions — the interaction primitives of a Network of Broadcasting Agents.

Every action an agent can perform is one of four kinds:

* :class:`Send` — point-to-point output on a channel (``c!``).
* :class:`Receive` — point-to-point input on a channel (``c?``).
* :class:`BroadcastSend` — broadcast output on a channel (``c!!``).
* :class:`BroadcastReceive` — broadcast input on a channel (``c??``).

All action types are immutable value objects: two instances of the same
subclass carrying equal channels are equal and hash-equivalent.  Instances
of *different* subclasses are never equal, even when their channels match,
because they represent distinct kinds of interaction.

Example::

    from pythia.models.nba.action import Send, Receive, BroadcastSend, BroadcastReceive
    from pythia.models.nba.channel import Channel

    c = Channel("data")

    assert Send(c) == Send(c)
    assert Send(c) != Receive(c)    # different kind
"""

from __future__ import annotations

import dataclasses
from abc import ABC

from pythia.models.nba.channel import Channel


class Action(ABC):
    """Abstract base class for NBA actions.

    All concrete actions (:class:`Send`, :class:`Receive`,
    :class:`BroadcastSend`, :class:`BroadcastReceive`) inherit from this
    class.  It exists solely to provide a common type for type annotations;
    equality and hashing are defined by each subclass.
    """


@dataclasses.dataclass(frozen=True)
class Send(Action):
    """Point-to-point output on *channel* (``channel!``).

    A :class:`Send` action synchronises with exactly one :class:`Receive`
    action on the same channel.

    Attributes:
        channel: The channel on which the output is performed.

    Example:
        >>> c = Channel("link")
        >>> Send(c) == Send(c)
        True
        >>> Send(c) == Receive(c)
        False
    """

    channel: Channel

    def __repr__(self) -> str:
        return f"Send({self.channel!r})"


@dataclasses.dataclass(frozen=True)
class Receive(Action):
    """Point-to-point input on *channel* (``channel?``).

    A :class:`Receive` action synchronises with exactly one :class:`Send`
    action on the same channel.

    Attributes:
        channel: The channel on which the input is performed.

    Example:
        >>> c = Channel("link")
        >>> Receive(c) == Receive(c)
        True
        >>> Receive(c) == Send(c)
        False
    """

    channel: Channel

    def __repr__(self) -> str:
        return f"Receive({self.channel!r})"


@dataclasses.dataclass(frozen=True)
class BroadcastSend(Action):
    """Broadcast output on *channel* (``channel!!``).

    A :class:`BroadcastSend` action simultaneously synchronises with all
    agents currently able to perform a :class:`BroadcastReceive` on the
    same channel.

    Attributes:
        channel: The channel on which the broadcast is performed.

    Example:
        >>> c = Channel("alert")
        >>> BroadcastSend(c) == BroadcastSend(c)
        True
        >>> BroadcastSend(c) == BroadcastReceive(c)
        False
    """

    channel: Channel

    def __repr__(self) -> str:
        return f"BroadcastSend({self.channel!r})"


@dataclasses.dataclass(frozen=True)
class BroadcastReceive(Action):
    """Broadcast input on *channel* (``channel??``).

    A :class:`BroadcastReceive` action is triggered when another agent
    performs a :class:`BroadcastSend` on the same channel.

    Attributes:
        channel: The channel on which the broadcast receive is performed.

    Example:
        >>> c = Channel("alert")
        >>> BroadcastReceive(c) == BroadcastReceive(c)
        True
        >>> BroadcastReceive(c) == BroadcastSend(c)
        False
    """

    channel: Channel

    def __repr__(self) -> str:
        return f"BroadcastReceive({self.channel!r})"
