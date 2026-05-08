"""Tests for pythia.models.nba.action."""

from __future__ import annotations

import pytest

from pythia.models.nba.action import (
    Action,
    BroadcastReceive,
    BroadcastSend,
    Receive,
    Send,
)
from pythia.models.nba.channel import Channel


C = Channel("c")
D = Channel("d")

# All four concrete types, for parametric tests.
_CHANNEL_TYPES: list[type] = [Send, Receive, BroadcastSend, BroadcastReceive]


# ---------------------------------------------------------------------------
# Action is an abstract base
# ---------------------------------------------------------------------------

class TestActionABC:
    def test_send_is_action(self) -> None:
        assert isinstance(Send(C), Action)

    def test_receive_is_action(self) -> None:
        assert isinstance(Receive(C), Action)

    def test_broadcast_send_is_action(self) -> None:
        assert isinstance(BroadcastSend(C), Action)

    def test_broadcast_receive_is_action(self) -> None:
        assert isinstance(BroadcastReceive(C), Action)


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

class TestSend:
    def test_channel_stored(self) -> None:
        assert Send(C).channel is C

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Send(C).channel = D  # type: ignore[misc]

    def test_equal_same_channel(self) -> None:
        assert Send(C) == Send(C)

    def test_not_equal_different_channel(self) -> None:
        assert Send(C) != Send(D)

    def test_hash_consistent(self) -> None:
        assert hash(Send(C)) == hash(Send(C))

    def test_usable_as_dict_key(self) -> None:
        d = {Send(C): 1}
        assert d[Send(C)] == 1

    def test_repr(self) -> None:
        assert repr(Send(C)) == "Send(Channel('c'))"


# ---------------------------------------------------------------------------
# Receive
# ---------------------------------------------------------------------------

class TestReceive:
    def test_channel_stored(self) -> None:
        assert Receive(C).channel is C

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Receive(C).channel = D  # type: ignore[misc]

    def test_equal_same_channel(self) -> None:
        assert Receive(C) == Receive(C)

    def test_not_equal_different_channel(self) -> None:
        assert Receive(C) != Receive(D)

    def test_hash_consistent(self) -> None:
        assert hash(Receive(C)) == hash(Receive(C))

    def test_usable_as_dict_key(self) -> None:
        d = {Receive(C): 2}
        assert d[Receive(C)] == 2

    def test_repr(self) -> None:
        assert repr(Receive(C)) == "Receive(Channel('c'))"


# ---------------------------------------------------------------------------
# BroadcastSend
# ---------------------------------------------------------------------------

class TestBroadcastSend:
    def test_channel_stored(self) -> None:
        assert BroadcastSend(C).channel is C

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            BroadcastSend(C).channel = D  # type: ignore[misc]

    def test_equal_same_channel(self) -> None:
        assert BroadcastSend(C) == BroadcastSend(C)

    def test_not_equal_different_channel(self) -> None:
        assert BroadcastSend(C) != BroadcastSend(D)

    def test_hash_consistent(self) -> None:
        assert hash(BroadcastSend(C)) == hash(BroadcastSend(C))

    def test_usable_as_dict_key(self) -> None:
        d = {BroadcastSend(C): 3}
        assert d[BroadcastSend(C)] == 3

    def test_repr(self) -> None:
        assert repr(BroadcastSend(C)) == "BroadcastSend(Channel('c'))"


# ---------------------------------------------------------------------------
# BroadcastReceive
# ---------------------------------------------------------------------------

class TestBroadcastReceive:
    def test_channel_stored(self) -> None:
        assert BroadcastReceive(C).channel is C

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            BroadcastReceive(C).channel = D  # type: ignore[misc]

    def test_equal_same_channel(self) -> None:
        assert BroadcastReceive(C) == BroadcastReceive(C)

    def test_not_equal_different_channel(self) -> None:
        assert BroadcastReceive(C) != BroadcastReceive(D)

    def test_hash_consistent(self) -> None:
        assert hash(BroadcastReceive(C)) == hash(BroadcastReceive(C))

    def test_usable_as_dict_key(self) -> None:
        d = {BroadcastReceive(C): 4}
        assert d[BroadcastReceive(C)] == 4

    def test_repr(self) -> None:
        assert repr(BroadcastReceive(C)) == "BroadcastReceive(Channel('c'))"


# ---------------------------------------------------------------------------
# Cross-type inequality (same channel, different action kind)
# ---------------------------------------------------------------------------

class TestCrossTypeInequality:
    def test_send_ne_receive(self) -> None:
        assert Send(C) != Receive(C)

    def test_send_ne_broadcast_send(self) -> None:
        assert Send(C) != BroadcastSend(C)

    def test_send_ne_broadcast_receive(self) -> None:
        assert Send(C) != BroadcastReceive(C)

    def test_receive_ne_broadcast_send(self) -> None:
        assert Receive(C) != BroadcastSend(C)

    def test_receive_ne_broadcast_receive(self) -> None:
        assert Receive(C) != BroadcastReceive(C)

    def test_broadcast_send_ne_broadcast_receive(self) -> None:
        assert BroadcastSend(C) != BroadcastReceive(C)

    def test_not_equal_to_channel(self) -> None:
        assert Send(C) != C

    def test_not_equal_to_string(self) -> None:
        assert Send(C) != "send"


# ---------------------------------------------------------------------------
# Actions with different channels are distinct dict keys
# ---------------------------------------------------------------------------

class TestActionsAsKeys:
    def test_send_different_channels_distinct_keys(self) -> None:
        d = {Send(C): 1, Send(D): 2}
        assert len(d) == 2

    def test_mixed_types_same_channel_distinct_keys(self) -> None:
        d = {Send(C): 1, Receive(C): 2, BroadcastSend(C): 3, BroadcastReceive(C): 4}
        assert len(d) == 4
