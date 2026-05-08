"""Tests for pythia.models.nba.agent."""

from __future__ import annotations

import pytest

from pythia.models.nba.agent import Agent


class TestAgentConstruction:
    def test_name_stored(self) -> None:
        assert Agent("idle").name == "idle"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Agent("")

    def test_frozen(self) -> None:
        a = Agent("x")
        with pytest.raises(AttributeError):
            a.name = "y"  # type: ignore[misc]

    def test_no_transitions_attribute(self) -> None:
        assert not hasattr(Agent("a"), "_transitions")


class TestAgentEquality:
    def test_same_name_equal(self) -> None:
        assert Agent("idle") == Agent("idle")

    def test_different_name_not_equal(self) -> None:
        assert Agent("idle") != Agent("busy")

    def test_not_equal_to_string(self) -> None:
        assert Agent("idle") != "idle"

    def test_not_equal_to_none(self) -> None:
        assert Agent("idle") != None  # noqa: E711


class TestAgentHashing:
    def test_same_name_same_hash(self) -> None:
        assert hash(Agent("a")) == hash(Agent("a"))

    def test_usable_as_dict_key(self) -> None:
        a = Agent("sender")
        d = {a: 99}
        assert d[Agent("sender")] == 99

    def test_usable_in_set(self) -> None:
        s = {Agent("a"), Agent("b"), Agent("a")}
        assert len(s) == 2


class TestAgentRepr:
    def test_repr(self) -> None:
        assert repr(Agent("idle")) == "Agent('idle')"
