"""Tests for pythia.models.lio.agent."""

from __future__ import annotations

import pytest

from pythia.models.lio.agent import Agent


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_name_stored(self) -> None:
        assert Agent("idle").name == "idle"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Agent("")

    def test_frozen_name(self) -> None:
        a = Agent("idle")
        with pytest.raises(AttributeError):
            a.name = "busy"  # type: ignore[misc]

    def test_no_transitions_attribute(self) -> None:
        a = Agent("x")
        assert not hasattr(a, "transitions")
        assert not hasattr(a, "_transitions")


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------

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
        assert hash(Agent("idle")) == hash(Agent("idle"))

    def test_usable_as_dict_key(self) -> None:
        a = Agent("running")
        d = {a: 42}
        assert d[Agent("running")] == 42

    def test_usable_in_set(self) -> None:
        agents = {Agent("a"), Agent("b"), Agent("a")}
        assert len(agents) == 2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestAgentRepr:
    def test_repr(self) -> None:
        assert repr(Agent("idle")) == "Agent('idle')"
