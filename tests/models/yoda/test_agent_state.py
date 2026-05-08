"""Tests for pythia.models.yoda.agent_state."""

from __future__ import annotations

import pytest

from pythia.models.yoda.agent_state import AgentState


class TestConstruction:
    def test_name_stored(self) -> None:
        assert AgentState("idle").name == "idle"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentState("")


class TestNameGetter:
    def test_returns_name(self) -> None:
        assert AgentState("busy").name == "busy"

    def test_name_not_reassignable(self) -> None:
        s = AgentState("idle")
        with pytest.raises(AttributeError):
            s.name = "other"  # type: ignore[misc]


class TestEquality:
    def test_same_name_equal(self) -> None:
        assert AgentState("idle") == AgentState("idle")

    def test_different_name_not_equal(self) -> None:
        assert AgentState("idle") != AgentState("busy")

    def test_not_equal_to_string(self) -> None:
        assert AgentState("idle") != "idle"

    def test_not_equal_to_none(self) -> None:
        assert AgentState("idle") != None  # noqa: E711

    def test_not_equal_to_agent(self) -> None:
        from pythia.models.yoda.agent import Agent
        assert AgentState("idle") != Agent("idle")


class TestHashing:
    def test_same_name_same_hash(self) -> None:
        assert hash(AgentState("idle")) == hash(AgentState("idle"))

    def test_usable_as_dict_key(self) -> None:
        s = AgentState("idle")
        d = {s: 42}
        assert d[AgentState("idle")] == 42

    def test_usable_in_set(self) -> None:
        states = {AgentState("a"), AgentState("a"), AgentState("b")}
        assert len(states) == 2


class TestRepr:
    def test_repr(self) -> None:
        assert repr(AgentState("idle")) == "AgentState('idle')"
