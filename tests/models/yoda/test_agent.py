"""Tests for pythia.models.yoda.agent."""

from __future__ import annotations

import pytest

from pythia.models.yoda.agent import Agent


class TestConstruction:
    def test_name_stored(self) -> None:
        assert Agent("alice").name == "alice"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            Agent("")


class TestNameGetter:
    def test_returns_name(self) -> None:
        assert Agent("bob").name == "bob"

    def test_name_not_reassignable(self) -> None:
        a = Agent("alice")
        with pytest.raises(AttributeError):
            a.name = "other"  # type: ignore[misc]


class TestEquality:
    def test_same_name_equal(self) -> None:
        assert Agent("a") == Agent("a")

    def test_different_name_not_equal(self) -> None:
        assert Agent("a") != Agent("b")

    def test_not_equal_to_string(self) -> None:
        assert Agent("a") != "a"

    def test_not_equal_to_none(self) -> None:
        assert Agent("a") != None  # noqa: E711


class TestHashing:
    def test_same_name_same_hash(self) -> None:
        assert hash(Agent("a")) == hash(Agent("a"))

    def test_usable_as_dict_key(self) -> None:
        a = Agent("alice")
        d = {a: 42}
        assert d[Agent("alice")] == 42

    def test_usable_in_set(self) -> None:
        s = {Agent("a"), Agent("a"), Agent("b")}
        assert len(s) == 2


class TestRepr:
    def test_repr(self) -> None:
        assert repr(Agent("alice")) == "Agent('alice')"
