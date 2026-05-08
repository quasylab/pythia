"""Tests for pythia.models.lio.action."""

from __future__ import annotations

import pytest

from pythia.models.lio.action import Action


class TestActionConstruction:
    def test_name_stored(self) -> None:
        assert Action("send").name == "send"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Action("")

    def test_frozen(self) -> None:
        a = Action("x")
        with pytest.raises(AttributeError):
            a.name = "y"  # type: ignore[misc]


class TestActionEquality:
    def test_same_name_equal(self) -> None:
        assert Action("move") == Action("move")

    def test_different_name_not_equal(self) -> None:
        assert Action("send") != Action("recv")

    def test_not_equal_to_string(self) -> None:
        assert Action("send") != "send"

    def test_not_equal_to_none(self) -> None:
        assert Action("send") != None  # noqa: E711


class TestActionHashing:
    def test_same_name_same_hash(self) -> None:
        assert hash(Action("a")) == hash(Action("a"))

    def test_usable_as_dict_key(self) -> None:
        a = Action("fire")
        d = {a: 42}
        assert d[Action("fire")] == 42

    def test_usable_in_set(self) -> None:
        s = {Action("a"), Action("b"), Action("a")}
        assert len(s) == 2


class TestActionRepr:
    def test_repr(self) -> None:
        assert repr(Action("jump")) == "Action('jump')"
