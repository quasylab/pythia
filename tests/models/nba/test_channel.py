"""Tests for pythia.models.nba.channel."""

from __future__ import annotations

import pytest

from pythia.models.nba.channel import Channel


class TestChannelConstruction:
    def test_name_stored(self) -> None:
        assert Channel("data").name == "data"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Channel("")

    def test_frozen(self) -> None:
        c = Channel("x")
        with pytest.raises(AttributeError):
            c.name = "y"  # type: ignore[misc]


class TestChannelEquality:
    def test_same_name_equal(self) -> None:
        assert Channel("alpha") == Channel("alpha")

    def test_different_name_not_equal(self) -> None:
        assert Channel("alpha") != Channel("beta")

    def test_not_equal_to_string(self) -> None:
        assert Channel("alpha") != "alpha"

    def test_not_equal_to_none(self) -> None:
        assert Channel("alpha") != None  # noqa: E711


class TestChannelHashing:
    def test_same_name_same_hash(self) -> None:
        assert hash(Channel("x")) == hash(Channel("x"))

    def test_usable_as_dict_key(self) -> None:
        c = Channel("ctrl")
        d = {c: 42}
        assert d[Channel("ctrl")] == 42

    def test_usable_in_set(self) -> None:
        s = {Channel("a"), Channel("b"), Channel("a")}
        assert len(s) == 2


class TestChannelRepr:
    def test_repr(self) -> None:
        assert repr(Channel("link")) == "Channel('link')"
