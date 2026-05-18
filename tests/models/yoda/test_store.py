"""Tests for pythia.models.yoda.store."""

from __future__ import annotations

import pytest

from pythia.models.yoda.store import Store


class TestConstruction:
    def test_empty_store(self) -> None:
        s = Store()
        assert len(s) == 0

    def test_from_dict(self) -> None:
        s = Store({"x": 1, "y": 2})
        assert s["x"] == 1
        assert s["y"] == 2

    def test_input_dict_not_mutated(self) -> None:
        d = {"x": 1}
        s = Store(d)
        d["x"] = 99
        assert s["x"] == 1

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError):
            Store({"": 1})


class TestImmutability:
    def test_setattr_raises(self) -> None:
        s = Store({"x": 1})
        with pytest.raises(AttributeError):
            s._data = {}  # type: ignore[misc]

    def test_original_unchanged_after_set(self) -> None:
        s = Store({"x": 1})
        s.set("x", 99)
        assert s["x"] == 1


class TestContains:
    def test_present_attribute(self) -> None:
        assert "x" in Store({"x": 1})

    def test_absent_attribute(self) -> None:
        assert "y" not in Store({"x": 1})

    def test_empty_store(self) -> None:
        assert "x" not in Store()


class TestLen:
    def test_empty(self) -> None:
        assert len(Store()) == 0

    def test_single(self) -> None:
        assert len(Store({"a": 1})) == 1

    def test_multiple(self) -> None:
        assert len(Store({"a": 1, "b": 2, "c": 3})) == 3


class TestGetItem:
    def test_existing_key(self) -> None:
        assert Store({"k": "v"})["k"] == "v"

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Store({"x": 1})["y"]

    def test_value_none(self) -> None:
        assert Store({"k": None})["k"] is None


class TestGet:
    def test_present_returns_value(self) -> None:
        assert Store({"x": 42}).get("x") == 42

    def test_absent_returns_none_by_default(self) -> None:
        assert Store().get("x") is None

    def test_absent_returns_custom_default(self) -> None:
        assert Store().get("x", 0) == 0

    def test_value_none_returned_as_is(self) -> None:
        assert Store({"k": None}).get("k", "default") is None


class TestAttributes:
    def test_empty_store(self) -> None:
        assert Store().attributes == frozenset()

    def test_returns_frozenset(self) -> None:
        assert isinstance(Store({"a": 1}).attributes, frozenset)

    def test_correct_keys(self) -> None:
        assert Store({"a": 1, "b": 2}).attributes == frozenset({"a", "b"})

    def test_attributes_not_affected_by_original_dict(self) -> None:
        d = {"a": 1}
        s = Store(d)
        d["b"] = 2
        assert s.attributes == frozenset({"a"})


class TestTest:
    def test_absent_attribute_returns_false(self) -> None:
        assert Store().test("x", lambda v: True) is False

    def test_predicate_true(self) -> None:
        assert Store({"x": 4}).test("x", lambda v: v > 3) is True

    def test_predicate_false(self) -> None:
        assert Store({"x": 4}).test("x", lambda v: v > 10) is False

    def test_predicate_not_called_when_absent(self) -> None:
        called = []
        Store().test("x", lambda v: called.append(v) or True)
        assert called == []

    def test_value_none_predicate_called(self) -> None:
        assert Store({"x": None}).test("x", lambda v: v is None) is True

    def test_predicate_receives_correct_value(self) -> None:
        received = []
        Store({"x": 42}).test("x", lambda v: received.append(v) or True)
        assert received == [42]

    def test_bool_predicate(self) -> None:
        assert Store({"flag": True}).test("flag", bool) is True
        assert Store({"flag": False}).test("flag", bool) is False


class TestHas:
    def test_present_attribute(self) -> None:
        assert Store({"x": 1}).has("x") is True

    def test_absent_attribute(self) -> None:
        assert Store({"x": 1}).has("y") is False

    def test_empty_store(self) -> None:
        assert Store().has("x") is False

    def test_consistent_with_contains(self) -> None:
        s = Store({"a": 1, "b": 2})
        for key in ("a", "b", "c"):
            assert s.has(key) == (key in s)


class TestSet:
    def test_update_existing(self) -> None:
        s2 = Store({"x": 1}).set("x", 10)
        assert s2["x"] == 10

    def test_add_new_attribute(self) -> None:
        s2 = Store({"x": 1}).set("y", 2)
        assert s2["y"] == 2
        assert s2["x"] == 1

    def test_returns_new_instance(self) -> None:
        s = Store({"x": 1})
        s2 = s.set("x", 2)
        assert s2 is not s

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError):
            Store().set("", 1)

    def test_set_on_empty_store(self) -> None:
        s = Store().set("a", 42)
        assert s["a"] == 42
        assert len(s) == 1


class TestEquality:
    def test_equal_stores(self) -> None:
        assert Store({"x": 1}) == Store({"x": 1})

    def test_different_values(self) -> None:
        assert Store({"x": 1}) != Store({"x": 2})

    def test_different_keys(self) -> None:
        assert Store({"x": 1}) != Store({"y": 1})

    def test_empty_stores_equal(self) -> None:
        assert Store() == Store()

    def test_not_equal_to_dict(self) -> None:
        assert Store({"x": 1}) != {"x": 1}


class TestHashing:
    def test_equal_stores_same_hash(self) -> None:
        assert hash(Store({"x": 1})) == hash(Store({"x": 1}))

    def test_usable_as_dict_key(self) -> None:
        s = Store({"x": 1})
        d = {s: "found"}
        assert d[Store({"x": 1})] == "found"

    def test_usable_in_set(self) -> None:
        a = Store({"x": 1})
        b = Store({"x": 1})
        c = Store({"x": 2})
        assert len({a, b, c}) == 2


class TestRepr:
    def test_empty(self) -> None:
        assert repr(Store()) == "Store({})"

    def test_with_values(self) -> None:
        assert repr(Store({"a": 1})) == "Store({'a': 1})"
