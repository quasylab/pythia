"""Tests for pythia.models.pm.species."""

from __future__ import annotations

import types

import pytest

from pythia.models.pm.species import Species


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------

class TestSpeciesProperties:
    def test_name_stored(self) -> None:
        s = Species("fox", {"habitat": "forest"}, index=0)
        assert s.name == "fox"

    def test_attributes_stored(self) -> None:
        s = Species("fox", {"habitat": "forest", "speed": 12}, index=0)
        assert s.attributes["habitat"] == "forest"
        assert s.attributes["speed"] == 12

    def test_index_stored(self) -> None:
        s = Species("deer", {"size": "medium"}, index=7)
        assert s.index == 7

    def test_index_is_int(self) -> None:
        s = Species("deer", {}, index=0)
        assert isinstance(s.index, int)

    def test_attributes_is_mapping_proxy(self) -> None:
        s = Species("bear", {"weight": 300}, index=0)
        assert isinstance(s.attributes, types.MappingProxyType)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestSpeciesImmutability:
    def test_cannot_set_name(self) -> None:
        s = Species("elk", {}, index=0)
        with pytest.raises((AttributeError, TypeError)):
            s.name = "moose"  # type: ignore[misc]

    def test_cannot_set_index(self) -> None:
        s = Species("elk", {}, index=0)
        with pytest.raises((AttributeError, TypeError)):
            s.index = 99  # type: ignore[misc]

    def test_cannot_set_attributes(self) -> None:
        s = Species("elk", {}, index=0)
        with pytest.raises((AttributeError, TypeError)):
            s.attributes = {}  # type: ignore[misc]

    def test_attributes_proxy_is_read_only(self) -> None:
        s = Species("elk", {"age": 5}, index=0)
        with pytest.raises(TypeError):
            s.attributes["age"] = 10  # type: ignore[index]

    def test_mutating_original_dict_does_not_affect_species(self) -> None:
        original = {"speed": 30}
        s = Species("cheetah", original, index=0)
        original["speed"] = 999
        assert s.attributes["speed"] == 30


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestSpeciesValidation:
    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Species("bad", {}, index=-1)

    def test_zero_index_is_valid(self) -> None:
        s = Species("ok", {}, index=0)
        assert s.index == 0


# ---------------------------------------------------------------------------
# Equality and hashing (index-based)
# ---------------------------------------------------------------------------

class TestSpeciesEqualityAndHash:
    def test_equal_when_same_index(self) -> None:
        a = Species("wolf", {"diet": "carnivore"}, index=3)
        b = Species("rabbit", {"diet": "herbivore"}, index=3)
        assert a == b

    def test_not_equal_when_different_index(self) -> None:
        a = Species("wolf", {"diet": "carnivore"}, index=0)
        b = Species("wolf", {"diet": "carnivore"}, index=1)
        assert a != b

    def test_not_equal_to_non_species(self) -> None:
        s = Species("wolf", {}, index=0)
        assert s != "wolf"
        assert s != 0
        assert s != None  # noqa: E711

    def test_hash_equal_for_same_index(self) -> None:
        a = Species("wolf", {"diet": "carnivore"}, index=5)
        b = Species("rabbit", {}, index=5)
        assert hash(a) == hash(b)

    def test_hash_differs_for_different_index(self) -> None:
        a = Species("wolf", {}, index=0)
        b = Species("wolf", {}, index=1)
        assert hash(a) != hash(b)

    def test_usable_as_dict_key(self) -> None:
        wolf = Species("wolf", {"diet": "carnivore"}, index=0)
        population: dict[Species, int] = {wolf: 42}
        same_wolf = Species("wolf", {"diet": "carnivore"}, index=0)
        assert population[same_wolf] == 42

    def test_usable_in_set(self) -> None:
        a = Species("wolf", {}, index=0)
        b = Species("wolf", {}, index=0)   # same index → same element
        c = Species("rabbit", {}, index=1)
        assert len({a, b, c}) == 2


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestSpeciesRepr:
    def test_repr_contains_name(self) -> None:
        s = Species("eagle", {"wingspan": 2}, index=0)
        assert "eagle" in repr(s)

    def test_repr_contains_index(self) -> None:
        s = Species("eagle", {"wingspan": 2}, index=4)
        assert "4" in repr(s)

    def test_repr_contains_attributes(self) -> None:
        s = Species("eagle", {"wingspan": 2}, index=0)
        assert "wingspan" in repr(s)
