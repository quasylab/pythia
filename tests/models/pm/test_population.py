"""Tests for pythia.models.pm.population."""

from __future__ import annotations

import pytest

from pythia.models.pm.population import Population
from pythia.models.pm.species import Species


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_population() -> tuple[Population, Species, Species]:
    """Return a population with wolf (index 0) and rabbit (index 1)."""
    pop = Population()
    wolf   = pop.register("wolf",   {"diet": "carnivore", "max_age": 15})
    rabbit = pop.register("rabbit", {"diet": "herbivore", "max_age": 8})
    return pop, wolf, rabbit


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestPopulationRegister:
    def test_returns_species(self) -> None:
        pop = Population()
        s = pop.register("fox", {"habitat": "forest"})
        assert isinstance(s, Species)

    def test_name_and_attributes_stored(self) -> None:
        pop = Population()
        s = pop.register("fox", {"habitat": "forest"})
        assert s.name == "fox"
        assert s.attributes["habitat"] == "forest"

    def test_first_species_gets_index_zero(self) -> None:
        pop = Population()
        s = pop.register("fox")
        assert s.index == 0

    def test_indices_assigned_sequentially(self) -> None:
        pop = Population()
        a = pop.register("a")
        b = pop.register("b")
        c = pop.register("c")
        assert [a.index, b.index, c.index] == [0, 1, 2]

    def test_same_content_returns_identical_instance(self) -> None:
        pop = Population()
        s1 = pop.register("wolf", {"diet": "carnivore"})
        s2 = pop.register("wolf", {"diet": "carnivore"})
        assert s1 is s2

    def test_default_attributes_is_empty(self) -> None:
        pop = Population()
        s = pop.register("ghost")
        assert dict(s.attributes) == {}

    def test_none_attributes_treated_as_empty(self) -> None:
        pop = Population()
        s = pop.register("ghost", None)
        assert dict(s.attributes) == {}

    def test_unhashable_attribute_value_raises(self) -> None:
        pop = Population()
        with pytest.raises(ValueError, match="unhashable"):
            pop.register("bad", {"data": [1, 2, 3]})

    def test_nested_dict_attribute_value_raises(self) -> None:
        pop = Population()
        with pytest.raises(ValueError, match="unhashable"):
            pop.register("bad", {"nested": {"key": "value"}})


# ---------------------------------------------------------------------------
# Index uniqueness invariant
# ---------------------------------------------------------------------------

class TestPopulationIndexInvariant:
    def test_same_name_same_attributes_same_index(self) -> None:
        pop = Population()
        a = pop.register("wolf", {"diet": "carnivore", "max_age": 15})
        b = pop.register("wolf", {"diet": "carnivore", "max_age": 15})
        assert a.index == b.index

    def test_different_name_different_index(self) -> None:
        pop, wolf, rabbit = make_population()
        assert wolf.index != rabbit.index

    def test_same_name_different_attributes_different_index(self) -> None:
        pop = Population()
        young = pop.register("wolf", {"max_age": 5})
        old   = pop.register("wolf", {"max_age": 20})
        assert young.index != old.index

    def test_different_name_same_attributes_different_index(self) -> None:
        pop = Population()
        lion  = pop.register("lion",  {"diet": "carnivore"})
        tiger = pop.register("tiger", {"diet": "carnivore"})
        assert lion.index != tiger.index

    def test_attribute_order_does_not_affect_index(self) -> None:
        pop = Population()
        s1 = pop.register("bat", {"wings": True, "nocturnal": True})
        s2 = pop.register("bat", {"nocturnal": True, "wings": True})
        assert s1.index == s2.index

    def test_empty_attributes_consistent(self) -> None:
        pop = Population()
        a = pop.register("ghost")
        b = pop.register("ghost", {})
        assert a.index == b.index

    def test_indices_unique_across_many_species(self) -> None:
        pop = Population()
        species = [pop.register(f"species_{i}", {"id": i}) for i in range(50)]
        indices = [s.index for s in species]
        assert len(set(indices)) == 50

    def test_independent_populations_have_independent_indices(self) -> None:
        pop_a = Population()
        pop_b = Population()
        _ = pop_a.register("placeholder")  # shifts indices in pop_a only
        wolf_a = pop_a.register("wolf")
        wolf_b = pop_b.register("wolf")
        assert wolf_a.index != wolf_b.index


# ---------------------------------------------------------------------------
# Collection interface
# ---------------------------------------------------------------------------

class TestPopulationLen:
    def test_empty_population(self) -> None:
        assert len(Population()) == 0

    def test_len_after_registration(self) -> None:
        pop, wolf, rabbit = make_population()
        assert len(pop) == 2

    def test_re_registration_does_not_increase_len(self) -> None:
        pop = Population()
        pop.register("wolf")
        pop.register("wolf")
        assert len(pop) == 1


class TestPopulationIter:
    def test_empty_iteration(self) -> None:
        assert list(Population()) == []

    def test_iteration_order_matches_index_order(self) -> None:
        pop, wolf, rabbit = make_population()
        species_list = list(pop)
        assert species_list[0] is wolf
        assert species_list[1] is rabbit

    def test_iteration_yields_all_species(self) -> None:
        pop, wolf, rabbit = make_population()
        assert set(pop) == {wolf, rabbit}


class TestPopulationContains:
    def test_registered_species_is_contained(self) -> None:
        pop, wolf, _ = make_population()
        assert wolf in pop

    def test_unregistered_species_not_contained(self) -> None:
        pop = Population()
        unregistered = Species("fox", {"habitat": "forest"}, index=0)
        assert unregistered not in pop

    def test_species_from_other_population_not_contained(self) -> None:
        pop_a = Population()
        pop_b = Population()
        wolf_b = pop_b.register("wolf")
        assert wolf_b not in pop_a

    def test_non_species_object_not_contained(self) -> None:
        pop, wolf, _ = make_population()
        assert "wolf" not in pop
        assert 0 not in pop

    def test_same_content_different_instance_is_contained(self) -> None:
        pop = Population()
        wolf = pop.register("wolf", {"diet": "carnivore"})
        # Construct independently with the same index
        other = Species("wolf", {"diet": "carnivore"}, index=wolf.index)
        assert other in pop


class TestPopulationGetItem:
    def test_lookup_by_index(self) -> None:
        pop, wolf, rabbit = make_population()
        assert pop[0] is wolf
        assert pop[1] is rabbit

    def test_out_of_range_raises_key_error(self) -> None:
        pop = Population()
        with pytest.raises(KeyError):
            pop[0]

    def test_negative_index_raises_key_error(self) -> None:
        pop, _, _ = make_population()
        with pytest.raises(KeyError):
            pop[-1]


class TestPopulationGetByName:
    def test_single_match(self) -> None:
        pop, wolf, _ = make_population()
        result = pop.get_by_name("wolf")
        assert result == [wolf]

    def test_multiple_matches(self) -> None:
        pop = Population()
        large = pop.register("wolf", {"size": "large"})
        small = pop.register("wolf", {"size": "small"})
        result = pop.get_by_name("wolf")
        assert result == [large, small]

    def test_no_match_returns_empty_list(self) -> None:
        pop, _, _ = make_population()
        assert pop.get_by_name("dragon") == []


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestPopulationRepr:
    def test_repr_contains_size(self) -> None:
        pop, _, _ = make_population()
        assert "2" in repr(pop)

    def test_empty_repr(self) -> None:
        assert "0" in repr(Population())
