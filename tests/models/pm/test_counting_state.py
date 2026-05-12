"""Tests for pythia.models.pm.counting_state."""

from __future__ import annotations

import types

import pytest

from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.species import Species


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pop() -> Population:
    p = Population()
    p.register("wolf",   {"diet": "carnivore"})  # index 0
    p.register("rabbit", {"diet": "herbivore"})  # index 1
    p.register("deer",   {"diet": "herbivore"})  # index 2
    return p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCountingStateConstruction:
    def test_stores_positive_counts(self, pop: Population) -> None:
        state = CountingState({pop[0]: 10, pop[1]: 50})
        assert state.counting_vector[0] == 10
        assert state.counting_vector[1] == 50

    def test_uses_species_index_as_key(self, pop: Population) -> None:
        wolf = pop[0]
        state = CountingState({wolf: 5})
        assert wolf.index in state.counting_vector
        assert state.counting_vector[wolf.index] == 5

    def test_zero_count_not_stored(self, pop: Population) -> None:
        state = CountingState({pop[0]: 0, pop[1]: 3})
        assert 0 not in state.counting_vector
        assert state.counting_vector[1] == 3

    def test_empty_mapping_gives_empty_vector(self) -> None:
        state = CountingState({})
        assert dict(state.counting_vector) == {}

    def test_all_zero_counts_gives_empty_vector(self, pop: Population) -> None:
        state = CountingState({pop[0]: 0, pop[1]: 0})
        assert dict(state.counting_vector) == {}


class TestCountingStateValidation:
    def test_negative_count_raises(self, pop: Population) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CountingState({pop[0]: -1})

    def test_float_count_raises(self, pop: Population) -> None:
        with pytest.raises(TypeError, match="integers"):
            CountingState({pop[0]: 3.5})  # type: ignore[dict-item]

    def test_string_count_raises(self, pop: Population) -> None:
        with pytest.raises(TypeError, match="integers"):
            CountingState({pop[0]: "many"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestSize:
    def test_empty_state_is_zero(self) -> None:
        assert CountingState({}).size == 0

    def test_single_species(self, pop: Population) -> None:
        assert CountingState({pop[0]: 5}).size == 5

    def test_multiple_species(self, pop: Population) -> None:
        assert CountingState({pop[0]: 3, pop[1]: 7}).size == 10

    def test_zero_counts_excluded(self, pop: Population) -> None:
        assert CountingState({pop[0]: 0, pop[1]: 4}).size == 4

    def test_size_after_add(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3})
        b = CountingState({pop[1]: 5})
        assert a.add(b).size == 8

    def test_size_after_remove(self, pop: Population) -> None:
        a = CountingState({pop[0]: 10, pop[1]: 5})
        b = CountingState({pop[0]: 3})
        assert a.remove(b).size == 12

    def test_returns_int(self, pop: Population) -> None:
        assert isinstance(CountingState({pop[0]: 2}).size, int)


class TestCount:
    def test_present_species(self, pop: Population) -> None:
        state = CountingState({pop[0]: 7, pop[1]: 3})
        assert state.count(pop[0]) == 7

    def test_another_present_species(self, pop: Population) -> None:
        state = CountingState({pop[0]: 7, pop[1]: 3})
        assert state.count(pop[1]) == 3

    def test_absent_species_returns_zero(self, pop: Population) -> None:
        state = CountingState({pop[0]: 5})
        assert state.count(pop[1]) == 0

    def test_empty_state_returns_zero(self, pop: Population) -> None:
        assert CountingState({}).count(pop[0]) == 0

    def test_returns_int(self, pop: Population) -> None:
        state = CountingState({pop[0]: 4})
        assert isinstance(state.count(pop[0]), int)

    def test_zero_stored_count_returns_zero(self, pop: Population) -> None:
        # CountingState drops zero counts at construction; verify count() returns 0
        state = CountingState({pop[0]: 0, pop[1]: 2})
        assert state.count(pop[0]) == 0


# ---------------------------------------------------------------------------
# counting_vector is immutable
# ---------------------------------------------------------------------------

class TestCountingVectorImmutability:
    def test_counting_vector_is_mapping_proxy(self, pop: Population) -> None:
        state = CountingState({pop[0]: 1})
        assert isinstance(state.counting_vector, types.MappingProxyType)

    def test_counting_vector_raises_on_set(self, pop: Population) -> None:
        state = CountingState({pop[0]: 1})
        with pytest.raises(TypeError):
            state.counting_vector[0] = 99  # type: ignore[index]


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestCountingStateEquality:
    def test_equal_when_same_counts(self, pop: Population) -> None:
        a = CountingState({pop[0]: 10, pop[1]: 5})
        b = CountingState({pop[0]: 10, pop[1]: 5})
        assert a == b

    def test_equal_when_zero_omitted_consistently(self, pop: Population) -> None:
        # Explicitly passing 0 vs omitting the key should produce equal states.
        a = CountingState({pop[0]: 3, pop[1]: 0})
        b = CountingState({pop[0]: 3})
        assert a == b

    def test_not_equal_when_counts_differ(self, pop: Population) -> None:
        a = CountingState({pop[0]: 10})
        b = CountingState({pop[0]: 9})
        assert a != b

    def test_not_equal_when_indices_differ(self, pop: Population) -> None:
        a = CountingState({pop[0]: 5})
        b = CountingState({pop[1]: 5})
        assert a != b

    def test_not_equal_to_non_counting_state(self, pop: Population) -> None:
        state = CountingState({pop[0]: 1})
        assert state != {0: 1}
        assert state != 1
        assert state != None  # noqa: E711

    def test_two_empty_states_are_equal(self) -> None:
        assert CountingState({}) == CountingState({})


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestCountingStateHash:
    def test_equal_states_have_equal_hash(self, pop: Population) -> None:
        a = CountingState({pop[0]: 4, pop[1]: 7})
        b = CountingState({pop[0]: 4, pop[1]: 7})
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self, pop: Population) -> None:
        state = CountingState({pop[0]: 2})
        d = {state: "stored"}
        same = CountingState({pop[0]: 2})
        assert d[same] == "stored"

    def test_usable_in_set(self, pop: Population) -> None:
        a = CountingState({pop[0]: 1})
        b = CountingState({pop[0]: 1})  # equal to a
        c = CountingState({pop[1]: 1})  # different
        assert len({a, b, c}) == 2


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------

class TestCountingStateContains:
    def test_identical_state_contains_itself(self, pop: Population) -> None:
        state = CountingState({pop[0]: 5, pop[1]: 10})
        assert state.contains(state)

    def test_larger_contains_smaller(self, pop: Population) -> None:
        large = CountingState({pop[0]: 10, pop[1]: 20})
        small = CountingState({pop[0]: 3,  pop[1]: 7})
        assert large.contains(small)

    def test_smaller_does_not_contain_larger(self, pop: Population) -> None:
        large = CountingState({pop[0]: 10})
        small = CountingState({pop[0]: 3})
        assert not small.contains(large)

    def test_contains_empty_state_always_true(self, pop: Population) -> None:
        state = CountingState({pop[0]: 5})
        assert state.contains(CountingState({}))

    def test_empty_state_contains_empty_state(self) -> None:
        assert CountingState({}).contains(CountingState({}))

    def test_empty_state_does_not_contain_non_empty(self, pop: Population) -> None:
        empty = CountingState({})
        non_empty = CountingState({pop[0]: 1})
        assert not empty.contains(non_empty)

    def test_contains_subset_of_species(self, pop: Population) -> None:
        # self has wolf+rabbit, other has only rabbit
        full    = CountingState({pop[0]: 5, pop[1]: 10})
        partial = CountingState({pop[1]: 8})
        assert full.contains(partial)

    def test_does_not_contain_when_one_species_short(self, pop: Population) -> None:
        a = CountingState({pop[0]: 5, pop[1]: 10})
        b = CountingState({pop[0]: 5, pop[1]: 11})  # one more rabbit
        assert not a.contains(b)

    def test_boundary_equal_counts_satisfies_contains(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3})
        b = CountingState({pop[0]: 3})
        assert a.contains(b)
        assert b.contains(a)

    def test_missing_species_in_self_counts_as_zero(self, pop: Population) -> None:
        # self has no wolf; other requires 1 wolf → not contained
        no_wolf = CountingState({pop[1]: 10})
        needs_wolf = CountingState({pop[0]: 1})
        assert not no_wolf.contains(needs_wolf)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestCountingStateAdd:
    def test_returns_new_instance(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3})
        b = CountingState({pop[0]: 2})
        c = a.add(b)
        assert c is not a
        assert c is not b

    def test_receiver_unchanged(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3})
        b = CountingState({pop[0]: 2})
        a.add(b)
        assert a.counting_vector[0] == 3

    def test_other_unchanged(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3})
        b = CountingState({pop[0]: 2})
        a.add(b)
        assert b.counting_vector[0] == 2

    def test_overlapping_counts_are_summed(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3, pop[1]: 10})
        b = CountingState({pop[0]: 2, pop[1]: 5})
        c = a.add(b)
        assert c.counting_vector[pop[0].index] == 5
        assert c.counting_vector[pop[1].index] == 15

    def test_disjoint_species_are_merged(self, pop: Population) -> None:
        a = CountingState({pop[0]: 4})
        b = CountingState({pop[1]: 7})
        c = a.add(b)
        assert c.counting_vector[pop[0].index] == 4
        assert c.counting_vector[pop[1].index] == 7

    def test_add_empty_returns_equal_state(self, pop: Population) -> None:
        a = CountingState({pop[0]: 5})
        c = a.add(CountingState({}))
        assert c == a

    def test_empty_add_other_returns_equal_to_other(self, pop: Population) -> None:
        b = CountingState({pop[0]: 5})
        c = CountingState({}).add(b)
        assert c == b

    def test_add_empty_to_empty_gives_empty(self) -> None:
        assert CountingState({}).add(CountingState({})) == CountingState({})

    def test_commutativity(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3, pop[1]: 10})
        b = CountingState({pop[0]: 7, pop[2]: 2})
        assert a.add(b) == b.add(a)

    def test_associativity(self, pop: Population) -> None:
        a = CountingState({pop[0]: 1})
        b = CountingState({pop[1]: 2})
        c = CountingState({pop[2]: 3})
        assert a.add(b).add(c) == a.add(b.add(c))

    def test_result_contains_both_operands(self, pop: Population) -> None:
        a = CountingState({pop[0]: 3, pop[1]: 10})
        b = CountingState({pop[0]: 2, pop[2]: 5})
        c = a.add(b)
        assert c.contains(a)
        assert c.contains(b)

    def test_result_is_counting_state(self, pop: Population) -> None:
        a = CountingState({pop[0]: 1})
        b = CountingState({pop[0]: 1})
        assert isinstance(a.add(b), CountingState)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestCountingStateRepr:
    def test_repr_contains_class_name(self, pop: Population) -> None:
        state = CountingState({pop[0]: 5})
        assert "CountingState" in repr(state)

    def test_repr_contains_index_and_count(self, pop: Population) -> None:
        state = CountingState({pop[0]: 7})
        r = repr(state)
        assert "0" in r
        assert "7" in r
