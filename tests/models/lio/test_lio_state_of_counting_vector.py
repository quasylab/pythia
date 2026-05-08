"""Tests for pythia.models.lio.lio_state_of_counting_vector."""

from __future__ import annotations

import random

import pytest

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state import LIOState
from pythia.models.lio.lio_state_of_counting_vector import LIOStateOfCountingVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

A = Agent("a")
B = Agent("b")
C = Agent("c")


def _uniform_wfun(agent: Agent):  # type: ignore[no-untyped-def]
    return lambda act: 1.0


def _make_loop_defs(*names: str) -> tuple[AgentDefinitions, list[Agent]]:
    tick = Action("tick")
    defs = AgentDefinitions()
    agents = [defs.get(n) for n in names]
    for agent in agents:
        defs.add(agent, tick, agent, 1.0)
    return defs, agents


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLIOStateOfCountingVectorConstruction:
    def test_single_agent(self) -> None:
        state = LIOStateOfCountingVector({A: 1})
        assert state.size == 1

    def test_multiple_agents(self) -> None:
        state = LIOStateOfCountingVector({A: 2, B: 3})
        assert state.size == 5

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LIOStateOfCountingVector({})

    def test_zero_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            LIOStateOfCountingVector({A: 0})

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            LIOStateOfCountingVector({A: -1})

    def test_non_integer_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            LIOStateOfCountingVector({A: 1.5})  # type: ignore[dict-item]

    def test_is_lio_state_subclass(self) -> None:
        assert isinstance(LIOStateOfCountingVector({A: 1}), LIOState)

    def test_immutable(self) -> None:
        state = LIOStateOfCountingVector({A: 1})
        with pytest.raises(AttributeError):
            state._counts = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# counts property
# ---------------------------------------------------------------------------

class TestCounts:
    def test_counts_returns_copy(self) -> None:
        state = LIOStateOfCountingVector({A: 2, B: 1})
        c = state.counts
        c[C] = 99  # mutate the returned copy
        assert state.count(C) == 0  # original unaffected

    def test_counts_content(self) -> None:
        state = LIOStateOfCountingVector({A: 3, B: 2})
        assert state.counts == {A: 3, B: 2}


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    def test_size_single_agent_count_one(self) -> None:
        assert LIOStateOfCountingVector({A: 1}).size == 1

    def test_size_single_agent_multiple_copies(self) -> None:
        assert LIOStateOfCountingVector({A: 5}).size == 5

    def test_size_multiple_agents(self) -> None:
        assert LIOStateOfCountingVector({A: 2, B: 3, C: 1}).size == 6


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_present(self) -> None:
        state = LIOStateOfCountingVector({A: 3, B: 1})
        assert state.count(A) == 3

    def test_count_single_copy(self) -> None:
        state = LIOStateOfCountingVector({A: 3, B: 1})
        assert state.count(B) == 1

    def test_count_absent(self) -> None:
        state = LIOStateOfCountingVector({A: 2, B: 1})
        assert state.count(C) == 0

    def test_count_only_agent(self) -> None:
        state = LIOStateOfCountingVector({A: 7})
        assert state.count(A) == 7


# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------

class TestFraction:
    def test_fraction_all_same(self) -> None:
        state = LIOStateOfCountingVector({A: 4})
        assert state.fraction(A) == pytest.approx(1.0)

    def test_fraction_none(self) -> None:
        state = LIOStateOfCountingVector({A: 2})
        assert state.fraction(B) == pytest.approx(0.0)

    def test_fraction_half(self) -> None:
        state = LIOStateOfCountingVector({A: 1, B: 1})
        assert state.fraction(A) == pytest.approx(0.5)

    def test_fraction_sums_to_one(self) -> None:
        state = LIOStateOfCountingVector({A: 2, B: 1, C: 1})
        total = state.fraction(A) + state.fraction(B) + state.fraction(C)
        assert total == pytest.approx(1.0)

    def test_fraction_equals_count_over_size(self) -> None:
        state = LIOStateOfCountingVector({A: 3, B: 1})
        assert state.fraction(A) == pytest.approx(state.count(A) / state.size)

    def test_fraction_three_quarters(self) -> None:
        state = LIOStateOfCountingVector({A: 3, B: 1})
        assert state.fraction(A) == pytest.approx(0.75)
        assert state.fraction(B) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestLIOStateOfCountingVectorStep:
    def test_returns_lio_state_of_counting_vector(self) -> None:
        defs, agents = _make_loop_defs("a", "b")
        state = LIOStateOfCountingVector({agents[0]: 2, agents[1]: 1})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert isinstance(result, LIOStateOfCountingVector)

    def test_size_preserved(self) -> None:
        defs, agents = _make_loop_defs("a", "b")
        state = LIOStateOfCountingVector({agents[0]: 3, agents[1]: 2})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert result.size == state.size

    def test_self_loop_agent_unchanged(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.add(a, tick, a, 1.0)
        state = LIOStateOfCountingVector({a: 5})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert result.count(a) == 5

    def test_deterministic_transition(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        state = LIOStateOfCountingVector({a: 3})
        for seed in range(10):
            result = state.step(random.Random(seed), _uniform_wfun, defs)
            assert result.count(b) == 3
            assert result.count(a) == 0

    def test_all_copies_stepped_independently(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)
        state = LIOStateOfCountingVector({a: 2, b: 3})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert result.count(b) == 2   # two a's became b's
        assert result.count(a) == 3   # three b's became a's

    def test_no_transitions_agent_returns_itself(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        state = LIOStateOfCountingVector({a: 4})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert result.count(a) == 4

    def test_mixed_transitions_and_no_transitions(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        state = LIOStateOfCountingVector({a: 2, c: 3})
        result = state.step(random.Random(0), _uniform_wfun, defs)
        assert result.count(b) == 2
        assert result.count(c) == 3
        assert result.count(a) == 0

    def test_result_agents_drawn_from_reachable_set(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, c, 1.0)
        state = LIOStateOfCountingVector({a: 1})
        reachable = set()
        for seed in range(30):
            result = state.step(random.Random(seed), _uniform_wfun, defs)
            reachable.update(result.counts.keys())
        assert reachable == {b, c}

    def test_wfun_receives_correct_agent(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, a, 1.0)
        defs.add(b, tick, b, 1.0)
        state = LIOStateOfCountingVector({a: 2, b: 3})
        called_with: list[Agent] = []

        def recording_wfun(agent: Agent):  # type: ignore[no-untyped-def]
            called_with.append(agent)
            return lambda act: 1.0

        state.step(random.Random(0), recording_wfun, defs)
        # wfun called once per distinct agent type, not once per copy
        assert sorted(called_with, key=lambda ag: ag.name) == [a, b]

    def test_original_not_mutated(self) -> None:
        defs, agents = _make_loop_defs("a", "b")
        state = LIOStateOfCountingVector({agents[0]: 2, agents[1]: 3})
        state.step(random.Random(0), _uniform_wfun, defs)
        assert state.count(agents[0]) == 2
        assert state.count(agents[1]) == 3

    def test_deterministic_with_fixed_seed(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, a, 1.0)
        state = LIOStateOfCountingVector({a: 10})

        def _outcome(seed: int) -> dict[Agent, int]:
            return state.step(random.Random(seed), _uniform_wfun, defs).counts

        assert _outcome(7) == _outcome(7)

    def test_different_seeds_may_produce_different_outcomes(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, a, 1.0)
        state = LIOStateOfCountingVector({a: 1})
        results = {
            frozenset(state.step(random.Random(seed), _uniform_wfun, defs).counts.items())
            for seed in range(20)
        }
        assert len(results) > 1
