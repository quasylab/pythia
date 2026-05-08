"""Tests for pythia.models.lio.lio_state_of_individuals."""

from __future__ import annotations

import random

import pytest

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(*names: str) -> LIOStateOfIndividuals:
    return LIOStateOfIndividuals([Agent(n) for n in names])


A = Agent("a")
B = Agent("b")
C = Agent("c")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLIOStateOfIndividualsConstruction:
    def test_single_agent(self) -> None:
        cfg = LIOStateOfIndividuals([A])
        assert cfg.size == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LIOStateOfIndividuals([])

    def test_immutable(self) -> None:
        cfg = LIOStateOfIndividuals([A, B])
        with pytest.raises(AttributeError):
            cfg._agents = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# agent(i)
# ---------------------------------------------------------------------------

class TestAgentAccess:
    def test_first_agent(self) -> None:
        cfg = LIOStateOfIndividuals([A, B, C])
        assert cfg.agent(0) == A

    def test_last_agent(self) -> None:
        cfg = LIOStateOfIndividuals([A, B, C])
        assert cfg.agent(2) == C

    def test_order_preserved(self) -> None:
        cfg = LIOStateOfIndividuals([C, A, B])
        assert cfg.agent(0) == C
        assert cfg.agent(1) == A
        assert cfg.agent(2) == B

    def test_out_of_range_raises(self) -> None:
        cfg = LIOStateOfIndividuals([A])
        with pytest.raises(IndexError):
            cfg.agent(1)


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    def test_size_one(self) -> None:
        assert LIOStateOfIndividuals([A]).size == 1

    def test_size_multiple(self) -> None:
        assert LIOStateOfIndividuals([A, A, B, C]).size == 4

    def test_duplicates_counted(self) -> None:
        assert LIOStateOfIndividuals([A, A, A]).size == 3


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_present(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, B])
        assert cfg.count(A) == 2

    def test_count_single(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, B])
        assert cfg.count(B) == 1

    def test_count_absent(self) -> None:
        cfg = LIOStateOfIndividuals([A, B])
        assert cfg.count(C) == 0

    def test_count_all_same(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, A])
        assert cfg.count(A) == 3


# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------

class TestFraction:
    def test_fraction_all_same(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, A])
        assert cfg.fraction(A) == pytest.approx(1.0)

    def test_fraction_none(self) -> None:
        cfg = LIOStateOfIndividuals([A, A])
        assert cfg.fraction(B) == pytest.approx(0.0)

    def test_fraction_half(self) -> None:
        cfg = LIOStateOfIndividuals([A, B])
        assert cfg.fraction(A) == pytest.approx(0.5)

    def test_fraction_sums_to_one(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, B, C])
        total = cfg.fraction(A) + cfg.fraction(B) + cfg.fraction(C)
        assert total == pytest.approx(1.0)

    def test_fraction_equals_count_over_size(self) -> None:
        cfg = LIOStateOfIndividuals([A, A, B, C])
        assert cfg.fraction(A) == pytest.approx(cfg.count(A) / cfg.size)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

def _uniform_wfun(agent: Agent):  # type: ignore[no-untyped-def]
    """Weight function that gives every action weight 1.0."""
    return lambda act: 1.0


def _make_loop_defs(*names: str) -> tuple[AgentDefinitions, list[Agent]]:
    """Create an AgentDefinitions with self-loop transitions via Action('tick')."""
    tick = Action("tick")
    defs = AgentDefinitions()
    agents = [defs.get(n) for n in names]
    for agent in agents:
        defs.add(agent, tick, agent, 1.0)
    return defs, agents


class TestLIOStateOfIndividualsStep:
    def test_returns_lio_state_of_individuals(self) -> None:
        defs, agents = _make_loop_defs("a", "b")
        cfg = LIOStateOfIndividuals(agents)
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert isinstance(result, LIOStateOfIndividuals)

    def test_size_preserved(self) -> None:
        defs, agents = _make_loop_defs("a", "b", "c")
        cfg = LIOStateOfIndividuals(agents)
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert result.size == cfg.size

    def test_self_loop_agent_unchanged(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.add(a, tick, a, 1.0)
        cfg = LIOStateOfIndividuals([a])
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert result.agent(0) == a

    def test_deterministic_transition(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        cfg = LIOStateOfIndividuals([a])
        for seed in range(10):
            result = cfg.step(random.Random(seed), _uniform_wfun, defs)
            assert result.agent(0) == b

    def test_each_agent_stepped_independently(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)
        cfg = LIOStateOfIndividuals([a, b])
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert result.agent(0) == b
        assert result.agent(1) == a

    def test_order_of_agents_preserved(self) -> None:
        defs, agents = _make_loop_defs("x", "y", "z")
        cfg = LIOStateOfIndividuals(agents)
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        for i, agent in enumerate(agents):
            assert result.agent(i) == agent

    def test_wfun_receives_correct_agent(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, a, 1.0)
        defs.add(b, tick, b, 1.0)
        cfg = LIOStateOfIndividuals([a, b])
        called_with: list[Agent] = []

        def recording_wfun(agent: Agent):  # type: ignore[no-untyped-def]
            called_with.append(agent)
            return lambda act: 1.0

        cfg.step(random.Random(0), recording_wfun, defs)
        assert called_with == [a, b]

    def test_no_transitions_agent_returns_itself(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        cfg = LIOStateOfIndividuals([a])
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert result.agent(0) == a

    def test_mixed_transitions_and_no_transitions(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        cfg = LIOStateOfIndividuals([a, c])
        result = cfg.step(random.Random(0), _uniform_wfun, defs)
        assert result.agent(0) == b
        assert result.agent(1) == c

    def test_result_agents_drawn_from_reachable_set(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, c, 1.0)
        cfg = LIOStateOfIndividuals([a])
        results = {
            cfg.step(random.Random(seed), _uniform_wfun, defs).agent(0)
            for seed in range(30)
        }
        assert results == {b, c}

    def test_original_not_mutated(self) -> None:
        defs, agents = _make_loop_defs("a", "b")
        cfg = LIOStateOfIndividuals(agents)
        cfg.step(random.Random(0), _uniform_wfun, defs)
        assert cfg.agent(0) == agents[0]
        assert cfg.agent(1) == agents[1]
