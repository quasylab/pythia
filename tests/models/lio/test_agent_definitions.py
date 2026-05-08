"""Tests for pythia.models.lio.agent_definitions."""

from __future__ import annotations

import random

import pytest

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(name: str = "go") -> Action:
    return Action(name)


# ---------------------------------------------------------------------------
# Agent registry — get
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_get_returns_agent_with_correct_name(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("idle").name == "idle"

    def test_get_same_name_returns_same_instance(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("x") is defs.get("x")

    def test_get_different_names_return_different_instances(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("a") is not defs.get("b")

    def test_get_empty_name_raises(self) -> None:
        defs = AgentDefinitions()
        with pytest.raises(ValueError, match="non-empty"):
            defs.get("")

    def test_registries_are_independent(self) -> None:
        d1, d2 = AgentDefinitions(), AgentDefinitions()
        assert d1.get("a") is not d2.get("a")


# ---------------------------------------------------------------------------
# add / weight
# ---------------------------------------------------------------------------

class TestTransitions:
    def test_weight_returns_zero_when_no_transition(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        assert defs.weight(a, _action(), b) == 0.0

    def test_add_sets_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.5)
        assert defs.weight(a, t, b) == pytest.approx(1.5)

    def test_add_accumulates_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, b, 0.5)
        assert defs.weight(a, t, b) == pytest.approx(1.5)

    def test_add_zero_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        with pytest.raises(ValueError, match="positive"):
            defs.add(a, _action(), b, 0.0)

    def test_add_negative_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        with pytest.raises(ValueError, match="positive"):
            defs.add(a, _action(), b, -1.0)

    def test_add_invalid_weight_does_not_modify_table(self) -> None:
        # A failed add must leave the existing weight unchanged.
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 2.0)
        with pytest.raises(ValueError):
            defs.add(a, t, b, -1.0)
        assert defs.weight(a, t, b) == pytest.approx(2.0)

    def test_self_transition(self) -> None:
        # A source may have a transition to itself.
        defs = AgentDefinitions()
        a = defs.get("a")
        t = _action()
        defs.add(a, t, a, 1.0)
        assert defs.weight(a, t, a) == pytest.approx(1.0)

    def test_weight_unknown_source_does_not_create_entry(self) -> None:
        # Querying weight for an unknown source must not add it to the table.
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        _ = defs.weight(a, _action(), b)
        # After the query, adding a transition for a different source must
        # not be affected.
        c = defs.get("c")
        defs.add(c, _action(), b, 1.0)
        assert defs.weight(a, _action(), b) == 0.0

    def test_different_actions_are_independent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 2.0)
        assert defs.weight(a, t2, b) == 0.0

    def test_different_targets_are_independent(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 3.0)
        assert defs.weight(a, t, c) == 0.0

    def test_different_sources_are_independent(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, c, 1.0)
        assert defs.weight(b, t, c) == 0.0

    def test_multiple_sources_coexist(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(b, t, c, 2.0)
        assert defs.weight(a, t, b) == pytest.approx(1.0)
        assert defs.weight(b, t, c) == pytest.approx(2.0)
        assert defs.weight(a, t, c) == 0.0
        assert defs.weight(b, t, b) == 0.0


# ---------------------------------------------------------------------------
# total_weight
# ---------------------------------------------------------------------------

class TestTotalWeight:
    def test_returns_zero_when_no_transitions(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.total_weight(a, _action()) == 0.0

    def test_returns_zero_for_unknown_source(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.total_weight(a, _action()) == 0.0

    def test_returns_zero_for_unknown_action_on_known_source(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        assert defs.total_weight(a, t2) == 0.0

    def test_single_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 2.5)
        assert defs.total_weight(a, t) == pytest.approx(2.5)

    def test_multiple_targets_summed(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 2.5)
        assert defs.total_weight(a, t) == pytest.approx(3.5)

    def test_accumulation_reflected_in_total(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, b, 0.5)
        assert defs.total_weight(a, t) == pytest.approx(1.5)

    def test_equals_sum_of_individual_weights(self) -> None:
        defs = AgentDefinitions()
        a, b, c, d = defs.get("a"), defs.get("b"), defs.get("c"), defs.get("d")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 2.0)
        defs.add(a, t, d, 0.5)
        expected = defs.weight(a, t, b) + defs.weight(a, t, c) + defs.weight(a, t, d)
        assert defs.total_weight(a, t) == pytest.approx(expected)

    def test_other_actions_not_included(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 3.0)
        assert defs.total_weight(a, t2) == 0.0

    def test_other_sources_not_included(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(b, t, c, 5.0)
        assert defs.total_weight(a, t) == 0.0


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_raises_when_no_transitions_for_source(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        with pytest.raises(ValueError, match="No transitions"):
            defs.execute(a, _action(), random.Random(0))

    def test_raises_when_action_unknown_for_source(self) -> None:
        # Source has transitions on t1 but execute is called with t2.
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        with pytest.raises(ValueError, match="No transitions"):
            defs.execute(a, t2, random.Random(0))

    def test_single_target_always_selected(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 5.0)
        for seed in range(10):
            assert defs.execute(a, t, random.Random(seed)) == b

    def test_self_transition_returns_source(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        t = _action()
        defs.add(a, t, a, 1.0)
        for seed in range(5):
            assert defs.execute(a, t, random.Random(seed)) == a

    def test_returns_one_of_the_reachable_agents(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        results = {defs.execute(a, t, random.Random(seed)) for seed in range(20)}
        assert results == {b, c}

    def test_result_never_outside_reachable_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c, d = defs.get("a"), defs.get("b"), defs.get("c"), defs.get("d")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        for seed in range(30):
            assert defs.execute(a, t, random.Random(seed)) in (b, c)
            assert defs.execute(a, t, random.Random(seed)) != d

    def test_execute_does_not_modify_transition_table(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.execute(a, t, random.Random(0))
        assert defs.weight(a, t, b) == pytest.approx(1.0)

    def test_weighted_distribution(self) -> None:
        # b has weight 3, c has weight 1 → b selected ~75% of the time.
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 3.0)
        defs.add(a, t, c, 1.0)
        rng = random.Random(42)
        counts: dict[Agent, int] = {b: 0, c: 0}
        n = 1000
        for _ in range(n):
            counts[defs.execute(a, t, rng)] += 1
        assert counts[b] > counts[c]
        assert abs(counts[b] / n - 0.75) < 0.05


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------

class TestSelectAction:
    def test_raises_when_no_transitions(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        with pytest.raises(ValueError, match="no defined transitions"):
            defs.select_action(a, lambda act: 1.0, random.Random(0))

    def test_raises_when_all_weights_zero(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, _action(), b, 1.0)
        with pytest.raises(ValueError, match="0.0"):
            defs.select_action(a, lambda act: 0.0, random.Random(0))

    def test_single_action_always_selected(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        for seed in range(10):
            assert defs.select_action(a, lambda act: 1.0, random.Random(seed)) == t

    def test_result_is_an_action_of_source(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, b, 1.0)
        for seed in range(20):
            assert defs.select_action(a, lambda act: 1.0, random.Random(seed)) in (t1, t2)

    def test_returns_one_of_available_actions(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, b, 1.0)
        results = {
            defs.select_action(a, lambda act: 1.0, random.Random(seed))
            for seed in range(20)
        }
        assert results == {t1, t2}

    def test_only_source_actions_considered(self) -> None:
        # b has transitions on t2; a only has t1.
        # select_action for a must never return t2.
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, c, 1.0)
        defs.add(b, t2, c, 1.0)
        for seed in range(20):
            assert defs.select_action(a, lambda act: 1.0, random.Random(seed)) == t1

    def test_zero_weight_action_never_selected(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, b, 1.0)
        results = {
            defs.select_action(a, lambda act: 0.0 if act == t2 else 1.0, random.Random(seed))
            for seed in range(20)
        }
        assert results == {t1}

    def test_deterministic_with_fixed_seed(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, b, 1.0)
        r1 = defs.select_action(a, lambda act: 1.0, random.Random(77))
        r2 = defs.select_action(a, lambda act: 1.0, random.Random(77))
        assert r1 == r2

    def test_weighted_distribution(self) -> None:
        # wf gives t1 weight 3 and t2 weight 1 → t1 selected ~75% of the time.
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, b, 1.0)
        weights = {t1: 3.0, t2: 1.0}
        rng = random.Random(42)
        counts: dict[Action, int] = {t1: 0, t2: 0}
        n = 1000
        for _ in range(n):
            counts[defs.select_action(a, lambda act: weights[act], rng)] += 1
        assert counts[t1] > counts[t2]
        assert abs(counts[t1] / n - 0.75) < 0.05


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_source_when_no_transitions(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.step(a, lambda act: 1.0, random.Random(0)) is a

    def test_returns_target_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        assert defs.step(a, lambda act: 1.0, random.Random(0)) == b

    def test_self_loop_returns_source(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        t = _action()
        defs.add(a, t, a, 1.0)
        for seed in range(5):
            assert defs.step(a, lambda act: 1.0, random.Random(seed)) == a

    def test_result_is_reachable_agent(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t1, t2 = Action("t1"), Action("t2")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, c, 1.0)
        results = {defs.step(a, lambda act: 1.0, random.Random(seed)) for seed in range(20)}
        assert results == {b, c}

    def test_result_never_outside_reachable_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c, d = defs.get("a"), defs.get("b"), defs.get("c"), defs.get("d")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        for seed in range(30):
            result = defs.step(a, lambda act: 1.0, random.Random(seed))
            assert result != d

    def test_wf_zero_for_all_actions_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, _action(), b, 1.0)
        with pytest.raises(ValueError):
            defs.step(a, lambda act: 0.0, random.Random(0))

    def test_step_does_not_modify_transition_table(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.step(a, lambda act: 1.0, random.Random(0))
        assert defs.weight(a, t, b) == pytest.approx(1.0)
        assert defs.total_weight(a, t) == pytest.approx(1.0)

    def test_deterministic_with_fixed_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        r1 = defs.step(a, lambda act: 1.0, random.Random(42))
        r2 = defs.step(a, lambda act: 1.0, random.Random(42))
        assert r1 == r2

    def test_sources_stepped_independently(self) -> None:
        # Stepping source a must not affect step results for source b.
        defs = AgentDefinitions()
        a, b, c, d = defs.get("a"), defs.get("b"), defs.get("c"), defs.get("d")
        t = _action()
        defs.add(a, t, c, 1.0)
        defs.add(b, t, d, 1.0)
        assert defs.step(a, lambda act: 1.0, random.Random(0)) == c
        assert defs.step(b, lambda act: 1.0, random.Random(0)) == d

    def test_chained_steps(self) -> None:
        # a --tick--> b --tick--> a, two steps must return to a.
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        t = _action()
        defs.add(a, t, b, 1.0)
        defs.add(b, t, a, 1.0)
        rng = random.Random(0)
        step1 = defs.step(a, lambda act: 1.0, rng)
        step2 = defs.step(step1, lambda act: 1.0, rng)
        assert step1 == b
        assert step2 == a
