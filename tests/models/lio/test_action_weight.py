"""Tests for pythia.models.lio.action_weight."""

from __future__ import annotations

import pytest

from pythia.models.lio.action import Action
from pythia.models.lio.action_weight import ActionWeightFunction
from pythia.models.lio.agent import Agent
from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

A = Agent("a")
B = Agent("b")
C = Agent("c")

MOVE = Action("move")
STAY = Action("stay")
FLEE = Action("flee")


def _cfg(*agents: Agent) -> LIOStateOfIndividuals:
    return LIOStateOfIndividuals(list(agents))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestActionWeightFunctionConstruction:
    def test_action_weights_initially_empty(self) -> None:
        awf = ActionWeightFunction()
        assert len(awf.action_weights) == 0

    def test_default_weights_initially_empty(self) -> None:
        awf = ActionWeightFunction()
        assert len(awf.default_weights) == 0


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

class TestSet:
    def test_set_stores_function(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 1.0)
        assert MOVE in awf.action_weights[A]

    def test_set_function_is_callable(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 2.5)
        cfg = _cfg(A)
        assert awf.action_weights[A][MOVE](cfg) == pytest.approx(2.5)

    def test_set_overwrites_existing(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 1.0)
        awf.set(A, MOVE, lambda cfg: 9.0)
        cfg = _cfg(A)
        assert awf.action_weights[A][MOVE](cfg) == pytest.approx(9.0)

    def test_set_multiple_actions_for_same_agent(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 1.0)
        awf.set(A, STAY, lambda cfg: 0.5)
        cfg = _cfg(A)
        assert awf.action_weights[A][MOVE](cfg) == pytest.approx(1.0)
        assert awf.action_weights[A][STAY](cfg) == pytest.approx(0.5)

    def test_set_same_action_for_different_agents(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: cfg.fraction(A))
        awf.set(B, MOVE, lambda cfg: cfg.fraction(B))
        cfg = _cfg(A, A, B)
        assert awf.action_weights[A][MOVE](cfg) == pytest.approx(2 / 3)
        assert awf.action_weights[B][MOVE](cfg) == pytest.approx(1 / 3)

    def test_set_does_not_affect_other_agents(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 1.0)
        assert MOVE not in awf.action_weights.get(B, {})


# ---------------------------------------------------------------------------
# set_default
# ---------------------------------------------------------------------------

class TestSetDefault:
    def test_set_default_stores_function(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 0.5)
        assert MOVE in awf.default_weights

    def test_set_default_function_is_callable(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 0.5)
        cfg = _cfg(A)
        assert awf.default_weights[MOVE](cfg) == pytest.approx(0.5)

    def test_set_default_overwrites_existing(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 0.5)
        awf.set_default(MOVE, lambda cfg: 3.0)
        cfg = _cfg(A)
        assert awf.default_weights[MOVE](cfg) == pytest.approx(3.0)

    def test_set_default_multiple_actions(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 1.0)
        awf.set_default(STAY, lambda cfg: 0.0)
        cfg = _cfg(A)
        assert awf.default_weights[MOVE](cfg) == pytest.approx(1.0)
        assert awf.default_weights[STAY](cfg) == pytest.approx(0.0)

    def test_set_default_does_not_touch_action_weights(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 1.0)
        assert len(awf.action_weights) == 0


# ---------------------------------------------------------------------------
# get — priority: action_weights > default_weights > 0.0
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_agent_specific_weight(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 7.0)
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(7.0)

    def test_agent_specific_takes_priority_over_default(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 7.0)
        awf.set_default(MOVE, lambda cfg: 1.0)
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(7.0)

    def test_falls_back_to_default_when_no_agent_entry(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: 0.5)
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(0.5)

    def test_falls_back_to_default_when_agent_present_but_action_missing(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, STAY, lambda cfg: 1.0)
        awf.set_default(MOVE, lambda cfg: 0.5)
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(0.5)

    def test_returns_zero_when_no_entry_at_all(self) -> None:
        awf = ActionWeightFunction()
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(0.0)

    def test_returns_zero_when_action_unknown_and_no_default(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, STAY, lambda cfg: 1.0)
        cfg = _cfg(A)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(0.0)

    def test_configuration_passed_to_agent_specific_function(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: cfg.fraction(A))
        cfg = _cfg(A, A, B)
        assert awf.get(cfg)(A)(MOVE) == pytest.approx(2 / 3)

    def test_configuration_passed_to_default_function(self) -> None:
        awf = ActionWeightFunction()
        awf.set_default(MOVE, lambda cfg: cfg.fraction(A))
        cfg = _cfg(A, A, B)
        assert awf.get(cfg)(B)(MOVE) == pytest.approx(2 / 3)

    def test_different_agents_resolved_independently(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 1.0)
        awf.set_default(MOVE, lambda cfg: 0.5)
        cfg = _cfg(A, B)
        f = awf.get(cfg)
        assert f(A)(MOVE) == pytest.approx(1.0)   # agent-specific
        assert f(B)(MOVE) == pytest.approx(0.5)   # default

    def test_get_does_not_mutate_action_weights(self) -> None:
        awf = ActionWeightFunction()
        cfg = _cfg(A)
        awf.get(cfg)(B)(MOVE)  # B has no entry
        assert B not in awf.action_weights

    def test_multiple_actions_resolved_correctly(self) -> None:
        awf = ActionWeightFunction()
        awf.set(A, MOVE, lambda cfg: 2.0)
        awf.set_default(STAY, lambda cfg: 0.3)
        cfg = _cfg(A)
        f = awf.get(cfg)
        assert f(A)(MOVE) == pytest.approx(2.0)   # agent-specific
        assert f(A)(STAY) == pytest.approx(0.3)   # default
        assert f(A)(FLEE) == pytest.approx(0.0)   # missing
