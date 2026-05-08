"""Tests for pythia.models.nba.agent_definitions."""

from __future__ import annotations

import random

import pytest

from pythia.models.nba.agent import Agent
from pythia.models.nba.agent_definitions import AgentDefinitions
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment


C = Channel("c")
D = Channel("d")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_get_returns_agent(self) -> None:
        defs = AgentDefinitions()
        assert isinstance(defs.get("idle"), Agent)

    def test_get_same_name_same_instance(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("a") is defs.get("a")

    def test_get_different_names_different_instances(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("a") is not defs.get("b")

    def test_get_name_stored(self) -> None:
        defs = AgentDefinitions()
        assert defs.get("idle").name == "idle"

    def test_get_empty_name_raises(self) -> None:
        defs = AgentDefinitions()
        with pytest.raises(ValueError, match="non-empty"):
            defs.get("")

    def test_separate_definitions_independent(self) -> None:
        d1 = AgentDefinitions()
        d2 = AgentDefinitions()
        assert d1.get("a") is not d2.get("a")


# ---------------------------------------------------------------------------
# send / receive / broadcast_send / broadcast_receive
# ---------------------------------------------------------------------------

class TestTransitionMethods:
    # --- send ---

    def test_send_stores_transition(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs._send_transitions[a][C][b] == pytest.approx(1.0)

    def test_send_accumulates_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, C, b, 0.5)
        assert defs._send_transitions[a][C][b] == pytest.approx(1.5)

    def test_send_multiple_targets(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, C, c_, 3.0)
        assert defs._send_transitions[a][C][b] == pytest.approx(1.0)
        assert defs._send_transitions[a][C][c_] == pytest.approx(3.0)

    # --- receive ---

    def test_receive_stores_transition(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs._receive_transitions[a][C][b] == pytest.approx(1.0)

    def test_receive_accumulates_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_receive_transition(a, C, b, 0.5)
        assert defs._receive_transitions[a][C][b] == pytest.approx(1.5)

    # --- broadcast_send ---

    def test_broadcast_send_stores_transition(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        assert defs._broadcast_send_transitions[a][C][b] == pytest.approx(1.0)

    def test_broadcast_send_accumulates_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, C, b, 0.5)
        assert defs._broadcast_send_transitions[a][C][b] == pytest.approx(1.5)

    # --- broadcast_receive ---

    def test_broadcast_receive_stores_transition(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs._broadcast_receive_transitions[a][C][b] == pytest.approx(1.0)

    def test_broadcast_receive_accumulates_weight(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        defs.add_broadcast_receive_transition(a, C, b, 0.5)
        assert defs._broadcast_receive_transitions[a][C][b] == pytest.approx(1.5)

    # --- weight validation (all four methods) ---

    def test_zero_weight_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        for method in (defs.add_send_transition, defs.add_receive_transition, defs.add_broadcast_send_transition, defs.add_broadcast_receive_transition):
            with pytest.raises(ValueError, match="strictly positive"):
                method(a, C, a, 0.0)

    def test_negative_weight_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        for method in (defs.add_send_transition, defs.add_receive_transition, defs.add_broadcast_send_transition, defs.add_broadcast_receive_transition):
            with pytest.raises(ValueError, match="strictly positive"):
                method(a, C, a, -1.0)

    # --- action types are independent of each other ---

    def test_different_action_types_independent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_send_transition(a, C, b, 2.0)
        assert defs._receive_transitions[a][C][b] == pytest.approx(1.0)
        assert defs._send_transitions[a][C][b] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# enabled_receive / enabled_broadcast_receive
# ---------------------------------------------------------------------------

class TestEnabledReceive:
    def test_no_transitions_returns_empty_set(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.enabled_receive(a) == set()

    def test_single_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.enabled_receive(a) == {C}

    def test_multiple_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_receive_transition(a, D, b, 1.0)
        assert defs.enabled_receive(a) == {C, D}

    def test_only_includes_receive_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)           # send, not receive
        defs.add_broadcast_receive_transition(a, D, b, 1.0)  # broadcast_receive, not receive
        assert defs.enabled_receive(a) == set()

    def test_independent_per_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.enabled_receive(b) == set()

    def test_does_not_create_entry(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.enabled_receive(a)
        tables = [defs._send_transitions, defs._receive_transitions,
                  defs._broadcast_send_transitions, defs._broadcast_receive_transitions]
        assert not any(a in table for table in tables)


class TestEnabledSend:
    def test_no_transitions_returns_empty_set(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.enabled_send(a) == set()

    def test_single_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs.enabled_send(a) == {C}

    def test_multiple_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        assert defs.enabled_send(a) == {C, D}

    def test_only_includes_send_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)           # receive, not send
        defs.add_broadcast_send_transition(a, D, b, 1.0)    # broadcast_send, not send
        assert defs.enabled_send(a) == set()

    def test_independent_per_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs.enabled_send(b) == set()

    def test_does_not_create_entry(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.enabled_send(a)
        tables = [defs._send_transitions, defs._receive_transitions,
                  defs._broadcast_send_transitions, defs._broadcast_receive_transitions]
        assert not any(a in table for table in tables)

    def test_predicate_filters_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        assert defs.enabled_send(a, p=lambda ch: ch == C) == {C}

    def test_predicate_all_false_returns_empty(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs.enabled_send(a, p=lambda ch: False) == set()

    def test_predicate_all_true_equals_no_predicate(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        assert defs.enabled_send(a, p=lambda ch: True) == defs.enabled_send(a)


class TestEnabledBroadcast:
    def test_no_transitions_returns_empty_set(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.enabled_broadcast(a) == set()

    def test_single_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        assert defs.enabled_broadcast(a) == {C}

    def test_multiple_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        assert defs.enabled_broadcast(a) == {C, D}

    def test_only_includes_broadcast_send_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)               # send, not broadcast_send
        defs.add_broadcast_receive_transition(a, D, b, 1.0)  # broadcast_receive, not broadcast_send
        assert defs.enabled_broadcast(a) == set()

    def test_independent_per_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        assert defs.enabled_broadcast(b) == set()

    def test_does_not_create_entry(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.enabled_broadcast(a)
        tables = [defs._send_transitions, defs._receive_transitions,
                  defs._broadcast_send_transitions, defs._broadcast_receive_transitions]
        assert not any(a in table for table in tables)


# ---------------------------------------------------------------------------
# weight
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# exit_rate
# ---------------------------------------------------------------------------

class TestExitRate:
    def test_no_transitions_returns_zero(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        env = Environment()
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(0.0)

    def test_single_send_enabled(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 3.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(3.0)

    def test_single_send_disabled(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 3.0)
        assert defs.exit_rate(a, env, lambda ch: False) == pytest.approx(0.0)

    def test_single_broadcast(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 5.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(5.0)

    def test_broadcast_not_filtered_by_enabled(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 5.0)
        # enabled predicate returns False but broadcast is always included
        assert defs.exit_rate(a, env, lambda ch: False) == pytest.approx(5.0)

    def test_send_and_broadcast_combined(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 2.0)
        env.set_broadcast_rate(a, D, 3.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(5.0)

    def test_multiple_send_channels_partially_enabled(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 2.0)
        env.set_send_rate(a, D, 4.0)
        # only C is enabled
        assert defs.exit_rate(a, env, lambda ch: ch == C) == pytest.approx(2.0)

    def test_uses_default_send_rate_from_env(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_default_send_rate(C, 7.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(7.0)

    def test_uses_default_broadcast_rate_from_env(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_default_broadcast_rate(C, 4.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(4.0)

    def test_receive_transitions_do_not_contribute(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 9.0)
        assert defs.exit_rate(a, env, lambda ch: True) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------

class TestReceive:
    def test_single_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.receive(a, C, random.Random(0)) == b

    def test_no_transitions_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        with pytest.raises(ValueError):
            defs.receive(a, C, random.Random(0))

    def test_no_transition_on_channel_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.receive(a, D, random.Random(0))

    def test_result_in_target_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_receive_transition(a, C, c_, 1.0)
        results = {defs.receive(a, C, random.Random(s)) for s in range(30)}
        assert results == {b, c_}

    def test_weighted_sampling(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_receive_transition(a, C, b, 999.0)
        defs.add_receive_transition(a, C, c_, 0.001)
        results = [defs.receive(a, C, random.Random(s)) for s in range(50)]
        assert results.count(b) > results.count(c_)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_receive_transition(a, C, c_, 1.0)
        assert (
            defs.receive(a, C, random.Random(7))
            == defs.receive(a, C, random.Random(7))
        )

    def test_independent_per_channel(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_receive_transition(a, C, b, 1.0)
        defs.add_receive_transition(a, D, c_, 1.0)
        assert defs.receive(a, C, random.Random(0)) == b
        assert defs.receive(a, D, random.Random(0)) == c_

    def test_send_transitions_not_used(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.receive(a, C, random.Random(0))


# ---------------------------------------------------------------------------
# broadcast_receive
# ---------------------------------------------------------------------------

class TestBroadcastReceive:
    def test_single_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.broadcast_receive(a, C, random.Random(0)) == b

    def test_no_transitions_returns_same_agent(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.broadcast_receive(a, C, random.Random(0)) == a

    def test_no_transition_on_channel_returns_same_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.broadcast_receive(a, D, random.Random(0)) == a

    def test_result_in_target_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        defs.add_broadcast_receive_transition(a, C, c_, 1.0)
        results = {defs.broadcast_receive(a, C, random.Random(s)) for s in range(30)}
        assert results == {b, c_}

    def test_weighted_sampling(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_receive_transition(a, C, b, 999.0)
        defs.add_broadcast_receive_transition(a, C, c_, 0.001)
        results = [defs.broadcast_receive(a, C, random.Random(s)) for s in range(50)]
        assert results.count(b) > results.count(c_)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        defs.add_broadcast_receive_transition(a, C, c_, 1.0)
        assert (
            defs.broadcast_receive(a, C, random.Random(7))
            == defs.broadcast_receive(a, C, random.Random(7))
        )

    def test_independent_per_channel(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        defs.add_broadcast_receive_transition(a, D, c_, 1.0)
        assert defs.broadcast_receive(a, C, random.Random(0)) == b
        assert defs.broadcast_receive(a, D, random.Random(0)) == c_

    def test_receive_transitions_not_used(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.broadcast_receive(a, C, random.Random(0)) == a


# ---------------------------------------------------------------------------
# is_receive_enabled
# ---------------------------------------------------------------------------

class TestIsReceiveEnabled:
    def test_no_transitions_returns_false(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.is_receive_enabled(a, C) is False

    def test_returns_true_after_transition_added(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.is_receive_enabled(a, C) is True

    def test_returns_false_for_different_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.is_receive_enabled(a, D) is False

    def test_returns_false_for_different_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.is_receive_enabled(b, C) is False

    def test_send_transition_does_not_count(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs.is_receive_enabled(a, C) is False

    def test_broadcast_receive_transition_does_not_count(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.is_receive_enabled(a, C) is False


# ---------------------------------------------------------------------------
# is_broadcast_receive_enabled
# ---------------------------------------------------------------------------

class TestIsBroadcastReceiveEnabled:
    def test_no_transitions_returns_false(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        assert defs.is_broadcast_receive_enabled(a, C) is False

    def test_returns_true_after_transition_added(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.is_broadcast_receive_enabled(a, C) is True

    def test_returns_false_for_different_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.is_broadcast_receive_enabled(a, D) is False

    def test_returns_false_for_different_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        assert defs.is_broadcast_receive_enabled(b, C) is False

    def test_receive_transition_does_not_count(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        assert defs.is_broadcast_receive_enabled(a, C) is False

    def test_broadcast_send_transition_does_not_count(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        assert defs.is_broadcast_receive_enabled(a, C) is False


# ---------------------------------------------------------------------------
# select_send_channel
# ---------------------------------------------------------------------------

class TestSelectSendChannel:
    def test_single_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        assert defs.select_send_channel(a, env, lambda ch: True, random.Random(0)) == C

    def test_no_enabled_channels_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        env = Environment()
        with pytest.raises(ValueError):
            defs.select_send_channel(a, env, lambda ch: True, random.Random(0))

    def test_predicate_filters_channels(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_send_rate(a, D, 1.0)
        assert defs.select_send_channel(a, env, lambda ch: ch == C, random.Random(0)) == C

    def test_all_filtered_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        with pytest.raises(ValueError):
            defs.select_send_channel(a, env, lambda ch: False, random.Random(0))

    def test_result_in_enabled_set(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_send_rate(a, D, 1.0)
        results = {defs.select_send_channel(a, env, lambda ch: True, random.Random(s)) for s in range(30)}
        assert results == {C, D}

    def test_weighted_by_send_rate(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 999.0)
        env.set_send_rate(a, D, 0.001)
        results = [defs.select_send_channel(a, env, lambda ch: True, random.Random(s)) for s in range(50)]
        assert results.count(C) > results.count(D)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_send_rate(a, D, 1.0)
        assert (
            defs.select_send_channel(a, env, lambda ch: True, random.Random(5))
            == defs.select_send_channel(a, env, lambda ch: True, random.Random(5))
        )

    def test_uses_default_send_rate(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_default_send_rate(C, 3.0)
        assert defs.select_send_channel(a, env, lambda ch: True, random.Random(0)) == C


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------

class TestSend:
    def test_single_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        assert defs.send(a, C, random.Random(0)) == b

    def test_no_transitions_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        with pytest.raises(ValueError):
            defs.send(a, C, random.Random(0))

    def test_no_transition_on_channel_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.send(a, D, random.Random(0))

    def test_result_in_target_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, C, c_, 1.0)
        results = {defs.send(a, C, random.Random(s)) for s in range(30)}
        assert results == {b, c_}

    def test_weighted_sampling(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_send_transition(a, C, b, 999.0)
        defs.add_send_transition(a, C, c_, 0.001)
        results = [defs.send(a, C, random.Random(s)) for s in range(50)]
        assert results.count(b) > results.count(c_)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, C, c_, 1.0)
        assert (
            defs.send(a, C, random.Random(7))
            == defs.send(a, C, random.Random(7))
        )

    def test_independent_per_channel(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, c_, 1.0)
        assert defs.send(a, C, random.Random(0)) == b
        assert defs.send(a, D, random.Random(0)) == c_

    def test_receive_transitions_not_used(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.send(a, C, random.Random(0))


# ---------------------------------------------------------------------------
# select_broadcast_send_channel
# ---------------------------------------------------------------------------

class TestSelectBroadcastSendChannel:
    def test_single_channel(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 1.0)
        assert defs.select_broadcast_send_channel(a, env, random.Random(0)) == C

    def test_no_transitions_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        env = Environment()
        with pytest.raises(ValueError):
            defs.select_broadcast_send_channel(a, env, random.Random(0))

    def test_result_in_channel_set(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 1.0)
        env.set_broadcast_rate(a, D, 1.0)
        results = {defs.select_broadcast_send_channel(a, env, random.Random(s)) for s in range(30)}
        assert results == {C, D}

    def test_weighted_by_broadcast_rate(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 999.0)
        env.set_broadcast_rate(a, D, 0.001)
        results = [defs.select_broadcast_send_channel(a, env, random.Random(s)) for s in range(50)]
        assert results.count(C) > results.count(D)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 1.0)
        env.set_broadcast_rate(a, D, 1.0)
        assert (
            defs.select_broadcast_send_channel(a, env, random.Random(5))
            == defs.select_broadcast_send_channel(a, env, random.Random(5))
        )

    def test_uses_default_broadcast_rate(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_default_broadcast_rate(C, 4.0)
        assert defs.select_broadcast_send_channel(a, env, random.Random(0)) == C

    def test_send_transitions_not_considered(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        with pytest.raises(ValueError):
            defs.select_broadcast_send_channel(a, env, random.Random(0))


# ---------------------------------------------------------------------------
# broadcast_send
# ---------------------------------------------------------------------------

class TestBroadcastSend:
    def test_single_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        assert defs.broadcast_send(a, C, random.Random(0)) == b

    def test_no_transitions_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        with pytest.raises(ValueError):
            defs.broadcast_send(a, C, random.Random(0))

    def test_no_transition_on_channel_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.broadcast_send(a, D, random.Random(0))

    def test_result_in_target_set(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, C, c_, 1.0)
        results = {defs.broadcast_send(a, C, random.Random(s)) for s in range(30)}
        assert results == {b, c_}

    def test_weighted_sampling(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_send_transition(a, C, b, 999.0)
        defs.add_broadcast_send_transition(a, C, c_, 0.001)
        results = [defs.broadcast_send(a, C, random.Random(s)) for s in range(50)]
        assert results.count(b) > results.count(c_)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, C, c_, 1.0)
        assert (
            defs.broadcast_send(a, C, random.Random(7))
            == defs.broadcast_send(a, C, random.Random(7))
        )

    def test_independent_per_channel(self) -> None:
        defs = AgentDefinitions()
        a, b, c_ = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, c_, 1.0)
        assert defs.broadcast_send(a, C, random.Random(0)) == b
        assert defs.broadcast_send(a, D, random.Random(0)) == c_

    def test_send_transitions_not_used(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        with pytest.raises(ValueError):
            defs.broadcast_send(a, C, random.Random(0))


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestStep:
    def test_send_only_returns_false(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        is_broadcast, ch, next_a = defs.step(a, env, lambda ch: True, random.Random(0))
        assert is_broadcast is False
        assert ch == C
        assert next_a == b

    def test_broadcast_only_returns_true(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 1.0)
        is_broadcast, ch, next_a = defs.step(a, env, lambda ch: True, random.Random(0))
        assert is_broadcast is True
        assert ch == C
        assert next_a == b

    def test_no_enabled_actions_raises(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        env = Environment()
        with pytest.raises(ValueError):
            defs.step(a, env, lambda ch: True, random.Random(0))

    def test_predicate_disabling_all_sends_with_no_broadcast_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        with pytest.raises(ValueError):
            defs.step(a, env, lambda ch: False, random.Random(0))

    def test_both_types_both_outcomes_reachable(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_broadcast_rate(a, D, 1.0)
        results = {defs.step(a, env, lambda ch: True, random.Random(s))[0] for s in range(50)}
        assert results == {False, True}

    def test_send_channel_in_result(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_send_rate(a, D, 1.0)
        for s in range(20):
            is_broadcast, ch, _ = defs.step(a, env, lambda ch: True, random.Random(s))
            assert is_broadcast is False
            assert ch in {C, D}

    def test_broadcast_channel_in_result(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, C, 1.0)
        env.set_broadcast_rate(a, D, 1.0)
        for s in range(20):
            is_broadcast, ch, _ = defs.step(a, env, lambda ch: True, random.Random(s))
            assert is_broadcast is True
            assert ch in {C, D}

    def test_weighted_action_type_selection(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 999.0)
        env.set_broadcast_rate(a, D, 0.001)
        results = [defs.step(a, env, lambda ch: True, random.Random(s))[0] for s in range(50)]
        assert results.count(False) > results.count(True)

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, C, b, 1.0)
        defs.add_broadcast_send_transition(a, D, b, 1.0)
        env = Environment()
        env.set_send_rate(a, C, 1.0)
        env.set_broadcast_rate(a, D, 1.0)
        assert (
            defs.step(a, env, lambda ch: True, random.Random(3))
            == defs.step(a, env, lambda ch: True, random.Random(3))
        )
