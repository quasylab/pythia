"""Tests for pythia.models.nba.nba_configuration."""

from __future__ import annotations

import random

import pytest

from pythia.models.nba.agent import Agent
from pythia.models.nba.agent_definitions import AgentDefinitions
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment
from pythia.models.nba.nba_configuration import NBAConfiguration


A = Agent("a")
B = Agent("b")
C = Agent("c")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_agent(self) -> None:
        cfg = NBAConfiguration({A: 1})
        assert cfg.count(A) == 1

    def test_multiple_agents(self) -> None:
        cfg = NBAConfiguration({A: 2, B: 3})
        assert cfg.count(A) == 2
        assert cfg.count(B) == 3

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one agent"):
            NBAConfiguration({})

    def test_zero_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive integer"):
            NBAConfiguration({A: 0})

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive integer"):
            NBAConfiguration({A: -1})

    def test_float_count_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive integer"):
            NBAConfiguration({A: 1.0})  # type: ignore[dict-item]

    def test_input_dict_not_mutated(self) -> None:
        d = {A: 2}
        NBAConfiguration(d)
        d[A] = 99
        cfg = NBAConfiguration({A: 2})
        assert cfg.count(A) == 2


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_setattr_raises(self) -> None:
        cfg = NBAConfiguration({A: 1})
        with pytest.raises(AttributeError):
            cfg._counts = {}  # type: ignore[misc]

    def test_counts_property_returns_copy(self) -> None:
        cfg = NBAConfiguration({A: 2})
        copy = cfg.counts
        copy[A] = 99
        assert cfg.count(A) == 2


# ---------------------------------------------------------------------------
# counts property
# ---------------------------------------------------------------------------

class TestCountsProperty:
    def test_returns_full_mapping(self) -> None:
        cfg = NBAConfiguration({A: 2, B: 3})
        assert cfg.counts == {A: 2, B: 3}

    def test_single_entry(self) -> None:
        cfg = NBAConfiguration({A: 5})
        assert cfg.counts == {A: 5}


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    def test_single_agent(self) -> None:
        assert NBAConfiguration({A: 4}).size == 4

    def test_multiple_agents(self) -> None:
        assert NBAConfiguration({A: 2, B: 3}).size == 5

    def test_three_agents(self) -> None:
        assert NBAConfiguration({A: 1, B: 1, C: 1}).size == 3


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_present_agent(self) -> None:
        cfg = NBAConfiguration({A: 7})
        assert cfg.count(A) == 7

    def test_absent_agent_returns_zero(self) -> None:
        cfg = NBAConfiguration({A: 1})
        assert cfg.count(B) == 0

    def test_multiple_agents(self) -> None:
        cfg = NBAConfiguration({A: 2, B: 5})
        assert cfg.count(A) == 2
        assert cfg.count(B) == 5


# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------

class TestFraction:
    def test_sole_agent_is_one(self) -> None:
        cfg = NBAConfiguration({A: 3})
        assert cfg.fraction(A) == pytest.approx(1.0)

    def test_absent_agent_is_zero(self) -> None:
        cfg = NBAConfiguration({A: 1})
        assert cfg.fraction(B) == pytest.approx(0.0)

    def test_equal_counts(self) -> None:
        cfg = NBAConfiguration({A: 1, B: 1})
        assert cfg.fraction(A) == pytest.approx(0.5)
        assert cfg.fraction(B) == pytest.approx(0.5)

    def test_fractions_sum_to_one(self) -> None:
        cfg = NBAConfiguration({A: 1, B: 3, C: 6})
        total = cfg.fraction(A) + cfg.fraction(B) + cfg.fraction(C)
        assert total == pytest.approx(1.0)

    def test_specific_values(self) -> None:
        cfg = NBAConfiguration({A: 1, B: 3})
        assert cfg.fraction(A) == pytest.approx(0.25)
        assert cfg.fraction(B) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# enabled_receive
# ---------------------------------------------------------------------------

class TestEnabledReceive:
    def test_returns_entry_for_every_agent(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 2, b: 1})
        result = cfg.enabled_receive(defs)
        assert set(result.keys()) == {a, b}

    def test_agent_with_receive_transition(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        result = cfg.enabled_receive(defs)
        assert result[a] == {c}

    def test_agent_without_receive_transition(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        result = cfg.enabled_receive(defs)
        assert result[b] == set()

    def test_multiple_channels_per_agent(self) -> None:
        c, d = Channel("c"), Channel("d")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        defs.add_receive_transition(a, d, b, 1.0)
        cfg = NBAConfiguration({a: 3})
        result = cfg.enabled_receive(defs)
        assert result[a] == {c, d}

    def test_only_agents_in_configuration_are_included(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        cfg = NBAConfiguration({a: 1})   # b is not in cfg
        result = cfg.enabled_receive(defs)
        assert list(result.keys()) == [a]

    def test_count_does_not_affect_result(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        assert NBAConfiguration({a: 1}).enabled_receive(defs) == \
               NBAConfiguration({a: 99}).enabled_receive(defs)


# ---------------------------------------------------------------------------
# get_receivers
# ---------------------------------------------------------------------------

class TestGetReceivers:
    def test_no_receive_transitions_returns_empty(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        cfg = NBAConfiguration({a: 1})
        assert cfg.get_receivers(defs) == {}

    def test_single_agent_single_channel(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        assert cfg.get_receivers(defs) == {c: {a}}

    def test_multiple_agents_same_channel(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        defs.add_receive_transition(b, c, a, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        assert cfg.get_receivers(defs) == {c: {a, b}}

    def test_multiple_channels(self) -> None:
        c, d = Channel("c"), Channel("d")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        defs.add_receive_transition(b, d, a, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        assert cfg.get_receivers(defs) == {c: {a}, d: {b}}

    def test_only_agents_in_configuration_considered(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        cfg = NBAConfiguration({a: 1})   # b not in cfg
        assert cfg.get_receivers(defs) == {}

    def test_count_does_not_affect_result(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(a, c, b, 1.0)
        assert NBAConfiguration({a: 1, b: 1}).get_receivers(defs) == \
               NBAConfiguration({a: 99, b: 1}).get_receivers(defs)


# ---------------------------------------------------------------------------
# get_is_enabled_predicate
# ---------------------------------------------------------------------------

class TestGetIsEnabledPredicate:
    def test_enabled_when_other_agent_can_receive(self) -> None:
        # a sends on c; b can receive on c — send is enabled
        c = Channel("c")
        a, b = Agent("a"), Agent("b")
        cfg = NBAConfiguration({a: 1, b: 1})
        receivers = {c: {b}}
        pred = cfg.get_is_enabled_predicate(receivers, a)
        assert pred(c) is True

    def test_disabled_when_no_receiver_exists(self) -> None:
        # a sends on c; nobody can receive — send is disabled
        c = Channel("c")
        a, b = Agent("a"), Agent("b")
        cfg = NBAConfiguration({a: 1, b: 1})
        pred = cfg.get_is_enabled_predicate({}, a)
        assert pred(c) is False

    def test_enabled_when_same_agent_receives_and_count_gt_1(self) -> None:
        # a can receive on c and there are 2 copies — one sends, one receives
        c = Channel("c")
        a = Agent("a")
        cfg = NBAConfiguration({a: 2})
        receivers = {c: {a}}
        pred = cfg.get_is_enabled_predicate(receivers, a)
        assert pred(c) is True

    def test_disabled_when_same_agent_receives_but_count_is_1(self) -> None:
        # a can receive on c but there is only 1 copy — cannot send to itself
        c = Channel("c")
        a = Agent("a")
        cfg = NBAConfiguration({a: 1})
        receivers = {c: {a}}
        pred = cfg.get_is_enabled_predicate(receivers, a)
        assert pred(c) is False

    def test_different_channel_not_enabled(self) -> None:
        # b can receive on d, not on c — send on c from a is disabled
        c, d = Channel("c"), Channel("d")
        a, b = Agent("a"), Agent("b")
        cfg = NBAConfiguration({a: 1, b: 1})
        receivers = {d: {b}}
        pred = cfg.get_is_enabled_predicate(receivers, a)
        assert pred(c) is False

    def test_multiple_receivers_enabled(self) -> None:
        # both b and d can receive on c — send from a is enabled
        c = Channel("c")
        a, b, d = Agent("a"), Agent("b"), Agent("d")
        cfg = NBAConfiguration({a: 1, b: 1, d: 1})
        receivers = {c: {b, d}}
        pred = cfg.get_is_enabled_predicate(receivers, a)
        assert pred(c) is True

    def test_returns_callable(self) -> None:
        c = Channel("c")
        a = Agent("a")
        cfg = NBAConfiguration({a: 1})
        pred = cfg.get_is_enabled_predicate({}, a)
        assert callable(pred)

    def test_predicate_is_per_agent(self) -> None:
        # pred for b should be False: b is the only receiver, count is 1
        c = Channel("c")
        a, b = Agent("a"), Agent("b")
        cfg = NBAConfiguration({a: 1, b: 1})
        receivers = {c: {b}}
        pred_b = cfg.get_is_enabled_predicate(receivers, b)
        assert pred_b(c) is False


# ---------------------------------------------------------------------------
# broadcast_send
# ---------------------------------------------------------------------------

C = Channel("c")
D = Channel("d")


class TestBroadcastSend:
    def test_source_not_present_raises(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({b: 1})
        with pytest.raises(ValueError):
            cfg.broadcast_send(random.Random(0), defs, a, C, b)

    def test_source_removed_when_multiple(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 3, b: 1})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, b)
        assert new_cfg.count(a) + new_cfg.count(b) == cfg.size

    def test_source_last_instance_removed(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 1, b: 2})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        # a had count 1: after removal it only returns via target or broadcast_receive
        assert new_cfg.size == cfg.size

    def test_target_added(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 1, b: 1})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, b)
        # b was 1; a (removed) gave no broadcast; target b added → b still 1 + target = 2
        assert new_cfg.count(b) == 2

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 2, b: 3})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg.size == cfg.size

    def test_no_broadcast_receive_transitions_remaining_unchanged(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 1, b: 2})
        # No broadcast_receive transitions: b stays b
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg.count(b) == 2
        assert new_cfg.count(a) == 1

    def test_broadcast_receive_applied_to_remaining(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(b, C, a, 1.0)
        cfg = NBAConfiguration({a: 1, b: 2})
        # source=a removed; both b's broadcast_receive → a; target=b added
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, b)
        assert new_cfg.count(a) == 2
        assert new_cfg.count(b) == 1

    def test_broadcast_receive_applied_per_individual(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(b, C, a, 1.0)
        cfg = NBAConfiguration({a: 1, b: 3})
        # source=a removed; 3 b's each → a; target=a added → a:4
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg.count(a) == 4
        assert new_cfg.count(b) == 0

    def test_source_also_undergoes_broadcast_receive_when_multiple(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_receive_transition(a, C, b, 1.0)
        cfg = NBAConfiguration({a: 3})
        # source=a, remove 1 → 2 remaining a's; each a → b via broadcast_receive; add target a
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg.count(b) == 2
        assert new_cfg.count(a) == 1

    def test_source_equals_target(self) -> None:
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 2, b: 1})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg.size == cfg.size

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        a, b, e = defs.get("a"), defs.get("b"), defs.get("e")
        defs.add_broadcast_receive_transition(b, C, a, 1.0)
        defs.add_broadcast_receive_transition(b, C, e, 1.0)
        cfg = NBAConfiguration({a: 1, b: 4})
        assert (
            cfg.broadcast_send(random.Random(42), defs, a, C, b).counts
            == cfg.broadcast_send(random.Random(42), defs, a, C, b).counts
        )

    def test_result_is_new_configuration(self) -> None:
        defs = AgentDefinitions()
        a = defs.get("a")
        cfg = NBAConfiguration({a: 2})
        new_cfg = cfg.broadcast_send(random.Random(0), defs, a, C, a)
        assert new_cfg is not cfg


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------

class TestSend:
    def _base(self) -> tuple[AgentDefinitions, Environment, Channel]:
        defs = AgentDefinitions()
        env = Environment()
        return defs, env, C

    def test_source_not_present_raises(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({b: 1})
        with pytest.raises(ValueError):
            cfg.send(random.Random(0), defs, env, a, c, a, {b})

    def test_no_receivers_in_remaining_raises(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        cfg = NBAConfiguration({a: 1})
        with pytest.raises(ValueError):
            cfg.send(random.Random(0), defs, env, a, c, a, {b})

    def test_source_is_only_receiver_with_count_1_raises(self) -> None:
        # After removing a, no a remains to receive
        defs, env, c = self._base()
        a = defs.get("a")
        defs.add_receive_transition(a, c, a, 1.0)
        env.set_receive_weight(a, c, 1.0)
        cfg = NBAConfiguration({a: 1})
        with pytest.raises(ValueError):
            cfg.send(random.Random(0), defs, env, a, c, a, {a})

    def test_size_preserved(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 2, b: 3})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {b})
        assert new_cfg.size == cfg.size

    def test_source_removed(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, b, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 2, b: 1})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {b})
        assert new_cfg.count(a) == 2  # 1 removed, 1 added as target

    def test_target_added(self) -> None:
        defs, env, c = self._base()
        a, b, e = defs.get("a"), defs.get("b"), defs.get("e")
        defs.add_receive_transition(b, c, b, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, e, {b})
        assert new_cfg.count(e) == 1

    def test_receiver_transitions(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {b})
        # b receives → a; target=a → a:2, b:0
        assert new_cfg.count(a) == 2
        assert new_cfg.count(b) == 0

    def test_single_receiver_always_selected(self) -> None:
        defs, env, c = self._base()
        a, b, e = defs.get("a"), defs.get("b"), defs.get("e")
        defs.add_receive_transition(b, c, e, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 2})
        for seed in range(10):
            new_cfg = cfg.send(random.Random(seed), defs, env, a, c, a, {b})
            # b is the only receiver; both b's present but only one is selected
            assert new_cfg.count(b) == 1  # one b transitioned to e
            assert new_cfg.count(e) == 1

    def test_weighted_selection_favours_higher_weight(self) -> None:
        defs, env, c = self._base()
        a, b, e = defs.get("a"), defs.get("b"), defs.get("e")
        defs.add_receive_transition(b, c, b, 1.0)
        defs.add_receive_transition(e, c, e, 1.0)
        env.set_receive_weight(b, c, 100.0)
        env.set_receive_weight(e, c, 0.001)
        cfg = NBAConfiguration({a: 1, b: 1, e: 1})
        results = [
            cfg.send(random.Random(s), defs, env, a, c, a, {b, e}).count(b)
            for s in range(50)
        ]
        # b is overwhelmingly more likely to be selected (receives → stays b)
        assert sum(results) > 40

    def test_count_affects_weight(self) -> None:
        defs, env, c = self._base()
        a, b, e = defs.get("a"), defs.get("b"), defs.get("e")
        defs.add_receive_transition(b, c, b, 1.0)
        defs.add_receive_transition(e, c, e, 1.0)
        env.set_receive_weight(b, c, 1.0)
        env.set_receive_weight(e, c, 1.0)
        # b has count 10, e has count 1 → b selected ~10× more often
        cfg = NBAConfiguration({a: 1, b: 10, e: 1})
        b_selected = sum(
            1 for s in range(100)
            if cfg.send(random.Random(s), defs, env, a, c, a, {b, e}).count(b) == 10
        )
        assert b_selected > 80

    def test_source_as_receiver_uses_reduced_count(self) -> None:
        defs, env, c = self._base()
        a = defs.get("a")
        defs.add_receive_transition(a, c, a, 1.0)
        env.set_receive_weight(a, c, 1.0)
        # a has count 2; after removing one, one a remains to receive
        cfg = NBAConfiguration({a: 2})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {a})
        assert new_cfg.size == cfg.size

    def test_deterministic_with_seed(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 2})
        assert (
            cfg.send(random.Random(7), defs, env, a, c, a, {b}).counts
            == cfg.send(random.Random(7), defs, env, a, c, a, {b}).counts
        )

    def test_result_is_new_configuration(self) -> None:
        defs, env, c = self._base()
        a, b = defs.get("a"), defs.get("b")
        defs.add_receive_transition(b, c, a, 1.0)
        env.set_receive_weight(b, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1})
        new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {b})
        assert new_cfg is not cfg


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestStep:
    def _point_to_point_setup(self) -> tuple[
        AgentDefinitions, Environment, Channel, Agent, Agent
    ]:
        """a --c!--> b (send);  b --c?--> b (receive, stays b)."""
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, c, b, 1.0)
        defs.add_receive_transition(b, c, b, 1.0)
        env = Environment()
        env.set_send_rate(a, c, 1.0)
        env.set_receive_weight(b, c, 1.0)
        return defs, env, c, a, b

    def _broadcast_setup(self) -> tuple[
        AgentDefinitions, Environment, Channel, Agent, Agent
    ]:
        """a --c!!--> b (broadcast send);  b --c??--> a (broadcast receive)."""
        c = Channel("c")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, c, b, 1.0)
        defs.add_broadcast_receive_transition(b, c, a, 1.0)
        env = Environment()
        env.set_broadcast_rate(a, c, 1.0)
        return defs, env, c, a, b

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_no_enabled_actions_returns_none(self) -> None:
        defs = AgentDefinitions()
        env = Environment()
        a = defs.get("a")
        cfg = NBAConfiguration({a: 1})
        assert cfg.step(random.Random(0), defs, env) is None

    def test_send_disabled_when_no_receiver_in_cfg_returns_none(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        env = Environment()
        a, b = defs.get("a"), defs.get("b")
        defs.add_send_transition(a, c, b, 1.0)
        env.set_send_rate(a, c, 1.0)
        # b not in cfg → send not enabled → total rate zero
        cfg = NBAConfiguration({a: 2})
        assert cfg.step(random.Random(0), defs, env) is None

    # ------------------------------------------------------------------
    # Structural invariants
    # ------------------------------------------------------------------

    def test_size_preserved_point_to_point(self) -> None:
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 1, b: 1})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.size == cfg.size

    def test_size_preserved_broadcast(self) -> None:
        defs, env, c, a, b = self._broadcast_setup()
        cfg = NBAConfiguration({a: 1, b: 3})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.size == cfg.size

    def test_result_is_new_configuration(self) -> None:
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 1, b: 1})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg is not cfg

    def test_deterministic_with_seed(self) -> None:
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 2, b: 2})
        t1, cfg1 = cfg.step(random.Random(9), defs, env)
        t2, cfg2 = cfg.step(random.Random(9), defs, env)
        assert t1 == t2
        assert cfg1.counts == cfg2.counts

    # ------------------------------------------------------------------
    # Time component
    # ------------------------------------------------------------------

    def test_time_is_positive(self) -> None:
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 1, b: 1})
        time, _ = cfg.step(random.Random(0), defs, env)
        assert time > 0.0

    def test_time_mean_close_to_inverse_total_rate(self) -> None:
        # total rate = send_rate(a, c) * count(a) = 1.0 * 3 = 3.0 → mean ≈ 1/3
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 3, b: 3})
        times = [cfg.step(random.Random(s), defs, env)[0] for s in range(2000)]
        assert abs(sum(times) / len(times) - 1 / 3.0) < 0.05

    # ------------------------------------------------------------------
    # Point-to-point send path
    # ------------------------------------------------------------------

    def test_point_to_point_outcome(self) -> None:
        # a sends on c → becomes b; b receives → stays b; add target b → {b: 2}
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 1, b: 1})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.count(a) == 0
        assert new_cfg.count(b) == 2

    def test_point_to_point_only_sender_selected(self) -> None:
        # only a has a non-zero exit rate so it must always be the sender
        defs, env, c, a, b = self._point_to_point_setup()
        cfg = NBAConfiguration({a: 1, b: 1})
        for seed in range(20):
            _, new_cfg = cfg.step(random.Random(seed), defs, env)
            assert new_cfg.count(b) == 2

    def test_send_enabled_self_receive_with_count_gt_1(self) -> None:
        # a can both send and receive on c; count 2 so one instance can send to the other
        c = Channel("c")
        defs = AgentDefinitions()
        env = Environment()
        a = defs.get("a")
        defs.add_send_transition(a, c, a, 1.0)
        defs.add_receive_transition(a, c, a, 1.0)
        env.set_send_rate(a, c, 1.0)
        env.set_receive_weight(a, c, 1.0)
        cfg = NBAConfiguration({a: 2})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.size == 2
        assert new_cfg.count(a) == 2

    # ------------------------------------------------------------------
    # Broadcast-send path
    # ------------------------------------------------------------------

    def test_broadcast_outcome(self) -> None:
        # a broadcasts → becomes b; both remaining b's → a; add target b → {a: 2, b: 1}
        defs, env, c, a, b = self._broadcast_setup()
        cfg = NBAConfiguration({a: 1, b: 2})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.count(a) == 2
        assert new_cfg.count(b) == 1

    def test_broadcast_enabled_without_receiver(self) -> None:
        # broadcast-send does not require a receiver in the configuration
        c = Channel("c")
        defs = AgentDefinitions()
        env = Environment()
        a, b = defs.get("a"), defs.get("b")
        defs.add_broadcast_send_transition(a, c, b, 1.0)
        env.set_broadcast_rate(a, c, 1.0)
        cfg = NBAConfiguration({a: 1})
        _, new_cfg = cfg.step(random.Random(0), defs, env)
        assert new_cfg.count(b) == 1
        assert new_cfg.count(a) == 0

    # ------------------------------------------------------------------
    # Agent-selection weights
    # ------------------------------------------------------------------

    def test_higher_exit_rate_selected_more_often(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        env = Environment()
        a, b, x, y, recv = (
            defs.get("a"), defs.get("b"), defs.get("x"), defs.get("y"), defs.get("recv")
        )
        # a → x on send; b → y on send; recv absorbs both
        defs.add_send_transition(a, c, x, 1.0)
        defs.add_send_transition(b, c, y, 1.0)
        defs.add_receive_transition(recv, c, recv, 1.0)
        env.set_send_rate(a, c, 100.0)
        env.set_send_rate(b, c, 1.0)
        env.set_receive_weight(recv, c, 1.0)
        cfg = NBAConfiguration({a: 1, b: 1, recv: 1})
        x_count = sum(
            1 for s in range(200)
            if cfg.step(random.Random(s), defs, env)[1].count(x) == 1
        )
        # x appears when a was the sender; should happen ~100/101 of the time
        assert x_count > 180

    def test_higher_count_selected_more_often(self) -> None:
        c = Channel("c")
        defs = AgentDefinitions()
        env = Environment()
        a, b, x, y, recv = (
            defs.get("a"), defs.get("b"), defs.get("x"), defs.get("y"), defs.get("recv")
        )
        # same rate; a has count 10, b has count 1 → a selected ~10× more
        defs.add_send_transition(a, c, x, 1.0)
        defs.add_send_transition(b, c, y, 1.0)
        defs.add_receive_transition(recv, c, recv, 1.0)
        env.set_send_rate(a, c, 1.0)
        env.set_send_rate(b, c, 1.0)
        env.set_receive_weight(recv, c, 1.0)
        cfg = NBAConfiguration({a: 10, b: 1, recv: 1})
        x_count = sum(
            1 for s in range(200)
            if cfg.step(random.Random(s), defs, env)[1].count(x) == 1
        )
        # x appears (count 1) when a was the sender; should happen ~10/11 of the time
        assert x_count > 160
