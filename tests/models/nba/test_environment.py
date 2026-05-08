"""Tests for pythia.models.nba.environment."""

from __future__ import annotations

import pytest

from pythia.models.nba.agent import Agent
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment


A = Agent("a")
B = Agent("b")
C = Channel("c")
D = Channel("d")


# ---------------------------------------------------------------------------
# send_rate
# ---------------------------------------------------------------------------

class TestSendRate:
    def test_default_is_zero(self) -> None:
        env = Environment()
        assert env.send_rate(A, C) == pytest.approx(0.0)

    def test_set_and_get(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 2.5)
        assert env.send_rate(A, C) == pytest.approx(2.5)

    def test_overwrite(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 1.0)
        env.set_send_rate(A, C, 3.0)
        assert env.send_rate(A, C) == pytest.approx(3.0)

    def test_independent_per_channel(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 1.0)
        env.set_send_rate(A, D, 2.0)
        assert env.send_rate(A, C) == pytest.approx(1.0)
        assert env.send_rate(A, D) == pytest.approx(2.0)

    def test_independent_per_agent(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 1.0)
        assert env.send_rate(B, C) == pytest.approx(0.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_send_rate(A, C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_send_rate(A, C, -1.0)


# ---------------------------------------------------------------------------
# broadcast_rate
# ---------------------------------------------------------------------------

class TestBroadcastRate:
    def test_default_is_zero(self) -> None:
        env = Environment()
        assert env.broadcast_rate(A, C) == pytest.approx(0.0)

    def test_set_and_get(self) -> None:
        env = Environment()
        env.set_broadcast_rate(A, C, 0.5)
        assert env.broadcast_rate(A, C) == pytest.approx(0.5)

    def test_overwrite(self) -> None:
        env = Environment()
        env.set_broadcast_rate(A, C, 1.0)
        env.set_broadcast_rate(A, C, 4.0)
        assert env.broadcast_rate(A, C) == pytest.approx(4.0)

    def test_independent_per_channel(self) -> None:
        env = Environment()
        env.set_broadcast_rate(A, C, 1.0)
        env.set_broadcast_rate(A, D, 2.0)
        assert env.broadcast_rate(A, C) == pytest.approx(1.0)
        assert env.broadcast_rate(A, D) == pytest.approx(2.0)

    def test_independent_per_agent(self) -> None:
        env = Environment()
        env.set_broadcast_rate(A, C, 1.0)
        assert env.broadcast_rate(B, C) == pytest.approx(0.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_broadcast_rate(A, C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_broadcast_rate(A, C, -0.1)


# ---------------------------------------------------------------------------
# receive_probability
# ---------------------------------------------------------------------------

class TestReceiveProbability:
    def test_default_is_zero(self) -> None:
        env = Environment()
        assert env.receive_probability(A, C) == pytest.approx(0.0)

    def test_set_and_get(self) -> None:
        env = Environment()
        env.set_receive_probability(A, C, 0.7)
        assert env.receive_probability(A, C) == pytest.approx(0.7)

    def test_overwrite(self) -> None:
        env = Environment()
        env.set_receive_probability(A, C, 0.3)
        env.set_receive_probability(A, C, 0.9)
        assert env.receive_probability(A, C) == pytest.approx(0.9)

    def test_independent_per_channel(self) -> None:
        env = Environment()
        env.set_receive_probability(A, C, 0.2)
        env.set_receive_probability(A, D, 0.8)
        assert env.receive_probability(A, C) == pytest.approx(0.2)
        assert env.receive_probability(A, D) == pytest.approx(0.8)

    def test_independent_per_agent(self) -> None:
        env = Environment()
        env.set_receive_probability(A, C, 0.5)
        assert env.receive_probability(B, C) == pytest.approx(0.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_receive_probability(A, C, 0.0)

    def test_one_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_receive_probability(A, C, 1.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_receive_probability(A, C, -0.1)

    def test_greater_than_one_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_receive_probability(A, C, 1.1)

    def test_boundary_values_accepted(self) -> None:
        env = Environment()
        env.set_receive_probability(A, C, 0.001)
        env.set_receive_probability(A, D, 0.999)
        assert env.receive_probability(A, C) == pytest.approx(0.001)
        assert env.receive_probability(A, D) == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# receive_weight
# ---------------------------------------------------------------------------

class TestReceiveWeight:
    def test_default_is_zero(self) -> None:
        env = Environment()
        assert env.receive_weight(A, C) == pytest.approx(0.0)

    def test_set_and_get(self) -> None:
        env = Environment()
        env.set_receive_weight(A, C, 4.0)
        assert env.receive_weight(A, C) == pytest.approx(4.0)

    def test_overwrite(self) -> None:
        env = Environment()
        env.set_receive_weight(A, C, 1.0)
        env.set_receive_weight(A, C, 7.0)
        assert env.receive_weight(A, C) == pytest.approx(7.0)

    def test_independent_per_channel(self) -> None:
        env = Environment()
        env.set_receive_weight(A, C, 1.0)
        env.set_receive_weight(A, D, 5.0)
        assert env.receive_weight(A, C) == pytest.approx(1.0)
        assert env.receive_weight(A, D) == pytest.approx(5.0)

    def test_independent_per_agent(self) -> None:
        env = Environment()
        env.set_receive_weight(A, C, 2.0)
        assert env.receive_weight(B, C) == pytest.approx(0.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_receive_weight(A, C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_receive_weight(A, C, -2.0)


# ---------------------------------------------------------------------------
# Fields are independent of each other
# ---------------------------------------------------------------------------

class TestFieldsAreIndependent:
    def test_send_rate_does_not_affect_others(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 1.0)
        assert env.broadcast_rate(A, C) == pytest.approx(0.0)
        assert env.receive_probability(A, C) == pytest.approx(0.0)
        assert env.receive_weight(A, C) == pytest.approx(0.0)

    def test_all_fields_set_independently(self) -> None:
        env = Environment()
        env.set_send_rate(A, C, 1.0)
        env.set_broadcast_rate(A, C, 2.0)
        env.set_receive_probability(A, C, 0.5)
        env.set_receive_weight(A, C, 3.0)
        assert env.send_rate(A, C) == pytest.approx(1.0)
        assert env.broadcast_rate(A, C) == pytest.approx(2.0)
        assert env.receive_probability(A, C) == pytest.approx(0.5)
        assert env.receive_weight(A, C) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Default send rate
# ---------------------------------------------------------------------------

class TestDefaultSendRate:
    def test_default_used_when_no_agent_entry(self) -> None:
        env = Environment()
        env.set_default_send_rate(C, 1.0)
        assert env.send_rate(A, C) == pytest.approx(1.0)

    def test_agent_specific_overrides_default(self) -> None:
        env = Environment()
        env.set_default_send_rate(C, 1.0)
        env.set_send_rate(A, C, 5.0)
        assert env.send_rate(A, C) == pytest.approx(5.0)

    def test_default_does_not_affect_other_channel(self) -> None:
        env = Environment()
        env.set_default_send_rate(C, 1.0)
        assert env.send_rate(A, D) == pytest.approx(0.0)

    def test_default_applies_to_all_agents(self) -> None:
        env = Environment()
        env.set_default_send_rate(C, 2.0)
        assert env.send_rate(A, C) == pytest.approx(2.0)
        assert env.send_rate(B, C) == pytest.approx(2.0)

    def test_overwrite_default(self) -> None:
        env = Environment()
        env.set_default_send_rate(C, 1.0)
        env.set_default_send_rate(C, 3.0)
        assert env.send_rate(A, C) == pytest.approx(3.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_send_rate(C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_send_rate(C, -1.0)


# ---------------------------------------------------------------------------
# Default broadcast rate
# ---------------------------------------------------------------------------

class TestDefaultBroadcastRate:
    def test_default_used_when_no_agent_entry(self) -> None:
        env = Environment()
        env.set_default_broadcast_rate(C, 0.5)
        assert env.broadcast_rate(A, C) == pytest.approx(0.5)

    def test_agent_specific_overrides_default(self) -> None:
        env = Environment()
        env.set_default_broadcast_rate(C, 0.5)
        env.set_broadcast_rate(A, C, 2.0)
        assert env.broadcast_rate(A, C) == pytest.approx(2.0)

    def test_default_does_not_affect_other_channel(self) -> None:
        env = Environment()
        env.set_default_broadcast_rate(C, 1.0)
        assert env.broadcast_rate(A, D) == pytest.approx(0.0)

    def test_default_applies_to_all_agents(self) -> None:
        env = Environment()
        env.set_default_broadcast_rate(C, 1.5)
        assert env.broadcast_rate(A, C) == pytest.approx(1.5)
        assert env.broadcast_rate(B, C) == pytest.approx(1.5)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_broadcast_rate(C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_broadcast_rate(C, -0.1)


# ---------------------------------------------------------------------------
# Default receive probability
# ---------------------------------------------------------------------------

class TestDefaultReceiveProbability:
    def test_default_used_when_no_agent_entry(self) -> None:
        env = Environment()
        env.set_default_receive_probability(C, 0.6)
        assert env.receive_probability(A, C) == pytest.approx(0.6)

    def test_agent_specific_overrides_default(self) -> None:
        env = Environment()
        env.set_default_receive_probability(C, 0.6)
        env.set_receive_probability(A, C, 0.9)
        assert env.receive_probability(A, C) == pytest.approx(0.9)

    def test_default_does_not_affect_other_channel(self) -> None:
        env = Environment()
        env.set_default_receive_probability(C, 0.5)
        assert env.receive_probability(A, D) == pytest.approx(0.0)

    def test_default_applies_to_all_agents(self) -> None:
        env = Environment()
        env.set_default_receive_probability(C, 0.3)
        assert env.receive_probability(A, C) == pytest.approx(0.3)
        assert env.receive_probability(B, C) == pytest.approx(0.3)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_default_receive_probability(C, 0.0)

    def test_one_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_default_receive_probability(C, 1.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_default_receive_probability(C, -0.1)

    def test_greater_than_one_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            env.set_default_receive_probability(C, 1.1)


# ---------------------------------------------------------------------------
# Default receive weight
# ---------------------------------------------------------------------------

class TestDefaultReceiveWeight:
    def test_default_used_when_no_agent_entry(self) -> None:
        env = Environment()
        env.set_default_receive_weight(C, 3.0)
        assert env.receive_weight(A, C) == pytest.approx(3.0)

    def test_agent_specific_overrides_default(self) -> None:
        env = Environment()
        env.set_default_receive_weight(C, 3.0)
        env.set_receive_weight(A, C, 7.0)
        assert env.receive_weight(A, C) == pytest.approx(7.0)

    def test_default_does_not_affect_other_channel(self) -> None:
        env = Environment()
        env.set_default_receive_weight(C, 1.0)
        assert env.receive_weight(A, D) == pytest.approx(0.0)

    def test_default_applies_to_all_agents(self) -> None:
        env = Environment()
        env.set_default_receive_weight(C, 4.0)
        assert env.receive_weight(A, C) == pytest.approx(4.0)
        assert env.receive_weight(B, C) == pytest.approx(4.0)

    def test_zero_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_receive_weight(C, 0.0)

    def test_negative_raises(self) -> None:
        env = Environment()
        with pytest.raises(ValueError, match="strictly positive"):
            env.set_default_receive_weight(C, -2.0)
