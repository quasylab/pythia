"""Tests for pythia.models.yoda.agent_configuration."""

from __future__ import annotations

import random

import pytest

from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.store import Store


RNG = random.Random(0)


A      = Agent("a")
STATE  = AgentState("idle")
STORE  = AgentStore(Store(), Store(), Store())


def _cfg(
    agent: Agent = A,
    agent_state: AgentState = STATE,
    agent_store: AgentStore = STORE,
) -> AgentConfiguration:
    return AgentConfiguration(agent, agent_state, agent_store)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_agent_stored(self) -> None:
        assert _cfg().agent is A

    def test_agent_state_stored(self) -> None:
        assert _cfg().agent_state is STATE

    def test_agent_store_stored(self) -> None:
        assert _cfg().agent_store is STORE


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_agent_not_reassignable(self) -> None:
        with pytest.raises(AttributeError):
            _cfg().agent = Agent("b")  # type: ignore[misc]

    def test_agent_state_not_reassignable(self) -> None:
        with pytest.raises(AttributeError):
            _cfg().agent_state = AgentState("busy")  # type: ignore[misc]

    def test_agent_store_not_reassignable(self) -> None:
        with pytest.raises(AttributeError):
            _cfg().agent_store = AgentStore(Store(), Store(), Store())  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestAgentProperty:
    def test_returns_correct_agent(self) -> None:
        a = Agent("worker")
        cfg = _cfg(agent=a)
        assert cfg.agent == a

    def test_is_same_object(self) -> None:
        a = Agent("worker")
        cfg = _cfg(agent=a)
        assert cfg.agent is a


class TestAgentStateProperty:
    def test_returns_correct_state(self) -> None:
        s = AgentState("busy")
        cfg = _cfg(agent_state=s)
        assert cfg.agent_state == s

    def test_is_same_object(self) -> None:
        s = AgentState("busy")
        cfg = _cfg(agent_state=s)
        assert cfg.agent_state is s


class TestAgentStoreProperty:
    def test_returns_correct_store(self) -> None:
        store = AgentStore(Store({"o": 1}), Store({"f": 2}), Store({"s": 3}))
        cfg = _cfg(agent_store=store)
        assert cfg.agent_store is store

    def test_store_data_accessible(self) -> None:
        store = AgentStore(Store(), Store(), Store({"energy": 10}))
        cfg = _cfg(agent_store=store)
        assert cfg.agent_store.state["energy"] == 10


# ---------------------------------------------------------------------------
# get_predicate
# ---------------------------------------------------------------------------

BUSY  = AgentState("busy")
B     = Agent("b")
STORE_WITH_ENERGY = AgentStore(Store(), Store(), Store({"energy": 5}))
EMPTY_STORE       = AgentStore(Store(), Store(), Store())

_has_energy: bool = False  # unused; predicates defined inline below


class TestGetPredicate:
    def test_returns_callable(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True)
        assert callable(p)

    # -- pred only --

    def test_pred_satisfied(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: s.get("energy") is not None
        )
        assert p(_cfg(agent_store=STORE_WITH_ENERGY)) is True

    def test_pred_not_satisfied(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: s.get("energy") is not None
        )
        assert p(_cfg(agent_store=EMPTY_STORE)) is False

    def test_pred_false_overrides_matching_agent_and_state(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: False, agent=A, agent_state=STATE
        )
        assert p(_cfg()) is False

    # -- agent filter --

    def test_agent_match(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent=A)
        assert p(_cfg(agent=A)) is True

    def test_agent_mismatch(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent=B)
        assert p(_cfg(agent=A)) is False

    def test_agent_none_ignores_agent(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent=None)
        assert p(_cfg(agent=A)) is True
        assert p(_cfg(agent=B)) is True

    # -- agent_state filter --

    def test_agent_state_match(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent_state=STATE)
        assert p(_cfg(agent_state=STATE)) is True

    def test_agent_state_mismatch(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent_state=BUSY)
        assert p(_cfg(agent_state=STATE)) is False

    def test_agent_state_none_ignores_state(self) -> None:
        p = AgentConfiguration.get_predicate(lambda s: True, agent_state=None)
        assert p(_cfg(agent_state=STATE)) is True
        assert p(_cfg(agent_state=BUSY)) is True

    # -- combined filters --

    def test_all_conditions_satisfied(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: s.get("energy") is not None,
            agent=A,
            agent_state=STATE,
        )
        assert p(AgentConfiguration(A, STATE, STORE_WITH_ENERGY)) is True

    def test_pred_fails_with_matching_agent_and_state(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: s.get("energy") is not None,
            agent=A,
            agent_state=STATE,
        )
        assert p(AgentConfiguration(A, STATE, EMPTY_STORE)) is False

    def test_agent_fails_with_satisfied_pred_and_state(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: True, agent=B, agent_state=STATE
        )
        assert p(AgentConfiguration(A, STATE, EMPTY_STORE)) is False

    def test_agent_state_fails_with_satisfied_pred_and_agent(self) -> None:
        p = AgentConfiguration.get_predicate(
            lambda s: True, agent=A, agent_state=BUSY
        )
        assert p(AgentConfiguration(A, STATE, EMPTY_STORE)) is False


# ---------------------------------------------------------------------------
# set_state
# ---------------------------------------------------------------------------

class TestSetState:
    def test_agent_state_updated(self) -> None:
        cfg = _cfg(agent_state=STATE)
        cfg2 = cfg.set_state(BUSY, Store())
        assert cfg2.agent_state == BUSY

    def test_state_layer_updated(self) -> None:
        store = AgentStore(Store(), Store(), Store({"x": 1}))
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_state(BUSY, Store({"x": 99}))
        assert cfg2.agent_store.state["x"] == 99

    def test_original_unchanged(self) -> None:
        store = AgentStore(Store(), Store(), Store({"x": 1}))
        cfg = AgentConfiguration(A, STATE, store)
        cfg.set_state(BUSY, Store({"x": 99}))
        assert cfg.agent_state == STATE
        assert cfg.agent_store.state["x"] == 1

    def test_returns_new_instance(self) -> None:
        cfg = _cfg()
        assert cfg.set_state(BUSY, Store()) is not cfg

    def test_agent_preserved(self) -> None:
        cfg = _cfg(agent=A)
        assert cfg.set_state(BUSY, Store()).agent is A

    def test_features_layer_preserved(self) -> None:
        store = AgentStore(Store(), Store({"speed": 5}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_state(BUSY, Store({"x": 1}))
        assert cfg2.agent_store.features["speed"] == 5

    def test_observations_layer_preserved(self) -> None:
        store = AgentStore(Store({"o": 3}), Store(), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_state(BUSY, Store({"x": 1}))
        assert cfg2.agent_store.observations["o"] == 3


# ---------------------------------------------------------------------------
# set_features
# ---------------------------------------------------------------------------

class TestSetFeatures:
    def test_features_updated(self) -> None:
        store = AgentStore(Store(), Store({"speed": 1}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_features(Store({"speed": 9}))
        assert cfg2.agent_store.features["speed"] == 9

    def test_original_features_unchanged(self) -> None:
        store = AgentStore(Store(), Store({"speed": 1}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg.set_features(Store({"speed": 9}))
        assert cfg.agent_store.features["speed"] == 1

    def test_returns_new_instance(self) -> None:
        cfg = _cfg()
        assert cfg.set_features(Store()) is not cfg

    def test_agent_preserved(self) -> None:
        cfg = _cfg(agent=A)
        assert cfg.set_features(Store()).agent is A

    def test_agent_state_preserved(self) -> None:
        cfg = _cfg(agent_state=STATE)
        assert cfg.set_features(Store()).agent_state is STATE

    def test_state_layer_preserved(self) -> None:
        store = AgentStore(Store(), Store(), Store({"energy": 10}))
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_features(Store({"speed": 5}))
        assert cfg2.agent_store.state["energy"] == 10

    def test_observations_layer_preserved(self) -> None:
        store = AgentStore(Store({"o": 3}), Store(), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_features(Store({"speed": 5}))
        assert cfg2.agent_store.observations["o"] == 3


# ---------------------------------------------------------------------------
# update_observations
# ---------------------------------------------------------------------------

class TestSetObservations:
    def test_observations_updated(self) -> None:
        cfg = AgentConfiguration(A, STATE, AgentStore(Store(), Store(), Store()))
        cfg2 = cfg.set_observations(Store({"seen": True}))
        assert cfg2.agent_store.observations["seen"] is True

    def test_original_observations_unchanged(self) -> None:
        cfg = AgentConfiguration(A, STATE, AgentStore(Store({"o": 1}), Store(), Store()))
        cfg.set_observations(Store({"seen": True}))
        assert cfg.agent_store.observations["o"] == 1

    def test_returns_new_instance(self) -> None:
        cfg = _cfg()
        assert cfg.set_observations(Store()) is not cfg

    def test_agent_preserved(self) -> None:
        cfg = _cfg(agent=A)
        assert cfg.set_observations(Store()).agent is A

    def test_agent_state_preserved(self) -> None:
        cfg = _cfg(agent_state=STATE)
        assert cfg.set_observations(Store()).agent_state is STATE

    def test_state_layer_preserved(self) -> None:
        store = AgentStore(Store(), Store(), Store({"energy": 10}))
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_observations(Store({"new": True}))
        assert cfg2.agent_store.state["energy"] == 10

    def test_features_layer_preserved(self) -> None:
        store = AgentStore(Store(), Store({"speed": 3}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_observations(Store({"new": True}))
        assert cfg2.agent_store.features["speed"] == 3

    def test_observation_uses_agent_store_data(self) -> None:
        store = AgentStore(Store(), Store({"v": 7}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.set_observations(Store({"copy": cfg.agent_store.features.get("v")}))
        assert cfg2.agent_store.observations["copy"] == 7


# ---------------------------------------------------------------------------
# apply_dynamics
# ---------------------------------------------------------------------------

_identity_dynamics = lambda rng, s, dt: s.features


class TestApplyDynamics:
    def test_features_updated(self) -> None:
        dyn = lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt)
        store = AgentStore(Store(), Store({"x": 1.0}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.apply_dynamics(RNG, 0.5, lambda agent: dyn)
        assert cfg2.agent_store.features["x"] == pytest.approx(1.5)

    def test_original_unchanged(self) -> None:
        store = AgentStore(Store(), Store({"x": 1.0}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg.apply_dynamics(RNG, 1.0, lambda agent: lambda rng, s, dt: s.features.set("x", 99.0))
        assert cfg.agent_store.features["x"] == pytest.approx(1.0)

    def test_returns_new_instance(self) -> None:
        cfg = _cfg()
        cfg2 = cfg.apply_dynamics(RNG, 1.0, lambda agent: _identity_dynamics)
        assert cfg2 is not cfg

    def test_agent_preserved(self) -> None:
        cfg = _cfg(agent=A)
        assert cfg.apply_dynamics(RNG, 1.0, lambda agent: _identity_dynamics).agent is A

    def test_agent_state_preserved(self) -> None:
        cfg = _cfg(agent_state=STATE)
        assert cfg.apply_dynamics(RNG, 1.0, lambda agent: _identity_dynamics).agent_state is STATE

    def test_agent_passed_to_agent_dynamics(self) -> None:
        received: list = []
        cfg = _cfg(agent=A)
        cfg.apply_dynamics(RNG, 1.0, lambda agent: received.append(agent) or _identity_dynamics)
        assert received[0] is A

    def test_registered_function_invoked(self) -> None:
        dyn = lambda rng, s, dt: s.features.set("tagged", True)
        store = AgentStore(Store(), Store(), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.apply_dynamics(RNG, 1.0, lambda agent: dyn)
        assert cfg2.agent_store.features["tagged"] is True

    def test_identity_dynamics_preserves_features(self) -> None:
        store = AgentStore(Store(), Store({"y": 5}), Store())
        cfg = AgentConfiguration(A, STATE, store)
        cfg2 = cfg.apply_dynamics(RNG, 1.0, lambda agent: _identity_dynamics)
        assert cfg2.agent_store.features["y"] == 5

    def test_rng_forwarded(self) -> None:
        received: list = []
        cfg = AgentConfiguration(A, STATE, AgentStore(Store(), Store(), Store()))
        dyn = lambda rng, s, dt: received.append(rng) or s.features
        cfg.apply_dynamics(RNG, 1.0, lambda agent: dyn)
        assert received[0] is RNG

    def test_dt_forwarded(self) -> None:
        received: list[float] = []
        cfg = AgentConfiguration(A, STATE, AgentStore(Store(), Store(), Store()))
        dyn = lambda rng, s, dt: received.append(dt) or s.features
        cfg.apply_dynamics(RNG, 2.5, lambda agent: dyn)
        assert received[0] == pytest.approx(2.5)
