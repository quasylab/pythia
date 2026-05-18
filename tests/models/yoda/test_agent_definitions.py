"""Tests for pythia.models.yoda.agent_definitions."""

from __future__ import annotations

import pytest

import random

from pythia.models.yoda.action import Action
from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_definitions import AgentDefinitions
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.configuration import SystemConfiguration
from pythia.models.yoda.store import Store


RNG = random.Random(0)
AS  = AgentStore(Store({"s": 1}), Store({"f": 2}), Store())


def _cfg(*agents: Agent) -> SystemConfiguration:
    """Build a minimal SystemConfiguration with one AgentConfiguration per agent."""
    return SystemConfiguration([
        AgentConfiguration(a, AgentState("s"), AgentStore(Store(), Store(), Store()))
        for a in agents
    ])


TICK = Action("tick", lambda rng, s, o: s)
STEP = Action("step", lambda rng, s, o: s)
IDLE = AgentState("idle")
BUSY = AgentState("busy")
DONE = AgentState("done")
A    = Agent("a")
B    = Agent("b")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_empty_on_init(self) -> None:
        assert AgentDefinitions().agents == {}

    def test_no_parameters(self) -> None:
        defs = AgentDefinitions()
        assert isinstance(defs, AgentDefinitions)


# ---------------------------------------------------------------------------
# add_transition
# ---------------------------------------------------------------------------

class TestAddTransition:
    def test_single_transition_stored(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY)]

    def test_multiple_outcomes_same_key(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, TICK, DONE, 2.0)
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY), (2.0, DONE)]

    def test_multiple_actions_same_source(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, STEP, DONE, 3.0)
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY)]
        assert defs.agents[A][IDLE][STEP] == [(3.0, DONE)]

    def test_multiple_source_states(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, BUSY, TICK, IDLE, 1.0)
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY)]
        assert defs.agents[A][BUSY][TICK] == [(1.0, IDLE)]

    def test_multiple_agents(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(B, IDLE, STEP, DONE, 2.0)
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY)]
        assert defs.agents[B][IDLE][STEP] == [(2.0, DONE)]

    def test_zero_weight_raises(self) -> None:
        defs = AgentDefinitions()
        with pytest.raises(ValueError):
            defs.add_transition(A, IDLE, TICK, BUSY, 0.0)

    def test_negative_weight_raises(self) -> None:
        defs = AgentDefinitions()
        with pytest.raises(ValueError):
            defs.add_transition(A, IDLE, TICK, BUSY, -1.0)

    def test_absent_agent_not_in_table(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        assert B not in defs.agents

    def test_order_preserved(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, DONE, 3.0)
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, TICK, IDLE, 2.0)
        outcomes = defs.agents[A][IDLE][TICK]
        assert outcomes[0] == (3.0, DONE)
        assert outcomes[1] == (1.0, BUSY)
        assert outcomes[2] == (2.0, IDLE)


# ---------------------------------------------------------------------------
# agents property — isolation
# ---------------------------------------------------------------------------

class TestAgentsProperty:
    def test_returns_empty_dict_initially(self) -> None:
        assert AgentDefinitions().agents == {}

    def test_mutation_of_returned_list_does_not_affect_internal(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        snapshot = defs.agents
        snapshot[A][IDLE][TICK].append((9.0, DONE))
        assert defs.agents[A][IDLE][TICK] == [(1.0, BUSY)]

    def test_mutation_of_returned_inner_dict_does_not_affect_internal(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        snapshot = defs.agents
        snapshot[A][IDLE][STEP] = [(9.0, DONE)]
        assert STEP not in defs.agents[A][IDLE]

    def test_absent_agent_not_in_result(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        assert B not in defs.agents


# ---------------------------------------------------------------------------
# add_dynamics / dynamics
# ---------------------------------------------------------------------------

class TestDynamics:
    def test_empty_by_default(self) -> None:
        assert AgentDefinitions().dynamics == {}

    def test_function_stored(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, dt: s.features
        defs.add_dynamics(A, f)
        assert defs.dynamics[A] is f

    def test_function_called_correctly(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(
            A,
            lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt),
        )
        store = AgentStore(Store(), Store({"x": 1.0}), Store())
        result = defs.dynamics[A](RNG, store, 0.5)
        assert result["x"] == pytest.approx(1.5)

    def test_rng_passed_to_function(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(rng) or s.features)
        defs.dynamics[A](RNG, AS, 1.0)
        assert received == [RNG]

    def test_agent_store_passed_to_function(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(s) or s.features)
        defs.dynamics[A](RNG, AS, 1.0)
        assert received == [AS]

    def test_dt_passed_to_function(self) -> None:
        received: list[float] = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(dt) or s.features)
        defs.dynamics[A](RNG, AS, 2.5)
        assert received == [pytest.approx(2.5)]

    def test_multiple_agents(self) -> None:
        defs = AgentDefinitions()
        fa = lambda rng, s, dt: s.features.set("agent", "a")
        fb = lambda rng, s, dt: s.features.set("agent", "b")
        defs.add_dynamics(A, fa)
        defs.add_dynamics(B, fb)
        assert defs.dynamics[A] is fa
        assert defs.dynamics[B] is fb

    def test_overwrite_replaces_function(self) -> None:
        defs = AgentDefinitions()
        old = lambda rng, s, dt: s.features
        new = lambda rng, s, dt: s.features.set("replaced", True)
        defs.add_dynamics(A, old)
        defs.add_dynamics(A, new)
        assert defs.dynamics[A] is new

    def test_absent_agent_not_in_dynamics(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features)
        assert B not in defs.dynamics

    def test_dynamics_property_returns_copy(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, dt: s.features
        defs.add_dynamics(A, f)
        snapshot = defs.dynamics
        del snapshot[A]
        assert A in defs.dynamics


# ---------------------------------------------------------------------------
# default_dynamics / get_dynamics
# ---------------------------------------------------------------------------

class TestGetDynamics:
    def test_default_dynamics_returns_features(self) -> None:
        defs = AgentDefinitions()
        result = defs.default_dynamics(RNG, AS, 1.0)
        assert result is AS.features

    def test_get_dynamics_returns_default_when_not_registered(self) -> None:
        defs = AgentDefinitions()
        assert defs.get_dynamics(A) is defs.default_dynamics

    def test_get_dynamics_returns_registered_function(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, dt: s.features.set("x", dt)
        defs.add_dynamics(A, f)
        assert defs.get_dynamics(A) is f

    def test_get_dynamics_returns_default_for_unregistered_agent(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features)
        assert defs.get_dynamics(B) is defs.default_dynamics

    def test_get_dynamics_invokes_registered_function(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", dt))
        store = AgentStore(Store(), Store(), Store())
        result = defs.get_dynamics(A)(RNG, store, 2.0)
        assert result["x"] == pytest.approx(2.0)

    def test_default_dynamics_is_mutable(self) -> None:
        defs = AgentDefinitions()
        custom = lambda rng, s, dt: s.features
        defs.default_dynamics = custom
        assert defs.get_dynamics(B) is custom

    def test_custom_default_used_when_agent_not_registered(self) -> None:
        defs = AgentDefinitions()
        defs.default_dynamics = lambda rng, s, dt: s.features.set("default", True)
        result = defs.get_dynamics(A)(RNG, AS, 1.0)
        assert result["default"] is True


# ---------------------------------------------------------------------------
# add_observation / observations / get_observation
# ---------------------------------------------------------------------------

CFG = _cfg(A, B)


class TestObservations:
    def test_empty_by_default(self) -> None:
        assert AgentDefinitions().observations == {}

    def test_function_stored(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, cfg: Store()
        defs.add_observation(A, f)
        assert defs.observations[A] is f

    def test_function_called_correctly(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"n": cfg.size()}))
        result = defs.observations[A](RNG, AS, CFG)
        assert result["n"] == CFG.size()

    def test_agent_store_passed(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(s) or Store())
        defs.observations[A](RNG, AS, CFG)
        assert received[0] is AS

    def test_configuration_passed(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(cfg) or Store())
        defs.observations[A](RNG, AS, CFG)
        assert received[0] is CFG

    def test_multiple_agents(self) -> None:
        defs = AgentDefinitions()
        fa = lambda rng, s, cfg: Store({"agent": "a"})
        fb = lambda rng, s, cfg: Store({"agent": "b"})
        defs.add_observation(A, fa)
        defs.add_observation(B, fb)
        assert defs.observations[A] is fa
        assert defs.observations[B] is fb

    def test_overwrite_replaces_function(self) -> None:
        defs = AgentDefinitions()
        old = lambda rng, s, cfg: Store({"v": 1})
        new = lambda rng, s, cfg: Store({"v": 2})
        defs.add_observation(A, old)
        defs.add_observation(A, new)
        assert defs.observations[A] is new

    def test_absent_agent_not_in_observations(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store())
        assert B not in defs.observations

    def test_observations_property_returns_copy(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, cfg: Store()
        defs.add_observation(A, f)
        snapshot = defs.observations
        del snapshot[A]
        assert A in defs.observations


class TestGetObservation:
    def test_default_observation_returns_empty_store(self) -> None:
        defs = AgentDefinitions()
        result = defs.default_observation(RNG, AS, CFG)
        assert len(result) == 0

    def test_get_observation_returns_default_when_not_registered(self) -> None:
        defs = AgentDefinitions()
        assert defs.get_observation(A) is defs.default_observation

    def test_get_observation_returns_registered_function(self) -> None:
        defs = AgentDefinitions()
        f = lambda rng, s, cfg: Store({"x": 1})
        defs.add_observation(A, f)
        assert defs.get_observation(A) is f

    def test_get_observation_returns_default_for_unregistered_agent(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store())
        assert defs.get_observation(B) is defs.default_observation

    def test_get_observation_invokes_registered_function(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"size": cfg.size()}))
        result = defs.get_observation(A)(RNG, AS, CFG)
        assert result["size"] == CFG.size()

    def test_default_observation_is_mutable(self) -> None:
        defs = AgentDefinitions()
        custom = lambda rng, s, cfg: Store({"custom": True})
        defs.default_observation = custom
        assert defs.get_observation(B) is custom

    def test_custom_default_used_when_not_registered(self) -> None:
        defs = AgentDefinitions()
        defs.default_observation = lambda rng, s, cfg: Store({"default": True})
        result = defs.get_observation(A)(RNG, AS, CFG)
        assert result["default"] is True


# ---------------------------------------------------------------------------
# apply_observations
# ---------------------------------------------------------------------------

class TestApplyDynamics:
    def _aconf(self, agent: Agent = A) -> AgentConfiguration:
        return AgentConfiguration(
            agent, AgentState("idle"), AgentStore(Store(), Store({"x": 1.0}), Store())
        )

    def test_features_updated(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
        result = defs.apply_dynamics(RNG, self._aconf(), 0.5)
        assert result.agent_store.features["x"] == pytest.approx(1.5)

    def test_returns_new_instance(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        assert defs.apply_dynamics(RNG, aconf, 1.0) is not aconf

    def test_original_unchanged(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", 99.0))
        aconf = self._aconf()
        defs.apply_dynamics(RNG, aconf, 1.0)
        assert aconf.agent_store.features["x"] == pytest.approx(1.0)

    def test_agent_preserved(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf(A)
        assert defs.apply_dynamics(RNG, aconf, 1.0).agent is A

    def test_agent_state_preserved(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        assert defs.apply_dynamics(RNG, aconf, 1.0).agent_state == AgentState("idle")

    def test_default_dynamics_used_when_not_registered(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        result = defs.apply_dynamics(RNG, aconf, 1.0)
        assert result.agent_store.features["x"] == pytest.approx(1.0)

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(rng) or s.features)
        defs.apply_dynamics(RNG, self._aconf(), 1.0)
        assert received[0] is RNG

    def test_dt_forwarded(self) -> None:
        received: list[float] = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(dt) or s.features)
        defs.apply_dynamics(RNG, self._aconf(), 2.5)
        assert received[0] == pytest.approx(2.5)

    def test_agent_store_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(s) or s.features)
        aconf = self._aconf()
        defs.apply_dynamics(RNG, aconf, 1.0)
        assert received[0] is aconf.agent_store


class TestDynamicsStep:
    def _conf(self, *agents: Agent) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(
                a, AgentState("idle"), AgentStore(Store(), Store({"x": 0.0}), Store())
            )
            for a in agents
        ])

    def test_features_updated_for_all_agents(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
        conf = self._conf(A, A)
        conf2 = defs.dynamics_step(RNG, conf, 1.0)
        assert conf2.get(0).agent_store.features["x"] == pytest.approx(1.0)
        assert conf2.get(1).agent_store.features["x"] == pytest.approx(1.0)

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A, B)
        assert defs.dynamics_step(RNG, conf, 1.0).size() == 2

    def test_returns_new_configuration(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A)
        assert defs.dynamics_step(RNG, conf, 1.0) is not conf

    def test_original_unchanged(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", 99.0))
        conf = self._conf(A)
        defs.dynamics_step(RNG, conf, 1.0)
        assert conf.get(0).agent_store.features["x"] == pytest.approx(0.0)

    def test_different_dynamics_per_agent(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", 1.0))
        defs.add_dynamics(B, lambda rng, s, dt: s.features.set("x", 2.0))
        conf = self._conf(A, B)
        conf2 = defs.dynamics_step(RNG, conf, 1.0)
        assert conf2.get(0).agent_store.features["x"] == pytest.approx(1.0)
        assert conf2.get(1).agent_store.features["x"] == pytest.approx(2.0)

    def test_dt_forwarded(self) -> None:
        received: list[float] = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(dt) or s.features)
        conf = self._conf(A)
        defs.dynamics_step(RNG, conf, 3.0)
        assert received[0] == pytest.approx(3.0)

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(rng) or s.features)
        conf = self._conf(A)
        defs.dynamics_step(RNG, conf, 1.0)
        assert received[0] is RNG


class TestAgentsStep:
    def _conf(self, agent: Agent, state: AgentState) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(
                agent, state,
                AgentStore(Store(), Store({"x": 0.0}), Store())
            )
        ])

    def test_dynamics_applied(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, IDLE, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
        conf2 = defs.agents_step(RNG, self._conf(A, IDLE), 2.0)
        assert conf2.get(0).agent_store.features["x"] == pytest.approx(2.0)

    def test_action_fires_after_dynamics(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        conf2 = defs.agents_step(RNG, self._conf(A, IDLE), 1.0)
        assert conf2.get(0).agent_state == BUSY

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        conf = SystemConfiguration([
            AgentConfiguration(A, IDLE, AgentStore(Store(), Store(), Store())),
            AgentConfiguration(B, IDLE, AgentStore(Store(), Store(), Store())),
        ])
        assert defs.agents_step(RNG, conf, 1.0).size() == 2

    def test_returns_new_configuration(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A, IDLE)
        assert defs.agents_step(RNG, conf, 1.0) is not conf

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(rng) or s.features)
        conf2 = defs.agents_step(RNG, self._conf(A, IDLE), 1.0)
        assert received[0] is RNG

    def test_dt_forwarded(self) -> None:
        received: list[float] = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(dt) or s.features)
        defs.agents_step(RNG, self._conf(A, IDLE), 4.5)
        assert received[0] == pytest.approx(4.5)


class TestActionExecutionsStep:
    def _conf(self, *pairs: tuple[Agent, AgentState]) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(a, s, AgentStore(Store(), Store(), Store()))
            for a, s in pairs
        ])

    def test_enabled_action_fires(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        conf = self._conf((A, IDLE))
        conf2 = defs.action_executions_step(RNG, conf)
        assert conf2.get(0).agent_state == BUSY

    def test_agent_without_action_unchanged(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        # B has no transitions → stays as-is
        conf = self._conf((A, IDLE), (B, IDLE))
        conf2 = defs.action_executions_step(RNG, conf)
        assert conf2.get(0).agent_state == BUSY
        assert conf2.get(1).agent_state == IDLE

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf((A, IDLE), (B, IDLE))
        assert defs.action_executions_step(RNG, conf).size() == 2

    def test_returns_new_configuration(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf((A, IDLE))
        assert defs.action_executions_step(RNG, conf) is not conf

    def test_all_agents_processed(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        conf = self._conf((A, IDLE), (A, IDLE), (A, IDLE))
        conf2 = defs.action_executions_step(RNG, conf)
        assert all(conf2.get(i).agent_state == BUSY for i in range(3))

    def test_rng_forwarded(self) -> None:
        received: list = []
        tick = Action("tick", lambda rng, s, o: received.append(rng) or s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        conf = self._conf((A, IDLE))
        defs.action_executions_step(RNG, conf)
        assert received[0] is RNG


class TestEnvironmentStep:
    def _conf(self, *agents: Agent) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(
                a, AgentState("idle"),
                AgentStore(Store(), Store({"x": 0.0}), Store())
            )
            for a in agents
        ])

    def test_dynamics_applied(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
        conf2 = defs.environment_step(RNG, self._conf(A), 2.0)
        assert conf2.get(0).agent_store.features["x"] == pytest.approx(2.0)

    def test_observations_applied_after_dynamics(self) -> None:
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
        defs.add_observation(A, lambda rng, s, cfg: Store({"x_seen": s.features.get("x")}))
        conf2 = defs.environment_step(RNG, self._conf(A), 1.0)
        # observations reflect the post-dynamics features
        assert conf2.get(0).agent_store.observations["x_seen"] == pytest.approx(1.0)

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A, B)
        assert defs.environment_step(RNG, conf, 1.0).size() == 2

    def test_returns_new_configuration(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A)
        assert defs.environment_step(RNG, conf, 1.0) is not conf

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(rng) or s.features)
        defs.environment_step(RNG, self._conf(A), 1.0)
        assert received[0] is RNG

    def test_dt_forwarded(self) -> None:
        received: list[float] = []
        defs = AgentDefinitions()
        defs.add_dynamics(A, lambda rng, s, dt: received.append(dt) or s.features)
        defs.environment_step(RNG, self._conf(A), 3.5)
        assert received[0] == pytest.approx(3.5)


class TestObservationsStep:
    def _conf(self, *agents: Agent) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(a, AgentState("idle"), AgentStore(Store(), Store(), Store()))
            for a in agents
        ])

    def test_observations_updated_for_all_agents(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"n": cfg.size()}))
        conf = self._conf(A, A)
        conf2 = defs.observations_step(RNG, conf)
        assert conf2.get(0).agent_store.observations["n"] == 2
        assert conf2.get(1).agent_store.observations["n"] == 2

    def test_size_preserved(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A, B)
        assert defs.observations_step(RNG, conf).size() == 2

    def test_returns_new_configuration(self) -> None:
        defs = AgentDefinitions()
        conf = self._conf(A)
        assert defs.observations_step(RNG, conf) is not conf

    def test_original_unchanged(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"seen": True}))
        conf = self._conf(A)
        defs.observations_step(RNG, conf)
        assert conf.get(0).agent_store.observations.get("seen") is None

    def test_conf_passed_as_context(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(cfg) or Store())
        conf = self._conf(A)
        defs.observations_step(RNG, conf)
        assert received[0] is conf

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(rng) or Store())
        conf = self._conf(A)
        defs.observations_step(RNG, conf)
        assert received[0] is RNG

    def test_different_observations_per_agent(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"tag": "a"}))
        defs.add_observation(B, lambda rng, s, cfg: Store({"tag": "b"}))
        conf = self._conf(A, B)
        conf2 = defs.observations_step(RNG, conf)
        assert conf2.get(0).agent_store.observations["tag"] == "a"
        assert conf2.get(1).agent_store.observations["tag"] == "b"


class TestApplyObservations:
    def _aconf(self, agent: Agent = A) -> AgentConfiguration:
        return AgentConfiguration(
            agent, AgentState("idle"), AgentStore(Store(), Store(), Store())
        )

    def test_observations_updated(self) -> None:
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: Store({"n": cfg.size()}))
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        result = defs.apply_observations(RNG, aconf, sconf)
        assert result.agent_store.observations["n"] == 1

    def test_returns_new_instance(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        assert defs.apply_observations(RNG, aconf, sconf) is not aconf

    def test_agent_preserved(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf(A)
        sconf = SystemConfiguration([aconf])
        assert defs.apply_observations(RNG, aconf, sconf).agent is A

    def test_agent_state_preserved(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        assert defs.apply_observations(RNG, aconf, sconf).agent_state == AgentState("idle")

    def test_rng_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(rng) or Store())
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        defs.apply_observations(RNG, aconf, sconf)
        assert received[0] is RNG

    def test_agent_store_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(s) or Store())
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        defs.apply_observations(RNG, aconf, sconf)
        assert received[0] is aconf.agent_store

    def test_sconf_forwarded(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.add_observation(A, lambda rng, s, cfg: received.append(cfg) or Store())
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        defs.apply_observations(RNG, aconf, sconf)
        assert received[0] is sconf

    def test_default_observation_used_when_not_registered(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        sconf = SystemConfiguration([aconf])
        result = defs.apply_observations(RNG, aconf, sconf)
        assert len(result.agent_store.observations) == 0


# ---------------------------------------------------------------------------
# set_action_weight / set_default_action_weight / get_action_weights
# ---------------------------------------------------------------------------

STORE_EMPTY = AgentStore(Store(), Store(), Store())


class TestExecuteStep:
    def _aconf(self, agent: Agent = A, state: AgentState = IDLE) -> AgentConfiguration:
        return AgentConfiguration(
            agent, state, AgentStore(Store(), Store(), Store())
        )

    def test_returns_aconf_when_no_action_enabled(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf()
        assert defs.execute_step(RNG, aconf) is aconf

    def test_returns_aconf_when_weight_is_zero(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 0.0)
        aconf = self._aconf()
        assert defs.execute_step(RNG, aconf) is aconf

    def test_transitions_when_action_enabled(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        result = defs.execute_step(RNG, self._aconf())
        assert result.agent_state == BUSY

    def test_returns_new_instance_when_action_fires(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        aconf = self._aconf()
        assert defs.execute_step(RNG, aconf) is not aconf

    def test_agent_preserved(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        assert defs.execute_step(RNG, self._aconf(A)).agent is A

    def test_state_updated_by_action(self) -> None:
        tick = Action("tick", lambda rng, s, o: s.set("done", True))
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        result = defs.execute_step(RNG, self._aconf())
        assert result.agent_store.state["done"] is True


class TestExecuteAction:
    def _aconf(
        self,
        agent: Agent = A,
        state: AgentState = IDLE,
        store_state: dict | None = None,
        obs: dict | None = None,
    ) -> AgentConfiguration:
        return AgentConfiguration(
            agent,
            state,
            AgentStore(
                Store(obs or {}),
                Store(),
                Store(store_state or {}),
            ),
        )

    def test_next_agent_state_set(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        result = defs.execute_action(RNG, self._aconf(), tick)
        assert result.agent_state == BUSY

    def test_state_store_updated_by_action(self) -> None:
        tick = Action("tick", lambda rng, s, o: s.set("ticked", True))
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        result = defs.execute_action(RNG, self._aconf(), tick)
        assert result.agent_store.state["ticked"] is True

    def test_action_step_receives_state_store(self) -> None:
        received: list = []
        tick = Action("tick", lambda rng, s, o: received.append(s) or s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        aconf = self._aconf(store_state={"k": 7})
        defs.execute_action(RNG, aconf, tick)
        assert received[0]["k"] == 7

    def test_action_step_receives_observations(self) -> None:
        received: list = []
        tick = Action("tick", lambda rng, s, o: received.append(o) or s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        aconf = self._aconf(obs={"signal": 42})
        defs.execute_action(RNG, aconf, tick)
        assert received[0]["signal"] == 42

    def test_action_step_receives_rng(self) -> None:
        received: list = []
        tick = Action("tick", lambda rng, s, o: received.append(rng) or s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        defs.execute_action(RNG, self._aconf(), tick)
        assert received[0] is RNG

    def test_agent_preserved(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        assert defs.execute_action(RNG, self._aconf(A), tick).agent is A

    def test_features_preserved(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        aconf = AgentConfiguration(A, IDLE, AgentStore(Store(), Store({"f": 9}), Store()))
        result = defs.execute_action(RNG, aconf, tick)
        assert result.agent_store.features["f"] == 9

    def test_weighted_state_selection(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 100.0)
        defs.add_transition(A, IDLE, tick, DONE, 1.0)
        results = [
            defs.execute_action(random.Random(seed), self._aconf(), tick).agent_state
            for seed in range(100)
        ]
        assert results.count(BUSY) > 90

    def test_missing_transition_raises(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        with pytest.raises(KeyError):
            defs.execute_action(RNG, self._aconf(), tick)

    def test_returns_new_instance(self) -> None:
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, BUSY, 1.0)
        aconf = self._aconf()
        assert defs.execute_action(RNG, aconf, tick) is not aconf


class TestSampleAction:
    def _aconf(self, agent: Agent, state: AgentState) -> AgentConfiguration:
        return AgentConfiguration(
            agent, state, AgentStore(Store(), Store(), Store())
        )

    def test_returns_none_when_no_transitions(self) -> None:
        defs = AgentDefinitions()
        aconf = self._aconf(A, IDLE)
        assert defs.sample_action(RNG, aconf) is None

    def test_returns_none_when_state_not_in_table(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        aconf = self._aconf(A, BUSY)
        assert defs.sample_action(RNG, aconf) is None

    def test_returns_none_when_all_weights_zero(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 0.0)
        aconf = self._aconf(A, IDLE)
        assert defs.sample_action(RNG, aconf) is None

    def test_single_enabled_action_always_selected(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        aconf = self._aconf(A, IDLE)
        for seed in range(10):
            assert defs.sample_action(random.Random(seed), aconf) is TICK

    def test_disabled_action_not_selected(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, STEP, DONE, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        defs.set_action_weight(A, STEP, lambda s: 0.0)
        aconf = self._aconf(A, IDLE)
        for seed in range(20):
            assert defs.sample_action(random.Random(seed), aconf) is TICK

    def test_higher_weight_selected_more_often(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, STEP, DONE, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 100.0)
        defs.set_action_weight(A, STEP, lambda s: 1.0)
        aconf = self._aconf(A, IDLE)
        tick_count = sum(
            1 for seed in range(200)
            if defs.sample_action(random.Random(seed), aconf) is TICK
        )
        assert tick_count > 180

    def test_action_must_be_in_transition_table(self) -> None:
        # Weight registered but no transition → action not sampled
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: 99.0)
        aconf = self._aconf(A, IDLE)
        assert defs.sample_action(RNG, aconf) is None

    def test_deterministic_with_seed(self) -> None:
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, TICK, BUSY, 1.0)
        defs.add_transition(A, IDLE, STEP, DONE, 1.0)
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        defs.set_action_weight(A, STEP, lambda s: 1.0)
        aconf = self._aconf(A, IDLE)
        assert (
            defs.sample_action(random.Random(7), aconf)
            is defs.sample_action(random.Random(7), aconf)
        )


class TestSetActionWeight:
    def test_weight_evaluated(self) -> None:
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: 2.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(2.0)

    def test_overwrite_replaces_function(self) -> None:
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        defs.set_action_weight(A, TICK, lambda s: 5.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(5.0)

    def test_different_agents_independent(self) -> None:
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: 1.0)
        defs.set_action_weight(B, TICK, lambda s: 2.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(1.0)
        assert defs.get_action_weight(B, STORE_EMPTY, TICK) == pytest.approx(2.0)

    def test_different_actions_independent(self) -> None:
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: 3.0)
        defs.set_action_weight(A, STEP, lambda s: 7.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(3.0)
        assert defs.get_action_weight(A, STORE_EMPTY, STEP) == pytest.approx(7.0)

    def test_weight_function_receives_agent_store(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: received.append(s) or 1.0)
        defs.get_action_weight(A, STORE_EMPTY, TICK)
        assert received[0] is STORE_EMPTY


class TestSetDefaultActionWeight:
    def test_default_used_when_no_agent_specific(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 4.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(4.0)

    def test_agent_specific_overrides_default(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 1.0)
        defs.set_action_weight(A, TICK, lambda s: 9.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(9.0)

    def test_default_overwrite_replaces(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 1.0)
        defs.set_default_action_weight(TICK, lambda s: 5.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(5.0)

    def test_different_actions_independent(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 2.0)
        defs.set_default_action_weight(STEP, lambda s: 8.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(2.0)
        assert defs.get_action_weight(A, STORE_EMPTY, STEP) == pytest.approx(8.0)


class TestGetActionWeight:
    def test_returns_float(self) -> None:
        defs = AgentDefinitions()
        result = defs.get_action_weight(A, STORE_EMPTY, TICK)
        assert isinstance(result, float)

    def test_unknown_action_returns_zero(self) -> None:
        defs = AgentDefinitions()
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(0.0)

    def test_agent_specific_takes_priority_over_default(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 1.0)
        defs.set_action_weight(A, TICK, lambda s: 99.0)
        assert defs.get_action_weight(A, STORE_EMPTY, TICK) == pytest.approx(99.0)

    def test_default_used_for_other_agent(self) -> None:
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: 3.0)
        defs.set_action_weight(A, TICK, lambda s: 99.0)
        assert defs.get_action_weight(B, STORE_EMPTY, TICK) == pytest.approx(3.0)

    def test_store_passed_to_weight_function(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.set_action_weight(A, TICK, lambda s: received.append(s) or 1.0)
        store = AgentStore(Store(), Store({"v": 5}), Store())
        defs.get_action_weight(A, store, TICK)
        assert received[0] is store

    def test_store_passed_to_default_function(self) -> None:
        received: list = []
        defs = AgentDefinitions()
        defs.set_default_action_weight(TICK, lambda s: received.append(s) or 1.0)
        defs.get_action_weight(A, STORE_EMPTY, TICK)
        assert received[0] is STORE_EMPTY
