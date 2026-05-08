"""Tests for pythia.models.yoda.yoda_model."""

from __future__ import annotations

import random

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.state import SimulationState
from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_definitions import AgentDefinitions
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.configuration import SystemConfiguration
from pythia.models.yoda.store import Store
from pythia.models.yoda.yoda_model import YodaModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

A    = Agent("a")
IDLE = AgentState("idle")
BUSY = AgentState("busy")


def _store() -> AgentStore:
    return AgentStore(Store(), Store({"x": 0.0}), Store())


def _aconf(agent: Agent = A, state: AgentState = IDLE) -> AgentConfiguration:
    return AgentConfiguration(agent, state, _store())


def _conf(*agents_states: tuple[Agent, AgentState]) -> SystemConfiguration:
    return SystemConfiguration([
        AgentConfiguration(a, s, _store()) for a, s in agents_states
    ])


def _initial_conf() -> SystemConfiguration:
    return _conf((A, IDLE))


def _model(
    env_step: float = 1.0,
    agent_step: float = 1.0,
    defs: AgentDefinitions | None = None,
) -> YodaModel:
    return YodaModel(
        initial_state_generator=lambda rng: _initial_conf(),
        agent_definitions=defs or AgentDefinitions(),
        environment_simulation_step=env_step,
        agents_simulation_step=agent_step,
    )


def _state(**data) -> SimulationState:
    s = SimulationState()
    s.data.update(data)
    return s


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_name(self) -> None:
        assert _model().name == "yoda"

    def test_defaults(self) -> None:
        m = _model()
        assert m._environment_simulation_step == 1.0
        assert m._agents_simulation_step == 1.0

    def test_custom_steps(self) -> None:
        m = _model(env_step=0.5, agent_step=2.0)
        assert m._environment_simulation_step == 0.5
        assert m._agents_simulation_step == 2.0


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

class TestInitialize:
    def _cfg(self, **meta) -> SimulationConfig:
        return SimulationConfig(max_time=10.0, random_seed=42, metadata=meta)

    def test_seeds_rng(self) -> None:
        m = _model()
        m.initialize(self._cfg())
        r1 = m._rng.random()
        m.initialize(self._cfg())
        r2 = m._rng.random()
        assert r1 == r2  # same seed → same first value

    def test_none_seed_is_non_deterministic(self) -> None:
        m = _model()
        m.initialize(SimulationConfig(max_time=10.0, random_seed=None))
        assert m._rng is not None

    def test_env_step_overridden_from_metadata(self) -> None:
        m = _model()
        m.initialize(self._cfg(**{"yoda.environment_simulation_step": 0.25}))
        assert m._environment_simulation_step == 0.25

    def test_agent_step_overridden_from_metadata(self) -> None:
        m = _model()
        m.initialize(self._cfg(**{"yoda.agents_simulation_step": 3.0}))
        assert m._agents_simulation_step == 3.0



# ---------------------------------------------------------------------------
# step — initial configuration
# ---------------------------------------------------------------------------

class TestStepInitialConfiguration:
    def test_generator_called_when_no_config_in_state(self) -> None:
        called = []
        conf = _initial_conf()
        m = YodaModel(
            initial_state_generator=lambda rng: called.append(True) or conf,
            agent_definitions=AgentDefinitions(),
        )
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        m.step(SimulationState())
        assert called == [True]

    def test_generator_not_called_when_config_present(self) -> None:
        called = []
        conf = _initial_conf()
        m = YodaModel(
            initial_state_generator=lambda rng: called.append(True) or conf,
            agent_definitions=AgentDefinitions(),
        )
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        state.data["yoda.configuration"] = conf
        m.step(state)
        assert called == []

    def test_configuration_stored_in_state(self) -> None:
        m = _model()
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        m.step(state)
        assert "yoda.configuration" in state.data


# ---------------------------------------------------------------------------
# step — scheduling: env_time < agent_time
# ---------------------------------------------------------------------------

class TestStepEnvFirst:
    def test_environment_step_called(self) -> None:
        defs = AgentDefinitions()
        called = []
        defs.add_dynamics(A, lambda rng, s, dt: called.append(dt) or s.features)
        m = YodaModel(
            initial_state_generator=lambda rng: _conf((A, IDLE)),
            agent_definitions=defs,
            environment_simulation_step=1.0,
            agents_simulation_step=3.0,
        )
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        dt, _ = m.step(state)
        assert dt == pytest.approx(1.0)
        assert called[0] == pytest.approx(1.0)

    def test_env_time_incremented(self) -> None:
        m = _model(env_step=1.0, agent_step=3.0)
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        m.step(state)
        assert state.data["yoda.time_of_next_environment_step"] == pytest.approx(2.0)
        assert state.data["yoda.time_of_next_agents_step"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# step — scheduling: env_time == agent_time
# ---------------------------------------------------------------------------

class TestStepBothTogether:
    def test_agents_step_called(self) -> None:
        from pythia.models.yoda.action import Action
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, 1.0, BUSY)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        m = YodaModel(
            initial_state_generator=lambda rng: _conf((A, IDLE)),
            agent_definitions=defs,
            environment_simulation_step=1.0,
            agents_simulation_step=1.0,
        )
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        dt, state = m.step(state)
        assert dt == pytest.approx(1.0)
        assert state.data["yoda.configuration"].get(0).agent_state == BUSY

    def test_both_times_incremented(self) -> None:
        m = _model(env_step=1.0, agent_step=1.0)
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        m.step(state)
        assert state.data["yoda.time_of_next_environment_step"] == pytest.approx(2.0)
        assert state.data["yoda.time_of_next_agents_step"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# step — scheduling: env_time > agent_time
# ---------------------------------------------------------------------------

class TestStepAgentFirst:
    def test_action_executions_step_called(self) -> None:
        from pythia.models.yoda.action import Action
        tick = Action("tick", lambda rng, s, o: s)
        defs = AgentDefinitions()
        defs.add_transition(A, IDLE, tick, 1.0, BUSY)
        defs.set_action_weight(A, tick, lambda s: 1.0)
        m = YodaModel(
            initial_state_generator=lambda rng: _conf((A, IDLE)),
            agent_definitions=defs,
            environment_simulation_step=3.0,
            agents_simulation_step=1.0,
        )
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        dt, state = m.step(state)
        assert dt == pytest.approx(1.0)
        assert state.data["yoda.configuration"].get(0).agent_state == BUSY

    def test_only_agent_time_incremented(self) -> None:
        m = _model(env_step=3.0, agent_step=1.0)
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        m.step(state)
        assert state.data["yoda.time_of_next_environment_step"] == pytest.approx(3.0)
        assert state.data["yoda.time_of_next_agents_step"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# step — time values persist across calls
# ---------------------------------------------------------------------------

class TestStepPersistence:
    def test_times_carried_across_steps(self) -> None:
        m = _model(env_step=2.0, agent_step=3.0)
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        # step 1: env_time=2 < agent_time=3 → environment step, dt=2
        dt1, state = m.step(state)
        state.advance(dt1)
        assert dt1 == pytest.approx(2.0)
        # step 2: env_time=4 > agent_time=3 → agent step, dt=1
        dt2, state = m.step(state)
        assert dt2 == pytest.approx(1.0)

    def test_configuration_carried_across_steps(self) -> None:
        m = _model()
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        _, state = m.step(state)
        conf1 = state.data["yoda.configuration"]
        _, state = m.step(state)
        # conf1 was used as input (not the generator)
        assert "yoda.configuration" in state.data


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_finalize_is_noop(self) -> None:
        m = _model()
        m.initialize(SimulationConfig(max_time=10.0, random_seed=0))
        state = SimulationState()
        m.finalize(state)  # should not raise
