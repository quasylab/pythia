"""Tests for pythia.models.lio.lio_model."""

from __future__ import annotations

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.runner import SimulationRunner
from pythia.api.state import SimulationState
from pythia.models.lio.action import Action
from pythia.models.lio.action_weight import ActionWeightFunction
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
from pythia.models.lio.lio_model import LioModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICK = Action("tick")


def _make_loop_defs(*names: str) -> tuple[AgentDefinitions, list[Agent]]:
    """Return AgentDefinitions whose agents each have a self-loop on TICK."""
    defs = AgentDefinitions()
    agents = [defs.get(n) for n in names]
    for agent in agents:
        defs.add(agent, TICK, agent, 1.0)
    return defs, agents


def _uniform_awf() -> ActionWeightFunction:
    awf = ActionWeightFunction()
    awf.set_default(TICK, lambda cfg: 1.0)
    return awf


def _make_model(
    names: list[str],
    awf: ActionWeightFunction | None = None,
) -> LioModel:
    defs, agents = _make_loop_defs(*names)
    return LioModel(
        initial_configuration=LIOStateOfIndividuals(agents),
        action_weights=awf if awf is not None else _uniform_awf(),
        agent_definitions=defs,
    )


def _make_config(max_time: float = 5.0, seed: int = 0) -> SimulationConfig:
    return SimulationConfig(max_time=max_time, random_seed=seed)


# ---------------------------------------------------------------------------
# name
# ---------------------------------------------------------------------------

class TestLioModelName:
    def test_name(self) -> None:
        assert _make_model(["a"]).name == "lio"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

class TestLioModelInitialize:
    def test_initialize_accepts_valid_config(self) -> None:
        _make_model(["a"]).initialize(_make_config())  # must not raise

    def test_initialize_with_none_seed_does_not_raise(self) -> None:
        _make_model(["a"]).initialize(SimulationConfig(max_time=5.0, random_seed=None))

    def test_same_seed_produces_same_rng_sequence(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)
        cfg = SimulationConfig(max_time=10.0, random_seed=7)

        m1 = LioModel(LIOStateOfIndividuals([a, b]), awf, defs)
        m2 = LioModel(LIOStateOfIndividuals([a, b]), awf, defs)
        m1.initialize(cfg)
        m2.initialize(cfg)

        state1, state2 = SimulationState(), SimulationState()
        for _ in range(5):
            _, state1 = m1.step(state1)
            _, state2 = m2.step(state2)

        c1 = state1.data["lio.configuration"]
        c2 = state2.data["lio.configuration"]
        assert [c1.agent(i) for i in range(c1.size)] == [
            c2.agent(i) for i in range(c2.size)
        ]

    def test_reinitialize_reseeds_rng(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)
        cfg = SimulationConfig(max_time=5.0, random_seed=42)
        model = LioModel(LIOStateOfIndividuals([a, b]), awf, defs)

        def _run_steps(n: int) -> list[Agent]:
            model.initialize(cfg)
            state = SimulationState()
            for _ in range(n):
                _, state = model.step(state)
            c = state.data["lio.configuration"]
            return [c.agent(i) for i in range(c.size)]

        assert _run_steps(4) == _run_steps(4)


# ---------------------------------------------------------------------------
# step — unit
# ---------------------------------------------------------------------------

class TestLioModelStep:
    def test_step_writes_lio_configuration(self) -> None:
        model = _make_model(["a", "b"])
        model.initialize(_make_config())
        _, state = model.step(SimulationState())
        assert "lio.configuration" in state.data

    def test_step_returns_dt_one(self) -> None:
        model = _make_model(["a"])
        model.initialize(_make_config())
        dt, _ = model.step(SimulationState())
        assert dt == pytest.approx(1.0)

    def test_step_uses_initial_configuration_on_first_call(self) -> None:
        defs, agents = _make_loop_defs("a")
        model = LioModel(LIOStateOfIndividuals(agents), _uniform_awf(), defs)
        model.initialize(_make_config())
        _, state = model.step(SimulationState())
        assert state.data["lio.configuration"].agent(0) == agents[0]

    def test_step_size_preserved(self) -> None:
        model = _make_model(["a", "b", "c"])
        model.initialize(_make_config())
        _, state = model.step(SimulationState())
        assert state.data["lio.configuration"].size == 3

    def test_step_result_becomes_next_input(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)

        model = LioModel(LIOStateOfIndividuals([a]), awf, defs)
        model.initialize(_make_config())
        state = SimulationState()

        _, state = model.step(state)
        assert state.data["lio.configuration"].agent(0) == b

        _, state = model.step(state)
        assert state.data["lio.configuration"].agent(0) == a

    def test_step_deterministic_with_fixed_seed(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)

        def _outcome(seed: int) -> Agent:
            model = LioModel(LIOStateOfIndividuals([a]), awf, defs)
            model.initialize(SimulationConfig(max_time=5.0, random_seed=seed))
            _, state = model.step(SimulationState())
            return state.data["lio.configuration"].agent(0)

        assert _outcome(99) == _outcome(99)

    def test_step_passes_action_weights_get_conf(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)

        seen: list[LIOStateOfIndividuals] = []

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: (seen.append(cfg) or 1.0))  # type: ignore[func-returns-value]

        model = LioModel(LIOStateOfIndividuals([a]), awf, defs)
        model.initialize(_make_config())
        model.step(SimulationState())

        assert len(seen) == 1

    def test_step_returns_same_state_object(self) -> None:
        model = _make_model(["a"])
        model.initialize(_make_config())
        state = SimulationState()
        _, returned = model.step(state)
        assert returned is state


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

class TestLioModelFinalize:
    def test_finalize_does_not_raise(self) -> None:
        model = _make_model(["a"])
        model.initialize(_make_config())
        model.finalize(SimulationState())  # must not raise


# ---------------------------------------------------------------------------
# full run via SimulationRunner
# ---------------------------------------------------------------------------

class TestLioModelFullRun:
    def test_state_contains_configuration_after_run(self) -> None:
        state = SimulationRunner(_make_model(["a", "b"]), _make_config(max_time=5.0)).run()
        assert "lio.configuration" in state.data

    def test_step_count_equals_max_time(self) -> None:
        state = SimulationRunner(_make_model(["a", "b"]), _make_config(max_time=10.0)).run()
        assert state.step == 10

    def test_elapsed_time_equals_max_time(self) -> None:
        state = SimulationRunner(_make_model(["a"]), _make_config(max_time=7.0)).run()
        assert state.elapsed_time == pytest.approx(7.0)

    def test_configuration_size_unchanged_after_run(self) -> None:
        state = SimulationRunner(_make_model(["a", "b", "c"]), _make_config(max_time=5.0)).run()
        assert state.data["lio.configuration"].size == 3

    def test_run_is_reproducible_with_same_seed(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)

        def _run(seed: int) -> list[Agent]:
            model = LioModel(LIOStateOfIndividuals([a, b]), awf, defs)
            state = SimulationRunner(model, SimulationConfig(max_time=8.0, random_seed=seed)).run()
            c = state.data["lio.configuration"]
            return [c.agent(i) for i in range(c.size)]

        assert _run(123) == _run(123)

    def test_different_seeds_produce_different_outcomes(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(a, tick, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(tick, lambda cfg: 1.0)

        finals: list[Agent] = []
        for seed in range(20):
            model = LioModel(LIOStateOfIndividuals([a]), awf, defs)
            state = SimulationRunner(model, SimulationConfig(max_time=1.0, random_seed=seed)).run()
            finals.append(state.data["lio.configuration"].agent(0))

        assert len(set(finals)) > 1

    def test_agent_specific_weight_overrides_default(self) -> None:
        t1, t2 = Action("t1"), Action("t2")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, t1, b, 1.0)
        defs.add(a, t2, a, 1.0)

        awf = ActionWeightFunction()
        awf.set_default(t1, lambda cfg: 1.0)
        awf.set_default(t2, lambda cfg: 1.0)
        awf.set(a, t1, lambda cfg: 1.0)
        awf.set(a, t2, lambda cfg: 0.0)  # suppress t2 for a

        model = LioModel(LIOStateOfIndividuals([a]), awf, defs)
        state = SimulationRunner(model, SimulationConfig(max_time=1.0, random_seed=0)).run()
        assert state.data["lio.configuration"].agent(0) == b
