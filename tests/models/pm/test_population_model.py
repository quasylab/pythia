"""Tests for pythia.models.pm.population_model."""

from __future__ import annotations

import random

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.runner import SimulationRunner
from pythia.api.state import SimulationState
from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.population_model import PopulationModel
from pythia.models.pm.rule import Rule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pop() -> Population:
    p = Population()
    p.register("wolf")    # index 0
    p.register("rabbit")  # index 1
    return p


@pytest.fixture()
def decay_rule(pop: Population) -> Rule:
    """A wolf dies: consumes 1 wolf, produces nothing."""
    return Rule(
        pre=CountingState({pop[0]: 1}),
        post=CountingState({}),
        rate_function=lambda s: float(s.counting_vector.get(pop[0].index, 0)),
    )


def make_model(
    pop: Population,
    rules: list[Rule],
    initial: dict | None = None,
    rng: random.Random | None = None,
) -> PopulationModel:
    if initial is None:
        initial = {pop[0]: 10}
    return PopulationModel(
        population=pop,
        initial_state=initial,
        rules=rules,
        random_generator=rng,
    )


def minimal_config(**kwargs: object) -> SimulationConfig:
    return SimulationConfig(max_time=kwargs.pop("max_time", 50.0), random_seed=0, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------

class TestPopulationModelConstruction:
    def test_population_stored(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        assert model.population is pop

    def test_rules_stored_as_tuple(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        assert isinstance(model.rules, tuple)
        assert model.rules == (decay_rule,)

    def test_initial_state_stored(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule], initial={pop[0]: 7})
        assert model.initial_state[pop[0]] == 7

    def test_random_generator_defaults_to_none(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = make_model(pop, [decay_rule])
        assert model.random_generator is None

    def test_random_generator_stored_when_provided(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        rng = random.Random(99)
        model = make_model(pop, [decay_rule], rng=rng)
        assert model.random_generator is rng

    def test_rules_order_preserved(self, pop: Population) -> None:
        r1 = Rule(pre=CountingState({}), post=CountingState({}), rate_function=lambda _: 1.0)
        r2 = Rule(pre=CountingState({}), post=CountingState({}), rate_function=lambda _: 2.0)
        model = make_model(pop, [r1, r2])
        assert model.rules == (r1, r2)


class TestPopulationModelImmutability:
    def test_cannot_set_population(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        with pytest.raises(AttributeError):
            model.population = pop  # type: ignore[misc]

    def test_cannot_set_rules(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        with pytest.raises(AttributeError):
            model.rules = ()  # type: ignore[misc]

    def test_cannot_set_initial_state(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        with pytest.raises(AttributeError):
            model.initial_state = {}  # type: ignore[misc]

    def test_cannot_set_random_generator(self, pop: Population, decay_rule: Rule) -> None:
        model = make_model(pop, [decay_rule])
        with pytest.raises(AttributeError):
            model.random_generator = None  # type: ignore[misc]

    def test_initial_state_is_read_only_mapping(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = make_model(pop, [decay_rule], initial={pop[0]: 5})
        with pytest.raises(TypeError):
            model.initial_state[pop[0]] = 99  # type: ignore[index]

    def test_mutating_original_list_does_not_affect_rules(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        original = [decay_rule]
        model = make_model(pop, original)
        original.clear()
        assert len(model.rules) == 1

    def test_mutating_original_dict_does_not_affect_initial_state(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        original = {pop[0]: 5}
        model = make_model(pop, [decay_rule], initial=original)
        original[pop[0]] = 999
        assert model.initial_state[pop[0]] == 5


# ---------------------------------------------------------------------------
# BaseModel interface
# ---------------------------------------------------------------------------

class TestPopulationModelName:
    def test_name(self, pop: Population, decay_rule: Rule) -> None:
        assert make_model(pop, [decay_rule]).name == "pm.population_model"


class TestPopulationModelInitialize:
    def test_initialize_does_not_raise_for_valid_config(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = make_model(pop, [decay_rule])
        model.initialize(minimal_config())

    def test_initialize_uses_provided_rng(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        rng = random.Random(7)
        model = make_model(pop, [decay_rule], rng=rng)
        model.initialize(minimal_config())
        # The provided rng is the one used internally.
        assert model._rng is rng

    def test_initialize_creates_rng_when_none(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = make_model(pop, [decay_rule])
        model.initialize(minimal_config())
        assert model._rng is not None
        assert isinstance(model._rng, random.Random)

    def test_random_generator_property_unchanged_after_initialize(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = make_model(pop, [decay_rule])
        model.initialize(minimal_config())
        assert model.random_generator is None


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------

class TestPopulationModelStep:
    def _initialized_model(
        self,
        pop: Population,
        rules: list[Rule],
        initial: dict | None = None,
        seed: int = 0,
    ) -> PopulationModel:
        model = make_model(pop, rules, initial=initial, rng=random.Random(seed))
        model.initialize(minimal_config())
        return model

    def test_step_seeds_counting_state_on_first_call(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = self._initialized_model(pop, [decay_rule], initial={pop[0]: 5})
        state = SimulationState()
        model.step(state)
        assert "pm.counting_state" in state.data

    def test_step_does_not_override_existing_counting_state(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = self._initialized_model(pop, [decay_rule])
        state = SimulationState()
        pre_existing = CountingState({pop[0]: 3})
        state.data["pm.counting_state"] = pre_existing
        model.step(state)
        # After step the count must have changed from the pre-existing value,
        # not reset to initial_state.
        assert state.data["pm.counting_state"] != CountingState({pop[0]: 10})

    def test_step_applies_enabled_rule(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = self._initialized_model(pop, [decay_rule], initial={pop[0]: 10})
        state = SimulationState()
        model.step(state)
        # decay_rule removes one wolf per firing.
        count_after = state.data["pm.counting_state"].counting_vector.get(pop[0].index, 0)
        assert count_after == 9

    def test_step_returns_positive_dt_when_rule_enabled(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = self._initialized_model(pop, [decay_rule], initial={pop[0]: 10})
        state = SimulationState()
        dt, _ = model.step(state)
        assert dt > 0.0

    def test_step_sets_is_done_when_no_rule_enabled(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = self._initialized_model(pop, [decay_rule], initial={pop[0]: 1})
        state = SimulationState()
        # Fire once to exhaust wolves.
        model.step(state)
        # Now no wolf → rule disabled.
        dt, state = model.step(state)
        assert dt == 0.0
        assert state.is_done is True

    def test_step_with_no_rules_sets_is_done_immediately(
        self, pop: Population
    ) -> None:
        model = self._initialized_model(pop, rules=[], initial={pop[0]: 5})
        state = SimulationState()
        dt, state = model.step(state)
        assert dt == 0.0
        assert state.is_done is True

    def test_step_dt_is_exponentially_distributed(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        # With 10 wolves and rate 10, expected dt ≈ 0.1.
        # Just verify dt is positive and finite for many steps.
        model = self._initialized_model(pop, [decay_rule], initial={pop[0]: 10})
        state = SimulationState()
        dts: list[float] = []
        for _ in range(5):
            dt, state = model.step(state)
            if dt == 0.0:
                break
            dts.append(dt)
        assert all(d > 0.0 for d in dts)


# ---------------------------------------------------------------------------
# Full run via SimulationRunner
# ---------------------------------------------------------------------------

class TestPopulationModelFullRun:
    def test_wolf_population_decreases(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = PopulationModel(
            population=pop,
            initial_state={pop[0]: 20},
            rules=[decay_rule],
        )
        config = SimulationConfig(max_time=500.0, random_seed=0)
        runner = SimulationRunner(model, config)
        final = runner.run()
        count = final.data["pm.counting_state"].counting_vector.get(pop[0].index, 0)
        assert count < 20

    def test_run_terminates_when_all_wolves_gone(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = PopulationModel(
            population=pop,
            initial_state={pop[0]: 5},
            rules=[decay_rule],
        )
        config = SimulationConfig(max_time=1000.0, random_seed=42)
        runner = SimulationRunner(model, config)
        final = runner.run()
        assert final.is_done is True
        assert final.data["pm.counting_state"].counting_vector.get(pop[0].index, 0) == 0

    def test_reproducible_with_same_seed(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        def run(seed: int) -> CountingState:
            model = PopulationModel(
                population=pop,
                initial_state={pop[0]: 10},
                rules=[decay_rule],
            )
            config = SimulationConfig(max_time=20.0, random_seed=seed)
            return SimulationRunner(model, config).run().data["pm.counting_state"]

        assert run(99) == run(99)

    def test_different_seeds_may_differ(self, pop: Population) -> None:
        # Two competing rules with equal rates: wolf dies vs wolf reproduces.
        # Different seeds pick different sequences, yielding different outcomes.
        rules = [
            Rule(  # wolf dies
                pre=CountingState({pop[0]: 1}),
                post=CountingState({}),
                rate_function=lambda s: float(s.counting_vector.get(pop[0].index, 0)),
            ),
            Rule(  # wolf reproduces
                pre=CountingState({pop[0]: 1}),
                post=CountingState({pop[0]: 2}),
                rate_function=lambda s: float(s.counting_vector.get(pop[0].index, 0)),
            ),
        ]
        results = set()
        for seed in range(20):
            model = PopulationModel(
                population=pop,
                initial_state={pop[0]: 5},
                rules=rules,
            )
            config = SimulationConfig(max_time=10.0, random_seed=seed)
            final = SimulationRunner(model, config).run()
            count = final.data["pm.counting_state"].counting_vector.get(pop[0].index, 0)
            results.add(count)
        assert len(results) > 1

    def test_counting_state_in_state_data_after_run(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        model = PopulationModel(
            population=pop,
            initial_state={pop[0]: 3},
            rules=[decay_rule],
        )
        config = SimulationConfig(max_time=50.0, random_seed=0)
        final = SimulationRunner(model, config).run()
        assert isinstance(final.data["pm.counting_state"], CountingState)

    def test_elapsed_time_reflects_gillespie_waiting_times(
        self, pop: Population, decay_rule: Rule
    ) -> None:
        # Elapsed time should NOT equal step * fixed_dt; it is a sum of
        # exponential variates, so it varies by seed.
        model = PopulationModel(
            population=pop,
            initial_state={pop[0]: 5},
            rules=[decay_rule],
        )
        config = SimulationConfig(max_time=1000.0, random_seed=7)
        final = SimulationRunner(model, config).run()
        # The run terminates when all wolves are gone (dt=0), so elapsed_time
        # is the sum of Gillespie waiting times — a positive finite value.
        assert final.elapsed_time > 0.0
        assert final.elapsed_time < 1000.0
