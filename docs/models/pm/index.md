# Population Models (`pm`)

The `pythia.models.pm` submodule provides a continuous-time stochastic
population simulator and the supporting value objects needed to describe a
population and its reactions.  The model is a concrete implementation of
[`BaseModel`](../../api/model.md) and follows the standard lifecycle
(`initialize` / `step` / `finalize`).

---

## Concepts

### Species and populations

A [`Species`](species.md) is an immutable descriptor for one type of
individual in the system.  Species are created and tracked by a
[`Population`](population.md) registry, which guarantees that each name maps
to exactly one instance and assigns each species a unique integer index.

### Population state

A [`CountingState`](counting_state.md) is an immutable snapshot that maps
each species to a non-negative integer count.  It is used both to represent
the current population and to describe the stoichiometry of reaction rules
(the *pre* and *post* states of each rule).

Key convenience accessors:

| Method / Property | Description |
|---|---|
| `size` | Total number of individuals across all species (O(1), updated by `add`/`remove`) |
| `count(species)` | Number of individuals of a given species; `0` if absent |

### Reaction rules

A [`Rule`](rule.md) encodes a single stochastic reaction: a *pre* condition
(`CountingState` of required individuals), a *post* result (`CountingState`
of produced individuals), and a *rate function* that maps the current state
to a non-negative propensity.  Rules are the sole driver of state change in
[`PopulationModel`](population_model.md).

### Simulation model

[`PopulationModel`](population_model.md) implements the **Gillespie
algorithm**: at each step it evaluates all rules, selects one with probability
proportional to its propensity, fires it, and samples the waiting time from
an exponential distribution with rate equal to the total propensity.  When no
rule is enabled the step returns `dt = 0.0`, halting the runner.

---

## Class map

| Class | Module | Purpose |
|---|---|---|
| [`Species`](species.md) | `pythia.models.pm.species` | Immutable species descriptor |
| [`Population`](population.md) | `pythia.models.pm.population` | Registry that creates and tracks species |
| [`CountingState`](counting_state.md) | `pythia.models.pm.counting_state` | Immutable species-count snapshot |
| [`Rule`](rule.md) | `pythia.models.pm.rule` | Single stochastic reaction rule |
| [`PopulationModel`](population_model.md) | `pythia.models.pm.population_model` | Gillespie simulation model |

---

## Quick start

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.pm import CountingState, Population, PopulationModel, Rule

# 1. Define species
pop = Population()
S   = pop.register("S")   # susceptible
I   = pop.register("I")   # infected
R   = pop.register("R")   # recovered

# 2. Define reaction rules (SIR model)
rules = [
    # S + I -> 2I  (infection, mass-action rate 0.3)
    Rule(
        pre=CountingState({S: 1, I: 1}),
        post=CountingState({I: 2}),
        rate_function=lambda s: 0.3 * s.counting_vector.get(S.index, 0)
                                      * s.counting_vector.get(I.index, 0),
    ),
    # I -> R  (recovery, rate 0.1 per infected individual)
    Rule(
        pre=CountingState({I: 1}),
        post=CountingState({R: 1}),
        rate_function=lambda s: 0.1 * s.counting_vector.get(I.index, 0),
    ),
]

# 3. Build and run the model
model = PopulationModel(
    population=pop,
    initial_state={S: 990, I: 10, R: 0},
    rules=rules,
)
state = SimulationRunner(model, SimulationConfig(max_time=50.0, random_seed=42)).run()

cs = state.data["pm.counting_state"]
print(f"S={cs.count(S)}  I={cs.count(I)}  R={cs.count(R)}")
print(f"Total: {cs.size}  Steps: {state.step}  Elapsed: {state.elapsed_time:.2f}")
```

---

## Conventions

- **State key**: `"pm.counting_state"` in `SimulationState.data` holds the
  current [`CountingState`](counting_state.md) after each step.
- **Step duration**: sampled from an exponential distribution whose rate is
  the sum of all enabled rule propensities — consistent with the Gillespie
  algorithm for continuous-time Markov chains.
- **Deadlock**: when no rule is enabled, `PopulationModel.step` returns
  `dt = 0.0`, halting the runner.
- **Reproducibility**: pass `random_seed` to `SimulationConfig` for fully
  deterministic runs.
