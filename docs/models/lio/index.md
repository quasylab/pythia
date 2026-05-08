# Language of Interacting Objects (`lio`)

The `pythia.models.lio` submodule implements simulation models based on the
**Language of Interacting Objects (LIO)**, a formal modelling language in
which a system is described as a collection of objects (agents) that evolve
by independently selecting and executing actions.

---

## Concepts

### Agents and actions

An **agent** represents a named behavioral state that a participant in the
system can occupy.  An **action** is a named interaction label.  Agents
transition between states by executing actions: each `(source, action,
target)` triple carries a positive weight that governs how likely the
transition is relative to alternatives.

Transitions are stored centrally in an
[`AgentDefinitions`](agent_definitions.md) object which also serves as a
factory ensuring each name maps to exactly one [`Agent`](agent.md) instance.

### System state

The state of a LIO system at any point in time is a population of agents.
Two concrete representations are provided:

| Class | Storage | Best for |
|---|---|---|
| [`LIOStateOfIndividuals`](lio_state_of_individuals.md) | Ordered list — one slot per individual | Small populations; preserving insertion order |
| [`LIOStateOfCountingVector`](lio_state_of_counting_vector.md) | `Agent → int` map | Large populations with few distinct types |

Both implement the abstract [`LIOState`](lio_state.md) interface which
mandates `fraction` and `step`.

### Action weights

An [`ActionWeightFunction`](action_weight.md) supplies, for each agent in the
current state, a weight for each action.  Weights may depend on the full
configuration — allowing density-dependent or environment-dependent selection.
Agent-specific overrides take priority over module-level defaults.

### Simulation model

[`LioModel`](lio_model.md) wires these pieces together as a
[`BaseModel`](../../api/model.md).  Each simulation tick advances every agent
by one step; the step duration is always `1.0`.

---

## Class map

| Class | Module | Purpose |
|---|---|---|
| [`Action`](action.md) | `pythia.models.lio.action` | Named action label |
| [`Agent`](agent.md) | `pythia.models.lio.agent` | Named behavioral state |
| [`AgentDefinitions`](agent_definitions.md) | `pythia.models.lio.agent_definitions` | Registry + transition table |
| [`ActionWeightFunction`](action_weight.md) | `pythia.models.lio.action_weight` | Per-agent action weight functions |
| [`LIOState`](lio_state.md) | `pythia.models.lio.lio_state` | Abstract population state |
| [`LIOStateOfIndividuals`](lio_state_of_individuals.md) | `pythia.models.lio.lio_state_of_individuals` | List-based state |
| [`LIOStateOfCountingVector`](lio_state_of_counting_vector.md) | `pythia.models.lio.lio_state_of_counting_vector` | Count-map state |
| [`LioModel`](lio_model.md) | `pythia.models.lio.lio_model` | Simulation model |

---

## Quick start

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.lio import (
    Action, AgentDefinitions, ActionWeightFunction,
    LIOStateOfIndividuals, LioModel,
)

# 1. Define agent types and transitions
defs = AgentDefinitions()
susceptible = defs.get("susceptible")
infected    = defs.get("infected")
recovered   = defs.get("recovered")

infect  = Action("infect")
recover = Action("recover")

defs.add(susceptible, infect,  infected,   1.0)
defs.add(infected,    recover, recovered,  1.0)

# 2. Define action weights (may depend on the current state)
awf = ActionWeightFunction()
awf.set_default(infect,  lambda cfg: cfg.fraction(infected))   # density-dependent
awf.set_default(recover, lambda cfg: 0.1)                      # constant rate

# 3. Build the initial population
initial = LIOStateOfIndividuals(
    [susceptible] * 90 + [infected] * 10
)

# 4. Run
model = LioModel(initial, awf, defs)
config = SimulationConfig(max_time=50.0, random_seed=42)
state = SimulationRunner(model, config).run()

cfg = state.data["lio.configuration"]
print(f"susceptible: {cfg.count(susceptible)}")
print(f"infected:    {cfg.count(infected)}")
print(f"recovered:   {cfg.count(recovered)}")
```

---

## Conventions

- **State key**: `"lio.configuration"` in `SimulationState.data` holds the
  current [`LIOState`](lio_state.md) after each step.
- **Step duration**: always `1.0` — `SimulationConfig.max_time` is therefore
  equal to the number of simulation ticks.
- **Reproducibility**: pass `random_seed` to `SimulationConfig` for fully
  deterministic runs.
