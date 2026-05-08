# YODA (`yoda`)

The `pythia.models.yoda` submodule provides building blocks for describing
and simulating **YODA (Yet anOther agent Description lAnguage)** specifications.
A YODA system is a population of agents, each carrying layered data stores,
that evolve continuously through dynamics and interact discretely through
typed actions.

---

## Concepts

### Data stores

A [`Store`](store.md) is an immutable key-value mapping (attribute name →
arbitrary value) that supports priority-order lookup, predicate testing, and
non-destructive update via `set`.

An [`AgentStore`](agent_store.md) groups three `Store` layers for a single
agent:

| Layer | Purpose |
|---|---|
| `state` | Private state variables owned by the agent |
| `features` | Derived or computed attributes set by the environment |
| `observations` | Values observed from the environment or other agents |

`AgentStore.get(attr)` resolves attributes in priority order: `state` →
`features` → `observations`.

### Agents and behavioural states

An [`Agent`](agent.md) identifies a participant in the system by name.  Two
`Agent` instances with the same name are equal and hash-equivalent.

An [`AgentState`](agent_state.md) represents one discrete behavioural state
that an agent can occupy (e.g. `"idle"`, `"busy"`).  Like `Agent`, equality
is name-based.

### Actions

An [`Action`](action.md) encodes a named transition an agent can execute.
It carries a **step function** `(rng, state_store, observations_store) -> Store`
that computes the new private state when the action fires.

### Agent configuration

An [`AgentConfiguration`](agent_configuration.md) binds an `Agent`, an
`AgentState`, and an `AgentStore` together as an immutable snapshot of a
single agent at one point in time.  Utility methods provide non-destructive
updates:

| Method | Effect |
|---|---|
| `set_state(agent_state, store)` | Replace behavioural state and state layer |
| `set_features(store)` | Replace features layer |
| `set_observations(store)` | Replace observations layer |
| `update_observations(f)` | Compute new observations via a function |
| `apply_dynamics(rng, dt, agent_dynamics)` | Advance features by *dt* |

### System configuration

A [`SystemConfiguration`](system_configuration.md) is an immutable ordered
list of `AgentConfiguration` objects representing the full population at one
point in time.  It provides:

- Positional access (`get(i)`), size, and element removal.
- Bulk transformation (`apply(f)`).
- Quantitative queries: `exists`, `forall`, `mean`, `min`, `max` — all
  optionally restricted to a subset of agents by index or predicate.

### Agent definitions

[`AgentDefinitions`](agent_definitions.md) is the central registry that
describes agent behaviour.  It stores five kinds of information:

| Table | Set via | Purpose |
|---|---|---|
| Transition table | `add_transition(agent, source, action, weight, target)` | Weighted next-state distribution for each `(agent, state, action)` triple |
| Dynamics | `add_dynamics(agent, f)` / `default_dynamics` | Continuous evolution of features: `(rng, AgentStore, dt) -> Store` |
| Observations | `add_observation(agent, f)` / `default_observation` | Compute observations from the full system: `(rng, AgentStore, SystemConfiguration) -> Store` |
| Action weights | `set_action_weight` / `set_default_action_weight` | Propensity of each action given the current store |

It also exposes high-level **step methods** that advance a
`SystemConfiguration`:

| Method | Description |
|---|---|
| `dynamics_step(rng, conf, dt)` | Apply dynamics to every agent |
| `observations_step(rng, conf)` | Update observations for every agent |
| `environment_step(rng, conf, dt)` | Dynamics then observations |
| `execute_step(rng, aconf)` | One agent samples and fires an action |
| `action_executions_step(rng, conf)` | Every agent independently executes one step |
| `agents_step(rng, conf, dt)` | Environment step followed by action executions |

### Simulation model

[`YodaModel`](yoda_model.md) wires all pieces together as a
[`BaseModel`](../../api/model.md).  It interleaves two independently
configurable schedules:

* **Environment steps** — dynamics + observations, every
  `environment_simulation_step` time units.
* **Agent-action steps** — each agent fires one action, every
  `agents_simulation_step` time units.

When both schedules coincide, `agents_step` (environment + actions) is used.

---

## Class map

| Class | Module | Purpose |
|---|---|---|
| [`Store`](store.md) | `pythia.models.yoda.store` | Immutable attribute-value mapping |
| [`AgentStore`](agent_store.md) | `pythia.models.yoda.agent_store` | Three-layer data store for one agent |
| [`Action`](action.md) | `pythia.models.yoda.action` | Named transition with a step function |
| [`Agent`](agent.md) | `pythia.models.yoda.agent` | Named participant |
| [`AgentState`](agent_state.md) | `pythia.models.yoda.agent_state` | Named behavioural state |
| [`AgentConfiguration`](agent_configuration.md) | `pythia.models.yoda.agent_configuration` | Immutable (agent, state, store) triple |
| [`AgentDefinitions`](agent_definitions.md) | `pythia.models.yoda.agent_definitions` | Transitions, dynamics, observations, weights |
| [`SystemConfiguration`](system_configuration.md) | `pythia.models.yoda.configuration` | Immutable ordered population snapshot |
| [`YodaModel`](yoda_model.md) | `pythia.models.yoda.yoda_model` | Simulation model (BaseModel) |

---

## Quick start

```python
import random
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.yoda import (
    Action, Agent, AgentConfiguration, AgentDefinitions,
    AgentState, AgentStore, SystemConfiguration, YodaModel,
)
from pythia.models.yoda.store import Store

# 1. Define agents and states
a    = Agent("worker")
idle = AgentState("idle")
busy = AgentState("busy")

# 2. Define an action: worker ticks a counter when working
tick = Action(
    "tick",
    step_function=lambda rng, state, obs: state.set(
        "count", state.get("count", 0) + 1
    ),
)

# 3. Register transitions and weights
defs = AgentDefinitions()
defs.add_transition(a, idle, tick, 1.0, busy)
defs.add_transition(a, busy, tick, 1.0, idle)
defs.set_default_action_weight(tick, lambda s: 1.0)

# 4. Register dynamics (features grow over time)
defs.add_dynamics(
    a,
    lambda rng, s, dt: s.features.set("energy", s.features.get("energy", 1.0) - 0.1 * dt),
)

# 5. Build initial configuration
def initial(rng: random.Random) -> SystemConfiguration:
    store = AgentStore(Store(), Store({"energy": 1.0}), Store())
    return SystemConfiguration([AgentConfiguration(a, idle, store)])

# 6. Run
model = YodaModel(
    initial_state_generator=initial,
    agent_definitions=defs,
    environment_simulation_step=0.1,
    agents_simulation_step=1.0,
)
state = SimulationRunner(model, SimulationConfig(max_time=10.0, random_seed=42)).run()
cfg = state.data["yoda.configuration"]
print(f"agent state: {cfg.get(0).agent_state.name}")
print(f"energy:      {cfg.get(0).agent_store.features.get('energy'):.3f}")
```

---

## Conventions

- **State key**: `"yoda.configuration"` in `SimulationState.data` holds the
  current `SystemConfiguration` after each step.
- **Scheduling**: Two independent clocks drive the simulation —
  `"yoda.time_of_next_environment_step"` and `"yoda.time_of_next_agents_step"`.
- **Reproducibility**: pass `random_seed` to `SimulationConfig` for fully
  deterministic runs.
- **Deadlock-free**: agents with no enabled action are silently skipped by
  `execute_step`; the model always makes forward progress.
