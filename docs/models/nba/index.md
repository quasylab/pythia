# Network of Broadcasting Agents (`nba`)

The `pythia.models.nba` submodule provides building blocks for systems based
on the **Network of Broadcasting Agents (NBA)** calculus — a process calculus
in which a population of agents interact through two synchronisation
mechanisms:

- **Point-to-point** — one sender (`c!`) pairs with exactly one receiver
  (`c?`) on the same channel `c`.
- **Broadcast** — one broadcaster (`c!!`) simultaneously synchronises with
  all agents currently able to receive (`c??`) on channel `c`.

---

## Concepts

### Channels

A [`Channel`](channel.md) is a named communication medium.  Channels are
value objects: two instances with the same name are equal and can be used as
dictionary keys.

### Actions

An [`Action`](action.md) describes the kind of interaction an agent performs.
There are four concrete subtypes, all immutable value objects:

| Type | Notation | Description |
|---|---|---|
| [`Send(c)`](action.md) | `c!` | Point-to-point output on channel `c` |
| [`Receive(c)`](action.md) | `c?` | Point-to-point input on channel `c` |
| [`BroadcastSend(c)`](action.md) | `c!!` | Broadcast output on channel `c` |
| [`BroadcastReceive(c)`](action.md) | `c??` | Broadcast input on channel `c` |

Two actions of *different* subtypes are never equal, even when their channels
match — `Send(c) != Receive(c)`.

### Agents and transitions

An [`Agent`](agent.md) is a named behavioral state.  Transitions between
states are stored in an [`AgentDefinitions`](agent_definitions.md) object,
which also serves as a factory ensuring that each name maps to a single
`Agent` instance.

Each transition `source --action--> target` carries a positive weight that
drives probabilistic selection inside
[`AgentDefinitions.step`](agent_definitions.md).

### Environment

An [`Environment`](environment.md) holds the numeric parameters that govern
how fast agents act and how receivers are chosen:

| Parameter | Setter | Role |
|---|---|---|
| **send rate** | `set_send_rate(agent, c, r)` | Rate at which an agent executes a point-to-point send on `c` |
| **broadcast rate** | `set_broadcast_rate(agent, c, r)` | Rate at which an agent executes a broadcast send on `c` |
| **receive weight** | `set_receive_weight(agent, c, w)` | Weight for selecting this agent as the point-to-point receiver |
| **receive probability** | `set_receive_probability(agent, c, p)` | Probability that an agent participates in a broadcast receive |

Every parameter has both an agent-specific override and a channel-level
default; when neither is set the value is `0.0`.

### Population state

An [`NBAConfiguration`](nba_configuration.md) is an **immutable counting
vector** that records how many copies of each agent type are currently
present.  It exposes:

- `count(agent)` / `fraction(agent)` — query individual agent counts.
- `send(rng, adef, env, source, c, target, receivers)` — apply a
  point-to-point send: remove one `source`, transition one selected receiver,
  add `target`.
- `broadcast_send(rng, adef, source, c, target)` — apply a broadcast send:
  remove one `source`, apply `broadcast_receive` to every remaining agent,
  add `target`.
- `step(rng, adef, env)` — select and execute one enabled action, returning
  `(time, next_cfg)` or `None` when no action is enabled.

### Simulation steps and timing

`NBAConfiguration.step` implements the **Gillespie algorithm**: it weights
each distinct agent type by `exit_rate(agent) × count(agent)`, samples the
acting agent, samples the sojourn time from an exponential distribution with
rate equal to the total weight, executes the chosen action, and returns
`(time, next_configuration)`.  When no agent has a positive exit rate the
method returns `None`, signalling that the system has reached a deadlock.

### Simulation model

[`NBAModel`](nba_model.md) wires all pieces together as a
[`BaseModel`](../../api/model.md).  It holds the initial agent counts, an
`AgentDefinitions`, and an `Environment`.  Each simulation tick delegates to
`NBAConfiguration.step`; a `None` result produces a step duration of `0.0`,
which causes the runner to stop.

---

## Class map

| Class | Module | Purpose |
|---|---|---|
| [`Channel`](channel.md) | `pythia.models.nba.channel` | Named communication channel |
| [`Action`](action.md) | `pythia.models.nba.action` | Abstract action base class |
| [`Send`](action.md) | `pythia.models.nba.action` | Point-to-point output |
| [`Receive`](action.md) | `pythia.models.nba.action` | Point-to-point input |
| [`BroadcastSend`](action.md) | `pythia.models.nba.action` | Broadcast output |
| [`BroadcastReceive`](action.md) | `pythia.models.nba.action` | Broadcast input |
| [`Agent`](agent.md) | `pythia.models.nba.agent` | Named behavioral state |
| [`AgentDefinitions`](agent_definitions.md) | `pythia.models.nba.agent_definitions` | Registry + transition table |
| [`Environment`](environment.md) | `pythia.models.nba.environment` | Per-agent rates and weights |
| [`NBAConfiguration`](nba_configuration.md) | `pythia.models.nba.nba_configuration` | Immutable population counting vector |
| [`NBAModel`](nba_model.md) | `pythia.models.nba.nba_model` | Simulation model (BaseModel) |

---

## Quick start

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.nba import AgentDefinitions, Channel, Environment, NBAModel

# 1. Define channels and agent types
c = Channel("sync")
defs = AgentDefinitions()
active  = defs.get("active")
standby = defs.get("standby")

# 2. Add transitions
#    active --c!--> standby  (point-to-point send)
defs.add_send_transition(active, c, standby, 1.0)
#    standby --c?--> active  (point-to-point receive)
defs.add_receive_transition(standby, c, active, 1.0)

# 3. Configure the environment
env = Environment()
env.set_send_rate(active, c, 1.0)        # rate at which active agents send
env.set_receive_weight(standby, c, 1.0)  # weight for standby to be chosen as receiver

# 4. Build and run the model
model = NBAModel({active: 5, standby: 5}, defs, env)
config = SimulationConfig(max_time=20.0, random_seed=42)
state = SimulationRunner(model, config).run()

cfg = state.data["nba.configuration"]
print(f"active:  {cfg.count(active)}")
print(f"standby: {cfg.count(standby)}")
print(f"elapsed: {state.elapsed_time:.3f}")
```

---

## Conventions

- **State key**: `"nba.configuration"` in `SimulationState.data` holds the
  current [`NBAConfiguration`](nba_configuration.md) after each step.
- **Step duration**: sampled from an exponential distribution whose rate is
  the total exit rate of the population — consistent with the Gillespie
  algorithm for continuous-time Markov chains.
- **Deadlock**: when no agent has an enabled action, `NBAConfiguration.step`
  returns `None` and `NBAModel.step` returns duration `0.0`, halting the
  runner.
- **Reproducibility**: pass `random_seed` to `SimulationConfig` for fully
  deterministic runs.

---

## Relation to LIO

| Concept | LIO | NBA |
|---|---|---|
| Interaction label | `Action(name)` — flat name | `Send(c)` / `Receive(c)` / etc. — typed with channel |
| Synchronisation | Independent per-agent steps | Point-to-point or broadcast pairs |
| Population state | `LIOState` (list or count map) | `NBAConfiguration` (count map only) |
| Timing | Fixed duration `1.0` per step | Exponentially distributed sojourn time |
| Agent states | `Agent(name)` | `Agent(name)` |
| Transition store | `AgentDefinitions` | `AgentDefinitions` |
