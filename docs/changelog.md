# Changelog

All notable changes to Pythia are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.0] — 2026-05-08

### Added

#### `pythia.models.nba`

- `NBAConfiguration.broadcast_send(rng, adef, source, c, target)` — applies a
  broadcast-send action: removes one `source`, applies `broadcast_receive` to
  every remaining agent, adds `target`
- `NBAConfiguration.send(rng, adef, env, source, c, target, receivers)` —
  applies a point-to-point send: removes one `source`, selects and transitions
  one receiver weighted by `count × receive_weight`, adds `target`
- `NBAConfiguration.step(rng, adef, env)` — Gillespie step returning
  `(time, next_configuration) | None`; sojourn time is sampled from
  `Exp(total_weight)`; returns `None` when no agent has an enabled action
- `NBAModel` — new `BaseModel` that simulates an NBA specification; state key
  `"nba.configuration"`; returns `dt = 0.0` when `step` returns `None`
- Full documentation for `Environment`, `NBAConfiguration`, and `NBAModel` in
  the NBA module docs; `mkdocs.yml` nav updated accordingly

#### `pythia.models.yoda` (new module)

- `Store` — immutable key-value mapping with `get`, `set`, `test`,
  `__contains__`, `__len__`, `__getitem__`, and value-semantic `__eq__` /
  `__hash__`
- `AgentStore` — three-layer immutable store (`observations`, `features`,
  `state`) with priority-order `get`; `set_state`, `set_features`,
  `set_observations`, and `apply_dynamics` mutation helpers
- `Action` — named agent transition carrying a step function
  `(rng, state_store, observations_store) -> Store`
- `Agent` — named participant; equality and hashing by name
- `AgentState` — named behavioural state; equality and hashing by name
- `AgentConfiguration` — immutable `(Agent, AgentState, AgentStore)` triple;
  `set_state`, `set_features`, `set_observations`, `update_observations`,
  `apply_dynamics` mutation helpers; static `get_predicate` factory
- `AgentDefinitions` — central registry for:
  - **Transitions** — `add_transition(agent, source, action, weight, target)`
    storing weighted `(AgentState → AgentState)` distributions per action
  - **Dynamics** — `add_dynamics` / `get_dynamics` / `default_dynamics`
    (`(rng, AgentStore, dt) -> Store`); `apply_dynamics`, `dynamics_step`
  - **Observations** — `add_observation` / `get_observation` /
    `default_observation` (`(rng, AgentStore, SystemConfiguration) -> Store`);
    `apply_observations`, `observations_step`
  - **Action weights** — `set_action_weight`, `set_default_action_weight`,
    `get_action_weight(agent, store, action) -> float`
  - **Step methods** — `environment_step`, `action_executions_step`,
    `agents_step`, `execute_step`, `execute_action`, `sample_action`,
    `environment_step`, `agents_step`
- `SystemConfiguration` — immutable ordered list of `AgentConfiguration`
  objects; `get`, `size`, `remove`, `apply`; `exists`, `forall` (with optional
  index exclusion); `mean`, `min`, `max` (with optional predicate and default)
- `YodaModel` — `BaseModel` that simulates a YODA specification by
  interleaving two independently-clocked schedules:
  - **Environment step** (`environment_simulation_step` interval) — dynamics
    then observations
  - **Agent-action step** (`agents_simulation_step` interval) — each agent
    independently samples and fires one action
  - When both clocks coincide, `agents_step` (environment + actions) is used
  - State keys: `"yoda.configuration"`, `"yoda.time_of_next_environment_step"`,
    `"yoda.time_of_next_agents_step"`
- Full MkDocs documentation for the `yoda` module (overview, quick start,
  class map, and individual API-reference pages for all nine classes)

### Changed

- `NBAConfiguration.broadcast_receive` in `AgentDefinitions`: returns the same
  agent unchanged (instead of raising `ValueError`) when no broadcast-receive
  transitions are defined for the given channel
- `NBAConfiguration.step` return type changed from `NBAConfiguration | None`
  to `tuple[float, NBAConfiguration] | None` — the float is the
  exponentially-distributed sojourn time
- `AgentDefinitions` (NBA) `add_broadcast_receive_transition` — weight of zero
  is now accepted; transition is skipped (identity) rather than raising

### Removed

- `pythia.models.pm.ExponentialGrowthModel` — deterministic ODE model removed;
  use `PopulationModel` with appropriate rules for continuous-time simulation
- `pythia.models.pm.LogisticGrowthModel` — same
- `pythia.models.pm.LotkaVolterraModel` — same
- Associated tests (`test_exponential.py`, `test_logistic.py`,
  `test_lotka_volterra.py`) and documentation pages removed
- `pythia.models.pm` module description updated to reflect that only the
  Gillespie-style `PopulationModel` remains

### Documentation

- `README.md` — updated project structure, available models table, and quick-
  start example to reflect the current codebase
- `docs/models/pm/index.md` — rewritten around `PopulationModel` and its
  supporting value objects; removed references to removed ODE models
- `docs/models/nba/index.md` — extended with Environment, NBAConfiguration,
  NBAModel concepts; quick-start updated to a complete simulation example
- `docs/models/yoda/` — new section with overview and nine API-reference pages
- `mkdocs.yml` — nav updated for NBA additions and new YODA section

---

## [0.2.0] — 2026-04-11

### Added

- `Measure` — frozen dataclass binding a name, a `SimulationState → float` function, and a
  sampling frequency in simulation-time units (`pythia.api.measure`)
- `DataCollector` — collects scalar measurements at uniform simulation-time intervals using a
  **zero-order hold** model (`pythia.api.collector`):
  - Timestamps are exact multiples of `frequency` (`k * frequency`), avoiding floating-point drift
  - `collect(state)` fills all scheduled points up to the current step time; retroactive points
    (between the previous and current step) reuse the cached value from the prior call
  - `finalize(state)` fills any remaining scheduled points up to `max_time` after the loop ends
  - `records` property returns an immutable `MappingProxyType` view keyed by measure name
- `SimulationRunner` accepts an optional `data_collector` keyword argument; when provided the
  runner calls `reset`, `collect` (at t = 0 and after every step), and `finalize` automatically
- Gillespie stochastic waiting time in `PopulationModel`: `dt` is now drawn from an exponential
  distribution with rate equal to the total enabled-rule rate (`rng.expovariate(total_rate)`)

### Changed

- **Breaking** — `SimulationConfig` no longer has `max_steps: int` or `time_step: float`;
  replaced by `max_time: float` (strictly positive simulation time horizon)
- **Breaking** — `BaseModel.step()` now returns `tuple[float, SimulationState]` instead of
  `SimulationState`; the first element is the time advance `dt` (returning `dt = 0.0` signals
  termination)
- `SimulationRunner.run()` loop condition changed from step-count to elapsed-time
  (`while elapsed_time < max_time`); breaks immediately when the model returns `dt ≤ 0`
- Deterministic population models (`ExponentialModel`, `LogisticModel`, `LotkaVolterraModel`)
  now read the fixed time step from `config.metadata["pm.time_step"]` instead of
  `config.time_step`

---

## [0.1.0] — 2026-04-09

### Added

- `pythia.api` module — stable public simulation interface:
  - `SimulationConfig` — immutable, frozen dataclass for run settings
  - `SimulationState` — mutable per-step state carrier with `advance()` and `snapshot()`
  - `BaseModel` — abstract base class defining the `initialize` / `step` / `finalize` lifecycle
  - `SimulationRunner` — run-loop orchestration with optional history recording
  - `EventBus` — lightweight synchronous publish-subscribe bus for lifecycle events
  - `EventKind` — enum of `RUN_STARTED`, `STEP_COMPLETED`, `RUN_FINISHED`, `RUN_FAILED`
  - `SimulationEvent` — immutable event payload
  - Exception hierarchy: `PythiaError`, `ConfigurationError`, `ModelError`, `RunnerError`
- Full test suite (54 tests, 100% pass rate)
- MkDocs-based documentation with Material theme and auto-generated API reference
- `py.typed` marker (PEP 561) for downstream type-checker support
