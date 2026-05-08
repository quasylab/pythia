# Architecture

## Module map

```
pythia/
├── api/                  ← stable public interface
│   ├── config.py         ← SimulationConfig  (what to run)
│   ├── state.py          ← SimulationState   (current run state)
│   ├── model.py          ← BaseModel         (how to step)
│   ├── runner.py         ← SimulationRunner  (orchestration)
│   ├── events.py         ← EventBus          (extension seam)
│   ├── collector.py      ← DataCollector     (time-series sampling)
│   ├── measure.py        ← Measure           (sampling descriptor)
│   ├── simulator.py      ← Simulator         (batch runs)
│   └── exceptions.py     ← PythiaError hierarchy
└── models/               ← concrete simulation models
    ├── pm/               ← Population Models (ODE + stochastic)
    ├── lio/              ← Language of Interacting Objects
    └── nba/              ← Network of Broadcasting Agents
```

The `api` package is the stable public interface.  All model subpackages
import from `pythia.api`; `pythia.api` never imports from them, keeping the
dependency graph acyclic.

---

## Data flow

```
SimulationConfig (frozen)
        │
        ▼
SimulationRunner.run()
        │
        ├── model.initialize(config)
        │
        │   ┌──────────────────────────────────────────────────────────┐
        │   │  loop: elapsed_time < max_time and not state.is_done     │
        │   │                                                          │
        │   │   dt, state = model.step(state)                          │
        │   │   state.advance(dt)          # step++ and time += dt     │
        │   │   collector.collect(state)   # optional measurements     │
        │   │   bus.emit(STEP_COMPLETED)                               │
        │   └──────────────────────────────────────────────────────────┘
        │
        ├── model.finalize(state)
        └── return SimulationState
```

---

## Key design decisions

### SimulationConfig is frozen

`SimulationConfig` is a `@dataclass(frozen=True)`.  This means:

- The same config object can be passed into multiple runners concurrently
  without synchronisation.
- Models cannot accidentally mutate global settings during `step`.
- Hashing and equality work out of the box, enabling config-keyed caches.

### SimulationState is mutable

`SimulationState` is a plain mutable dataclass.  Models may mutate it
in place inside `step` — no allocation cost per iteration.  When you need an
immutable copy (e.g. for history recording), call `state.snapshot()`.

### step returns (dt, state)

The `step` method returns a `(float, SimulationState)` pair.  The float is
the simulated time elapsed during that tick and is accumulated by the runner
into `state.elapsed_time`.  Returning a variable `dt` allows models to use
event-driven time (e.g. Gillespie's next-reaction method) alongside
fixed-step Euler integration, in the same framework.

### data dict as an escape hatch

Both `SimulationConfig.metadata` and `SimulationState.data` are open-ended
`dict[str, Any]` mappings.  This allows domain modules to store their keys
without forking the core dataclasses.

Use namespaced string keys to avoid collisions:

```python
state.data["lio.configuration"] = my_state
state.data["pm.counting_state"] = counts
```

### EventBus as the extension seam

The runner emits four events at well-defined lifecycle points:

| Event | When |
|---|---|
| `RUN_STARTED` | Before `model.initialize` |
| `STEP_COMPLETED` | After each successful `model.step` |
| `RUN_FINISHED` | After `model.finalize` returns normally |
| `RUN_FAILED` | When any exception aborts the run |

External components (loggers, visualisers, checkpointers) subscribe to these
events without modifying or importing from the runner.  Handler exceptions are
caught and logged — they never abort the simulation.

### Dependency inversion

```
pythia.api.runner  ──depends on──▶  pythia.api.model  (abstract)
                                           ▲
                                           │ implements
                         ┌─────────────────┼──────────────────────┐
                   pythia.models.pm   pythia.models.lio   pythia.models.nba
```

The runner never imports a concrete model.  Model libraries import from
`pythia.api`; `pythia.api` never imports from them.

---

## Exception hierarchy

```
PythiaError
├── ConfigurationError   — bad or missing config fields
├── ModelError           — model violated its contract at runtime
└── RunnerError          — orchestration-level failure
```

Catch `PythiaError` for a broad net or a specific subclass for targeted
handling.
