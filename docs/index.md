# Pythia

**A modular simulation framework in Python.**

Pythia provides a clean, extensible environment for implementing and running
simulation models. The framework enforces a strict separation between
configuration, state, model logic, and run orchestration — making models easy
to test in isolation, swap between experiments, and extend without modifying
the core.

---

## Highlights

- **Protocol-first design** — every model implements a simple three-method
  lifecycle (`initialize` / `step` / `finalize`), nothing more.
- **Immutable configuration, mutable state** — `SimulationConfig` is frozen
  so it is safe to share across runs; `SimulationState` is mutable so models
  pay no allocation cost per step.
- **Event-driven extensibility** — an `EventBus` lets loggers, visualisers,
  and checkpointers attach to any run without touching the runner.
- **Zero runtime dependencies** — the `api` layer is pure Python stdlib.
- **Fully typed** — ships with `py.typed`, verified with mypy in strict mode.

---

## Quick example

```python
from pythia.api import BaseModel, SimulationConfig, SimulationRunner, SimulationState

class CounterModel(BaseModel):
    @property
    def name(self) -> str:
        return "counter"

    def initialize(self, config: SimulationConfig) -> None:
        self.validate_config(config)

    def step(self, state: SimulationState) -> SimulationState:
        state.data["count"] = state.data.get("count", 0) + 1
        return state

    def finalize(self, state: SimulationState) -> None:
        print(f"Final count: {state.data['count']}")

config = SimulationConfig(max_steps=100, time_step=0.1, random_seed=42)
runner = SimulationRunner(CounterModel(), config)
final_state = runner.run()
```

---

## Navigation

| Section | Contents |
|---|---|
| [Getting Started](getting-started.md) | Installation, first run |
| [Architecture](guide/architecture.md) | Design overview and module map |
| [Writing a Model](guide/writing-a-model.md) | Step-by-step guide |
| [Observing Runs](guide/observing-runs.md) | Logging, events, history recording |
| [API Reference](api/index.md) | Full class and method documentation |
| [Population Models](models/pm/index.md) | ODE and stochastic population dynamics (`pm`) |
| [LIO Models](models/lio/index.md) | Language of Interacting Objects (`lio`) |
| [NBA Models](models/nba/index.md) | Network of Broadcasting Agents (`nba`) |
