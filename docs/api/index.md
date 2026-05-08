# API Reference

All public symbols are available from the top-level `pythia` package and from
`pythia.api`:

```python
from pythia import SimulationConfig, SimulationRunner, BaseModel
# or equivalently:
from pythia.api import SimulationConfig, SimulationRunner, BaseModel
```

---

## Quick reference

| Class / Type | Module | Purpose |
|---|---|---|
| [`SimulationConfig`](config.md) | `pythia.api.config` | Immutable run settings |
| [`SimulationState`](state.md) | `pythia.api.state` | Mutable per-step carrier |
| [`BaseModel`](model.md) | `pythia.api.model` | Abstract model contract |
| [`SimulationRunner`](runner.md) | `pythia.api.runner` | Run-loop orchestration |
| [`EventBus`](events.md) | `pythia.api.events` | Publish-subscribe bus |
| [`EventKind`](events.md) | `pythia.api.events` | Lifecycle event enum |
| [`SimulationEvent`](events.md) | `pythia.api.events` | Immutable event payload |
| [`PythiaError`](exceptions.md) | `pythia.api.exceptions` | Root exception |
| [`ConfigurationError`](exceptions.md) | `pythia.api.exceptions` | Bad configuration |
| [`ModelError`](exceptions.md) | `pythia.api.exceptions` | Model contract violation |
| [`RunnerError`](exceptions.md) | `pythia.api.exceptions` | Orchestration failure |
