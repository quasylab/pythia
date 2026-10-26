# pythia

A modular simulation framework in Python.

## Overview

Pythia provides a structured environment for building and running discrete-event
and continuous-time simulations.  The framework is built around a clean,
extensible API that separates configuration, state, model logic, and
orchestration.  Ready-made model modules cover population dynamics, agent
interaction, and broadcast communication, while the utility layer adds
data collection, visualisation, and statistical analysis tools.

## Project structure

```
pythia/
├── pythia/
│   ├── api/                    # Stable public interface
│   │   ├── config.py           # SimulationConfig — immutable run settings
│   │   ├── state.py            # SimulationState — mutable per-step carrier
│   │   ├── model.py            # BaseModel — abstract contract for all models
│   │   ├── runner.py           # SimulationRunner — run-loop orchestration
│   │   ├── simulator.py        # Simulator — multi-run batch execution
│   │   ├── collector.py        # DataCollector — uniform-interval measurement
│   │   ├── data_table.py       # DataTable — time-indexed float table
│   │   ├── data_tracer.py      # DataTracer — per-object trace collection
│   │   ├── tracer.py           # Tracer — single-run trace driver
│   │   ├── reachability.py     # ReachabilityEstimator — Monte Carlo analysis
│   │   ├── events.py           # EventBus — lifecycle hooks for observers
│   │   └── exceptions.py       # Domain-specific exceptions
│   ├── models/
│   │   ├── pm/                 # Population Models (Gillespie stochastic)
│   │   ├── lio/                # Language of Interacting Objects
│   │   ├── nba/                # Network of Broadcasting Agents
│   │   └── yoda/               # Yet anOther agent Description lAnguage
│   └── util/
│       └── result_plotter.py   # ResultPlotter — Matplotlib visualisation
└── tests/                      # Test suite mirroring the package structure
```

## Available modules

### Model modules

| Module | Description |
|---|---|
| `pythia.models.pm` | Continuous-time stochastic population simulator (`PopulationModel`) driven by Gillespie-style reaction rules |
| `pythia.models.lio` | Agent-based models using the Language of Interacting Objects calculus |
| `pythia.models.nba` | Continuous-time agent-based models using the Network of Broadcasting Agents calculus |
| `pythia.models.yoda` | Multi-agent simulation with layered data stores (`Store`, `AgentStore`), typed actions, dynamics, observations, weighted transitions (default weight 1.0), and a dual-clock scheduler (`YodaModel`) |

### API utilities

| Class | Description |
|---|---|
| `SimulationRunner` | Single-run lifecycle orchestration |
| `Simulator` | Batch execution over *N* independent runs; results stored as `DataTable` per run |
| `DataCollector` | Records measures at a shared mean frequency into a single `DataTable` |
| `DataTable` | Immutable time-indexed float table with fixed column labels |
| `DataTracer` | Collects per-object `DataTable` observations via a user-supplied trace function |
| `Tracer` | Drives a single model run, feeding a `DataTracer` at every step |
| `ReachabilityEstimator` | Monte Carlo estimator: probability and standard deviation of reaching a target state within a deadline |

### Utility modules

| Class | Description |
|---|---|
| `DataStatistics` | Computes mean, std, Q1/Q2/Q3, min and max across a `list[DataTable]` using NumPy (requires `pip install 'pythia[visualization]'`) |
| `ResultPlotter` | Generates raw-run and statistical-summary Matplotlib figures from a `list[DataTable]` (requires `pip install 'pythia[visualization]'`) |
| `Trajectory2D` | Immutable time-ordered 2-D trajectory from a `DataTable` and a trace function; supports linear-time normalisation onto a uniform grid |
| `TraceViewer` | Builds and returns (without showing) a `FuncAnimation` from a `DataTracer`, animating every named object through 2-D space; call `plt.show()` or `anim.save(...)` on the result (requires `pip install 'pythia[visualization]'`) |

## Quick start

### Running a simulation

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.nba import AgentDefinitions, Channel, Environment, NBAModel

c = Channel("sync")
defs = AgentDefinitions()
active  = defs.get("active")
standby = defs.get("standby")

defs.add_send_transition(active, c, standby, 1.0)
defs.add_receive_transition(standby, c, active, 1.0)

env = Environment()
env.set_send_rate(active, c, 1.0)
env.set_receive_weight(standby, c, 1.0)

model = NBAModel({active: 5, standby: 5}, defs, env)
state = SimulationRunner(model, SimulationConfig(max_time=20.0, random_seed=42)).run()

cfg = state.data["nba.configuration"]
print(f"active:  {cfg.count(active)}")
print(f"standby: {cfg.count(standby)}")
```

### Collecting measurements

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.api.collector import DataCollector
from pythia.api.measure import Measure

collector = DataCollector([
    Measure("active",  lambda s: float(s.data["nba.configuration"].count(active)),  frequency=1.0),
    Measure("standby", lambda s: float(s.data["nba.configuration"].count(standby)), frequency=1.0),
])

runner = SimulationRunner(model, SimulationConfig(max_time=20.0, random_seed=42),
                          data_collector=collector)
runner.run()

table = collector.records          # single DataTable with columns "active" and "standby"
for t in table.times:
    row = table.get(t)
    print(f"t={t:.1f}  active={row['active']:.0f}  standby={row['standby']:.0f}")
```

### Estimating reachability

```python
from pythia.api import SimulationConfig
from pythia.api.reachability import ReachabilityEstimator

estimator = ReachabilityEstimator(model, SimulationConfig(max_time=20.0, random_seed=0))
prob, std = estimator.estimate(
    n_runs=500,
    deadline=20.0,
    final_predicate=lambda s: s.data["nba.configuration"].count(active) == 10,
)
print(f"P(all active within t=20) = {prob:.3f} ± {std:.3f}")
```

### Custom models

Any model that subclasses `BaseModel` plugs straight into the runner:

```python
import random
from pythia.api import BaseModel, SimulationConfig, SimulationRunner, SimulationState

class RandomWalkModel(BaseModel):
    @property
    def name(self) -> str:
        return "random_walk"

    def initialize(self, config: SimulationConfig) -> None:
        self.validate_config(config)
        self._rng = random.Random(config.random_seed)

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        state.data["position"] = state.data.get("position", 0.0) + self._rng.choice([-1.0, 1.0])
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass

state = SimulationRunner(
    RandomWalkModel(),
    SimulationConfig(max_time=100.0, random_seed=0),
).run()
print(f"Final position: {state.data['position']}")
```

## Installation

```bash
git clone https://github.com/quasylab/pythia.git
cd pythia
pip install -e ".[dev]"
```

To also install documentation and visualisation dependencies:

```bash
pip install -e ".[dev,docs,visualization]"
```

## Running tests

```bash
pytest
```

## Building the documentation

```bash
mkdocs serve       # live-reload preview at http://localhost:8000
mkdocs build       # static site written to site/
```

## Development

This project uses:

- **pytest** for testing
- **mypy** for static type checking
- **ruff** for linting and formatting

```bash
mypy pythia/
ruff check pythia/ tests/
```

## Disclaimer on AI-Assisted Development

This project was developed with partial assistance from [Claude](https://www.anthropic.com/claude), an AI language model by Anthropic. The use of this tool was limited to supporting and accelerating certain aspects of the coding process.

All design decisions, software architecture, structural choices, and technical direction are the sole responsibility of the authors and are grounded in their own expertise and professional judgment. The authors retain full intellectual and creative ownership of this work.