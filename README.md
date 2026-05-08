# pythia

A modular simulation framework in Python.

## Overview

Pythia provides a structured environment for building and running discrete-event
and continuous-time simulations.  The framework is built around a clean,
extensible API that separates configuration, state, model logic, and
orchestration.  Ready-made model modules cover population dynamics, agent
interaction, and broadcast communication.

## Project structure

```
pythia/
├── pythia/
│   ├── api/                    # Stable public interface
│   │   ├── config.py           # SimulationConfig — immutable run settings
│   │   ├── state.py            # SimulationState — mutable per-step carrier
│   │   ├── model.py            # BaseModel — abstract contract for all models
│   │   ├── runner.py           # SimulationRunner — run-loop orchestration
│   │   ├── events.py           # EventBus — lifecycle hooks for observers
│   │   └── exceptions.py       # Domain-specific exceptions
│   └── models/
│       ├── pm/                 # Population Models (discrete-time ODEs)
│       ├── lio/                # Language of Interacting Objects
│       └── nba/                # Network of Broadcasting Agents
└── tests/                      # Test suite mirroring the package structure
```

## Available model modules

| Module | Description |
|---|---|
| `pythia.models.pm` | Continuous-time stochastic population simulator (`PopulationModel`) driven by Gillespie-style reaction rules |
| `pythia.models.lio` | Agent-based models using the Language of Interacting Objects calculus |
| `pythia.models.nba` | Continuous-time agent-based models using the Network of Broadcasting Agents calculus |

## Quick start

The example below runs a point-to-point communication system with the NBA
module: ten *active* agents each send on channel `c` at rate 1, causing one
*standby* agent to become active in return.

```python
from pythia.api import SimulationConfig, SimulationRunner
from pythia.models.nba import AgentDefinitions, Channel, Environment, NBAModel

# Define channel and agent types
c = Channel("sync")
defs = AgentDefinitions()
active  = defs.get("active")
standby = defs.get("standby")

# Transitions: active --c!--> standby, standby --c?--> active
defs.add_send_transition(active, c, standby, 1.0)
defs.add_receive_transition(standby, c, active, 1.0)

# Environment: send rate and receiver weight
env = Environment()
env.set_send_rate(active, c, 1.0)
env.set_receive_weight(standby, c, 1.0)

# Run
model = NBAModel({active: 5, standby: 5}, defs, env)
state = SimulationRunner(model, SimulationConfig(max_time=20.0, random_seed=42)).run()

cfg = state.data["nba.configuration"]
print(f"active:  {cfg.count(active)}")
print(f"standby: {cfg.count(standby)}")
print(f"elapsed: {state.elapsed_time:.3f}")
```

## Custom models

Any model that subclasses `BaseModel` and implements the three lifecycle
methods plugs straight into the runner:

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
        position = state.data.get("position", 0.0)
        state.data["position"] = position + self._rng.choice([-1.0, 1.0])
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
git clone https://github.com/loreti/pythia.git
cd pythia
pip install -e ".[dev]"
```

To also install documentation dependencies:

```bash
pip install -e ".[dev,docs]"
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
