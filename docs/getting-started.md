# Getting Started

## Requirements

- Python **3.11** or later

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/quasylab/pythia.git
cd pythia
pip install -e ".[dev]"
```

To also install documentation dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Verify the installation

```bash
python -c "import pythia; print(pythia.__version__)"
# 0.1.0
```

Run the test suite to confirm everything works:

```bash
pytest
```

---

## Your first simulation

### 1. Define a model

Every model subclasses [`BaseModel`](api/model.md) and implements three methods.

```python
from pythia.api import BaseModel, SimulationConfig, SimulationState

class RandomWalkModel(BaseModel):
    """A 1-D random walk simulation."""

    def __init__(self) -> None:
        import random
        self._rng = random.Random()

    @property
    def name(self) -> str:
        return "random_walk"

    def initialize(self, config: SimulationConfig) -> None:
        self.validate_config(config)
        seed = config.random_seed
        self._rng.seed(seed)

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        position = state.data.get("position", 0.0)
        state.data["position"] = position + self._rng.choice([-1.0, 1.0])
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass
```

### 2. Configure the run

[`SimulationConfig`](api/config.md) is immutable and must be created before the runner:

```python
from pythia.api import SimulationConfig

config = SimulationConfig(
    max_time=1000.0,
    random_seed=42,
)
```

### 3. Run the simulation

```python
from pythia.api import SimulationRunner

runner = SimulationRunner(RandomWalkModel(), config)
final_state = runner.run()

print(f"Final position: {final_state.data['position']}")
print(f"Steps taken:    {final_state.step}")
```

### 4. Record history (optional)

Pass `record_history=True` to keep a snapshot of the state after every step:

```python
runner = SimulationRunner(RandomWalkModel(), config, record_history=True)
final_state = runner.run()

positions = [s.data["position"] for s in runner.history]
```

### 5. Attach an observer (optional)

Use the [`EventBus`](api/events.md) to react to run events without modifying the runner:

```python
from pythia.api import EventBus, EventKind, SimulationRunner

bus = EventBus()

def on_step(event):
    if event.step % 100 == 0:
        pos = event.state.data.get("position", 0)
        print(f"Step {event.step}: position={pos:.1f}")

bus.subscribe(EventKind.STEP_COMPLETED, on_step)

runner = SimulationRunner(RandomWalkModel(), config, event_bus=bus)
runner.run()
```

---

## Next steps

- [Architecture](guide/architecture.md) — understand how the pieces fit together.
- [Writing a Model](guide/writing-a-model.md) — detailed guide with best practices.
- [Observing Runs](guide/observing-runs.md) — logging, events, and history.
