# Writing a Model

Every simulation model is a subclass of [`BaseModel`](../api/model.md).  This
guide walks through the full lifecycle from skeleton to a tested, production-
ready implementation.

---

## The lifecycle methods

The runner calls lifecycle methods in the following order:

1. `initialize(config)` — allocate resources, seed RNGs.
2. `setup(state)` *(optional override)* — populate `state.data` with the true
   initial state so that measures collected at `t = 0` are correct.
3. Step loop: `step(state)` repeatedly until `dt = 0` or `max_time` reached.
4. `finalize(state)` — flush buffers, compute summaries.

```python
from pythia.api import BaseModel, SimulationConfig, SimulationState

class MyModel(BaseModel):

    @property
    def name(self) -> str:
        """Short identifier used in logs and events."""
        return "my_model"

    def initialize(self, config: SimulationConfig) -> None:
        """Called once before the run loop. Allocate resources here."""
        ...

    def setup(self, state: SimulationState) -> None:
        """Optional: pre-populate state.data so t=0 measures are correct."""
        state.data["my_model.value"] = 0.0   # initial value visible at t=0

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the simulation by one tick. Returns (dt, state)."""
        ...
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        """Called once after the run loop. Flush buffers, compute summaries."""
        ...
```

---

## Step 1 — validate configuration

Call `self.validate_config(config)` at the top of `initialize`.  The base
implementation checks that `max_time` is positive; add domain-specific checks
after:

```python
from pythia.api.exceptions import ConfigurationError

def initialize(self, config: SimulationConfig) -> None:
    self.validate_config(config)  # base checks

    if "population" not in config.metadata:
        raise ConfigurationError("'population' is required in config.metadata")
    if config.metadata["population"] <= 0:
        raise ConfigurationError("'population' must be positive")

    self._population = config.metadata["population"]
```

---

## Step 2 — seed the RNG

Always use `config.random_seed` so runs are reproducible:

```python
import random

def initialize(self, config: SimulationConfig) -> None:
    self.validate_config(config)
    self._rng = random.Random(config.random_seed)  # None → non-deterministic
```

---

## Step 3 — implement step

`step` receives the current `SimulationState` and must return a
`(dt, state)` pair where `dt` is the time elapsed during the step.  The
runner accumulates `dt` values into `state.elapsed_time` and stops when it
reaches `config.max_time`.  Mutating `state` in place and returning it is
the standard pattern:

```python
def step(self, state: SimulationState) -> tuple[float, SimulationState]:
    # read
    count = state.data.get("count", 0)

    # compute
    count += self._rng.randint(1, 6)

    # write back
    state.data["count"] = count

    # optionally request early termination
    if count >= 100:
        state.is_done = True

    return 1.0, state
```

!!! warning "Always return `(dt, state)`"
    The runner unpacks the tuple.  Returning only `state` (the old one-value
    signature) raises a `ModelError` at runtime.

---

## Step 4 — finalize

Use `finalize` for teardown that must happen regardless of whether the run
completed normally or stopped early via `state.is_done`:

```python
def finalize(self, state: SimulationState) -> None:
    final_count = state.data.get("count", 0)
    print(f"[{self.name}] finished at step {state.step}, count={final_count}")
```

---

## Full example — SIR epidemiological model

```python
import random
from pythia.api import BaseModel, SimulationConfig, SimulationState
from pythia.api.exceptions import ConfigurationError


class SIRModel(BaseModel):
    """Discrete-time SIR (Susceptible-Infected-Recovered) model.

    Required metadata keys:
        population (int): Total population size.
        beta (float): Transmission rate per contact (0 < beta <= 1).
        gamma (float): Recovery rate per step (0 < gamma <= 1).
        initial_infected (int): Number of infected individuals at t=0.
    """

    @property
    def name(self) -> str:
        return "sir"

    def initialize(self, config: SimulationConfig) -> None:
        self.validate_config(config)

        m = config.metadata
        for key in ("population", "beta", "gamma", "initial_infected"):
            if key not in m:
                raise ConfigurationError(f"'{key}' is required in config.metadata")

        self._rng = random.Random(config.random_seed)
        self._beta = float(m["beta"])
        self._gamma = float(m["gamma"])

        n = int(m["population"])
        i0 = int(m["initial_infected"])
        self._initial = {"S": n - i0, "I": i0, "R": 0}

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        if not state.data:
            state.data.update(self._initial)

        S, I, R = state.data["S"], state.data["I"], state.data["R"]
        N = S + I + R

        new_infected = round(self._beta * S * I / N)
        new_recovered = round(self._gamma * I)

        state.data["S"] = max(0, S - new_infected)
        state.data["I"] = max(0, I + new_infected - new_recovered)
        state.data["R"] = R + new_recovered

        if state.data["I"] == 0:
            state.is_done = True

        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass
```

Run it:

```python
from pythia.api import SimulationConfig, SimulationRunner

config = SimulationConfig(
    max_time=365.0,
    random_seed=0,
    metadata={
        "population": 10_000,
        "beta": 0.3,
        "gamma": 0.05,
        "initial_infected": 10,
    },
)

runner = SimulationRunner(SIRModel(), config, record_history=True)
final = runner.run()
print(f"Epidemic ended at step {final.step}, recovered={final.data['R']}")
```

---

## Testing your model

Test models in isolation by calling their lifecycle methods directly — no
runner required:

```python
from pythia.api import SimulationConfig, SimulationState

def test_sir_step_reduces_susceptible():
    model = SIRModel()
    config = SimulationConfig(
        max_time=10.0,
        metadata={"population": 100, "beta": 0.5, "gamma": 0.1, "initial_infected": 5},
    )
    model.initialize(config)

    state = SimulationState()
    _dt, state = model.step(state)  # first step also seeds state.data

    assert state.data["S"] < 95   # some new infections
    assert state.data["I"] > 0
```
