# Observing Runs

Pythia provides two complementary mechanisms for observing a simulation
without modifying the runner or the model: **history recording** and the
**event bus**.

---

## History recording

Pass `record_history=True` to the runner to keep a
[`SimulationState.snapshot()`](../api/state.md) after every step:

```python
from pythia.api import SimulationRunner, SimulationConfig

runner = SimulationRunner(my_model, config, record_history=True)
final = runner.run()

# runner.history is a list[SimulationState], one entry per step
for state in runner.history:
    print(state.step, state.data)
```

!!! note "Memory cost"
    Each snapshot is a deep copy of the state.  For large `state.data`
    payloads or very long runs, prefer the event bus approach instead.

### Extracting a time series

```python
steps     = [s.step for s in runner.history]
positions = [s.data["position"] for s in runner.history]
```

---

## The event bus

The [`EventBus`](../api/events.md) lets you attach observers that react to
lifecycle events in real time, without storing the full history.

### Available events

| `EventKind` | When it fires |
|---|---|
| `RUN_STARTED` | Before `model.initialize` |
| `STEP_COMPLETED` | After each successful `model.step` |
| `RUN_FINISHED` | After `model.finalize` returns normally |
| `RUN_FAILED` | When an exception aborts the run |

### Basic usage

```python
from pythia.api import EventBus, EventKind, SimulationRunner

bus = EventBus()

def log_progress(event):
    if event.step % 50 == 0:
        print(f"step {event.step}")

bus.subscribe(EventKind.STEP_COMPLETED, log_progress)

runner = SimulationRunner(my_model, config, event_bus=bus)
runner.run()
```

### Collecting metrics without full history

```python
metrics = []

def collect(event):
    metrics.append({
        "step": event.step,
        "value": event.state.data.get("value"),
    })

bus = EventBus()
bus.subscribe(EventKind.STEP_COMPLETED, collect)
SimulationRunner(my_model, config, event_bus=bus).run()
```

### Handling run failures

```python
def on_failure(event):
    print(f"Run failed at step {event.step}!")

bus.subscribe(EventKind.RUN_FAILED, on_failure)
```

### Unsubscribing a handler

```python
bus.unsubscribe(EventKind.STEP_COMPLETED, log_progress)
```

---

## Standard library logging

The runner emits structured log records via Python's `logging` module under
the `pythia.api.runner` logger.  Wire it up with your preferred handler:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

Log levels used:

| Level | Messages |
|---|---|
| `INFO` | Run start / finish with model name and step count |
| `DEBUG` | Per-step completion with elapsed time |
| `WARNING` | Event handler exceptions (never propagated) |

---

## Combining history and events

You can use both at the same time.  History is useful for post-run analysis;
events are useful for real-time feedback:

```python
bus = EventBus()
bus.subscribe(EventKind.STEP_COMPLETED, lambda e: print(".", end="", flush=True))
bus.subscribe(EventKind.RUN_FINISHED,   lambda e: print(" done."))

runner = SimulationRunner(my_model, config, event_bus=bus, record_history=True)
final = runner.run()

# post-run: inspect full history
counts = [s.data["count"] for s in runner.history]
```
