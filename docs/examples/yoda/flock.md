# Boids Flocking (`models.yoda`)

**File:** `examples/yoda/flock.py`

A YODA implementation of Reynolds' Boids model in which agents (boids) move
on a 2-D plane and steer according to three local rules, producing emergent
flocking behaviour:

| Rule | Effect |
|---|---|
| Separation | Steer away from neighbours that are too close |
| Alignment | Steer towards the average heading of nearby neighbours |
| Cohesion | Steer towards the centre of mass of nearby neighbours |

## Store

Each agent keeps the following entries in its `AgentStore`:

| Section | Key | Meaning |
|---|---|---|
| features | `pos` | Position on the plane |
| state | `id` | Agent identifier (used to name the trace) |
| state | `velocity` | Unit vector giving the current heading |
| observations | `separation` | Direction pointing away from close neighbours |
| observations | `cmas` | Vector from the agent to the neighbours' centre of mass |
| observations | `alignment` | Average velocity of nearby neighbours |

## Behaviour

- **Observations** — at each step every boid computes `separation`, `cmas`
  and `alignment` from the agents within the respective distances.
- **Action** — the single `steer` action adds the three observation vectors,
  each scaled by its strength, to a copy of the current velocity and
  normalises the result.
- **Dynamics** — the position advances by `velocity * SPEED * dt`.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `N` | 100 | Number of boids |
| `cohesion_distance` | 1000 | Radius used for cohesion |
| `alignment_distance` | 10 | Radius used for alignment |
| `separation_distance` | 10 | Radius used for separation |
| `cohesion_strength` | 0.005 | Weight of the cohesion vector |
| `separation_strength` | 0.1 | Weight of the separation vector |
| `alignment_strength` | 0.03 | Weight of the alignment vector |
| `SPEED` | 200 | Speed multiplier applied by the dynamics |

## Running

```bash
python examples/yoda/flock.py
```

The run is traced with a `DataTracer` and animated with `TraceViewer`
over `[0, 100]` with a 0.25 time step.  The animation is saved to
`examples/yoda/flock.gif` (requires `pip install 'pythia[visualization]'`
and Pillow); to view it interactively instead, replace the `ani.save(...)`
call with `plt.show()`.
