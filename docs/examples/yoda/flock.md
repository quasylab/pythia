# Boids Flocking (`models.yoda`)

**File:** `examples/yoda/flock.py`

A YODA implementation of Reynolds' Boids model in which agents (boids) move
on a 2-D plane and align their velocity with nearby neighbours, producing
emergent flocking behaviour.

## State

Each agent tracks its current heading and position in its `AgentStore`.

## Running

```bash
python examples/yoda/flock.py
```
