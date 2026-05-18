# Random Walk (`models.yoda`)

**File:** `examples/yoda/randomwalk.py`

A group of robots move on a bounded 2-D arena.  Each robot randomly selects
a direction and walks until it hits a boundary, at which point it stops and
waits before choosing a new direction.

## States

| State | Meaning |
|---|---|
| walk | Robot is moving in the current direction |
| stop | Robot is waiting before selecting a new direction |

## Running

```bash
python examples/yoda/randomwalk.py
```
