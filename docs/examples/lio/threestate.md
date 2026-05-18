# Three State Consensus Protocol (`models.lio`)

**File:** `examples/lio/threestate.py`

The Three State protocol is a well-known distributed probabilistic algorithm
by which a population of agents reaches consensus on a binary decision without
global synchronisation.

## States

| State | Meaning |
|---|---|
| B0 | Agent believes the decision is 0 |
| B1 | Agent believes the decision is 1 |
| DUB | Agent is doubtful about the decision |

## Transitions

| Interaction | Effect |
|---|---|
| B0 meets B1 | B0 → DUB |
| B1 meets B0 | B1 → DUB |
| DUB meets B1 | DUB → B1 |
| DUB meets B0 | DUB → B0 |

## Running

```bash
python examples/lio/threestate.py
```
