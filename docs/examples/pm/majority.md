# Majority Population Protocol (`models.pm`)

**File:** `examples/pm/majority.py`

The Majority model is a classical population protocol in which agents must
agree on a common colour (Blue or Red) through pairwise interactions.

## States

| State | Meaning |
|---|---|
| B | Agent currently supporting Blue |
| R | Agent currently supporting Red |

## Reactions

Agents update their colour when two agents of opposing colours meet, following
majority-rule dynamics.

## Running

```bash
python examples/pm/majority.py
```
