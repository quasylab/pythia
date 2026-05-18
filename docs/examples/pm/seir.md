# SEIR Epidemic Model (`models.pm`)

**File:** `examples/pm/seir.py`

The SEIR model is a classical compartmental epidemic model that extends SIR
with an *Exposed* compartment for individuals who are infected but not yet
infectious.

## Compartments

| State | Meaning |
|---|---|
| S | Susceptible — healthy individuals who can be infected |
| E | Exposed — infected but in the incubation period |
| I | Infectious — actively spreading the disease |
| R | Recovered — immune, no longer infectious |

## Reactions

| Rule | Rate |
|---|---|
| S + I → E + I | β · S · I / N |
| E → I | σ · E |
| I → R | γ · I |

## Running

```bash
python examples/pm/seir.py
```

Produces a printed summary and (when `matplotlib` is available) a time-series
plot saved to `examples/pm/seir.png`.
