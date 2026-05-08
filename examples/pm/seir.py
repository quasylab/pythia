"""SEIR compartmental model — pythia.models.pm example.

The SEIR model extends the classic SIR epidemic model with an *Exposed* (E)
compartment for individuals who have been infected but are not yet infectious.

Compartments
------------
S   Susceptible  — healthy individuals who can be infected
E   Exposed      — infected but in the incubation period (not yet infectious)
I   Infectious   — actively spreading the disease
R   Recovered    — no longer infectious (assumed immune)

Reactions
---------
S + I → E + I   infection:   a susceptible individual is exposed by contact
                              with an infectious one
                              rate = β · S · I / N

E     → I       progression: an exposed individual becomes infectious after
                              the incubation period
                              rate = σ · E

I     → R       recovery:    an infectious individual recovers
                              rate = γ · I

Parameters (defaults)
---------------------
β (beta)  = 0.4    transmission rate
σ (sigma) = 0.2    progression rate  (mean incubation ≈ 5 time units)
γ (gamma) = 0.1    recovery rate     (mean infectious period ≈ 10 time units)
N         = 10 000 total (fixed) population
S₀        = 9 990  initial susceptible
E₀        = 5      initial exposed
I₀        = 5      initial infectious
R₀        = 0      initial recovered

Usage
-----
    python examples/pm/seir.py

Produces a printed summary and, if matplotlib is available, a time-series
plot saved to ``examples/pm/seir.png``.
"""

from __future__ import annotations

from pythia.api import SimulationConfig, SimulationRunner
from pythia.api.collector import DataCollector
from pythia.api.measure import Measure
from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.population_model import PopulationModel
from pythia.models.pm.rule import Rule

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BETA  = 0.4    # transmission rate
SIGMA = 0.2    # E → I progression rate
GAMMA = 0.1    # I → R recovery rate
N     = 10_000 # total population (conserved)

S0, E0, I0, R0 = 9_990, 5, 5, 0
MAX_TIME = 200.0
SEED     = 42

# ---------------------------------------------------------------------------
# Population and rules
# ---------------------------------------------------------------------------

pop = Population()
S = pop.register("S")
E = pop.register("E")
I = pop.register("I")
R = pop.register("R")

rules = [
    # S + I → E + I  (infection via mass-action)
    Rule(
        pre=CountingState({S: 1, I: 1}),
        post=CountingState({E: 1, I: 1}),
        rate_function=lambda s: (
            BETA
            * s.counting_vector.get(S.index, 0)
            * s.counting_vector.get(I.index, 0)
            / N
        ),
    ),
    # E → I  (incubation completes)
    Rule(
        pre=CountingState({E: 1}),
        post=CountingState({I: 1}),
        rate_function=lambda s: SIGMA * s.counting_vector.get(E.index, 0),
    ),
    # I → R  (recovery)
    Rule(
        pre=CountingState({I: 1}),
        post=CountingState({R: 1}),
        rate_function=lambda s: GAMMA * s.counting_vector.get(I.index, 0),
    ),
]

# ---------------------------------------------------------------------------
# Helpers to read compartment counts from the simulation state
# ---------------------------------------------------------------------------

def _count(state, species) -> float:
    cs = state.data.get("pm.counting_state")
    return float(cs.counting_vector.get(species.index, 0)) if cs else 0.0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

model = PopulationModel(
    population=pop,
    initial_state={S: S0, E: E0, I: I0, R: R0},
    rules=rules,
)

collector = DataCollector([
    Measure("S", lambda s: _count(s, S), frequency=1.0),
    Measure("E", lambda s: _count(s, E), frequency=1.0),
    Measure("I", lambda s: _count(s, I), frequency=1.0),
    Measure("R", lambda s: _count(s, R), frequency=1.0),
])

config = SimulationConfig(max_time=MAX_TIME, random_seed=SEED)
runner = SimulationRunner(model, config, data_collector=collector)
runner.run()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

table = collector.records
final_t = table.times[-1]
final   = table.get(final_t)

print("SEIR model summary")
print(f"  β={BETA}  σ={SIGMA}  γ={GAMMA}  N={N}")
print(f"  Initial : S={S0}  E={E0}  I={I0}  R={R0}")
print(f"  Final   : S={final['S']:.0f}  E={final['E']:.0f}"
      f"  I={final['I']:.0f}  R={final['R']:.0f}"
      f"  (t={final_t:.1f})")
print(f"  Peak I  : {max(table.get(t)['I'] for t in table.times):.0f}")

# ---------------------------------------------------------------------------
# Plot (optional — requires matplotlib)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    times = list(table.times)
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, colour in (("S", "steelblue"), ("E", "orange"),
                           ("I", "crimson"),  ("R", "seagreen")):
        values = [table.get(t)[label] for t in times]
        ax.plot(times, values, label=label, color=colour)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population")
    ax.set_title(f"SEIR model  (β={BETA}, σ={SIGMA}, γ={GAMMA}, N={N:,})")
    ax.legend()
    ax.set_xlim(0, MAX_TIME)
    ax.set_ylim(0, N)

    out = "examples/pm/seir.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.close(fig)

except ImportError:
    print("  (matplotlib not installed — skipping plot)")
