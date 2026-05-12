"""Majority population model — pythia.models.pm example.

The Majority model is a classical population protocol that can be used by a
group of agents to agree on the same decision. In this example we assume the
agents must agree on the colour to use: either blue (B) or red (R).

In the protocol each agent can be either active (A) or passive (P). Active
agents are the ones that will trigger the decision, while passive agents just
follow the indications of the active ones.

At the beginning each agent is active and has a colour. When two active agents
with different colours meet, become passive. A passive agent changes its colour
when it meets an active with a different colour.

Parameters (defaults)
---------------------
A         = 100    dimension of the areas where the agents operate
N         = 100    number of agents
CA        = 3      Communication Area 

̄𝜶 (alpha) = 0.4    agents interaction rate rate

β (beta)  = 0.4    fraction of blue agents

Species
------------
A_B  - Active Blue Agent
A_R  - Active Red Agent
P_B  - Passive Blue Agent
P_R  - Passive Red Agent


Reactions
---------
A_B + A_R → P_B + P_R   cancellation:  two active agents with different
                            colours became passive.
                            rate = 2 * 𝜶 * A_B * A_R * CA / A 

A_B + P_R → A_B + P_B   color update B: a passive red agent meets an active
                            blue agent and becomes blue.
                            rate = 𝜶 * A_B * P_R * CA / A 

A_R + P_B → A_R + P_R   color update R: a passive blue agent meets an active
                            red agent and becomes red.
                            rate = 𝜶 * A_R * P_B * CA / A 



Usage
-----
    python examples/pm/majority.py

Produces a printed summary and, if matplotlib is available, a time-series
plot saved to ``examples/pm/majority.png``.
"""

from __future__ import annotations

from pythia.api import SimulationConfig, SimulationRunner, Simulator, DataTable, ReachabilityEstimator
from pythia.api.collector import DataCollector
from pythia.api.measure import Measure
from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.population_model import PopulationModel
from pythia.models.pm.rule import Rule
from pythia.util import ResultPlotter

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALPHA  = 0.4   # communication rate

BETA   = 0.4   # fraction of red agents

A      = 100   # dimension of the areas where the agents operate
CA     = 10    # communication area
N      = 100   # number of agents

INIT_B = int(BETA*N)
INIT_R = N - INIT_B

MAX_TIME = 20.0
SEED     = 42

# ---------------------------------------------------------------------------
# Population and rules
# ---------------------------------------------------------------------------

pop = Population()
A_R = pop.register("A_R")
A_B = pop.register("A_B")
P_R = pop.register("P_R")
P_B = pop.register("P_B")

rules = [
    # A_B + A_R → P_B + P_R   cancellation
    Rule(
        pre=CountingState({A_R: 1, A_B: 1}),
        post=CountingState({P_R: 1, P_B: 1}),
        rate_function=lambda s: (
            ALPHA
            * s.counting_vector.get(A_R.index, 0)
            * s.counting_vector.get(A_B.index, 0)
            * CA / N
        ),
    ),
    # A_B + P_R → A_B + P_B   color update B
    Rule(
        pre=CountingState({A_B: 1, P_R: 1}),
        post=CountingState({A_B: 1, P_B: 1}),
        # 𝜶 * A_B * P_R * CA / A
        rate_function=lambda s: (
                                ALPHA
                                * s.counting_vector.get(A_B.index, 0)
                                * s.counting_vector.get(P_R.index, 0)
                                * CA / N
        ),
    ),
    # I → R  (recovery)
    Rule(
        pre=CountingState({A_R: 1, P_B: 1}),
        post=CountingState({A_R: 1, P_R: 1}),
        rate_function=lambda s: (
                                ALPHA
                                * s.counting_vector.get(A_R.index, 0)
                                * s.counting_vector.get(P_B.index, 0)
                                * CA / N
        ),
    ),
]

model = PopulationModel(
    population=pop,
    initial_state={A_B: INIT_B, A_R: INIT_R},
    rules=rules,
)


# ---------------------------------------------------------------------------
# Helpers to read number of agents from the simulation state
# ---------------------------------------------------------------------------

def _count(state, species) -> float:
    cs = state.data.get("pm.counting_state")
    return float(cs.counting_vector.get(species.index, 0)) if cs else 0.0

# ---------------------------------------------------------------------------
# Single Simulation Run
# ---------------------------------------------------------------------------

measures = [
    Measure("RED", lambda s: _count(s, A_R)+_count(s, P_R), frequency=0.25),
    Measure("BLUE", lambda s: _count(s, A_B)+_count(s, P_B), frequency=0.25),
    Measure("ACTIVE", lambda s: _count(s, A_B)+_count(s, A_R), frequency=0.25),
    Measure("PASSIVE", lambda s: _count(s, P_B)+_count(s, P_R), frequency=0.25),
]
collector = DataCollector(measures)

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
print(f"  A={A}  N={N}  CA={CA}")
print(f"  𝜶={ALPHA}  β={BETA}")
print(f"  Initial : A_B={INIT_B}  A_R={INIT_R}")
print(f"  Final   : RED={final['RED']:.0f}  BLUE={final['BLUE']:.0f}"
      f"  ACTIVE={final['ACTIVE']:.0f}  PASSIVE={final['PASSIVE']:.0f}"
      f"  (t={final_t:.1f})")

# ---------------------------------------------------------------------------
# Plot (optional — requires matplotlib)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    times = list(table.times)
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, colour in (("BLUE", "steelblue"), ("RED", "red"),
                           ("ACTIVE", "crimson"),  ("PASSIVE", "seagreen")):
        values = [table.get(t)[label] for t in times]
        ax.plot(times, values, label=label, color=colour)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population")
    ax.set_title(f"Majority model  (A={A}  N={N}  CA={CA} 𝜶={ALPHA}  β={BETA})")
    ax.legend()
    ax.set_xlim(0, MAX_TIME)
    ax.set_ylim(0, N)

    out = "examples/pm/majority.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.close(fig)

except ImportError:
    print("  (matplotlib not installed — skipping plot)")

# ---------------------------------------------------------------------------
# Multiple Runs
# ---------------------------------------------------------------------------

simulator = Simulator(model, config, measures=measures)
simulator.run(10)

plotter = ResultPlotter(simulator.results)
plotter.plot(output_folder="examples/pm", prefix="majority")
plotter.plot_statistics(output_folder="examples/pm", prefix="majority_statistics")

# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

reachability = ReachabilityEstimator(model, config)

dt = 0.25
probs = reachability.estimate_over_time(1000,20, dt, lambda s: _count(s, A_R)+_count(s, P_R) == N)

try:
    import matplotlib.pyplot as plt

    times = [dt*i for i in range(len(probs))]
    values = [p[0] for p in probs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, values, label="Probability")

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_title(f"Majority model  (A={A}  N={N}  CA={CA} 𝜶={ALPHA}  β={BETA})")
    ax.legend()
    ax.set_xlim(0, MAX_TIME)
    ax.set_ylim(0, 1.1)

    out = "examples/pm/majority_probability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.close(fig)

except ImportError:
    print("  (matplotlib not installed — skipping plot)")