"""Three state protocol — pythia.models.lio example.

The Three State protocol is a well known distributed probabilistic protocol
that can be used by a group of agents to agree on the same decision.
Differently from other protocols, agents have a limited communication capacities
and synchronisation is not possible.

In this example we assume the agents must agree on the result of a consultation
where a group of agents each of which can either take a decision or be agnostic.

Three states are possible:

- B0: is an agent that believes that the final decision is 0
- B1: is an agent that believes that the final decision is 1
- DUB: is an agent that doubts about the final decision

At each step, depending on its observations, an agent can change its point of view:
- if B0 meets an agent B1, it starts doubting about the final decision
- if B1 meets an agent B0, it starts doubting about the final decision
- if DUB meets an agent B1, it starts believing that 1 is the final decision
- if DUB meets an agent B0, it starts believing taht 0 is the final decision


Parameters (defaults)
---------------------
N          = 100    Number of agents

̄p_meet     = 0.4    Probability to meet another agent

β0 (beta0) = 0.25   Fraction of agents supporting decision 0

β1 (beta1) = 0.25   Fraction of agents supporting decision 1


Actions
------------
Each agent can perform two actions:
- meet0: an agent supporting 0 has been observed in the last time interval
- meet1: an agent supporting 1 has been observed in the last time interval
- skip: no agent is observed in the last time interval


Transitions
------------
S0  = meet1.IND + meet0.S0 + skip.S0
S1  = meet0.IND + meet1.S1 + skip.S1
IND = meet0.S0  + meet1.S1 + skip.IND

Action probability
------------------------
meet0 = p_meet * S0/N
meet1 = p_meet * S1/N
skip = p_meet * IND/N + (1-p_meet)



Usage
-----
    python examples/lio/threestate.py

Produces a printed summary and, if matplotlib is available, a time-series
plot saved to ``examples/lio/threestate.png``.
"""

from __future__ import annotations

from pythia.api import SimulationConfig, SimulationRunner, Simulator, DataTable, ReachabilityEstimator
from pythia.api.collector import DataCollector
from pythia.api.measure import Measure
from pythia.models.lio import AgentDefinitions, LioModel, LIOState, LIOStateOfCountingVector, Action, action_weight, \
    ActionWeightFunction
from pythia.util import ResultPlotter

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N          = 100    # Number of agents

p_meet     = 0.4    # Probability to meet another agent

BETA0      = 0.25   # Fraction of agents supporting decision 0

BETA1      = 0.25   # Fraction of agents supporting decision 1

INIT_S0  = int(BETA0*N)
INIT_S1  = int(BETA1*N)
INIT_IND = N-INIT_S1-INIT_S0

MAX_TIME = 70.0
SEED     = 42

# ---------------------------------------------------------------------------
# Agents, Actions and Definitions
# ---------------------------------------------------------------------------

defs = AgentDefinitions()
S0 = defs.get("S0")
S1 = defs.get("S1")
IND = defs.get("IND")

meet0 = Action("meet0")
meet1 = Action("meet1")
skip = Action("skip")

defs.add(S0, meet0, S0, 1)
defs.add(S0, meet1, IND, 1)
defs.add(S0, skip, S0, 1)

defs.add(S1, meet0, IND, 1)
defs.add(S1, meet1, S1, 1)
defs.add(S1, skip, S1, 1)

defs.add(IND, meet0, S0, 1)
defs.add(IND, meet1, S1, 1)
defs.add(IND, skip, IND, 1)

action_weights = ActionWeightFunction()

action_weights.set_default(meet0, lambda s: p_meet*s.fraction(S0))
action_weights.set_default(meet1, lambda s: p_meet*s.fraction(S1))
action_weights.set_default(skip, lambda s: p_meet*s.fraction(IND)+(1-p_meet))

state = LIOStateOfCountingVector({S0: INIT_S0, S1: INIT_S1, IND: INIT_IND})

model = LioModel(
    state,
    action_weights,
    defs
)


# ---------------------------------------------------------------------------
# Helpers to read number of agents from the simulation state
# ---------------------------------------------------------------------------

def _count(state, agent) -> float:
    cs = state.data.get("lio.configuration")
    return float(cs.fraction(agent)) if cs else 0.0

# ---------------------------------------------------------------------------
# Single Simulation Run
# ---------------------------------------------------------------------------

measures = [
    Measure("S0", lambda s: _count(s, S0), frequency=1.0),
    Measure("S1", lambda s: _count(s, S1), frequency=1.0),
    Measure("IND", lambda s: _count(s, IND), frequency=1.0),
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


print("ThreeState model summary")
print(f"  N={N}  p_meet={p_meet} β0={BETA0}  β1={BETA1}")
print(f"  Initial : S0={INIT_S0}  S1={INIT_S1}  IND={IND}")
print(f"  Final   : S0={final['S0']*N:.0f}  S1={final['S1']*N:.0f}  IND={final['IND']*N:.0f}")

# ---------------------------------------------------------------------------
# Plot (optional — requires matplotlib)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    times = list(table.times)
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, colour in (("S0", "steelblue"), ("S1", "red"),("IND", "seagreen")):
        values = [table.get(t)[label] for t in times]
        ax.plot(times, values, label=label, color=colour)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population")
    ax.set_title(f"Three State model  (N={N}  p_meet={p_meet} β0={BETA0}  β1={BETA1})")
    ax.legend()
    ax.set_xlim(0, MAX_TIME)
    ax.set_ylim(0, 1)

    out = "examples/lio/threestate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.close(fig)

except ImportError:
    print("  (matplotlib not installed — skipping plot)")

# ---------------------------------------------------------------------------
# Multiple Runs
# ---------------------------------------------------------------------------

simulator = Simulator(model, config, measures=measures)
simulator.run(100)

plotter = ResultPlotter(simulator.results)
plotter.plot(output_folder="examples/lio", prefix="threestate")
plotter.plot_statistics(output_folder="examples/lio", prefix="threestate_statistics")

# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

reachability = ReachabilityEstimator(model, config)

dt = 1
probs = reachability.estimate_over_time(1000,70, dt, lambda s: (_count(s, S0) == 1) or (_count(s, S1) == 1))

try:
    import matplotlib.pyplot as plt

    times = [dt*i for i in range(len(probs))]
    values = [p[0] for p in probs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, values, label="Probability")

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_title(f"Three State model  (N={N}  p_meet={p_meet} β0={BETA0}  β1={BETA1})")
    ax.legend()
    ax.set_xlim(0, MAX_TIME)
    ax.set_ylim(0, 1.1)

    out = "examples/lio/threestate_probability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")
    plt.close(fig)

except ImportError:
    print("  (matplotlib not installed — skipping plot)")