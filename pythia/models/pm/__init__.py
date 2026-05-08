"""Population models (``pm``) submodule.

Provides a continuous-time stochastic population simulator built on
:class:`~pythia.api.model.BaseModel`, together with the supporting value
objects needed to describe a population and its reactions.

Available model:

- :class:`PopulationModel` — Gillespie-style stochastic simulator driven by
  :class:`Rule` objects; each waiting time is drawn from an exponential
  distribution whose rate is the total propensity of all enabled rules.

Supporting value objects:

- :class:`Species` — immutable species descriptor
- :class:`Population` — mutable registry that creates and tracks species
- :class:`CountingState` — immutable snapshot mapping species to integer counts
- :class:`Rule` — single stochastic reaction rule consumed by :class:`PopulationModel`

All classes use namespaced state keys (``"pm.*"``) to avoid collisions with
other modules.

Example::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.pm import CountingState, Population, PopulationModel, Rule

    pop = Population()
    S   = pop.register("S")
    I   = pop.register("I")
    R   = pop.register("R")

    rules = [
        # S + I -> 2I  (infection)
        Rule(
            pre=CountingState({S: 1, I: 1}),
            post=CountingState({I: 2}),
            rate_function=lambda s: 0.3 * s.counting_vector.get(S.index, 0)
                                      * s.counting_vector.get(I.index, 0),
        ),
        # I -> R  (recovery)
        Rule(
            pre=CountingState({I: 1}),
            post=CountingState({R: 1}),
            rate_function=lambda s: 0.1 * s.counting_vector.get(I.index, 0),
        ),
    ]

    model = PopulationModel(
        population=pop,
        initial_state={S: 990, I: 10, R: 0},
        rules=rules,
    )
    state = SimulationRunner(model, SimulationConfig(max_time=50.0)).run()
    print(state.data["pm.counting_state"].counting_vector)
"""

from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.population_model import PopulationModel
from pythia.models.pm.rule import Rule
from pythia.models.pm.species import Species

__all__ = [
    "CountingState",
    "Population",
    "PopulationModel",
    "Rule",
    "Species",
]
