"""YODA (Yet anOther agent Description lAnguage) submodule.

Provides building blocks for describing and simulating YODA specifications.
A YODA system is a population of agents, each carrying layered data stores,
that evolve continuously through dynamics and interact discretely through
typed actions.

Available classes:

- :class:`Store` — immutable attribute-value mapping.
- :class:`AgentStore` — three-layer data store (state, features, observations)
  for a single agent.
- :class:`Action` — named transition with a step function
  ``(rng, state, observations) -> Store``.
- :class:`Agent` — named participant in a YODA system.
- :class:`AgentState` — named behavioural state of an agent.
- :class:`AgentConfiguration` — immutable binding of an :class:`Agent`, an
  :class:`AgentState`, and an :class:`AgentStore`.
- :class:`AgentDefinitions` — central registry for transitions, dynamics,
  observations, action weights, and high-level step methods.
- :class:`SystemConfiguration` — immutable ordered collection of
  :class:`AgentConfiguration` objects representing the full population.
- :class:`YodaModel` — :class:`~pythia.api.model.BaseModel` that simulates a
  YODA specification by interleaving environment and agent-action steps.

Example::

    import random
    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.yoda import (
        Action, Agent, AgentConfiguration, AgentDefinitions,
        AgentState, AgentStore, SystemConfiguration, YodaModel,
    )
    from pythia.models.yoda.store import Store

    a    = Agent("worker")
    idle = AgentState("idle")
    busy = AgentState("busy")
    tick = Action("tick", step_function=lambda rng, s, o: s)

    defs = AgentDefinitions()
    defs.add_transition(a, idle, tick, 1.0, busy)
    defs.set_default_action_weight(tick, lambda s: 1.0)

    def initial(rng: random.Random) -> SystemConfiguration:
        return SystemConfiguration([
            AgentConfiguration(a, idle, AgentStore(Store(), Store(), Store()))
        ])

    model = YodaModel(initial_state_generator=initial, agent_definitions=defs)
    state = SimulationRunner(model, SimulationConfig(max_time=5.0, random_seed=0)).run()
    print(state.data["yoda.configuration"].get(0).agent_state.name)
"""

from pythia.models.yoda.action import Action
from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_definitions import AgentDefinitions
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.configuration import SystemConfiguration
from pythia.models.yoda.store import Store
from pythia.models.yoda.yoda_model import YodaModel

__all__ = [
    "Action",
    "Agent",
    "AgentConfiguration",
    "AgentDefinitions",
    "AgentState",
    "AgentStore",
    "SystemConfiguration",
    "Store",
    "YodaModel",
]
