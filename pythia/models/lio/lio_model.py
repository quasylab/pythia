"""LioModel — a BaseModel that simulates a Language of Interacting Objects system.

Each simulation step advances a
:class:`~pythia.models.lio.lio_state_of_individuals.LIOState`
by one tick via
:meth:`~pythia.models.lio.lio_state_of_individuals.LIOState.step`,
using the action weights provided by an
:class:`~pythia.models.lio.action_weight.ActionWeightFunction` and the transition
table owned by an
:class:`~pythia.models.lio.agent_definitions.AgentDefinitions`.

State keys written:

- ``"lio.configuration"`` *(LIOState)* — the current state.

Example::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.lio.action import Action
    from pythia.models.lio.action_weight import ActionWeightFunction
    from pythia.models.lio.agent_definitions import AgentDefinitions
    from pythia.models.lio.lio_state_of_individuals import LIOState
    from pythia.models.lio.lio_model import LioModel

    defs = AgentDefinitions()
    a, b = defs.get("a"), defs.get("b")
    tick = Action("tick")
    defs.add(a, tick, b, 1.0)
    defs.add(b, tick, a, 1.0)

    awf = ActionWeightFunction()
    awf.set_default(tick, lambda cfg: 1.0)

    model = LioModel(LIOState([a, b]), awf, defs)
    state = SimulationRunner(model, SimulationConfig(max_time=10.0, random_seed=42)).run()
    cfg = state.data["lio.configuration"]
"""

from __future__ import annotations

import random

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState
from pythia.models.lio.action_weight import ActionWeightFunction
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state import LIOState


class LioModel(BaseModel):
    """Simulation model for a Language of Interacting Objects system.

    At each step the current
    :class:`~pythia.models.lio.lio_state_of_individuals.LIOState`
    is advanced by calling
    ``conf.step(rng, action_weights.get(conf), agent_definitions)``, and the
    result is stored back under the ``"lio.configuration"`` key of the
    simulation state.  Every step has duration ``1.0``.

    Args:
        initial_configuration: The starting
            :class:`~pythia.models.lio.lio_state_of_individuals.LIOState`
            used on the first step of a run.
        action_weights: The weight functions that govern action selection for
            every agent.
        agent_definitions: The registry and transition table that owns the
            transitions used to advance agents.

    Attributes:
        action_weights: The :class:`~pythia.models.lio.action_weight.ActionWeightFunction`
            supplied at construction time.
        agent_definitions: The :class:`~pythia.models.lio.agent_definitions.AgentDefinitions`
            supplied at construction time.

    State keys written:
        lio.configuration (LIOState): The state at the current step.

    Examples:
        >>> from pythia.api import SimulationConfig, SimulationRunner
        >>> from pythia.models.lio.action import Action
        >>> from pythia.models.lio.action_weight import ActionWeightFunction
        >>> from pythia.models.lio.agent_definitions import AgentDefinitions
        >>> from pythia.models.lio.lio_state_of_individuals import LIOState
        >>> defs = AgentDefinitions()
        >>> a, b = defs.get("a"), defs.get("b")
        >>> tick = Action("tick")
        >>> defs.add(a, tick, b, 1.0)
        >>> defs.add(b, tick, a, 1.0)
        >>> awf = ActionWeightFunction()
        >>> awf.set_default(tick, lambda cfg: 1.0)
        >>> model = LioModel(LIOState([a, b]), awf, defs)
        >>> state = SimulationRunner(model, SimulationConfig(max_time=3.0, random_seed=0)).run()
        >>> state.data["lio.configuration"].size
        2
    """

    def __init__(
        self,
        initial_configuration: LIOState,
        action_weights: ActionWeightFunction,
        agent_definitions: AgentDefinitions,
    ) -> None:
        """Initialise the model with a starting configuration, weight function, and definitions."""
        self._initial_configuration = initial_configuration
        self.action_weights = action_weights
        self.agent_definitions = agent_definitions
        self._rng: random.Random = random.Random()

    @property
    def name(self) -> str:
        """Model identifier."""
        return "lio"

    def initialize(self, config: SimulationConfig) -> None:
        """Seed the random-number generator for the upcoming run.

        Args:
            config: The run configuration.  ``config.random_seed`` is used to
                seed the internal :class:`random.Random` instance; ``None``
                produces a non-deterministic run.
        """
        self.validate_config(config)
        self._rng = random.Random(config.random_seed)

    def setup(self, state: SimulationState) -> None:
        """Write the initial LIO configuration into ``state.data``.

        Sets ``state.data["lio.configuration"]`` so that measures collected
        at ``t = 0`` reflect the true starting configuration.

        Args:
            state: The simulation state at ``elapsed_time = 0``.
        """
        state.data["lio.configuration"] = self._initial_configuration

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the configuration by one step.

        Reads the current configuration from ``state.data["lio.configuration"]``
        (pre-populated by :meth:`setup`) and replaces it with the next state
        produced by ``conf.step(rng, action_weights.get(conf), agent_definitions)``.

        Args:
            state: The current simulation state.

        Returns:
            A pair ``(1.0, state)`` — every LIO step has duration ``1.0``.
        """
        conf: LIOState = state.data.get(
            "lio.configuration", self._initial_configuration
        )
        state.data["lio.configuration"] = conf.step(
            self._rng, self.action_weights.get(conf), self.agent_definitions
        )
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        """No-op finalisation.

        Args:
            state: The final simulation state.
        """
