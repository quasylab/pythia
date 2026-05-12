"""NBAModel — a BaseModel that simulates a Network of Broadcasting Agents system.

Each simulation step advances the current
:class:`~pythia.models.nba.nba_configuration.NBAConfiguration` by one tick via
:meth:`~pythia.models.nba.nba_configuration.NBAConfiguration.step`.  When no
transition is enabled the step returns duration ``0.0``, signalling the runner
to stop.

State keys written:

- ``"nba.configuration"`` *(NBAConfiguration)* — the current population state.

Example::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.nba.agent_definitions import AgentDefinitions
    from pythia.models.nba.channel import Channel
    from pythia.models.nba.environment import Environment
    from pythia.models.nba.nba_model import NBAModel

    c = Channel("c")
    defs = AgentDefinitions()
    a, b = defs.get("a"), defs.get("b")
    defs.add_send_transition(a, c, b, 1.0)
    defs.add_receive_transition(b, c, a, 1.0)

    env = Environment()
    env.set_send_rate(a, c, 1.0)
    env.set_receive_weight(b, c, 1.0)

    model = NBAModel({a: 3, b: 3}, defs, env)
    state = SimulationRunner(model, SimulationConfig(max_time=10.0, random_seed=0)).run()
    cfg = state.data["nba.configuration"]
"""

from __future__ import annotations

import random

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState
from pythia.models.nba.agent import Agent
from pythia.models.nba.agent_definitions import AgentDefinitions
from pythia.models.nba.environment import Environment
from pythia.models.nba.nba_configuration import NBAConfiguration


class NBAModel(BaseModel):
    """Simulation model for a Network of Broadcasting Agents system.

    At each step the current
    :class:`~pythia.models.nba.nba_configuration.NBAConfiguration` is advanced
    by calling :meth:`~pythia.models.nba.nba_configuration.NBAConfiguration.step`.
    When that call returns ``None`` (no enabled transition), the step duration
    is ``0.0``, which signals the runner to stop.  Otherwise the returned
    ``(time, next_cfg)`` pair is forwarded directly.

    Args:
        initial_state: Mapping from :class:`~pythia.models.nba.agent.Agent` to
            a strictly positive integer count, used to build the initial
            :class:`~pythia.models.nba.nba_configuration.NBAConfiguration`.
        agent_definitions: Registry and transition table that owns the NBA
            transitions.
        environment: Per-agent, per-channel rates and weights.

    Attributes:
        initial_state: The initial agent-count mapping supplied at construction.
        agent_definitions: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
            supplied at construction.
        environment: The :class:`~pythia.models.nba.environment.Environment`
            supplied at construction.

    State keys written:
        nba.configuration (NBAConfiguration): The population at the current step.

    Example:
        >>> from pythia.api import SimulationConfig, SimulationRunner
        >>> from pythia.models.nba.agent_definitions import AgentDefinitions
        >>> from pythia.models.nba.channel import Channel
        >>> from pythia.models.nba.environment import Environment
        >>> c = Channel("c")
        >>> defs = AgentDefinitions()
        >>> a, b = defs.get("a"), defs.get("b")
        >>> defs.add_send_transition(a, c, b, 1.0)
        >>> defs.add_receive_transition(b, c, a, 1.0)
        >>> env = Environment()
        >>> env.set_send_rate(a, c, 1.0)
        >>> env.set_receive_weight(b, c, 1.0)
        >>> model = NBAModel({a: 1, b: 1}, defs, env)
        >>> state = SimulationRunner(model, SimulationConfig(max_time=5.0, random_seed=0)).run()
        >>> state.data["nba.configuration"].size
        2
    """

    def __init__(
        self,
        initial_state: dict[Agent, int],
        agent_definitions: AgentDefinitions,
        environment: Environment,
    ) -> None:
        self.initial_state = initial_state
        self.agent_definitions = agent_definitions
        self.environment = environment
        self._initial_cfg = NBAConfiguration(initial_state)
        self._rng: random.Random = random.Random()

    @property
    def name(self) -> str:
        """Model identifier."""
        return "nba"

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
        """Write the initial NBA configuration into ``state.data``.

        Sets ``state.data["nba.configuration"]`` so that measures collected
        at ``t = 0`` reflect the true starting population.

        Args:
            state: The simulation state at ``elapsed_time = 0``.
        """
        state.data["nba.configuration"] = self._initial_cfg

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the configuration by one step.

        Reads the current configuration from ``state.data["nba.configuration"]``
        (pre-populated by :meth:`setup`).  Calls
        :meth:`~pythia.models.nba.nba_configuration.NBAConfiguration.step`;
        if it returns ``None`` (no enabled transition) the step duration is
        ``0.0`` and the configuration is left unchanged, signalling the runner
        to stop.  Otherwise the new configuration and the sampled time are
        returned.

        Args:
            state: The current simulation state.

        Returns:
            A pair ``(dt, state)`` where *dt* is ``0.0`` when no transition is
            enabled, or the exponentially-distributed sojourn time otherwise.
        """
        cfg: NBAConfiguration = state.data.get(
            "nba.configuration", self._initial_cfg
        )
        result = cfg.step(self._rng, self.agent_definitions, self.environment)
        if result is None:
            state.data["nba.configuration"] = cfg
            return 0.0, state
        dt, next_cfg = result
        state.data["nba.configuration"] = next_cfg
        return dt, state

    def finalize(self, state: SimulationState) -> None:
        """No-op finalisation.

        Args:
            state: The final simulation state.
        """
