"""YodaModel — a BaseModel that simulates a YODA specification.

Each simulation step advances the current
:class:`~pythia.models.yoda.configuration.SystemConfiguration` by
interleaving environment evolution (dynamics + observations) and agent action
execution according to two configurable time-step schedules.

State keys written:

- ``"yoda.configuration"`` *(SystemConfiguration)* — the current system state.
- ``"yoda.time_of_next_environment_step"`` *(float)* — simulated time at which
  the next environment step is due.
- ``"yoda.time_of_next_agents_step"`` *(float)* — simulated time at which the
  next agent-action step is due.

Configuration metadata keys (optional):

- ``"yoda.environment_simulation_step"`` *(float, > 0)* — overrides the
  constructor default for the environment step interval.
- ``"yoda.agents_simulation_step"`` *(float, > 0)* — overrides the constructor
  default for the agent-action step interval.

Example::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.yoda import AgentDefinitions, YodaModel
    from pythia.models.yoda.configuration import SystemConfiguration

    defs = AgentDefinitions()
    # ... populate defs ...

    model = YodaModel(
        initial_state_generator=lambda rng: SystemConfiguration([...]),
        agent_definitions=defs,
    )
    state = SimulationRunner(model, SimulationConfig(max_time=10.0, random_seed=0)).run()
    cfg = state.data["yoda.configuration"]
"""

from __future__ import annotations

import random
from collections.abc import Callable

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState
from pythia.models.yoda.agent_definitions import AgentDefinitions
from pythia.models.yoda.configuration import SystemConfiguration


class YodaModel(BaseModel):
    """Simulation model for a YODA multi-agent system.

    Interleaves two kinds of step:

    * **Environment step** — applies agent dynamics then updates observations
      (via :meth:`~pythia.models.yoda.agent_definitions.AgentDefinitions.environment_step`).
    * **Agent-action step** — each agent independently samples and fires one
      action
      (via :meth:`~pythia.models.yoda.agent_definitions.AgentDefinitions.action_executions_step`).

    When both steps are due at the same simulated time, both are applied
    together via
    :meth:`~pythia.models.yoda.agent_definitions.AgentDefinitions.agents_step`.

    Args:
        initial_state_generator: A callable ``rng -> SystemConfiguration``
            used to produce the initial system state on the first step.
        agent_definitions: The
            :class:`~pythia.models.yoda.agent_definitions.AgentDefinitions`
            that owns transitions, dynamics, observations, and action weights.
        environment_simulation_step: Interval (in simulated time units)
            between environment steps.  Defaults to ``1.0``.
        agents_simulation_step: Interval (in simulated time units) between
            agent-action steps.  Defaults to ``1.0``.

    State keys written:
        yoda.configuration (SystemConfiguration): Current system state.
        yoda.time_of_next_environment_step (float): Next environment step time.
        yoda.time_of_next_agents_step (float): Next agent-action step time.
    """

    def __init__(
        self,
        initial_state_generator: Callable[[random.Random], SystemConfiguration],
        agent_definitions: AgentDefinitions,
        environment_simulation_step: float = 1.0,
        agents_simulation_step: float = 1.0,
    ) -> None:
        self._initial_state_generator = initial_state_generator
        self.agent_definitions = agent_definitions
        self._environment_simulation_step = environment_simulation_step
        self._agents_simulation_step = agents_simulation_step
        self._rng: random.Random = random.Random()

    @property
    def name(self) -> str:
        """Model identifier."""
        return "yoda"

    def initialize(self, config: SimulationConfig) -> None:
        """Prepare the model for a new run.

        Seeds the internal random-number generator from *config.random_seed*
        and resolves the two step intervals (constructor defaults may be
        overridden via ``config.metadata``).  Also resets the internal
        time-of-next-step counters to the resolved interval values.

        Args:
            config: The run configuration.
        """
        self.validate_config(config)
        self._rng = random.Random(config.random_seed)
        self._environment_simulation_step = config.metadata.get(
            "yoda.environment_simulation_step", self._environment_simulation_step
        )
        self._agents_simulation_step = config.metadata.get(
            "yoda.agents_simulation_step", self._agents_simulation_step
        )

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the system by one scheduled event.

        Selects the next pending event (environment step or agent-action step)
        based on their scheduled times, executes it, updates the simulation
        state, and returns the elapsed duration.

        Args:
            state: The current simulation state.

        Returns:
            A pair ``(dt, state)`` where *dt* is the simulated time consumed
            by this step.
        """
        conf: SystemConfiguration | None = state.data.get("yoda.configuration")
        if conf is None:
            conf = self._initial_state_generator(self._rng)

        env_time: float = state.data.get(
            "yoda.time_of_next_environment_step",
            state.elapsed_time + self._environment_simulation_step,
        )
        agent_time: float = state.data.get(
            "yoda.time_of_next_agents_step",
            state.elapsed_time + self._agents_simulation_step,
        )

        if env_time < agent_time:
            dt = env_time - state.elapsed_time
            next_conf = self.agent_definitions.environment_step(self._rng, conf, dt)
            env_time += self._environment_simulation_step
        elif env_time == agent_time:
            dt = agent_time - state.elapsed_time
            next_conf = self.agent_definitions.agents_step(self._rng, conf, dt)
            env_time += self._environment_simulation_step
            agent_time += self._agents_simulation_step
        else:
            dt = agent_time - state.elapsed_time
            next_conf = self.agent_definitions.agents_step(self._rng, conf, dt)
            agent_time += self._agents_simulation_step

        state.data["yoda.configuration"] = next_conf
        state.data["yoda.time_of_next_environment_step"] = env_time
        state.data["yoda.time_of_next_agents_step"] = agent_time

        return dt, state

    def finalize(self, state: SimulationState) -> None:
        """No-op finalisation.

        Args:
            state: The final simulation state.
        """
