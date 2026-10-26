"""AgentDefinitions — transition table for a YODA system.

:class:`AgentDefinitions` owns the complete set of agents and their
behavioural transitions.  Transitions are added incrementally via
:meth:`add_transition` and stored as a three-level mapping:

    ``Agent → AgentState (source) → Action → [(weight, AgentState (target))]``

Each ``(weight, target)`` pair represents one possible outcome of firing the
action from the given source state, with *weight* being a strictly positive
float that governs how likely that outcome is relative to others.

Example::

    from pythia.models.yoda.action import Action
    from pythia.models.yoda.agent import Agent
    from pythia.models.yoda.agent_definitions import AgentDefinitions
    from pythia.models.yoda.agent_state import AgentState
    from pythia.models.yoda.store import Store

    idle = AgentState("idle")
    busy = AgentState("busy")
    a    = Agent("worker")
    tick = Action("tick", lambda s, o: 1.0, lambda s, o: s)

    defs = AgentDefinitions()
    defs.add_transition(a, idle, tick, busy, 1.0)

    defs.agents[a][idle][tick]   # [(1.0, AgentState('busy'))]
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from pythia.models.yoda.action import Action
from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.store import Store

from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.configuration import SystemConfiguration


# Agent → source state → action → [(weight, target state)]
_TransitionTable = dict[Agent, dict[AgentState, dict[Action, list[tuple[float, AgentState]]]]]

# Agent → dynamics function (rng, agent_store, dt) → new features
_DynamicsTable = dict[Agent, Callable[[random.Random, AgentStore, float], Store]]

# Agent → observation function (agent_store, configuration) → observed store
_ObservationTable = dict[Agent, Callable[[random.Random, AgentStore, SystemConfiguration], Store]]

# Weight function: AgentStore → non-negative float
_WeightFunction = Callable[[AgentStore], float]

# Agent → (Action → weight function)
_ActionWeightTable = dict[Agent, dict[Action, _WeightFunction]]

# Action → default weight function
_DefaultActionWeightTable = dict[Action, _WeightFunction]


class AgentDefinitions:
    """Transition table for a YODA multi-agent system.

    Stores weighted transitions of the form
    ``(agent, source_state, action) → [(weight, target_state)]``.
    Transitions are added one at a time via :meth:`add_transition`; the
    table starts empty.

    Examples:
        >>> from pythia.models.yoda.action import Action
        >>> from pythia.models.yoda.agent import Agent
        >>> from pythia.models.yoda.agent_state import AgentState
        >>> from pythia.models.yoda.store import Store
        >>> idle = AgentState("idle")
        >>> busy = AgentState("busy")
        >>> a    = Agent("worker")
        >>> tick = Action("tick", lambda s, o: 1.0, lambda s, o: s)
        >>> defs = AgentDefinitions()
        >>> defs.add_transition(a, idle, tick, busy, 1.0)
        >>> defs.agents[a][idle][tick]
        [(1.0, AgentState('busy'))]
    """

    def __init__(self) -> None:
        self._agents: _TransitionTable = {}
        self._dynamics: _DynamicsTable = {}
        self.default_dynamics: Callable[[random.Random, AgentStore, float], Store] = (
            lambda rng, s, dt: s.features
        )
        self._observations: _ObservationTable = {}
        self.default_observation: Callable[[random.Random, AgentStore, SystemConfiguration], Store] = (
            lambda rng, s, cfg: Store()
        )
        self._action_weights: _ActionWeightTable = {}
        self._default_action_weights: _DefaultActionWeightTable = {}

    @property
    def agents(self) -> _TransitionTable:
        """The transition table as a nested mapping.

        Returns a copy of the containers at every level so that external
        mutations do not affect the internal state.

        Examples:
            >>> from pythia.models.yoda.action import Action
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.store import Store
            >>> idle = AgentState("idle")
            >>> a    = Agent("a")
            >>> tick = Action("tick", lambda s, o: 1.0, lambda s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, idle, 2.0)
            >>> defs.agents[a][idle][tick]
            [(2.0, AgentState('idle'))]
        """
        return _copy(self._agents)

    def add_transition(
        self,
        agent: Agent,
        source: AgentState,
        action: Action,
        target: AgentState,
        weight: float = 1.0,
    ) -> None:
        """Record a weighted transition for *agent*.

        Appends the pair ``(weight, target)`` to the list stored at
        ``agents[agent][source][action]``, creating intermediate mappings
        as needed.

        Args:
            agent: The agent to which the transition belongs.
            source: The source behavioural state.
            action: The action that triggers the transition.
            target: The resulting behavioural state.
            weight: A strictly positive float representing the propensity of
                this outcome.  Defaults to ``1.0``.

        Raises:
            ValueError: If *weight* is not strictly positive.

        Examples:
            >>> from pythia.models.yoda.action import Action
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.store import Store
            >>> idle = AgentState("idle")
            >>> busy = AgentState("busy")
            >>> a    = Agent("a")
            >>> tick = Action("tick", lambda s, o: 1.0, lambda s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, busy, 1.0)
            >>> defs.agents[a][idle][tick]
            [(1.0, AgentState('busy'))]
        """
        if weight <= 0.0:
            raise ValueError(
                f"Transition weight must be strictly positive, got {weight!r}"
            )
        (
            self._agents
            .setdefault(agent, {})
            .setdefault(source, {})
            .setdefault(action, [])
            .append((weight, target))
        )

    @property
    def dynamics(self) -> _DynamicsTable:
        """The dynamics table mapping each agent to its dynamics function.

        Returns a shallow copy of the mapping so that external mutations do
        not affect the internal table.  The callable values are not copied
        since functions are immutable.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_dynamics(a, lambda rng, s, dt: s.features.set("t", dt))
            >>> defs.dynamics[a](random.Random(0), AgentStore(Store(), Store(), Store()), 0.5)["t"]
            0.5
        """
        return dict(self._dynamics)

    def add_dynamics(
        self,
        agent: Agent,
        f: Callable[[random.Random, AgentStore, float], Store],
    ) -> None:
        """Register the dynamics function for *agent*.

        The dynamics function describes how the agent's features evolve
        over time: given a random generator *rng*, the agent's current
        :class:`~pythia.models.yoda.agent_store.AgentStore` *s*, and a
        time step *dt* (strictly positive), it returns a new
        :class:`~pythia.models.yoda.store.Store` representing the updated
        features after *dt* time units.

        Calling this method again for the same agent replaces the previously
        registered function.

        Args:
            agent: The agent whose dynamics are being defined.
            f: A callable ``(rng, s, dt) -> Store`` where *rng* is a
                :class:`random.Random` instance, *s* is an
                :class:`~pythia.models.yoda.agent_store.AgentStore`, and
                *dt* is a strictly positive float.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_dynamics(
            ...     a,
            ...     lambda rng, s, dt: s.features.set(
            ...         "x", s.features.get("x", 0) + dt
            ...     ),
            ... )
            >>> f = defs.dynamics[a]
            >>> store = AgentStore(Store(), Store({"x": 1.0}), Store())
            >>> f(random.Random(0), store, 0.5)["x"]
            1.5
        """
        self._dynamics[agent] = f

    def get_dynamics(
        self, agent: Agent
    ) -> Callable[[random.Random, AgentStore, float], Store]:
        """Return the dynamics function registered for *agent*.

        Falls back to :attr:`default_dynamics` when no function has been
        registered for *agent* via :meth:`add_dynamics`.

        Args:
            agent: The agent whose dynamics function is requested.

        Returns:
            The registered dynamics callable, or :attr:`default_dynamics`
            if none has been recorded for *agent*.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.get_dynamics(a) is defs.default_dynamics
            True
            >>> defs.add_dynamics(a, lambda rng, s, dt: s.features.set("x", dt))
            >>> store = AgentStore(Store(), Store(), Store())
            >>> defs.get_dynamics(a)(random.Random(0), store, 1.0)["x"]
            1.0
        """
        return self._dynamics.get(agent, self.default_dynamics)

    @property
    def observations(self) -> _ObservationTable:
        """The observation table mapping each agent to its observation function.

        Returns a shallow copy of the mapping so that external mutations do
        not affect the internal table.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_observation(a, lambda rng, s, cfg: s.features)
            >>> defs.observations[a] is not None
            True
        """
        return dict(self._observations)

    def add_observation(
        self,
        agent: Agent,
        f: Callable[[random.Random, AgentStore, SystemConfiguration], Store],
    ) -> None:
        """Register the observation function for *agent*.

        The observation function computes what the agent observes from its
        own store and the current system configuration: given the agent's
        :class:`~pythia.models.yoda.agent_store.AgentStore` *s* and the
        full :class:`~pythia.models.yoda.configuration.SystemConfiguration`
        *cfg*, it returns a :class:`~pythia.models.yoda.store.Store`
        representing the agent's observations.

        Calling this method again for the same agent replaces the previously
        registered function.

        Args:
            agent: The agent whose observation function is being defined.
            f: A callable ``(s, cfg) -> Store`` where *s* is the agent's
                :class:`~pythia.models.yoda.agent_store.AgentStore` and
                *cfg* is the current
                :class:`~pythia.models.yoda.configuration.SystemConfiguration`.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_observation(a, lambda rng, s, cfg: Store({"n": cfg.size()}))
            >>> defs.observations[a] is not None
            True
        """
        self._observations[agent] = f

    def get_observation(
        self, agent: Agent
    ) -> Callable[[random.Random, AgentStore, Any], Store]:
        """Return the observation function registered for *agent*.

        Falls back to :attr:`default_observation` when no function has been
        registered for *agent* via :meth:`add_observation`.

        Args:
            agent: The agent whose observation function is requested.

        Returns:
            The registered observation callable, or
            :attr:`default_observation` if none has been recorded for
            *agent*.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.get_observation(a) is defs.default_observation
            True
            >>> defs.add_observation(a, lambda rng, s, cfg: Store({"n": cfg.size()}))
            >>> defs.get_observation(a) is defs.default_observation
            False
        """
        return self._observations.get(agent, self.default_observation)

    def apply_dynamics(
        self,
        rng: random.Random,
        aconf: AgentConfiguration,
        dt: float,
    ) -> AgentConfiguration:
        """Return a new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        with updated features.

        Evaluates the dynamics function registered for ``aconf.agent``
        (falling back to :attr:`default_dynamics`) on *rng*,
        ``aconf.agent_store``, and *dt*, then replaces the features layer
        of the agent store with the result.

        Args:
            rng: Random number generator forwarded to the dynamics function.
            aconf: The agent configuration whose features are to be updated.
            dt: Time step (strictly positive) forwarded to the dynamics
                function.

        Returns:
            A new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
            with the features layer of its store updated.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_dynamics(
            ...     a,
            ...     lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt),
            ... )
            >>> store = AgentStore(Store(), Store({"x": 1.0}), Store())
            >>> aconf = AgentConfiguration(a, AgentState("idle"), store)
            >>> result = defs.apply_dynamics(random.Random(0), aconf, 0.5)
            >>> result.agent_store.features["x"]
            1.5
        """
        new_features = self.get_dynamics(aconf.agent)(rng, aconf.agent_store, dt)
        return aconf.set_features(new_features)

    def dynamics_step(
        self,
        rng: random.Random,
        conf: SystemConfiguration,
        dt: float,
    ) -> SystemConfiguration:
        """Return a new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
        with agent dynamics applied to every agent.

        Equivalent to ``conf.apply(lambda aconf: self.apply_dynamics(rng, aconf, dt))``:
        every :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        in *conf* is replaced with the result of advancing its features by
        *dt* time units using the registered dynamics function.

        Args:
            rng: Random number generator forwarded to each dynamics function.
            conf: The current system configuration.
            dt: Time step (strictly positive) forwarded to each dynamics
                function.

        Returns:
            A new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
            of the same size.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_dynamics(
            ...     a,
            ...     lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt),
            ... )
            >>> store = AgentStore(Store(), Store({"x": 0.0}), Store())
            >>> aconf = AgentConfiguration(a, AgentState("idle"), store)
            >>> conf  = SystemConfiguration([aconf])
            >>> conf2 = defs.dynamics_step(random.Random(0), conf, 1.0)
            >>> conf2.get(0).agent_store.features["x"]
            1.0
        """
        return conf.apply(lambda aconf: self.apply_dynamics(rng, aconf, dt))

    def agents_step(
        self,
        rng: random.Random,
        conf: SystemConfiguration,
        dt: float,
    ) -> SystemConfiguration:
        """Advance the system by *dt* time units followed by agent action execution.

        Composes :meth:`environment_step` (dynamics + observations) with
        :meth:`action_executions_step` (each agent fires one action):

        .. code-block:: python

            self.action_executions_step(rng, self.environment_step(rng, conf, dt))

        Args:
            rng: Random number generator forwarded to both steps.
            conf: The current system configuration.
            dt: Time step forwarded to :meth:`environment_step`.

        Returns:
            A new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
            reflecting the environmental evolution over *dt* time units
            followed by one action per agent.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> idle = AgentState("idle")
            >>> busy = AgentState("busy")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, busy, 1.0)
            >>> defs.set_action_weight(a, tick, lambda s: 1.0)
            >>> defs.add_dynamics(a, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt))
            >>> store = AgentStore(Store(), Store({"x": 0.0}), Store())
            >>> aconf = AgentConfiguration(a, idle, store)
            >>> conf  = SystemConfiguration([aconf])
            >>> conf2 = defs.agents_step(random.Random(0), conf, 1.0)
            >>> conf2.get(0).agent_state.name
            'busy'
            >>> conf2.get(0).agent_store.features["x"]
            1.0
        """
        return self.action_executions_step(rng, self.environment_step(rng, conf, dt))

    def action_executions_step(
        self,
        rng: random.Random,
        conf: SystemConfiguration,
    ) -> SystemConfiguration:
        """Return the configuration after each agent independently executes one action.

        Applies :meth:`execute_step` to every agent configuration in *conf*.
        Agents for which no action is enabled remain unchanged.

        Equivalent to:

        .. code-block:: python

            conf.apply(lambda aconf: self.execute_step(rng, aconf))

        Args:
            rng: Random number generator forwarded to :meth:`execute_step`
                for every agent.
            conf: The current system configuration.

        Returns:
            A new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
            of the same size.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> idle = AgentState("idle")
            >>> busy = AgentState("busy")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, busy, 1.0)
            >>> defs.set_action_weight(a, tick, lambda s: 1.0)
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, idle, store)
            >>> conf  = SystemConfiguration([aconf])
            >>> defs.action_executions_step(random.Random(0), conf).get(0).agent_state.name
            'busy'
        """
        return conf.apply(lambda aconf: self.execute_step(rng, aconf))

    def environment_step(
        self,
        rng: random.Random,
        conf: SystemConfiguration,
        dt: float,
    ) -> SystemConfiguration:
        """Advance the system by *dt* time units without agent actions.

        Applies agent dynamics then updates observations:

        .. code-block:: python

            self.observations_step(rng, self.dynamics_step(rng, conf, dt))

        Args:
            rng: Random number generator forwarded to both steps.
            conf: The current system configuration.
            dt: Time step forwarded to :meth:`dynamics_step`.

        Returns:
            A new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
            reflecting the environmental evolution over *dt* time units.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_dynamics(
            ...     a, lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt)
            ... )
            >>> defs.add_observation(a, lambda rng, s, cfg: Store({"n": cfg.size()}))
            >>> store = AgentStore(Store(), Store({"x": 0.0}), Store())
            >>> aconf = AgentConfiguration(a, AgentState("idle"), store)
            >>> conf  = SystemConfiguration([aconf])
            >>> conf2 = defs.environment_step(random.Random(0), conf, 1.0)
            >>> conf2.get(0).agent_store.features["x"]
            1.0
            >>> conf2.get(0).agent_store.observations["n"]
            1
        """
        return self.observations_step(rng, self.dynamics_step(rng, conf, dt))

    def observations_step(
        self,
        rng: random.Random,
        conf: SystemConfiguration,
    ) -> SystemConfiguration:
        """Return a new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
        with observations updated for every agent.

        Equivalent to ``conf.apply(lambda aconf: self.apply_observations(rng, aconf, conf))``:
        every :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        in *conf* is replaced with the result of evaluating its registered
        observation function against the full system configuration.

        Args:
            rng: Random number generator forwarded to each observation
                function.
            conf: The current system configuration, passed both as the
                collection to iterate and as the context for each observation
                function.

        Returns:
            A new :class:`~pythia.models.yoda.configuration.SystemConfiguration`
            of the same size.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_observation(a, lambda rng, s, cfg: Store({"n": cfg.size()}))
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, AgentState("idle"), store)
            >>> conf  = SystemConfiguration([aconf])
            >>> conf2 = defs.observations_step(random.Random(0), conf)
            >>> conf2.get(0).agent_store.observations["n"]
            1
        """
        return conf.apply(lambda aconf: self.apply_observations(rng, aconf, conf))

    def apply_observations(
        self,
        rng: random.Random,
        aconf: AgentConfiguration,
        sconf: SystemConfiguration,
    ) -> AgentConfiguration:
        """Return a new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        with updated observations.

        Evaluates the observation function registered for ``aconf.agent``
        (falling back to :attr:`default_observation`) on *rng*,
        ``aconf.agent_store``, and *sconf*, then replaces the observations
        layer of the agent store with the result.

        Args:
            rng: Random number generator forwarded to the observation function.
            aconf: The agent configuration whose observations are to be
                updated.
            sconf: The current system configuration passed to the observation
                function.

        Returns:
            A new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
            with the observations layer of its store updated.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.configuration import SystemConfiguration
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> defs = AgentDefinitions()
            >>> defs.add_observation(
            ...     a,
            ...     lambda rng, s, cfg: Store({"n": cfg.size()}),
            ... )
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, AgentState("idle"), store)
            >>> sconf = SystemConfiguration([aconf])
            >>> result = defs.apply_observations(random.Random(0), aconf, sconf)
            >>> result.agent_store.observations["n"]
            1
        """
        return aconf.set_observations(self.get_observation(aconf.agent)(rng, aconf.agent_store, sconf))


    def get_action_weight(
        self,
        agent: Agent,
        store: AgentStore,
        action: Action,
    ) -> float:
        """Return the weight of *action* for *agent* evaluated on *store*.

        Resolution order:

        1. ``_action_weights[agent][action](store)`` — agent-specific weight
           function registered via :meth:`set_action_weight`.
        2. ``_default_action_weights[action](store)`` — default weight
           function registered via :meth:`set_default_action_weight`.
        3. ``0.0`` — when neither is found.

        Args:
            agent: The agent performing the action.
            store: The current :class:`~pythia.models.yoda.agent_store.AgentStore`
                passed to the weight function.
            action: The action whose weight is requested.

        Returns:
            A non-negative float.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> b    = Agent("b")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.set_action_weight(a, tick, lambda s: 3.0)
            >>> defs.set_default_action_weight(tick, lambda s: 1.0)
            >>> store = AgentStore(Store(), Store(), Store())
            >>> defs.get_action_weight(a, store, tick)
            3.0
            >>> defs.get_action_weight(b, store, tick)
            1.0
            >>> defs.get_action_weight(b, store, Action("other", lambda rng, s, o: s))
            0.0
        """
        agent_specific = self._action_weights.get(agent, {})
        if action in agent_specific:
            return agent_specific[action](store)
        if action in self._default_action_weights:
            return self._default_action_weights[action](store)
        return 0.0


    def sample_action(
        self,
        rng: random.Random,
        aconf: AgentConfiguration,
    ) -> Action | None:
        """Sample an enabled action for *aconf*, or return ``None``.

        An action is *enabled* when it appears in the transition table for
        ``(aconf.agent, aconf.agent_state)`` **and** its weight, as returned
        by :meth:`get_action_weight`, is strictly positive.

        One enabled action is chosen at random with probability proportional
        to its weight.  ``None`` is returned when no action is enabled.

        Args:
            rng: Random number generator used for the weighted selection.
            aconf: The agent configuration for which an action is sampled.

        Returns:
            A randomly selected enabled :class:`~pythia.models.yoda.action.Action`,
            or ``None`` if no action is enabled.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> idle = AgentState("idle")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, idle, 1.0)
            >>> defs.set_action_weight(a, tick, lambda s: 1.0)
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, idle, store)
            >>> defs.sample_action(random.Random(0), aconf) is tick
            True
            >>> defs.sample_action(random.Random(0), AgentConfiguration(a, AgentState("busy"), store)) is None
            True
        """
        available = self._agents.get(aconf.agent, {}).get(aconf.agent_state, {})
        actions: list[Action] = []
        weights: list[float] = []
        for act in available:
            w = self.get_action_weight(aconf.agent, aconf.agent_store, act)
            if w > 0.0:
                actions.append(act)
                weights.append(w)
        if not actions:
            return None
        return rng.choices(actions, weights=weights, k=1)[0]

    def execute_action(
        self,
        rng: random.Random,
        aconf: AgentConfiguration,
        action: Action,
    ) -> AgentConfiguration:
        """Return the :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        produced by firing *action* for *aconf*.

        Selects the next behavioural state by sampling from the transitions
        ``self.agents[aconf.agent][aconf.agent_state][action]`` with
        probability proportional to the stored weights, then builds the new
        configuration:

        .. code-block:: python

            aconf.set_state(
                next_state,
                action.step(rng, aconf.agent_store.state,
                            aconf.agent_store.observations),
            )

        Args:
            rng: Random number generator used for both the state selection
                and the action step function.
            aconf: The agent configuration from which the action is executed.
            action: The action to fire.

        Returns:
            A new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`.

        Raises:
            KeyError: If *action* has no transitions defined for
                ``(aconf.agent, aconf.agent_state)``.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> idle = AgentState("idle")
            >>> busy = AgentState("busy")
            >>> tick = Action("tick", lambda rng, s, o: s.set("ticked", True))
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, busy, 1.0)
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, idle, store)
            >>> result = defs.execute_action(random.Random(0), aconf, tick)
            >>> result.agent_state.name
            'busy'
            >>> result.agent_store.state["ticked"]
            True
        """
        transitions = self._agents[aconf.agent][aconf.agent_state][action]
        states  = [s for _, s in transitions]
        weights = [w for w, _ in transitions]
        next_state = rng.choices(states, weights=weights, k=1)[0]
        new_state_store = action.step(
            rng, aconf.agent_store.state, aconf.agent_store.observations
        )
        return aconf.set_state(next_state, new_state_store)

    def execute_step(
        self,
        rng: random.Random,
        aconf: AgentConfiguration,
    ) -> AgentConfiguration:
        """Advance *aconf* by one step.

        Samples an enabled action via :meth:`sample_action`.  If no action
        is enabled the configuration is returned unchanged; otherwise the
        selected action is fired via :meth:`execute_action`.

        Args:
            rng: Random number generator forwarded to both
                :meth:`sample_action` and :meth:`execute_action`.
            aconf: The agent configuration to advance.

        Returns:
            A new :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
            produced by the selected action, or *aconf* itself when no
            action is enabled.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> idle = AgentState("idle")
            >>> busy = AgentState("busy")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.add_transition(a, idle, tick, busy, 1.0)
            >>> defs.set_action_weight(a, tick, lambda s: 1.0)
            >>> store = AgentStore(Store(), Store(), Store())
            >>> aconf = AgentConfiguration(a, idle, store)
            >>> defs.execute_step(random.Random(0), aconf).agent_state.name
            'busy'
        """
        action = self.sample_action(rng, aconf)
        if action is None:
            return aconf
        return self.execute_action(rng, aconf, action)

    def set_action_weight(
        self,
        agent: Agent,
        action: Action,
        f: _WeightFunction,
    ) -> None:
        """Register a weight function for *agent* on *action*.

        The weight function maps an
        :class:`~pythia.models.yoda.agent_store.AgentStore` to a
        non-negative float representing the propensity of *action* for
        *agent* given the current store.

        Calling this method again for the same ``(agent, action)`` pair
        replaces the previously registered function.

        Args:
            agent: The agent for which the weight function applies.
            action: The action being weighted.
            f: A callable ``AgentStore -> float`` returning a non-negative
                value.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.set_action_weight(a, tick, lambda s: 2.0)
            >>> defs.get_action_weight(a,AgentStore(Store(), Store(), Store()),tick)
            2.0
        """
        self._action_weights.setdefault(agent, {})[action] = f

    def set_default_action_weight(
        self,
        action: Action,
        f: _WeightFunction,
    ) -> None:
        """Register a default weight function for *action*.

        Used as a fallback by :meth:`get_action_weight` when no
        agent-specific weight function has been registered for the given
        ``(agent, action)`` pair.

        Calling this method again for the same *action* replaces the
        previously registered default.

        Args:
            action: The action for which the default weight applies.
            f: A callable ``AgentStore -> float`` returning a non-negative
                value.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> b    = Agent("b")
            >>> tick = Action("tick", lambda rng, s, o: s)
            >>> defs = AgentDefinitions()
            >>> defs.set_default_action_weight(tick, lambda s: 1.0)
            >>> defs.get_action_weight(Agent("b"), AgentStore(Store(), Store(), Store()), tick)
            1.0
        """
        self._default_action_weights[action] = f

    def __repr__(self) -> str:
        return f"AgentDefinitions(agents={self._agents!r})"


def _copy(table: _TransitionTable) -> _TransitionTable:
    """Return a copy of *table* rebuilding every container level.

    Agent, AgentState, and Action instances are value objects and are not
    duplicated; only the dicts and lists that hold them are replaced.
    """
    return {
        agent: {
            source: {
                action: list(outcomes)
                for action, outcomes in actions.items()
            }
            for source, actions in sources.items()
        }
        for agent, sources in table.items()
    }
