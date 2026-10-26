"""AgentConfiguration — the runtime state of a single agent in a YODA system.

An :class:`AgentConfiguration` binds together the three components that fully
describe an agent at a point in time:

* the :class:`~pythia.models.yoda.agent.Agent` — the participant.
* the :class:`~pythia.models.yoda.agent_state.AgentState` — the current
  behavioural state the agent occupies.
* the :class:`~pythia.models.yoda.agent_store.AgentStore` — the layered
  data store (observations, features, and private state) associated with
  the agent.

Example::

    from pythia.models.yoda.agent import Agent
    from pythia.models.yoda.agent_configuration import AgentConfiguration
    from pythia.models.yoda.agent_state import AgentState
    from pythia.models.yoda.agent_store import AgentStore
    from pythia.models.yoda.store import Store

    a     = Agent("worker")
    state = AgentState("idle")
    store = AgentStore(Store(), Store(), Store({"energy": 10}))

    cfg = AgentConfiguration(a, state, store)
    cfg.agent.name          # 'worker'
    cfg.agent_state.name    # 'idle'
    cfg.agent_store.state["energy"]  # 10
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Iterator

from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.store import Store


class AgentConfiguration:
    """Immutable binding of an agent, its behavioural state, and its data store.

    Args:
        agent: The participant this configuration belongs to.
        agent_state: The behavioural state the agent currently occupies.
        agent_store: The layered data store associated with the agent.

    Examples:
        >>> from pythia.models.yoda.agent import Agent
        >>> from pythia.models.yoda.agent_state import AgentState
        >>> from pythia.models.yoda.agent_store import AgentStore
        >>> from pythia.models.yoda.store import Store
        >>> a     = Agent("worker")
        >>> state = AgentState("idle")
        >>> store = AgentStore(Store(), Store(), Store({"energy": 10}))
        >>> cfg   = AgentConfiguration(a, state, store)
        >>> cfg.agent.name
        'worker'
        >>> cfg.agent_state.name
        'idle'
        >>> cfg.agent_store.state["energy"]
        10
    """

    def __init__(
        self,
        agent: Agent,
        agent_state: AgentState,
        agent_store: AgentStore,
    ) -> None:
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_agent_state", agent_state)
        object.__setattr__(self, "_agent_store", agent_store)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AgentConfiguration is immutable")

    @property
    def agent(self) -> Agent:
        """The participant this configuration belongs to.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> cfg = AgentConfiguration(Agent("a"), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> cfg.agent
            Agent('a')
        """
        return self._agent  # type: ignore[return-value]

    @property
    def agent_state(self) -> AgentState:
        """The behavioural state the agent currently occupies.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> cfg = AgentConfiguration(Agent("a"), AgentState("idle"), AgentStore(Store(), Store(), Store()))
            >>> cfg.agent_state
            AgentState('idle')
        """
        return self._agent_state  # type: ignore[return-value]

    @property
    def agent_store(self) -> AgentStore:
        """The layered data store associated with the agent.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> store = AgentStore(Store(), Store(), Store({"x": 7}))
            >>> cfg   = AgentConfiguration(Agent("a"), AgentState("s"), store)
            >>> cfg.agent_store.state["x"]
            7
        """
        return self._agent_store  # type: ignore[return-value]

    def set_observations(
        self,
        observations: Store
    ) -> AgentConfiguration:
        """Return a new :class:`AgentConfiguration` with updated observations.

        Returns a new :class:`AgentConfiguration` where the :attr:`agent_store`'s
        observations layer has been replaced `òbservations``.  The
        :attr:`agent`, :attr:`agent_state`, and the other store layers
        (state and features) are carried over unchanged.

        Args:
            observations: A ``Store`` that contains the new observations for this agent.

        Returns:
            A new :class:`AgentConfiguration` with updated observations.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a     = Agent("a")
            >>> store = AgentStore(Store(), Store(), Store({"energy": 5}))
            >>> cfg   = AgentConfiguration(a, AgentState("idle"), store)
            >>> cfg2  = cfg.update_observations(
            ...     lambda agent, s: Store({"agent_name": agent.name})
            ... )
            >>> cfg2.agent_store.observations["agent_name"]
            'a'
            >>> cfg2.agent_store.state["energy"]
            5
        """
        return AgentConfiguration(
            agent=self.agent,
            agent_state=self.agent_state,
            agent_store=self.agent_store.set_observations(
                observations
            ),
        )

    def set_state(self, agent_state: AgentState, state: Store) -> AgentConfiguration:
        """Return a new :class:`AgentConfiguration` with updated state.

        Replaces the behavioural state with *agent_state* and the state layer
        of the agent store with *state*.  The :attr:`agent`, observations,
        and features layers are carried over unchanged.

        Args:
            agent_state: The new behavioural state for the agent.
            state: A :class:`~pythia.models.yoda.store.Store` containing
                the new private state variables for this agent.

        Returns:
            A new :class:`AgentConfiguration` with the updated
            :attr:`agent_state` and state layer.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a    = Agent("a")
            >>> store = AgentStore(Store(), Store({"speed": 3}), Store({"x": 1}))
            >>> cfg  = AgentConfiguration(a, AgentState("idle"), store)
            >>> cfg2 = cfg.set_state(AgentState("busy"), Store({"x": 99}))
            >>> cfg2.agent_state.name
            'busy'
            >>> cfg2.agent_store.state["x"]
            99
            >>> cfg2.agent_store.features["speed"]
            3
        """
        return AgentConfiguration(
            agent=self.agent,
            agent_state=agent_state,
            agent_store=self.agent_store.set_state(state),
        )

    def set_features(self, features: Store) -> AgentConfiguration:
        """Return a new :class:`AgentConfiguration` with updated features.

        The :attr:`agent_store`'s features layer is replaced with *features*.
        The :attr:`agent`, :attr:`agent_state`, and the other store layers
        (observations and state) are carried over unchanged.

        Args:
            features: A :class:`~pythia.models.yoda.store.Store` containing
                the new features for this agent.

        Returns:
            A new :class:`AgentConfiguration` with updated features.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a     = Agent("a")
            >>> store = AgentStore(Store(), Store({"speed": 1}), Store({"energy": 5}))
            >>> cfg   = AgentConfiguration(a, AgentState("idle"), store)
            >>> cfg2  = cfg.set_features(Store({"speed": 9}))
            >>> cfg2.agent_store.features["speed"]
            9
            >>> cfg2.agent_store.state["energy"]
            5
        """
        return AgentConfiguration(
            agent=self.agent,
            agent_state=self.agent_state,
            agent_store=self.agent_store.set_features(features),
        )

    def apply_dynamics(
        self,
        rng: random.Random,
        dt: float,
        agent_dynamics: Callable[[Agent], Callable[[random.Random, AgentStore, float], Store]],
    ) -> AgentConfiguration:
        """Return a new :class:`AgentConfiguration` with updated features.

        Calls ``agent_dynamics(self.agent)`` to obtain the dynamics function
        for this agent, then delegates to
        :meth:`~pythia.models.yoda.agent_store.AgentStore.apply_dynamics`,
        producing a new :attr:`agent_store` whose features reflect the
        evolution of the agent over *dt* time units.  The :attr:`agent` and
        :attr:`agent_state` fields are carried over unchanged.

        Args:
            rng: Random number generator forwarded to the dynamics function.
            dt: Time step (strictly positive) forwarded to the dynamics
                function.
            agent_dynamics: A callable that, given an
                :class:`~pythia.models.yoda.agent.Agent`, returns the
                corresponding dynamics function
                ``(rng, AgentStore, dt) -> Store``.

        Returns:
            A new :class:`AgentConfiguration` with an updated
            :attr:`agent_store`.

        Examples:
            >>> import random
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a     = Agent("a")
            >>> dyn   = lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt)
            >>> store = AgentStore(Store(), Store({"x": 1.0}), Store())
            >>> cfg   = AgentConfiguration(a, AgentState("idle"), store)
            >>> cfg2  = cfg.apply_dynamics(random.Random(0), 0.5, lambda agent: dyn)
            >>> cfg2.agent_store.features["x"]
            1.5
            >>> cfg.agent_store.features["x"]
            1.0
        """
        return AgentConfiguration(
            agent=self.agent,
            agent_state=self.agent_state,
            agent_store=self.agent_store.apply_dynamics(
                rng, dt, agent_dynamics(self.agent)
            ),
        )

    @staticmethod
    def get_predicate(
        pred: Callable[[AgentStore], bool],
        agent: Agent | None = None,
        agent_state: AgentState | None = None,
    ) -> Callable[[AgentConfiguration], bool]:
        """Return a predicate over :class:`AgentConfiguration` instances.

        The returned predicate is satisfied by an instance *aconf* if and
        only if:

        * ``pred(aconf.agent_store)`` is ``True``, **and**
        * ``aconf.agent == agent`` when *agent* is not ``None``, **and**
        * ``aconf.agent_state == agent_state`` when *agent_state* is not
          ``None``.

        Args:
            pred: A predicate over :class:`~pythia.models.yoda.agent_store.AgentStore`.
            agent: When not ``None``, the returned predicate also requires
                the configuration's agent to equal this value.
            agent_state: When not ``None``, the returned predicate also
                requires the configuration's behavioural state to equal
                this value.

        Returns:
            A callable ``AgentConfiguration -> bool``.

        Examples:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> a     = Agent("worker")
            >>> idle  = AgentState("idle")
            >>> busy  = AgentState("busy")
            >>> store = AgentStore(Store(), Store(), Store({"energy": 5}))
            >>> has_energy = AgentConfiguration.get_predicate(
            ...     lambda s: s.get("energy", 0) > 0,
            ...     agent=a,
            ...     agent_state=idle,
            ... )
            >>> has_energy(AgentConfiguration(a, idle, store))
            True
            >>> has_energy(AgentConfiguration(a, busy, store))
            False
            >>> has_energy(AgentConfiguration(Agent("other"), idle, store))
            False
        """
        def predicate(aconf: AgentConfiguration) -> bool:
            if not pred(aconf.agent_store):
                return False
            if agent is not None and aconf.agent != agent:
                return False
            if agent_state is not None and aconf.agent_state != agent_state:
                return False
            return True

        return predicate

    def __repr__(self) -> str:
        return (
            f"AgentConfiguration("
            f"agent={self._agent!r}, "  # type: ignore[attr-defined]
            f"agent_state={self._agent_state!r}, "  # type: ignore[attr-defined]
            f"agent_store={self._agent_store!r})"  # type: ignore[attr-defined]
        )


