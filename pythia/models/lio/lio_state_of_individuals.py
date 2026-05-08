"""LIOStateOfIndividuals — an immutable array of agents in a LIO system.

A :class:`LIOStateOfIndividuals` represents a snapshot of the system population:
an ordered, immutable sequence of :class:`~pythia.models.lio.agent.Agent`
instances.  It provides access by index, size, and count/fraction queries.

Example::

    from pythia.models.lio.agent import Agent
    from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals

    a = Agent("a")
    b = Agent("b")
    cfg = LIOStateOfIndividuals([a, a, b])

    cfg.size          # 3
    cfg.agent(0)      # Agent('a')
    cfg.count(a)      # 2
    cfg.fraction(a)   # 0.666...
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state import LIOState


class LIOStateOfIndividuals(LIOState):
    """An immutable, ordered collection of agents.

    Args:
        agents: The agents that form the state, in order.

    Raises:
        ValueError: If *agents* is empty.

    Example:
        >>> a = Agent("a")
        >>> b = Agent("b")
        >>> cfg = LIOStateOfIndividuals([a, a, b])
        >>> cfg.size
        3
        >>> cfg.count(a)
        2
        >>> cfg.fraction(b)
        0.3333333333333333
    """

    def __init__(self, agents: Iterable[Agent]) -> None:
        agents_tuple: tuple[Agent, ...] = tuple(agents)
        if not agents_tuple:
            raise ValueError("LIOStateOfIndividuals must contain at least one agent")
        object.__setattr__(self, "_agents", agents_tuple)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LIOStateOfIndividuals is immutable")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def agent(self, i: int) -> Agent:
        """Return the agent at position *i*.

        Args:
            i: Zero-based index into the state.

        Returns:
            The :class:`~pythia.models.lio.agent.Agent` at position *i*.

        Raises:
            IndexError: If *i* is out of range.

        Example:
            >>> cfg = LIOStateOfIndividuals([Agent("a"), Agent("b")])
            >>> cfg.agent(0)
            Agent('a')
            >>> cfg.agent(1)
            Agent('b')
        """
        return self._agents[i]

    @property
    def size(self) -> int:
        """The number of agents in the state.

        Example:
            >>> LIOStateOfIndividuals([Agent("a"), Agent("a"), Agent("b")]).size
            3
        """
        return len(self._agents)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self, agt: Agent) -> int:
        """Return the number of occurrences of *agt* in the state.

        Args:
            agt: The agent to count.

        Returns:
            The number of agents in the state equal to *agt*.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> cfg = LIOStateOfIndividuals([a, a, b])
            >>> cfg.count(a)
            2
            >>> cfg.count(b)
            1
        """
        return sum(1 for a in self._agents if a == agt)

    def fraction(self, agt: Agent) -> float:
        """Return the fraction of agents equal to *agt*.

        Computed as ``count(agt) / size``.

        Args:
            agt: The agent whose fraction to compute.

        Returns:
            A float in ``[0.0, 1.0]``.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> cfg = LIOStateOfIndividuals([a, a, b])
            >>> cfg.fraction(a)
            0.6666666666666666
            >>> cfg.fraction(b)
            0.3333333333333333
        """
        return self.count(agt) / self.size

    def step(
        self,
        rng: random.Random,
        wfun: Callable[[Agent], Callable[[Action], float]],
        definitions: AgentDefinitions,
    ) -> LIOState:
        """Return the next state by advancing every agent one step.

        Each agent independently selects and executes an action via
        :meth:`~pythia.models.lio.agent_definitions.AgentDefinitions.step`.
        The new state is assembled from the resulting agents in the same
        order as the original.

        Args:
            rng: A :class:`random.Random` instance forwarded to each
                :meth:`~pythia.models.lio.agent_definitions.AgentDefinitions.step`
                call.
            wfun: A function ``agent -> (action -> float)`` that provides the
                action-selection weights for each agent.  Typically obtained
                from :meth:`~pythia.models.lio.action_weight.ActionWeightFunction.get`.
            definitions: The :class:`~pythia.models.lio.agent_definitions.AgentDefinitions`
                that owns the transition table used to advance each agent.

        Returns:
            A new :class:`LIOStateOfIndividuals` of the same size whose *i*-th
            agent is ``definitions.step(agents[i], wfun(agents[i]), rng)``.

        Example:
            >>> import random
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t = Action("tick")
            >>> defs.add(a, t, b, 1.0)
            >>> defs.add(b, t, a, 1.0)
            >>> cfg = LIOStateOfIndividuals([a, b])
            >>> next_cfg = cfg.step(random.Random(0), lambda agent: lambda act: 1.0, defs)
            >>> next_cfg.size
            2
        """
        return LIOStateOfIndividuals(
            [definitions.step(agent, wfun(agent), rng) for agent in self._agents]
        )

    def __repr__(self) -> str:
        return f"LIOStateOfIndividuals({list(self._agents)!r})"
