
"""LIOStateOfCountingVector — a compact LIO state stored as agent→count pairs.

Rather than keeping one entry per individual, a
:class:`LIOStateOfCountingVector` stores each distinct
:class:`~pythia.models.lio.agent.Agent` together with the number of copies
present in the population.  This representation is more efficient when the
population is large but the number of distinct agent types is small.

Example::

    from pythia.models.lio.agent import Agent
    from pythia.models.lio.lio_state_of_counting_vector import LIOStateOfCountingVector

    a = Agent("a")
    b = Agent("b")
    state = LIOStateOfCountingVector({a: 3, b: 1})

    state.size          # 4
    state.count(a)      # 3
    state.fraction(a)   # 0.75
"""

from __future__ import annotations

import random
from collections.abc import Callable

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state import LIOState


class LIOStateOfCountingVector(LIOState):
    """An immutable LIO state that stores agents as a counting vector.

    The population is represented as a mapping ``Agent -> int`` where each
    value is the number of copies of that agent present.  Every count must
    be a strictly positive integer; the mapping must be non-empty.

    Args:
        counts: A mapping from :class:`~pythia.models.lio.agent.Agent` to
            a strictly positive integer indicating how many copies of that
            agent are in the state.

    Raises:
        ValueError: If *counts* is empty or if any count is not a strictly
            positive integer.

    Example:
        >>> a = Agent("a")
        >>> b = Agent("b")
        >>> state = LIOStateOfCountingVector({a: 2, b: 1})
        >>> state.size
        3
        >>> state.count(a)
        2
        >>> state.fraction(b)
        0.3333333333333333
    """

    def __init__(self, counts: dict[Agent, int]) -> None:
        if not counts:
            raise ValueError("LIOStateOfCountingVector must contain at least one agent")
        for agent, n in counts.items():
            if not isinstance(n, int) or n <= 0:
                raise ValueError(
                    f"Count for agent {agent!r} must be a strictly positive integer, got {n!r}"
                )
        object.__setattr__(self, "_counts", dict(counts))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LIOStateOfCountingVector is immutable")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def counts(self) -> dict[Agent, int]:
        """A copy of the agent→count mapping.

        Example:
            >>> a = Agent("a")
            >>> state = LIOStateOfCountingVector({a: 3})
            >>> state.counts == {a: 3}
            True
        """
        return dict(self._counts)

    @property
    def size(self) -> int:
        """The total number of agents (sum of all counts).

        Example:
            >>> a = Agent("a")
            >>> b = Agent("b")
            >>> LIOStateOfCountingVector({a: 2, b: 3}).size
            5
        """
        return sum(self._counts.values())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self, agt: Agent) -> int:
        """Return the number of copies of *agt* in the state.

        Args:
            agt: The agent to look up.

        Returns:
            The count for *agt*, or ``0`` if *agt* is not present.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> state = LIOStateOfCountingVector({a: 4})
            >>> state.count(a)
            4
            >>> state.count(b)
            0
        """
        return self._counts.get(agt, 0)

    def fraction(self, agt: Agent) -> float:
        """Return the fraction of agents equal to *agt*.

        Computed as ``count(agt) / size``.

        Args:
            agt: The agent whose fraction to compute.

        Returns:
            A float in ``[0.0, 1.0]``.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> state = LIOStateOfCountingVector({a: 1, b: 3})
            >>> state.fraction(a)
            0.25
            >>> state.fraction(b)
            0.75
        """
        return self.count(agt) / self.size

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        rng: random.Random,
        wfun: Callable[[Agent], Callable[[Action], float]],
        definitions: AgentDefinitions,
    ) -> LIOState:
        """Return the next state by advancing every agent copy one tick.

        For each ``(agent, n)`` pair in the counting vector, *n* independent
        steps are performed via
        :meth:`~pythia.models.lio.agent_definitions.AgentDefinitions.step`,
        all using the same weight function ``wfun(agent)``.  The results are
        accumulated into a new :class:`LIOStateOfCountingVector`.

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
            A new :class:`LIOStateOfCountingVector` whose counts reflect the
            population after one tick.

        Example:
            >>> import random
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t = Action("tick")
            >>> defs.add(a, t, b, 1.0)
            >>> defs.add(b, t, a, 1.0)
            >>> state = LIOStateOfCountingVector({a: 2, b: 1})
            >>> next_state = state.step(random.Random(0), lambda agent: lambda act: 1.0, defs)
            >>> next_state.size
            3
        """
        new_counts: dict[Agent, int] = {}
        for agent, n in self._counts.items():
            wf = wfun(agent)
            for _ in range(n):
                result = definitions.step(agent, wf, rng)
                new_counts[result] = new_counts.get(result, 0) + 1
        return LIOStateOfCountingVector(new_counts)

    def __repr__(self) -> str:
        return f"LIOStateOfCountingVector({self._counts!r})"
