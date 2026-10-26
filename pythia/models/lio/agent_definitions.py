"""AgentDefinitions — registry and transition table for LIO agents.

:class:`AgentDefinitions` has two responsibilities:

1. **Agent registry** — it acts as a factory and cache for
   :class:`~pythia.models.lio.agent.Agent` instances.  Requesting the same
   name twice always returns the *same* object.

2. **Transition table** — it owns the transition weights for every agent it
   manages.  The methods :meth:`add`, :meth:`weight`, :meth:`total_weight`,
   :meth:`execute`, :meth:`select_action`, and :meth:`step` all take a
   *source* agent as their first parameter.

Example::

    from pythia.models.lio.action import Action
    from pythia.models.lio.agent_definitions import AgentDefinitions

    defs = AgentDefinitions()
    a = defs.get("a")
    b = defs.get("b")
    tick = Action("tick")

    defs.add(a, tick, b, 1.0)
    defs.add(b, tick, a, 1.0)

    assert defs.get("a") is a      # same instance
    assert defs.weight(a, tick, b) == 1.0
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent


class AgentDefinitions:
    """Registry and transition table for :class:`~pythia.models.lio.agent.Agent` instances.

    Each name maps to exactly one :class:`~pythia.models.lio.agent.Agent`
    instance (:meth:`get`).  Transitions between agents are accumulated via
    :meth:`add` and queried or executed via :meth:`weight`,
    :meth:`total_weight`, :meth:`execute`, :meth:`select_action`, and
    :meth:`step`.

    Examples:
        >>> from pythia.models.lio.action import Action
        >>> defs = AgentDefinitions()
        >>> a = defs.get("a")
        >>> b = defs.get("b")
        >>> tick = Action("tick")
        >>> defs.add(a, tick, b, 1.0)
        >>> defs.weight(a, tick, b)
        1.0
        >>> defs.get("a") is a
        True
    """

    def __init__(self) -> None:
        """Initialise an empty registry and an empty transition table."""
        self._agents: dict[str, Agent] = {}
        self._transitions: dict[Agent, dict[Action, dict[Agent, float]]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def get(self, name: str) -> Agent:
        """Return the :class:`~pythia.models.lio.agent.Agent` registered under *name*.

        If no agent with *name* exists yet, one is created, stored, and
        returned.  Subsequent calls with the same *name* always return the
        same instance.

        Args:
            name: The agent name.  Must be a non-empty string (enforced by
                :class:`~pythia.models.lio.agent.Agent` itself).

        Returns:
            The unique :class:`~pythia.models.lio.agent.Agent` for *name*.

        Raises:
            ValueError: If *name* is empty.

        Examples:
            >>> defs = AgentDefinitions()
            >>> x = defs.get("x")
            >>> defs.get("x") is x
            True
        """
        if name not in self._agents:
            self._agents[name] = Agent(name)
        return self._agents[name]

    # ------------------------------------------------------------------
    # Transition table
    # ------------------------------------------------------------------

    def add(self, source: Agent, action: Action, target: Agent, w: float) -> None:
        """Accumulate weight *w* on the transition ``source --action--> target``.

        If no transition exists yet for the given ``(source, action, target)``
        triple, it is created with weight *w*.  Otherwise *w* is added to the
        existing weight.

        Args:
            source: The agent initiating the transition.
            action: The action labelling the transition.
            target: The agent reached after the transition.
            w: A strictly positive weight to add.

        Raises:
            ValueError: If *w* is not strictly positive.

        Examples:
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t = Action("tick")
            >>> defs.add(a, t, b, 1.0)
            >>> defs.weight(a, t, b)
            1.0
            >>> defs.add(a, t, b, 0.5)
            >>> defs.weight(a, t, b)
            1.5
        """
        if w <= 0.0:
            raise ValueError(f"Transition weight must be strictly positive, got {w!r}")
        inner = self._transitions[source].setdefault(action, {})
        inner[target] = inner.get(target, 0.0) + w

    def weight(self, source: Agent, action: Action, target: Agent) -> float:
        """Return the accumulated weight of ``source --action--> target``.

        Args:
            source: The agent initiating the transition.
            action: The action labelling the transition.
            target: The agent reached after the transition.

        Returns:
            The cumulative weight, or ``0.0`` if no such transition is defined.

        Examples:
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t = Action("tick")
            >>> defs.weight(a, t, b)
            0.0
            >>> defs.add(a, t, b, 2.0)
            >>> defs.weight(a, t, b)
            2.0
        """
        return self._transitions.get(source, {}).get(action, {}).get(target, 0.0)

    def total_weight(self, source: Agent, action: Action) -> float:
        """Return the sum of all weights on transitions from *source* labelled *action*.

        Args:
            source: The agent initiating the transitions.
            action: The action whose outgoing weights are summed.

        Returns:
            The sum of weights over all reachable targets, or ``0.0`` if no
            transitions are defined for ``(source, action)``.

        Examples:
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
            >>> t = Action("tick")
            >>> defs.total_weight(a, t)
            0.0
            >>> defs.add(a, t, b, 1.0)
            >>> defs.add(a, t, c, 2.5)
            >>> defs.total_weight(a, t)
            3.5
        """
        return sum(self._transitions.get(source, {}).get(action, {}).values())

    def execute(self, source: Agent, action: Action, rng: random.Random) -> Agent:
        """Randomly select a target for ``source --action-->`` using weighted sampling.

        Args:
            source: The agent initiating the transition.
            action: The action to execute.
            rng: A :class:`random.Random` instance used for the selection.

        Returns:
            The selected target :class:`~pythia.models.lio.agent.Agent`.

        Raises:
            ValueError: If no transitions are defined for ``(source, action)``.

        Examples:
            >>> import random
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
            >>> t = Action("tick")
            >>> defs.add(a, t, b, 1.0)
            >>> defs.add(a, t, c, 3.0)
            >>> defs.execute(a, t, random.Random(0)) in (b, c)
            True
        """
        inner = self._transitions.get(source, {}).get(action)
        if not inner:
            raise ValueError(f"No transitions defined for action {action!r}")
        targets = list(inner.keys())
        weights = [inner[tgt] for tgt in targets]
        return rng.choices(targets, weights=weights, k=1)[0]

    def select_action(
        self,
        source: Agent,
        wf: Callable[[Action], float],
        rng: random.Random,
    ) -> Action:
        """Randomly select an action for *source* weighted by *wf*.

        Args:
            source: The agent selecting an action.
            wf: A function mapping each available action to a non-negative
                float weight.  Actions for which *wf* returns ``0.0`` are
                never selected.
            rng: A :class:`random.Random` instance used for the selection.

        Returns:
            The selected :class:`~pythia.models.lio.action.Action`.

        Raises:
            ValueError: If *source* has no defined transitions, or if *wf*
                returns ``0.0`` for every available action.

        Examples:
            >>> import random
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t1, t2 = Action("t1"), Action("t2")
            >>> defs.add(a, t1, b, 1.0)
            >>> defs.add(a, t2, b, 1.0)
            >>> defs.select_action(a, lambda act: 1.0, random.Random(0)) in (t1, t2)
            True
        """
        source_transitions = self._transitions.get(source, {})
        if not source_transitions:
            raise ValueError("Agent has no defined transitions")
        actions = list(source_transitions.keys())
        weights = [wf(act) for act in actions]
        if sum(weights) == 0.0:
            raise ValueError("wf returns 0.0 for every available action")
        return rng.choices(actions, weights=weights, k=1)[0]

    def step(
        self,
        source: Agent,
        wf: Callable[[Action], float],
        rng: random.Random,
    ) -> Agent:
        """Select an action for *source* and return the resulting next agent.

        Selects an action via :meth:`select_action` (weighted by *wf*), then
        returns the target agent chosen by :meth:`execute`.  If *source* has
        no defined transitions, *source* itself is returned.

        Args:
            source: The agent performing the step.
            wf: A function mapping each available action to a non-negative
                float weight used to select the action.
            rng: A :class:`random.Random` instance used for both selections.

        Returns:
            The next :class:`~pythia.models.lio.agent.Agent`, or *source* if
            no transitions are defined for it.

        Examples:
            >>> import random
            >>> from pythia.models.lio.action import Action
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> t = Action("tick")
            >>> defs.add(a, t, b, 1.0)
            >>> defs.step(a, lambda act: 1.0, random.Random(0)) == b
            True
            >>> defs.step(b, lambda act: 1.0, random.Random(0)) == b
            True
        """
        if not self._transitions.get(source):
            return source
        action = self.select_action(source, wf, rng)
        return self.execute(source, action, rng)
