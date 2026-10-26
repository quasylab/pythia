"""ActionWeightFunction — per-agent, per-action weight functions over configurations.

An :class:`ActionWeightFunction` stores, for each :class:`~pythia.models.lio.agent.Agent`
and each of its :class:`~pythia.models.lio.action.Action`\\ s, a callable that maps a
:class:`~pythia.models.lio.lio_state.LIOState` to a ``float`` weight.

Example::

    from pythia.models.lio.action import Action
    from pythia.models.lio.agent import Agent
    from pythia.models.lio.action_weight import ActionWeightFunction
    from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals

    a = Agent("a")
    move = Action("move")

    awf = ActionWeightFunction()
    awf.set(a, move, lambda cfg: cfg.fraction(a))
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.lio_state import LIOState

# Type alias: a function from a LIOState to a float weight.
WeightFunction = Callable[[LIOState], float]


class ActionWeightFunction:
    """Maps (agent, action) pairs to weight functions over configurations.

    Attributes:
        action_weights: Nested mapping
            ``Agent -> Action -> (LIOState -> float)``.
            Accessing a missing agent key returns an empty ``dict``
            automatically (the outer mapping uses :class:`~collections.defaultdict`).
        default_weights: Mapping ``Action -> (LIOState -> float)`` used
            as fallback weights that apply to all agents unless overridden in
            :attr:`action_weights`.

    Examples:
        >>> from pythia.models.lio.agent import Agent
        >>> from pythia.models.lio.action import Action
        >>> from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
        >>> a = Agent("a")
        >>> move = Action("move")
        >>> awf = ActionWeightFunction()
        >>> awf.set(a, move, lambda cfg: 1.0)
        >>> cfg = LIOStateOfIndividuals([a])
        >>> awf.get(cfg)(a)(move)
        1.0
    """

    def __init__(self) -> None:
        """Initialise with empty weight tables."""
        self.action_weights: dict[Agent, dict[Action, WeightFunction]] = defaultdict(dict)
        self.default_weights: dict[Action, WeightFunction] = {}

    def set(self, agent: Agent, act: Action, wfun: WeightFunction) -> None:
        """Register or replace the weight function for (*agent*, *act*).

        After this call ``action_weights[agent][act]`` is *wfun*.

        Args:
            agent: The agent whose action weight is being defined.
            act: The action to associate the weight function with.
            wfun: A callable that accepts a
                :class:`~pythia.models.lio.lio_state.LIOState`
                and returns a ``float`` representing the weight of *act* for
                *agent* in that configuration.

        Examples:
            >>> from pythia.models.lio.agent import Agent
            >>> from pythia.models.lio.action import Action
            >>> from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
            >>> a = Agent("a")
            >>> stay = Action("stay")
            >>> awf = ActionWeightFunction()
            >>> awf.set(a, stay, lambda cfg: cfg.fraction(a))
            >>> cfg = LIOStateOfIndividuals([a, a, Agent("b")])
            >>> awf.get(cfg)(a)(stay)
            0.6666666666666666
        """
        self.action_weights[agent][act] = wfun

    def set_default(self, act: Action, wfun: WeightFunction) -> None:
        """Register or replace the default weight function for *act*.

        After this call ``default_weights[act]`` is *wfun*.

        Args:
            act: The action to associate the default weight function with.
            wfun: A callable that accepts a
                :class:`~pythia.models.lio.lio_state.LIOState`
                and returns a ``float`` representing the default weight of *act*
                in that configuration.

        Examples:
            >>> from pythia.models.lio.agent import Agent
            >>> from pythia.models.lio.action import Action
            >>> from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
            >>> move = Action("move")
            >>> awf = ActionWeightFunction()
            >>> awf.set_default(move, lambda cfg: 0.5)
            >>> a = Agent("a")
            >>> cfg = LIOStateOfIndividuals([a])
            >>> awf.default_weights[move](cfg)
            0.5
        """
        self.default_weights[act] = wfun

    def get(self, conf: LIOState) -> Callable[[Agent], Callable[[Action], float]]:
        """Return a curried weight lookup bound to *conf*.

        The returned function has the signature
        ``agent -> (action -> float)`` and resolves weights with the
        following priority:

        1. ``action_weights[agent][act](conf)`` — agent-specific weight.
        2. ``default_weights[act](conf)`` — action-level default weight.
        3. ``0.0`` — when neither mapping contains *act*.

        Args:
            conf: The configuration against which all weight functions are
                evaluated.

        Returns:
            A callable ``f`` such that ``f(agent)(act)`` yields the weight
            of *act* for *agent* in configuration *conf*.

        Examples:
            >>> from pythia.models.lio.agent import Agent
            >>> from pythia.models.lio.action import Action
            >>> from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
            >>> a, b = Agent("a"), Agent("b")
            >>> move, stay = Action("move"), Action("stay")
            >>> awf = ActionWeightFunction()
            >>> awf.set(a, move, lambda cfg: cfg.fraction(a))
            >>> awf.set_default(stay, lambda cfg: 0.5)
            >>> cfg = LIOStateOfIndividuals([a, a, b])
            >>> f = awf.get(cfg)
            >>> f(a)(move)   # agent-specific
            0.6666666666666666
            >>> f(b)(stay)   # default weight
            0.5
            >>> f(a)(stay)   # default weight (no agent-specific entry)
            0.5
            >>> f(b)(move)   # no entry at all
            0.0
        """
        def by_agent(agent: Agent) -> Callable[[Action], float]:
            def by_action(act: Action) -> float:
                agent_map = self.action_weights.get(agent, {})
                if act in agent_map:
                    return agent_map[act](conf)
                if act in self.default_weights:
                    return self.default_weights[act](conf)
                return 0.0
            return by_action
        return by_agent
