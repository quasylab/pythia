"""LIOState — abstract base class for LIO system states.

A :class:`LIOState` represents a snapshot of a LIO system population.
Concrete subclasses must implement :meth:`count`, :meth:`fraction`, and
:meth:`step`.

Example::

    from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals
    from pythia.models.lio.agent import Agent

    a = Agent("a")
    state: LIOState = LIOStateOfIndividuals([a])
    state.size      # 1
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions


class LIOState(ABC):
    """Abstract base class for a LIO system state.

    A state holds a population of agents and exposes an aggregate query
    (:meth:`fraction`) and a :meth:`step` method that advances every agent
    by one tick.

    Subclasses must implement :meth:`fraction` and :meth:`step`.
    """

    @abstractmethod
    def fraction(self, agt: Agent) -> float:
        """Return the fraction of agents equal to *agt*.

        Args:
            agt: The agent whose fraction to compute.

        Returns:
            A float in ``[0.0, 1.0]``.
        """

    @abstractmethod
    def step(
        self,
        rng: random.Random,
        wfun: Callable[[Agent], Callable[[Action], float]],
        definitions: AgentDefinitions,
    ) -> LIOState:
        """Return the next state by advancing every agent one tick.

        Args:
            rng: A :class:`random.Random` instance for stochastic selection.
            wfun: A function ``agent -> (action -> float)`` providing
                action-selection weights.  Typically obtained from
                :meth:`~pythia.models.lio.action_weight.ActionWeightFunction.get`.
            definitions: The :class:`~pythia.models.lio.agent_definitions.AgentDefinitions`
                that owns the transition table used to advance each agent.

        Returns:
            A new :class:`LIOState` representing the population after one tick.
        """
