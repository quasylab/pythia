"""Stochastic population model driven by reaction rules.

:class:`PopulationModel` implements :class:`~pythia.api.model.BaseModel` and
combines a :class:`~pythia.models.pm.population.Population` registry, an
initial :class:`~pythia.models.pm.counting_state.CountingState`, and a set of
:class:`~pythia.models.pm.rule.Rule` objects into a continuous-time stochastic
simulation.

Each call to :meth:`~PopulationModel.step` performs one Gillespie step: all
enabled rules are evaluated, one is selected with probability proportional to
its rate, the resulting state is stored, and the waiting time until the next
event is sampled from an exponential distribution with rate equal to the sum
of all enabled rule rates.  The step returns ``dt = 0.0`` when no rule is
enabled, signalling the runner to terminate.

The current :class:`~pythia.models.pm.counting_state.CountingState` is
stored in ``state.data["pm.counting_state"]`` and is accessible to event
observers throughout the run.

Example::

    from pythia.api import SimulationConfig, SimulationRunner
    from pythia.models.pm.counting_state import CountingState
    from pythia.models.pm.population import Population
    from pythia.models.pm.population_model import PopulationModel
    from pythia.models.pm.rule import Rule

    pop    = Population()
    wolf   = pop.register("wolf")
    rabbit = pop.register("rabbit")

    model = PopulationModel(
        population=pop,
        initial_state={wolf: 10, rabbit: 100},
        rules=[
            Rule(                                   # rabbit reproduces
                pre=CountingState({rabbit: 1}),
                post=CountingState({rabbit: 2}),
                rate_function=lambda s: 0.1 * s.counting_vector.get(rabbit.index, 0),
            ),
            Rule(                                   # wolf eats rabbit
                pre=CountingState({wolf: 1, rabbit: 1}),
                post=CountingState({wolf: 2}),
                rate_function=lambda s: 0.02
                    * s.counting_vector.get(wolf.index, 0)
                    * s.counting_vector.get(rabbit.index, 0),
            ),
            Rule(                                   # wolf dies
                pre=CountingState({wolf: 1}),
                post=CountingState({}),
                rate_function=lambda s: 0.1 * s.counting_vector.get(wolf.index, 0),
            ),
        ],
    )

    config = SimulationConfig(max_time=1000.0, random_seed=42)
    runner = SimulationRunner(model, config)
    final  = runner.run()
    print(final.data["pm.counting_state"].counting_vector)
"""

from __future__ import annotations

import random as _random
import types
from collections.abc import Callable, Mapping
from typing import Any

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState
from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.rule import Rule
from pythia.models.pm.species import Species


class PopulationModel(BaseModel):
    """Stochastic population model driven by a list of reaction rules.

    Each step performs a single Gillespie rule selection and fires the
    chosen rule.  The waiting time until the next event is sampled from
    ``Exp(total_rate)`` and returned as the step duration *dt*.  The run
    terminates early (``dt = 0.0``) when no rule is enabled.

    All constructor-set fields are immutable after construction.  The
    :attr:`random_generator` field reflects the value passed at construction
    time (``None`` by default); the effective RNG used during simulation is
    resolved in :meth:`initialize`.

    Args:
        population: The :class:`~pythia.models.pm.population.Population`
            registry that owns the species referenced by the rules.
        initial_state: Mapping from :class:`~pythia.models.pm.species.Species`
            to positive integer counts representing the population at ``t=0``.
        rules: Ordered list of :class:`~pythia.models.pm.rule.Rule` objects
            that define the possible transitions.
        random_generator: Optional :class:`random.Random` instance used for
            rule selection.  When ``None`` (the default), a new
            :class:`random.Random` is created in :meth:`initialize` seeded
            with ``config.random_seed``.

    Example:
        >>> pop  = Population()
        >>> wolf = pop.register("wolf")
        >>> rule = Rule(
        ...     pre=CountingState({wolf: 1}),
        ...     post=CountingState({}),
        ...     rate_function=lambda s: float(s.counting_vector.get(wolf.index, 0)),
        ... )
        >>> model = PopulationModel(
        ...     population=pop,
        ...     initial_state={wolf: 5},
        ...     rules=[rule],
        ... )
        >>> model.population is pop
        True
        >>> model.random_generator is None
        True
    """

    def __init__(
        self,
        population: Population,
        initial_state: Mapping[Species, int],
        rules: list[Rule],
        random_generator: _random.Random | None = None,
    ) -> None:
        """Initialise the model with its immutable fields."""
        self._population: Population = population
        self._initial_state: Mapping[Species, int] = types.MappingProxyType(
            dict(initial_state)
        )
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._random_generator: _random.Random | None = random_generator
        # Resolved in initialize(); not set by the constructor.
        self._rng: _random.Random | None = None

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def population(self) -> Population:
        """The :class:`~pythia.models.pm.population.Population` registry."""
        return self._population

    @property
    def initial_state(self) -> Mapping[Species, int]:
        """Immutable mapping from species to their initial element counts."""
        return self._initial_state

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Immutable sequence of reaction rules."""
        return self._rules

    @property
    def random_generator(self) -> _random.Random | None:
        """The :class:`random.Random` instance provided at construction, or ``None``."""
        return self._random_generator

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Model identifier."""
        return "pm.population_model"

    def initialize(self, config: SimulationConfig) -> None:
        """Validate configuration and resolve the random generator.

        If :attr:`random_generator` is ``None``, a new :class:`random.Random`
        is created and seeded with ``config.random_seed``.  The resolved
        generator is used internally during :meth:`step` and does not replace
        the value returned by :attr:`random_generator`.

        Args:
            config: The immutable run configuration.

        Raises:
            ConfigurationError: If ``config.max_time`` violates its
                constraints.
        """
        self.validate_config(config)
        self._rng = (
            self._random_generator
            if self._random_generator is not None
            else _random.Random(config.random_seed)
        )

    def setup(self, state: SimulationState) -> None:
        """Write the initial counting state into ``state.data``.

        Sets ``state.data["pm.counting_state"]`` so that measures collected
        at ``t = 0`` reflect the true starting population.

        Args:
            state: The simulation state at ``elapsed_time = 0``.
        """
        state.data["pm.counting_state"] = CountingState(dict(self._initial_state))

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the population by one Gillespie step.

        The current :class:`~pythia.models.pm.counting_state.CountingState`
        is read from ``state.data["pm.counting_state"]`` (pre-populated by
        :meth:`setup`).

        Subsequent calls read the current state from that key, evaluate all
        rules, and — if at least one is enabled — select one with probability
        proportional to its rate and apply it.  The waiting time until the
        next event is sampled from an exponential distribution with rate equal
        to the total propensity.

        Args:
            state: The current simulation state.

        Returns:
            A pair ``(dt, state)`` where *dt* is the Gillespie waiting time
            sampled from ``Exp(total_rate)``, and ``state.data["pm.counting_state"]``
            reflects the post-transition population.  Returns ``(0.0, state)``
            when no rule is enabled.
        """
        if "pm.counting_state" not in state.data:
            state.data["pm.counting_state"] = CountingState(
                dict(self._initial_state)
            )

        counting: CountingState = state.data["pm.counting_state"]

        # Evaluate all rules and collect enabled (rate, producer) pairs.
        enabled: list[tuple[float, Callable[[], CountingState]]] = []
        for rule in self._rules:
            result = rule.eval(counting)
            if result is not None:
                enabled.append(result)

        if not enabled:
            state.is_done = True
            return 0.0, state

        # Gillespie weighted selection: pick a rule proportional to its rate.
        total_rate = sum(rate for rate, _ in enabled)
        threshold = self._rng.random() * total_rate  # type: ignore[union-attr]
        cumulative = 0.0
        chosen: Callable[[], CountingState] = enabled[-1][1]
        for rate, producer in enabled:
            cumulative += rate
            if cumulative >= threshold:
                chosen = producer
                break

        state.data["pm.counting_state"] = chosen()

        # Gillespie waiting time: Exp(total_rate).
        dt = self._rng.expovariate(total_rate)  # type: ignore[union-attr]
        return dt, state

    def finalize(self, state: SimulationState) -> None:
        """No-op finalisation.

        Args:
            state: The final simulation state.
        """
