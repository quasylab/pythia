"""ReachabilityEstimator — Monte Carlo reachability probability estimation.

:class:`ReachabilityEstimator` estimates the probability that a simulation
reaches a *final* state within a given deadline, passing only through states
that satisfy a *transient* predicate.

It repeats the simulation *n* times and computes:

* **Probability estimate** — fraction of runs in which the final predicate is
  satisfied before the deadline.
* **Standard deviation** — :math:`\\sqrt{\\hat p \\cdot (1 - \\hat p) / n}`,
  the standard deviation of the Bernoulli estimator.

Example::

    from pythia.api import ReachabilityEstimator, SimulationConfig
    from my_model import MyModel

    model = MyModel()
    config = SimulationConfig(max_time=10.0, random_seed=42)
    estimator = ReachabilityEstimator(model, config)

    prob, std = estimator.estimate(
        n_runs=1000,
        deadline=10.0,
        final_predicate=lambda s: s.data.get("reached", False),
    )
    print(f"P = {prob:.4f}  ±  {std:.4f}")
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from pythia.api.config import SimulationConfig
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState


class ReachabilityEstimator:
    """Monte Carlo estimator for first-passage reachability probability.

    Runs the simulation *n* times.  Each run succeeds when a state satisfying
    *final_predicate* is encountered at or before *deadline*.  If the
    *transient_predicate* is violated at any point, the run is aborted and
    counted as a failure.

    Per-run random seeds are derived deterministically from
    ``config.random_seed`` so that results are reproducible.  Pass
    ``random_seed=None`` in the config for non-deterministic runs.

    Args:
        model: A concrete :class:`~pythia.api.model.BaseModel` instance.
        config: Base configuration.  ``max_time`` is overridden by the
            *deadline* argument of :meth:`estimate`; all other fields
            (``metadata``, ``random_seed``) are inherited by each run.

    Raises:
        ValueError: If *n_runs* is not strictly positive when :meth:`estimate`
            is called.

    Example:
        >>> from pythia.api import BaseModel, SimulationConfig, SimulationState
        >>> class AlwaysReachModel(BaseModel):
        ...     @property
        ...     def name(self): return "always"
        ...     def initialize(self, config): pass
        ...     def step(self, state):
        ...         state.data["done"] = True
        ...         return 0.0, state
        ...     def finalize(self, state): pass
        >>> est = ReachabilityEstimator(
        ...     AlwaysReachModel(),
        ...     SimulationConfig(max_time=1.0, random_seed=0),
        ... )
        >>> p, std = est.estimate(100, 1.0, lambda s: s.data.get("done", False))
        >>> p
        1.0
        >>> std
        0.0
    """

    def __init__(self, model: BaseModel, config: SimulationConfig) -> None:
        self._model = model
        self._config = config

    def estimate(
        self,
        n_runs: int,
        deadline: float,
        final_predicate: Callable[[SimulationState], bool],
        transient_predicate: Callable[[SimulationState], bool] = lambda s: True,
    ) -> tuple[float, float]:
        """Estimate reachability probability over *n_runs* independent runs.

        Each run initialises the model with a unique seed derived from the
        base configuration, then advances the simulation step by step.  A run
        *succeeds* when *final_predicate* is satisfied for the first time; it
        *fails* when the *transient_predicate* is violated or the elapsed time
        reaches *deadline* without the final predicate having been satisfied.

        Args:
            n_runs: Number of independent simulation runs.  Must be strictly
                positive.
            deadline: Maximum simulated time for each run.
            final_predicate: A callable ``SimulationState -> bool`` that
                returns ``True`` when the target state has been reached.
            transient_predicate: An optional callable ``SimulationState ->
                bool`` that must remain ``True`` throughout the run.  If it
                returns ``False`` at any state (including the initial one),
                the run is immediately aborted as a failure.  Defaults to
                ``lambda s: True`` (no constraint).

        Returns:
            A pair ``(probability, std_dev)`` where *probability* is the
            fraction of successful runs and *std_dev* is
            :math:`\\sqrt{\\hat p (1 - \\hat p) / n}`.

        Raises:
            ValueError: If *n_runs* is not strictly positive.

        Example:
            >>> from pythia.api import BaseModel, SimulationConfig, SimulationState
            >>> class NeverReachModel(BaseModel):
            ...     @property
            ...     def name(self): return "never"
            ...     def initialize(self, config): pass
            ...     def step(self, state): return 0.0, state
            ...     def finalize(self, state): pass
            >>> est = ReachabilityEstimator(
            ...     NeverReachModel(),
            ...     SimulationConfig(max_time=1.0, random_seed=0),
            ... )
            >>> p, std = est.estimate(100, 1.0, lambda s: False)
            >>> p
            0.0
            >>> std
            0.0
        """
        if n_runs <= 0:
            raise ValueError(
                f"n_runs must be strictly positive, got {n_runs!r}"
            )

        master_rng = random.Random(self._config.random_seed)
        n_success = 0

        for _ in range(n_runs):
            run_seed = master_rng.randint(0, 2**31 - 1)
            run_config = SimulationConfig(
                max_time=deadline,
                random_seed=run_seed,
                metadata=dict(self._config.metadata),
            )
            self._model.initialize(run_config)
            state = SimulationState()

            if self._run(state, deadline, final_predicate, transient_predicate):
                n_success += 1

        p = n_success / n_runs
        std = math.sqrt(p * (1.0 - p) / n_runs)
        return p, std

    def _run(
        self,
        state: SimulationState,
        deadline: float,
        final_predicate: Callable[[SimulationState], bool],
        transient_predicate: Callable[[SimulationState], bool],
    ) -> bool:
        """Execute one run and return ``True`` if the final state is reached."""
        if final_predicate(state):
            return True
        if not transient_predicate(state):
            return False

        while state.elapsed_time < deadline:
            dt, state = self._model.step(state)
            if dt <= 0.0:
                break
            state.advance(dt)
            if final_predicate(state):
                return True
            if not transient_predicate(state):
                break

        return False
