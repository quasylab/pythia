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

    def estimate_over_time(
        self,
        n_runs: int,
        deadline: float,
        dt: float,
        final_predicate: Callable[[SimulationState], bool],
        transient_predicate: Callable[[SimulationState], bool] = lambda s: True,
    ) -> tuple[tuple[float, float], ...]:
        """Estimate reachability probability at every ``k * dt`` time step.

        Runs the simulation *n_runs* times.  Each run records the *first
        passage time* — the earliest simulated time at which *final_predicate*
        is satisfied.  For every time bucket ``i * dt`` (``i`` from 1 to
        ``k``) a counter is incremented when the first passage time of that
        run is at most ``i * dt``.

        The number of buckets is:

        .. math::

            k = \\min\\{k \\in \\mathbb{N} : \\text{deadline} \\leq k \\cdot dt\\}

        (i.e. ``k = ceil(deadline / dt)``), so the last bucket always covers
        the full *deadline*.

        Args:
            n_runs: Number of independent simulation runs.  Must be strictly
                positive.
            deadline: Maximum simulated time for each run.
            dt: Width of each time bucket.  Must be strictly positive.
            final_predicate: A callable ``SimulationState -> bool`` that
                returns ``True`` when the target state has been reached.
            transient_predicate: An optional callable ``SimulationState ->
                bool`` that must remain ``True`` throughout the run.  Defaults
                to ``lambda s: True`` (no constraint).

        Returns:
            A tuple of ``k`` pairs ``(probability, std_dev)`` where element
            ``i`` (0-indexed) corresponds to time ``(i + 1) * dt``:

            * *probability* — fraction of runs that reached the final state
              by time ``(i + 1) * dt``.
            * *std_dev* — :math:`\\sqrt{\\hat p_i (1 - \\hat p_i) / n}`.

        Raises:
            ValueError: If *n_runs* is not strictly positive or *dt* is not
                strictly positive.

        Example:
            >>> from pythia.api import BaseModel, SimulationConfig, SimulationState
            >>> class StepModel(BaseModel):
            ...     @property
            ...     def name(self): return "step"
            ...     def initialize(self, config): pass
            ...     def step(self, state):
            ...         state.data["done"] = True
            ...         return 1.0, state
            ...     def finalize(self, state): pass
            >>> est = ReachabilityEstimator(
            ...     StepModel(),
            ...     SimulationConfig(max_time=3.0, random_seed=0),
            ... )
            >>> results = est.estimate_over_time(
            ...     100, 3.0, 1.0, lambda s: s.data.get("done", False)
            ... )
            >>> len(results)
            3
            >>> results[0][0]   # all runs reach the goal within 1 step
            1.0
        """
        if n_runs <= 0:
            raise ValueError(
                f"n_runs must be strictly positive, got {n_runs!r}"
            )
        if dt <= 0.0:
            raise ValueError(
                f"dt must be strictly positive, got {dt!r}"
            )

        k = math.ceil(deadline / dt)
        counts: list[int] = [0] * k

        master_rng = random.Random(self._config.random_seed)

        for _ in range(n_runs):
            run_seed = master_rng.randint(0, 2**31 - 1)
            run_config = SimulationConfig(
                max_time=deadline,
                random_seed=run_seed,
                metadata=dict(self._config.metadata),
            )
            self._model.initialize(run_config)
            state = SimulationState()

            passage_time = self._run_timed(
                state, deadline, final_predicate, transient_predicate
            )

            if passage_time is not None:
                # Increment every bucket whose upper bound >= passage_time.
                # Bucket i (0-indexed) covers time (i+1)*dt.
                # The first such bucket is ceil(passage_time/dt) - 1, clamped
                # to 0.  A small epsilon handles exact floating-point equality
                # (e.g. passage_time == dt).
                first_bucket = max(
                    0, math.ceil(passage_time / dt - 1e-9) - 1
                )
                for i in range(first_bucket, k):
                    counts[i] += 1

        result: list[tuple[float, float]] = []
        for c in counts:
            p = c / n_runs
            std = math.sqrt(p * (1.0 - p) / n_runs)
            result.append((p, std))

        return tuple(result)

    def _run_timed(
        self,
        state: SimulationState,
        deadline: float,
        final_predicate: Callable[[SimulationState], bool],
        transient_predicate: Callable[[SimulationState], bool],
    ) -> float | None:
        """Execute one run and return the first passage time, or ``None``.

        Returns the value of ``state.elapsed_time`` the first time
        *final_predicate* is satisfied, or ``None`` if the run ends without
        reaching the goal (transient predicate violated, model halted, or
        deadline exceeded).
        """
        if final_predicate(state):
            return state.elapsed_time
        if not transient_predicate(state):
            return None

        while state.elapsed_time < deadline:
            dt_step, state = self._model.step(state)
            if dt_step <= 0.0:
                break
            state.advance(dt_step)
            if final_predicate(state):
                return state.elapsed_time
            if not transient_predicate(state):
                break

        return None

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
