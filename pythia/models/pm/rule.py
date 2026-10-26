"""Reaction rule for population models.

A :class:`Rule` encodes a single transition that can be applied to a
:class:`~pythia.models.pm.counting_state.CountingState`.  It is the
building block for stochastic and deterministic population simulators:

- **pre** — the multiset of individuals consumed by the transition.
- **post** — the multiset of individuals produced by the transition.
- **rate_function** — a callable that maps the current state to a
  non-negative rate (firing frequency).

A rule is *enabled* in a state when the state contains at least as many
individuals as ``pre`` *and* the rate is strictly positive.  Applying an
enabled rule removes ``pre`` from the state and adds ``post``.

Example::

    from pythia.models.pm.counting_state import CountingState
    from pythia.models.pm.population import Population
    from pythia.models.pm.rule import Rule

    pop = Population()
    wolf   = pop.register("wolf")
    rabbit = pop.register("rabbit")

    # A wolf eats a rabbit: consumes one rabbit, produces one wolf.
    predation = Rule(
        pre=CountingState({rabbit: 1}),
        post=CountingState({wolf: 1}),
        rate_function=lambda state: (
            0.02
            * state.counting_vector.get(wolf.index, 0)
            * state.counting_vector.get(rabbit.index, 0)
        ),
    )

    state = CountingState({wolf: 10, rabbit: 50})
    print(predation.is_enabled(state))  # True

    rate, next_state = predation.eval(state)
    print(rate)                         # 10.0
    print(next_state().counting_vector) # {0: 11, 1: 49}
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from pythia.models.pm.counting_state import CountingState


@dataclasses.dataclass
class Rule:
    """A transition rule that consumes and produces individuals in a population.

    Attributes:
        pre: Multiset of individuals required (and consumed) by the transition.
        post: Multiset of individuals produced by the transition.
        rate_function: Callable mapping a :class:`~pythia.models.pm.counting_state.CountingState`
            to a non-negative float rate.  A rate of zero means the rule
            cannot fire even if ``pre`` is satisfied.

    Examples:
        >>> from pythia.models.pm.population import Population
        >>> pop = Population()
        >>> wolf   = pop.register("wolf")
        >>> rabbit = pop.register("rabbit")
        >>> rule = Rule(
        ...     pre=CountingState({rabbit: 1}),
        ...     post=CountingState({wolf: 1}),
        ...     rate_function=lambda s: 0.02
        ...         * s.counting_vector.get(wolf.index, 0)
        ...         * s.counting_vector.get(rabbit.index, 0),
        ... )
        >>> state = CountingState({wolf: 10, rabbit: 50})
        >>> rule.is_enabled(state)
        True
    """

    pre: CountingState
    post: CountingState
    rate_function: Callable[[CountingState], float]

    def is_enabled(self, state: CountingState) -> bool:
        """Return ``True`` if the rule can fire in *state*.

        A rule is enabled when *state* contains at least as many individuals
        as :attr:`pre` (i.e. ``state.contains(pre)``) *and* the rate
        evaluated at *state* is strictly positive.

        Args:
            state: The current population state.

        Returns:
            ``True`` when both the precondition and the rate condition are
            satisfied; ``False`` otherwise.

        Examples:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> a = pop.register("a")
            >>> rule = Rule(
            ...     pre=CountingState({a: 1}),
            ...     post=CountingState({}),
            ...     rate_function=lambda s: float(s.counting_vector.get(a.index, 0)),
            ... )
            >>> rule.is_enabled(CountingState({a: 3}))
            True
            >>> rule.is_enabled(CountingState({}))
            False
        """
        return state.contains(self.pre) and self.rate_function(state) > 0

    def apply(self, state: CountingState) -> CountingState:
        """Apply the rule to *state* and return the resulting state.

        Removes :attr:`pre` from *state* and adds :attr:`post`, equivalent
        to ``state.remove(pre).add(post)``.  *state* is not modified.

        Args:
            state: The current population state.  Should satisfy
                :meth:`is_enabled`; behaviour is well-defined even when the
                rule is not enabled (counts are clamped to zero by
                :meth:`~pythia.models.pm.counting_state.CountingState.remove`).

        Returns:
            A new :class:`~pythia.models.pm.counting_state.CountingState`
            representing the population after the transition fires.

        Examples:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> a = pop.register("a")
            >>> b = pop.register("b")
            >>> rule = Rule(
            ...     pre=CountingState({a: 1}),
            ...     post=CountingState({b: 1}),
            ...     rate_function=lambda s: 1.0,
            ... )
            >>> result = rule.apply(CountingState({a: 3}))
            >>> result.counting_vector[a.index]
            2
            >>> result.counting_vector[b.index]
            1
        """
        return state.remove(self.pre).add(self.post)

    def eval(
        self,
        state: CountingState,
    ) -> tuple[float, Callable[[], CountingState]] | None:
        """Evaluate the rule against *state*, returning a rate and a lazy next-state producer.

        If the rule is not enabled — either because *state* does not contain
        :attr:`pre` or because the rate is zero or negative — ``None`` is
        returned.  Otherwise a pair ``(rate, f)`` is returned where:

        - *rate* is the result of applying :attr:`rate_function` to *state*.
        - *f* is a zero-argument callable that, **when invoked**, computes
          and returns ``state.remove(pre).add(post)``.  The computation is
          deferred: calling :meth:`eval` does *not* compute the next state.

        Args:
            state: The current population state.

        Returns:
            ``None`` when the rule is not enabled.  Otherwise a
            ``(float, Callable[[], CountingState])`` pair where the first
            element is the firing rate and the second is a lazy producer of
            the post-transition state.

        Examples:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> a = pop.register("a")
            >>> rule = Rule(
            ...     pre=CountingState({a: 2}),
            ...     post=CountingState({a: 1}),
            ...     rate_function=lambda s: float(s.counting_vector.get(a.index, 0)),
            ... )
            >>> result = rule.eval(CountingState({a: 4}))
            >>> result is not None
            True
            >>> rate, f = result
            >>> rate
            4.0
            >>> f().counting_vector[a.index]
            3
            >>> rule.eval(CountingState({})) is None
            True
        """
        if not state.contains(self.pre):
            return None
        rate = float(self.rate_function(state))
        if rate <= 0:
            return None
        return rate, lambda: state.remove(self.pre).add(self.post)
