"""Tests for pythia.models.pm.rule."""

from __future__ import annotations

import pytest

from pythia.models.pm.counting_state import CountingState
from pythia.models.pm.population import Population
from pythia.models.pm.rule import Rule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pop() -> Population:
    p = Population()
    p.register("wolf")    # index 0
    p.register("rabbit")  # index 1
    p.register("deer")    # index 2
    return p


def constant_rate(value: float) -> object:
    """Return a rate function that always returns *value*."""
    return lambda _: value


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRuleConstruction:
    def test_fields_stored(self, pop: Population) -> None:
        pre  = CountingState({pop[0]: 1})
        post = CountingState({pop[1]: 1})
        rate = constant_rate(1.0)
        rule = Rule(pre=pre, post=post, rate_function=rate)
        assert rule.pre is pre
        assert rule.post is post
        assert rule.rate_function is rate


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestRuleIsEnabled:
    def test_enabled_when_pre_satisfied_and_rate_positive(
        self, pop: Population
    ) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 2}),
            post=CountingState({}),
            rate_function=constant_rate(1.0),
        )
        assert rule.is_enabled(CountingState({pop[0]: 5}))

    def test_disabled_when_pre_not_satisfied(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 3}),
            post=CountingState({}),
            rate_function=constant_rate(1.0),
        )
        assert not rule.is_enabled(CountingState({pop[0]: 2}))

    def test_disabled_when_rate_zero(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=constant_rate(0.0),
        )
        assert not rule.is_enabled(CountingState({pop[0]: 5}))

    def test_disabled_when_rate_negative(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=constant_rate(-1.0),
        )
        assert not rule.is_enabled(CountingState({pop[0]: 5}))

    def test_disabled_when_state_empty(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=constant_rate(1.0),
        )
        assert not rule.is_enabled(CountingState({}))

    def test_enabled_when_pre_is_empty(self, pop: Population) -> None:
        # Empty pre is always satisfied.
        rule = Rule(
            pre=CountingState({}),
            post=CountingState({pop[0]: 1}),
            rate_function=constant_rate(0.5),
        )
        assert rule.is_enabled(CountingState({}))

    def test_rate_function_receives_current_state(self, pop: Population) -> None:
        # Rate depends on state content — only enabled when wolf count >= 10.
        rule = Rule(
            pre=CountingState({pop[1]: 1}),
            post=CountingState({}),
            rate_function=lambda s: float(s.counting_vector.get(pop[0].index, 0) - 10),
        )
        assert not rule.is_enabled(CountingState({pop[0]: 9, pop[1]: 1}))
        assert rule.is_enabled(CountingState({pop[0]: 11, pop[1]: 1}))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

class TestRuleApply:
    def test_pre_consumed(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 2}),
            post=CountingState({}),
            rate_function=constant_rate(1.0),
        )
        result = rule.apply(CountingState({pop[0]: 5}))
        assert result.counting_vector.get(pop[0].index, 0) == 3

    def test_post_added(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({}),
            post=CountingState({pop[1]: 3}),
            rate_function=constant_rate(1.0),
        )
        result = rule.apply(CountingState({pop[1]: 2}))
        assert result.counting_vector[pop[1].index] == 5

    def test_pre_consumed_and_post_added(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[1]: 1}),   # consume 1 rabbit
            post=CountingState({pop[0]: 1}),  # produce 1 wolf
            rate_function=constant_rate(1.0),
        )
        state  = CountingState({pop[0]: 4, pop[1]: 10})
        result = rule.apply(state)
        assert result.counting_vector[pop[0].index] == 5
        assert result.counting_vector[pop[1].index] == 9

    def test_receiver_unchanged(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({pop[1]: 1}),
            rate_function=constant_rate(1.0),
        )
        state = CountingState({pop[0]: 3})
        rule.apply(state)
        assert state.counting_vector[pop[0].index] == 3

    def test_apply_clamped_when_pre_exceeds_state(self, pop: Population) -> None:
        # apply is defined even when the rule is not enabled; remove clamps to 0.
        rule = Rule(
            pre=CountingState({pop[0]: 10}),
            post=CountingState({pop[1]: 1}),
            rate_function=constant_rate(1.0),
        )
        result = rule.apply(CountingState({pop[0]: 3}))
        assert result.counting_vector.get(pop[0].index, 0) == 0
        assert result.counting_vector[pop[1].index] == 1

    def test_apply_equals_remove_then_add(self, pop: Population) -> None:
        pre  = CountingState({pop[0]: 1})
        post = CountingState({pop[1]: 2})
        rule = Rule(pre=pre, post=post, rate_function=constant_rate(1.0))
        state = CountingState({pop[0]: 4, pop[1]: 1})
        assert rule.apply(state) == state.remove(pre).add(post)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

class TestRuleEval:
    def test_returns_none_when_pre_not_satisfied(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 5}),
            post=CountingState({}),
            rate_function=constant_rate(1.0),
        )
        assert rule.eval(CountingState({pop[0]: 3})) is None

    def test_returns_none_when_rate_zero(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=constant_rate(0.0),
        )
        assert rule.eval(CountingState({pop[0]: 5})) is None

    def test_returns_none_when_rate_negative(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=constant_rate(-0.5),
        )
        assert rule.eval(CountingState({pop[0]: 5})) is None

    def test_returns_pair_when_enabled(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({pop[1]: 1}),
            rate_function=constant_rate(2.5),
        )
        result = rule.eval(CountingState({pop[0]: 3}))
        assert result is not None
        assert len(result) == 2

    def test_rate_value_is_correct(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[1]: 1}),
            post=CountingState({}),
            rate_function=lambda s: 0.02
                * s.counting_vector.get(pop[0].index, 0)
                * s.counting_vector.get(pop[1].index, 0),
        )
        state = CountingState({pop[0]: 10, pop[1]: 50})
        result = rule.eval(state)
        assert result is not None
        rate, _ = result
        assert rate == pytest.approx(10.0)

    def test_producer_returns_correct_state(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[1]: 1}),   # consume 1 rabbit
            post=CountingState({pop[0]: 1}),  # produce 1 wolf
            rate_function=constant_rate(1.0),
        )
        state  = CountingState({pop[0]: 4, pop[1]: 10})
        result = rule.eval(state)
        assert result is not None
        _, f = result
        next_state = f()
        assert next_state.counting_vector[pop[0].index] == 5
        assert next_state.counting_vector[pop[1].index] == 9

    def test_producer_is_lazy(self, pop: Population) -> None:
        """Calling eval must not compute the next state immediately."""
        computed = []

        def tracking_rate(s: CountingState) -> float:
            return 1.0

        class _TrackingCountingState(CountingState):
            def remove(self, other: CountingState) -> CountingState:
                computed.append(True)
                return super().remove(other)

        state = _TrackingCountingState({pop[0]: 3})
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({}),
            rate_function=tracking_rate,
        )
        result = rule.eval(state)
        assert result is not None
        assert computed == [], "next state must not be computed before f() is called"
        _, f = result
        f()
        assert computed == [True], "next state must be computed when f() is called"

    def test_producer_callable_returns_counting_state(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({pop[1]: 1}),
            rate_function=constant_rate(1.0),
        )
        _, f = rule.eval(CountingState({pop[0]: 2}))
        assert isinstance(f(), CountingState)

    def test_producer_result_equals_apply(self, pop: Population) -> None:
        rule = Rule(
            pre=CountingState({pop[0]: 1}),
            post=CountingState({pop[1]: 1}),
            rate_function=constant_rate(1.0),
        )
        state = CountingState({pop[0]: 3, pop[1]: 2})
        _, f = rule.eval(state)
        assert f() == rule.apply(state)
