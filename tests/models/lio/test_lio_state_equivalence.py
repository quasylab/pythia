"""Cross-representation equivalence tests.

These tests verify that :class:`~pythia.models.lio.lio_state_of_counting_vector.LIOStateOfCountingVector`
and :class:`~pythia.models.lio.lio_state_of_individuals.LIOStateOfIndividuals` produce
the same aggregate outcomes for equivalent populations.

Two states are *equivalent* when they represent the same multiset of agents —
i.e. ``state_a.count(x) == state_b.count(x)`` for every agent ``x``.

The tests are organised into three groups:

* **Static queries** — ``size``, ``count``, and ``fraction`` must agree on
  every equivalent pair.
* **Deterministic-step agreement** — when every transition is deterministic
  (unique action, unique target), both representations must produce identical
  aggregate counts after one or more steps, for every random seed.
* **Statistical agreement** — for stochastic transitions the empirical
  distribution of aggregate outcomes must match: expected counts must converge
  to the same value over many independent trials.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from pythia.models.lio.action import Action
from pythia.models.lio.agent import Agent
from pythia.models.lio.agent_definitions import AgentDefinitions
from pythia.models.lio.lio_state import LIOState
from pythia.models.lio.lio_state_of_counting_vector import LIOStateOfCountingVector
from pythia.models.lio.lio_state_of_individuals import LIOStateOfIndividuals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pair(
    agents: list[Agent],
) -> tuple[LIOStateOfIndividuals, LIOStateOfCountingVector]:
    """Build an equivalent (LIOStateOfIndividuals, LIOStateOfCountingVector) pair."""
    counts: dict[Agent, int] = {}
    for a in agents:
        counts[a] = counts.get(a, 0) + 1
    return LIOStateOfIndividuals(agents), LIOStateOfCountingVector(counts)


def _agg(state: LIOState, agents: list[Agent]) -> dict[Agent, int]:
    """Extract aggregate counts for a set of agents from any LIOState."""
    return {a: state.count(a) for a in agents}


def _uniform_wfun(agent: Agent):  # type: ignore[no-untyped-def]
    return lambda act: 1.0


# ---------------------------------------------------------------------------
# Static-query equivalence
# ---------------------------------------------------------------------------

class TestStaticEquivalence:
    """size, count, and fraction must agree on equivalent states."""

    def test_size_single_type(self) -> None:
        a = Agent("a")
        indiv, counting = _pair([a, a, a])
        assert indiv.size == counting.size

    def test_size_multiple_types(self) -> None:
        a, b, c = Agent("a"), Agent("b"), Agent("c")
        indiv, counting = _pair([a, a, b, c, c, c])
        assert indiv.size == counting.size

    def test_count_present_agent(self) -> None:
        a, b = Agent("a"), Agent("b")
        indiv, counting = _pair([a, a, b])
        assert indiv.count(a) == counting.count(a)
        assert indiv.count(b) == counting.count(b)

    def test_count_absent_agent(self) -> None:
        a, b = Agent("a"), Agent("b")
        indiv, counting = _pair([a, a])
        assert indiv.count(b) == counting.count(b) == 0

    def test_fraction_agrees(self) -> None:
        a, b = Agent("a"), Agent("b")
        indiv, counting = _pair([a, a, b])
        assert indiv.fraction(a) == pytest.approx(counting.fraction(a))
        assert indiv.fraction(b) == pytest.approx(counting.fraction(b))

    def test_fraction_sums_match(self) -> None:
        a, b, c = Agent("a"), Agent("b"), Agent("c")
        indiv, counting = _pair([a, a, b, b, c])
        for agent in (a, b, c):
            assert indiv.fraction(agent) == pytest.approx(counting.fraction(agent))

    def test_fraction_absent_agent_agrees(self) -> None:
        a, b = Agent("a"), Agent("b")
        indiv, counting = _pair([a, a, a])
        assert indiv.fraction(b) == pytest.approx(counting.fraction(b)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Deterministic-step equivalence
# ---------------------------------------------------------------------------

class TestDeterministicStepEquivalence:
    """When transitions are deterministic, both representations must produce
    identical aggregate counts after every step, for every random seed."""

    def _step_agg(
        self,
        state: LIOState,
        agents: list[Agent],
        rng: random.Random,
        defs: AgentDefinitions,
    ) -> dict[Agent, int]:
        return _agg(state.step(rng, _uniform_wfun, defs), agents)

    def test_single_agent_self_loop(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a = defs.get("a")
        defs.add(a, tick, a, 1.0)
        indiv, counting = _pair([a, a, a])
        for seed in range(10):
            r1, r2 = random.Random(seed), random.Random(seed)
            assert (
                _agg(indiv.step(r1, _uniform_wfun, defs), [a])
                == _agg(counting.step(r2, _uniform_wfun, defs), [a])
            )

    def test_full_swap(self) -> None:
        """a→b and b→a: counts swap deterministically each step."""
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)
        indiv, counting = _pair([a, a, b])
        agents = [a, b]
        for seed in range(10):
            r1, r2 = random.Random(seed), random.Random(seed)
            agg_i = _agg(indiv.step(r1, _uniform_wfun, defs), agents)
            agg_c = _agg(counting.step(r2, _uniform_wfun, defs), agents)
            assert agg_i == agg_c

    def test_three_agent_cycle(self) -> None:
        """a→b, b→c, c→a: the triple cycles deterministically."""
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, c, 1.0)
        defs.add(c, tick, a, 1.0)
        indiv, counting = _pair([a, a, b, c, c])
        agents = [a, b, c]
        for seed in range(10):
            r1, r2 = random.Random(seed), random.Random(seed)
            agg_i = _agg(indiv.step(r1, _uniform_wfun, defs), agents)
            agg_c = _agg(counting.step(r2, _uniform_wfun, defs), agents)
            assert agg_i == agg_c

    def test_chained_steps_agree(self) -> None:
        """Multiple deterministic steps: aggregate counts stay in lockstep."""
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)
        agents = [a, b]
        n_steps = 6

        si, sc = _pair([a, a, a, b, b])
        for step in range(n_steps):
            ri, rc = random.Random(step), random.Random(step)
            si = si.step(ri, _uniform_wfun, defs)
            sc = sc.step(rc, _uniform_wfun, defs)
            assert _agg(si, agents) == _agg(sc, agents), f"mismatch at step {step}"

    def test_mixed_with_no_transitions(self) -> None:
        """Agents without transitions stay put; agents with transitions move."""
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, tick, b, 1.0)
        # c has no transitions — it stays as c
        indiv, counting = _pair([a, a, c, c, c])
        agents = [a, b, c]
        for seed in range(10):
            r1, r2 = random.Random(seed), random.Random(seed)
            agg_i = _agg(indiv.step(r1, _uniform_wfun, defs), agents)
            agg_c = _agg(counting.step(r2, _uniform_wfun, defs), agents)
            assert agg_i == agg_c

    def test_size_preserved_after_deterministic_step(self) -> None:
        tick = Action("tick")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, tick, b, 1.0)
        defs.add(b, tick, a, 1.0)
        indiv, counting = _pair([a, a, b])
        r1, r2 = random.Random(0), random.Random(0)
        assert indiv.step(r1, _uniform_wfun, defs).size == counting.step(r2, _uniform_wfun, defs).size


# ---------------------------------------------------------------------------
# Statistical equivalence
# ---------------------------------------------------------------------------

class TestStatisticalEquivalence:
    """For stochastic transitions the empirical expected count must converge
    to the same value for both representations."""

    N_TRIALS = 500
    TOLERANCE = 0.08   # well above ~3σ for N=500

    def _expected_count(
        self,
        state: LIOState,
        target: Agent,
        defs: AgentDefinitions,
        n_trials: int,
    ) -> float:
        total = 0
        for seed in range(n_trials):
            result = state.step(random.Random(seed), _uniform_wfun, defs)
            total += result.count(target)
        return total / n_trials

    def test_expected_count_agrees_single_agent_two_targets(self) -> None:
        """a splits equally to b and c: expected 0.5 each from both representations."""
        t = Action("t")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        indiv, counting = _pair([a])
        for target in (b, c):
            ei = self._expected_count(indiv, target, defs, self.N_TRIALS)
            ec = self._expected_count(counting, target, defs, self.N_TRIALS)
            assert ei == pytest.approx(ec, abs=self.TOLERANCE)

    def test_expected_count_agrees_multiple_copies(self) -> None:
        """With n copies of a, expected count of b after one step is n/2."""
        t = Action("t")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, t, b, 1.0)
        defs.add(a, t, a, 1.0)
        indiv, counting = _pair([a, a, a, a])   # 4 copies
        ei = self._expected_count(indiv, b, defs, self.N_TRIALS)
        ec = self._expected_count(counting, b, defs, self.N_TRIALS)
        assert ei == pytest.approx(ec, abs=self.TOLERANCE)

    def test_expected_count_agrees_mixed_population(self) -> None:
        """Mixed population: a→b and b→a, each with 50/50 split to itself."""
        t = Action("t")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, t, b, 1.0)
        defs.add(a, t, a, 1.0)
        defs.add(b, t, a, 1.0)
        defs.add(b, t, b, 1.0)
        indiv, counting = _pair([a, a, b, b, b])
        for target in (a, b):
            ei = self._expected_count(indiv, target, defs, self.N_TRIALS)
            ec = self._expected_count(counting, target, defs, self.N_TRIALS)
            assert ei == pytest.approx(ec, abs=self.TOLERANCE)

    def test_reachable_outcome_sets_agree(self) -> None:
        """The set of aggregate outcomes reachable in one step must be identical."""
        t = Action("t")
        defs = AgentDefinitions()
        a, b, c = defs.get("a"), defs.get("b"), defs.get("c")
        defs.add(a, t, b, 1.0)
        defs.add(a, t, c, 1.0)
        indiv, counting = _pair([a, a])
        agents = [a, b, c]

        def _reachable(state: LIOState, n: int) -> set[tuple[int, ...]]:
            outcomes = set()
            for seed in range(n):
                nxt = state.step(random.Random(seed), _uniform_wfun, defs)
                outcomes.add(tuple(nxt.count(ag) for ag in agents))
            return outcomes

        assert _reachable(indiv, 200) == _reachable(counting, 200)

    def test_fraction_distribution_agrees(self) -> None:
        """Empirical mean fraction of agent a must agree across representations."""
        t = Action("t")
        defs = AgentDefinitions()
        a, b = defs.get("a"), defs.get("b")
        defs.add(a, t, b, 1.0)
        defs.add(a, t, a, 1.0)
        defs.add(b, t, a, 1.0)
        defs.add(b, t, b, 1.0)
        indiv, counting = _pair([a, a, a, b])

        def _mean_fraction(state: LIOState, n: int) -> float:
            return sum(
                state.step(random.Random(seed), _uniform_wfun, defs).fraction(a)
                for seed in range(n)
            ) / n

        fi = _mean_fraction(indiv, self.N_TRIALS)
        fc = _mean_fraction(counting, self.N_TRIALS)
        assert fi == pytest.approx(fc, abs=self.TOLERANCE)
