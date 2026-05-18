"""Tests for pythia.models.yoda.configuration."""

from __future__ import annotations

import pytest

from pythia.models.yoda.agent import Agent
from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_state import AgentState
from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.configuration import SystemConfiguration
from pythia.models.yoda.store import Store


def _ac(name: str = "a", state: str = "idle") -> AgentConfiguration:
    return AgentConfiguration(
        Agent(name),
        AgentState(state),
        AgentStore(Store(), Store(), Store()),
    )


AC1 = _ac("a", "idle")
AC2 = _ac("b", "busy")
AC3 = _ac("c", "done")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_element(self) -> None:
        cfg = SystemConfiguration([AC1])
        assert cfg.size() == 1

    def test_multiple_elements(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.size() == 3

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            SystemConfiguration([])

    def test_input_iterable_not_mutated(self) -> None:
        lst = [AC1, AC2]
        cfg = SystemConfiguration(lst)
        lst.append(AC3)
        assert cfg.size() == 2


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_setattr_raises(self) -> None:
        cfg = SystemConfiguration([AC1])
        with pytest.raises(AttributeError):
            cfg._elements = ()  # type: ignore[misc]

    def test_elements_property_returns_tuple(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        assert isinstance(cfg.elements, tuple)

    def test_elements_tuple_is_same_object(self) -> None:
        cfg = SystemConfiguration([AC1])
        assert cfg.elements is cfg.elements


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

BUSY_STATE = AgentState("busy")


class TestApply:
    def test_all_elements_transformed(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.apply(lambda ac: AgentConfiguration(ac.agent, BUSY_STATE, ac.agent_store))
        assert all(result.get(i).agent_state == BUSY_STATE for i in range(3))

    def test_size_preserved(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.apply(lambda ac: ac).size() == 3

    def test_returns_new_configuration(self) -> None:
        cfg = SystemConfiguration([AC1])
        assert cfg.apply(lambda ac: ac) is not cfg

    def test_original_unchanged(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        cfg.apply(lambda ac: AgentConfiguration(ac.agent, BUSY_STATE, ac.agent_store))
        assert cfg.get(0).agent_state != BUSY_STATE

    def test_order_preserved(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.apply(lambda ac: ac)
        for i in range(3):
            assert result.get(i) is cfg.get(i)

    def test_identity_function(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        result = cfg.apply(lambda ac: ac)
        assert result.get(0) is AC1
        assert result.get(1) is AC2

    def test_function_receives_each_element(self) -> None:
        received: list = []
        cfg = SystemConfiguration([AC1, AC2, AC3])
        cfg.apply(lambda ac: received.append(ac) or ac)
        assert received == [AC1, AC2, AC3]


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

IDLE  = AgentState("idle")
BUSY  = AgentState("busy")
DONE  = AgentState("done")


def _ac_with_store(name: str, energy: float) -> AgentConfiguration:
    """AgentConfiguration with 'energy' in the state layer."""
    return AgentConfiguration(
        Agent(name),
        AgentState("s"),
        AgentStore(Store(), Store(), Store({"energy": energy})),
    )


class TestEval:
    def test_returns_list(self) -> None:
        cfg = SystemConfiguration([AC1])
        assert isinstance(cfg.eval(lambda s: True, lambda ac: ac.agent_state.name), list)

    def test_matching_agents_included(self) -> None:
        cfg = SystemConfiguration([
            _ac_with_store("a", 5.0),
            _ac_with_store("b", 0.0),
        ])
        result = cfg.eval(
            lambda s: s.get("energy") is not None and s.get("energy") > 0,
            lambda ac: ac.agent.name,
        )
        assert result == ["a"]

    def test_multiple_matching_agents(self) -> None:
        cfg = SystemConfiguration([
            _ac_with_store("a", 5.0),
            _ac_with_store("b", 0.0),
            _ac_with_store("c", 3.0),
        ])
        result = cfg.eval(
            lambda s: s.get("energy") is not None and s.get("energy") > 0,
            lambda ac: ac.agent.name,
        )
        assert result == ["a", "c"]

    def test_no_match_returns_empty(self) -> None:
        cfg = SystemConfiguration([_ac_with_store("a", 0.0)])
        result = cfg.eval(
            lambda s: s.get("energy") is not None and s.get("energy") > 0,
            lambda ac: ac.agent.name,
        )
        assert result == []

    def test_all_match(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.eval(lambda s: True, lambda ac: ac.agent_state.name)
        assert result == ["idle", "busy", "done"]

    def test_function_receives_full_configuration(self) -> None:
        cfg = SystemConfiguration([_ac_with_store("x", 1.0)])
        result = cfg.eval(lambda s: True, lambda ac: (ac.agent.name, ac.agent_state.name))
        assert result == [("x", "s")]

    def test_predicate_uses_agent_store(self) -> None:
        cfg = SystemConfiguration([_ac_with_store("a", 2.0), _ac_with_store("b", 0.0)])
        result = cfg.eval(lambda s: s.has("energy") and s.get("energy") > 1, lambda ac: ac.agent.name)
        assert result == ["a"]

    def test_order_preserved(self) -> None:
        cfg = SystemConfiguration([
            _ac_with_store("c", 0.0),
            _ac_with_store("a", 1.0),
            _ac_with_store("b", 0.0),
        ])
        result = cfg.eval(
            lambda s: s.get("energy") is not None and s.get("energy") > 0,
            lambda ac: ac.agent.name,
        )
        assert result == ["a"]

    def test_generic_return_type(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        result = cfg.eval(lambda s: True, lambda ac: 42)
        assert result == [42, 42]


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_removes_first(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.remove(0)
        assert result.size() == 2
        assert result.get(0) is AC2
        assert result.get(1) is AC3

    def test_removes_middle(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.remove(1)
        assert result.get(0) is AC1
        assert result.get(1) is AC3

    def test_removes_last(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.remove(2)
        assert result.get(0) is AC1
        assert result.get(1) is AC2

    def test_negative_index(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.remove(-1).get(-1) is AC2

    def test_out_of_range_raises(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        with pytest.raises(IndexError):
            cfg.remove(2)

    def test_negative_out_of_range_raises(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        with pytest.raises(IndexError):
            cfg.remove(-3)

    def test_single_element_raises(self) -> None:
        with pytest.raises(ValueError):
            SystemConfiguration([AC1]).remove(0)

    def test_elements_are_same_references(self) -> None:
        # The surviving AgentConfiguration objects must be the exact same instances.
        cfg = SystemConfiguration([AC1, AC2, AC3])
        result = cfg.remove(1)
        assert result.get(0) is AC1
        assert result.get(1) is AC3

    def test_original_unchanged(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        cfg.remove(0)
        assert cfg.size() == 3


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    def test_size_one(self) -> None:
        assert SystemConfiguration([AC1]).size() == 1

    def test_size_three(self) -> None:
        assert SystemConfiguration([AC1, AC2, AC3]).size() == 3


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_first(self) -> None:
        assert SystemConfiguration([AC1, AC2]).get(0) is AC1

    def test_get_last(self) -> None:
        assert SystemConfiguration([AC1, AC2, AC3]).get(2) is AC3

    def test_get_middle(self) -> None:
        assert SystemConfiguration([AC1, AC2, AC3]).get(1) is AC2

    def test_get_negative_index(self) -> None:
        assert SystemConfiguration([AC1, AC2]).get(-1) is AC2

    def test_get_out_of_range_raises(self) -> None:
        with pytest.raises(IndexError):
            SystemConfiguration([AC1]).get(1)

    def test_order_preserved(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.get(0) is AC1
        assert cfg.get(1) is AC2
        assert cfg.get(2) is AC3


# ---------------------------------------------------------------------------
# elements property
# ---------------------------------------------------------------------------

class TestElements:
    def test_returns_all_elements(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        assert cfg.elements == (AC1, AC2)

    def test_single_element(self) -> None:
        cfg = SystemConfiguration([AC1])
        assert cfg.elements == (AC1,)


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------

# Predicates keyed on agent name for clarity.
_is_a = lambda ac: ac.agent == Agent("a")
_is_b = lambda ac: ac.agent == Agent("b")
_is_c = lambda ac: ac.agent == Agent("c")
_never = lambda ac: False
_always = lambda ac: True


class TestExists:
    def test_true_when_other_matches(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_is_a, idx=1) is True

    def test_false_when_no_other_matches(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_never, idx=0) is False

    def test_excluded_index_not_considered(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_is_a, idx=0) is False

    def test_true_when_match_exists_among_others(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_is_b, idx=0) is True

    def test_single_element_with_idx_always_false(self) -> None:
        assert SystemConfiguration([AC1]).exists(_always, idx=0) is False

    def test_multiple_matches_returns_true(self) -> None:
        ac_a2 = _ac("a", "busy")
        cfg = SystemConfiguration([AC1, ac_a2, AC2])
        assert cfg.exists(_is_a, idx=0) is True

    def test_last_index_excluded(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_is_c, idx=2) is False

    def test_no_idx_considers_all(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_is_a) is True

    def test_no_idx_false_when_none_match(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.exists(_never) is False

    def test_no_idx_single_element_true(self) -> None:
        assert SystemConfiguration([AC1]).exists(_always) is True


# ---------------------------------------------------------------------------
# forall
# ---------------------------------------------------------------------------

class TestForall:
    def test_true_when_all_others_match(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.forall(_always, idx=2) is True

    def test_false_when_one_other_fails(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.forall(_is_b, idx=0) is False

    def test_excluded_index_not_considered(self) -> None:
        cfg = SystemConfiguration([AC1, AC2])
        assert cfg.forall(_is_b, idx=0) is True

    def test_single_element_vacuously_true(self) -> None:
        assert SystemConfiguration([AC1]).forall(_never, idx=0) is True

    def test_true_when_all_others_share_state(self) -> None:
        idle1 = _ac("a", "idle")
        idle2 = _ac("b", "idle")
        busy  = _ac("c", "busy")
        is_idle = lambda ac: ac.agent_state == AgentState("idle")
        cfg = SystemConfiguration([busy, idle1, idle2])
        assert cfg.forall(is_idle, idx=0) is True

    def test_false_when_any_other_fails(self) -> None:
        idle1 = _ac("a", "idle")
        busy  = _ac("b", "busy")
        is_idle = lambda ac: ac.agent_state == AgentState("idle")
        cfg = SystemConfiguration([idle1, busy, idle1])
        assert cfg.forall(is_idle, idx=0) is False

    def test_no_idx_considers_all_true(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.forall(_always) is True

    def test_no_idx_considers_all_false(self) -> None:
        cfg = SystemConfiguration([AC1, AC2, AC3])
        assert cfg.forall(_is_a) is False

    def test_no_idx_single_element_true(self) -> None:
        assert SystemConfiguration([AC1]).forall(_always) is True

    def test_no_idx_single_element_false(self) -> None:
        assert SystemConfiguration([AC1]).forall(_never) is False


# ---------------------------------------------------------------------------
# Helpers for mean / min / max tests
# ---------------------------------------------------------------------------

def _ac_v(v: float) -> AgentConfiguration:
    """Agent configuration carrying a single numeric value 'v' in state."""
    return AgentConfiguration(
        Agent("a"),
        AgentState("s"),
        AgentStore(Store(), Store(), Store({"v": v})),
    )


_f = lambda s: s.get("v")          # extract numeric value from store
_is_idle_pred = lambda ac: ac.agent_state == AgentState("idle")


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_all_elements(self) -> None:
        cfg = SystemConfiguration([_ac_v(2.0), _ac_v(4.0), _ac_v(6.0)])
        assert cfg.mean(_f) == pytest.approx(4.0)

    def test_with_idx(self) -> None:
        cfg = SystemConfiguration([_ac_v(2.0), _ac_v(4.0), _ac_v(6.0)])
        assert cfg.mean(_f, idx=2) == pytest.approx(3.0)

    def test_with_pred(self) -> None:
        idle = AgentConfiguration(Agent("a"), AgentState("idle"),
                                  AgentStore(Store(), Store(), Store({"v": 10.0})))
        busy = AgentConfiguration(Agent("b"), AgentState("busy"),
                                  AgentStore(Store(), Store(), Store({"v": 2.0})))
        cfg = SystemConfiguration([idle, busy, idle])
        assert cfg.mean(_f, pred=_is_idle_pred) == pytest.approx(10.0)

    def test_with_pred_and_idx(self) -> None:
        idle = AgentConfiguration(Agent("a"), AgentState("idle"),
                                  AgentStore(Store(), Store(), Store({"v": 10.0})))
        busy = AgentConfiguration(Agent("b"), AgentState("busy"),
                                  AgentStore(Store(), Store(), Store({"v": 2.0})))
        cfg = SystemConfiguration([idle, idle, busy])
        # exclude idx=0 (idle,10), pred=idle → only idx=1 (idle,10) qualifies
        assert cfg.mean(_f, pred=_is_idle_pred, idx=0) == pytest.approx(10.0)

    def test_single_element(self) -> None:
        assert SystemConfiguration([_ac_v(7.0)]).mean(_f) == pytest.approx(7.0)

    def test_no_qualifying_returns_none_by_default(self) -> None:
        cfg = SystemConfiguration([_ac_v(1.0)])
        assert cfg.mean(_f, idx=0) is None

    def test_no_qualifying_returns_custom_default(self) -> None:
        cfg = SystemConfiguration([_ac_v(1.0)])
        assert cfg.mean(_f, idx=0, default=-1.0) == pytest.approx(-1.0)

    def test_no_qualifying_pred_returns_none(self) -> None:
        cfg = SystemConfiguration([_ac_v(1.0)])
        assert cfg.mean(_f, pred=_is_idle_pred) is None


# ---------------------------------------------------------------------------
# min
# ---------------------------------------------------------------------------

class TestMin:
    def test_all_elements(self) -> None:
        cfg = SystemConfiguration([_ac_v(3.0), _ac_v(1.0), _ac_v(2.0)])
        assert cfg.min(_f) == pytest.approx(1.0)

    def test_with_idx_excludes_min(self) -> None:
        cfg = SystemConfiguration([_ac_v(1.0), _ac_v(3.0), _ac_v(5.0)])
        assert cfg.min(_f, idx=0) == pytest.approx(3.0)

    def test_with_pred(self) -> None:
        idle = AgentConfiguration(Agent("a"), AgentState("idle"),
                                  AgentStore(Store(), Store(), Store({"v": 4.0})))
        busy = AgentConfiguration(Agent("b"), AgentState("busy"),
                                  AgentStore(Store(), Store(), Store({"v": 1.0})))
        cfg = SystemConfiguration([idle, busy, idle])
        assert cfg.min(_f, pred=_is_idle_pred) == pytest.approx(4.0)

    def test_single_element(self) -> None:
        assert SystemConfiguration([_ac_v(9.0)]).min(_f) == pytest.approx(9.0)

    def test_no_qualifying_returns_none_by_default(self) -> None:
        assert SystemConfiguration([_ac_v(1.0)]).min(_f, idx=0) is None

    def test_no_qualifying_returns_custom_default(self) -> None:
        assert SystemConfiguration([_ac_v(1.0)]).min(_f, idx=0, default=99.0) == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------

class TestMax:
    def test_all_elements(self) -> None:
        cfg = SystemConfiguration([_ac_v(3.0), _ac_v(7.0), _ac_v(2.0)])
        assert cfg.max(_f) == pytest.approx(7.0)

    def test_with_idx_excludes_max(self) -> None:
        cfg = SystemConfiguration([_ac_v(1.0), _ac_v(3.0), _ac_v(5.0)])
        assert cfg.max(_f, idx=2) == pytest.approx(3.0)

    def test_with_pred(self) -> None:
        idle = AgentConfiguration(Agent("a"), AgentState("idle"),
                                  AgentStore(Store(), Store(), Store({"v": 8.0})))
        busy = AgentConfiguration(Agent("b"), AgentState("busy"),
                                  AgentStore(Store(), Store(), Store({"v": 99.0})))
        cfg = SystemConfiguration([idle, busy, idle])
        assert cfg.max(_f, pred=_is_idle_pred) == pytest.approx(8.0)

    def test_single_element(self) -> None:
        assert SystemConfiguration([_ac_v(5.0)]).max(_f) == pytest.approx(5.0)

    def test_no_qualifying_returns_none_by_default(self) -> None:
        assert SystemConfiguration([_ac_v(1.0)]).max(_f, idx=0) is None

    def test_no_qualifying_returns_custom_default(self) -> None:
        assert SystemConfiguration([_ac_v(1.0)]).max(_f, idx=0, default=42.0) == pytest.approx(42.0)
