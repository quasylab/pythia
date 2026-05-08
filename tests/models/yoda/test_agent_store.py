"""Tests for pythia.models.yoda.agent_store."""

from __future__ import annotations

import random

import pytest

from pythia.models.yoda.agent_store import AgentStore
from pythia.models.yoda.store import Store


RNG = random.Random(0)


def _make(
    state: dict | None = None,
    features: dict | None = None,
    observations: dict | None = None,
) -> AgentStore:
    return AgentStore(
        state=Store(state or {}),
        features=Store(features or {}),
        observations=Store(observations or {}),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_stores_assigned(self) -> None:
        st = Store({"x": 1})
        ft = Store({"y": 2})
        ob = Store({"z": 3})
        s = AgentStore(state=st, features=ft, observations=ob)
        assert s.state is st
        assert s.features is ft
        assert s.observations is ob

    def test_all_empty_stores(self) -> None:
        s = _make()
        assert len(s.state) == 0
        assert len(s.features) == 0
        assert len(s.observations) == 0


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_setattr_raises(self) -> None:
        s = _make(state={"x": 1})
        with pytest.raises(AttributeError):
            s.state = Store()  # type: ignore[misc]

    def test_features_not_reassignable(self) -> None:
        s = _make(features={"x": 1})
        with pytest.raises(AttributeError):
            s.features = Store()  # type: ignore[misc]

    def test_observations_not_reassignable(self) -> None:
        s = _make(observations={"x": 1})
        with pytest.raises(AttributeError):
            s.observations = Store()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get — priority order
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_none_when_absent_everywhere(self) -> None:
        assert _make().get("x") is None

    def test_returns_from_state(self) -> None:
        s = _make(state={"x": 1})
        assert s.get("x") == 1

    def test_returns_from_features_when_absent_in_state(self) -> None:
        s = _make(features={"x": 2})
        assert s.get("x") == 2

    def test_returns_from_observations_when_absent_in_state_and_features(self) -> None:
        s = _make(observations={"x": 3})
        assert s.get("x") == 3

    def test_state_shadows_features(self) -> None:
        s = _make(state={"x": 1}, features={"x": 2})
        assert s.get("x") == 1

    def test_state_shadows_observations(self) -> None:
        s = _make(state={"x": 1}, observations={"x": 3})
        assert s.get("x") == 1

    def test_features_shadows_observations(self) -> None:
        s = _make(features={"x": 2}, observations={"x": 3})
        assert s.get("x") == 2

    def test_state_has_highest_priority(self) -> None:
        s = _make(state={"x": 1}, features={"x": 2}, observations={"x": 3})
        assert s.get("x") == 1

    def test_explicit_none_in_state_returned(self) -> None:
        s = _make(state={"x": None}, features={"x": 99})
        assert s.get("x") is None

    def test_explicit_none_in_features_returned(self) -> None:
        s = _make(features={"x": None}, observations={"x": 99})
        assert s.get("x") is None

    def test_different_attributes_from_different_layers(self) -> None:
        s = _make(
            state={"a": 1},
            features={"b": 2},
            observations={"c": 3},
        )
        assert s.get("a") == 1
        assert s.get("b") == 2
        assert s.get("c") == 3
        assert s.get("d") is None


# ---------------------------------------------------------------------------
# set_state / set_features / set_observations
# ---------------------------------------------------------------------------

class TestSetState:
    def test_state_updated(self) -> None:
        s = _make(state={"x": 1})
        s2 = s.set_state(Store({"x": 99}))
        assert s2.state["x"] == 99

    def test_original_unchanged(self) -> None:
        s = _make(state={"x": 1})
        s.set_state(Store({"x": 99}))
        assert s.state["x"] == 1

    def test_returns_new_instance(self) -> None:
        s = _make()
        assert s.set_state(Store()) is not s

    def test_features_preserved(self) -> None:
        s = _make(features={"f": 7})
        s2 = s.set_state(Store({"x": 1}))
        assert s2.features["f"] == 7

    def test_observations_preserved(self) -> None:
        s = _make(observations={"o": 5})
        s2 = s.set_state(Store({"x": 1}))
        assert s2.observations["o"] == 5


class TestSetFeatures:
    def test_features_updated(self) -> None:
        s = _make(features={"y": 2})
        s2 = s.set_features(Store({"y": 99}))
        assert s2.features["y"] == 99

    def test_original_unchanged(self) -> None:
        s = _make(features={"y": 2})
        s.set_features(Store({"y": 99}))
        assert s.features["y"] == 2

    def test_returns_new_instance(self) -> None:
        s = _make()
        assert s.set_features(Store()) is not s

    def test_state_preserved(self) -> None:
        s = _make(state={"x": 1})
        s2 = s.set_features(Store({"y": 2}))
        assert s2.state["x"] == 1

    def test_observations_preserved(self) -> None:
        s = _make(observations={"o": 5})
        s2 = s.set_features(Store({"y": 2}))
        assert s2.observations["o"] == 5


class TestSetObservations:
    def test_observations_updated(self) -> None:
        s = _make(observations={"z": 3})
        s2 = s.set_observations(Store({"z": 99}))
        assert s2.observations["z"] == 99

    def test_original_unchanged(self) -> None:
        s = _make(observations={"z": 3})
        s.set_observations(Store({"z": 99}))
        assert s.observations["z"] == 3

    def test_returns_new_instance(self) -> None:
        s = _make()
        assert s.set_observations(Store()) is not s

    def test_state_preserved(self) -> None:
        s = _make(state={"x": 1})
        s2 = s.set_observations(Store({"z": 3}))
        assert s2.state["x"] == 1

    def test_features_preserved(self) -> None:
        s = _make(features={"f": 7})
        s2 = s.set_observations(Store({"z": 3}))
        assert s2.features["f"] == 7


# ---------------------------------------------------------------------------
# apply_dynamics
# ---------------------------------------------------------------------------

class TestApplyDynamics:
    def test_features_updated(self) -> None:
        s = _make(features={"x": 0.0})
        dyn = lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt)
        s2 = s.apply_dynamics(RNG, 0.5, dyn)
        assert s2.features["x"] == pytest.approx(0.5)

    def test_original_features_unchanged(self) -> None:
        s = _make(features={"x": 1.0})
        s.apply_dynamics(RNG, 1.0, lambda rng, s, dt: s.features.set("x", 99.0))
        assert s.features["x"] == pytest.approx(1.0)

    def test_returns_new_instance(self) -> None:
        s = _make()
        s2 = s.apply_dynamics(RNG, 1.0, lambda rng, s, dt: s.features)
        assert s2 is not s

    def test_agent_store_passed_to_dyn(self) -> None:
        received: list = []
        s = _make(state={"k": 7})
        s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: received.append(agent_store) or agent_store.features)
        assert received[0] is s

    def test_state_accessible_via_agent_store(self) -> None:
        s = _make(state={"k": 7})
        result = s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: agent_store.features.set("k", agent_store.state.get("k")))
        assert result.features["k"] == 7

    def test_features_accessible_via_agent_store(self) -> None:
        s = _make(features={"v": 3})
        result = s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: agent_store.features)
        assert result.features["v"] == 3

    def test_rng_passed_to_dyn(self) -> None:
        received: list = []
        s = _make()
        s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: received.append(rng) or agent_store.features)
        assert received[0] is RNG

    def test_dt_passed_to_dyn(self) -> None:
        received: list[float] = []
        s = _make()
        s.apply_dynamics(RNG, 2.5, lambda rng, agent_store, dt: received.append(dt) or Store())
        assert received[0] == pytest.approx(2.5)

    def test_state_preserved(self) -> None:
        s = _make(state={"s": 9})
        s2 = s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: agent_store.features)
        assert s2.state["s"] == 9

    def test_observations_preserved(self) -> None:
        s = _make(observations={"o": 4})
        s2 = s.apply_dynamics(RNG, 1.0, lambda rng, agent_store, dt: agent_store.features)
        assert s2.observations["o"] == 4
