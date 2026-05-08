"""Tests for pythia.models.yoda.action."""

from __future__ import annotations

import random

import pytest

from pythia.models.yoda.action import Action
from pythia.models.yoda.store import Store


RNG = random.Random(0)
S   = Store({"x": 1})
O   = Store({"y": 2})


def _action(name: str = "a", result: Store | None = None) -> Action:
    r = result if result is not None else Store()
    return Action(name=name, step_function=lambda rng, s, o: r)


class TestConstruction:
    def test_name_stored(self) -> None:
        assert _action(name="move").name == "move"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            Action("", lambda rng, s, o: Store())

    def test_step_function_stored(self) -> None:
        sf = lambda rng, s, o: Store({"k": 1})
        a = Action("a", sf)
        assert a.step_function is sf


class TestImmutability:
    def test_name_not_reassignable(self) -> None:
        a = _action()
        with pytest.raises(AttributeError):
            a.name = "other"  # type: ignore[misc]

    def test_step_function_not_reassignable(self) -> None:
        a = _action()
        with pytest.raises(AttributeError):
            a.step_function = lambda rng, s, o: Store()  # type: ignore[misc]


class TestStep:
    def test_returns_result_store(self) -> None:
        result = Store({"done": True})
        a = _action(result=result)
        assert a.step(RNG, S, O) is result

    def test_rng_passed_to_step_function(self) -> None:
        received: list[random.Random] = []
        a = Action("a", lambda rng, s, o: received.append(rng) or s)  # type: ignore[arg-type]
        a.step(RNG, S, O)
        assert received == [RNG]

    def test_state_passed_to_step_function(self) -> None:
        a = Action(
            "a",
            step_function=lambda rng, state, observations: state.set("incremented", True),
        )
        out = a.step(RNG, Store({"x": 1}), Store())
        assert out["incremented"] is True
        assert out["x"] == 1

    def test_observations_passed_to_step_function(self) -> None:
        a = Action(
            "a",
            step_function=lambda rng, state, observations: state.set(
                "seen", observations.get("signal")
            ),
        )
        out = a.step(RNG, Store(), Store({"signal": 42}))
        assert out["seen"] == 42

    def test_both_state_and_observations_passed(self) -> None:
        a = Action(
            "a",
            step_function=lambda rng, state, observations: state.set(
                "total",
                state.get("v", 0) + observations.get("v", 0),
            ),
        )
        out = a.step(RNG, Store({"v": 3}), Store({"v": 4}))
        assert out["total"] == 7

    def test_rng_usable_in_step_function(self) -> None:
        a = Action(
            "a",
            step_function=lambda rng, state, observations: state.set(
                "coin", rng.choice(["heads", "tails"])
            ),
        )
        out = a.step(random.Random(0), Store(), Store())
        assert out["coin"] in {"heads", "tails"}


class TestRepr:
    def test_repr(self) -> None:
        assert repr(_action(name="tick")) == "Action('tick')"
