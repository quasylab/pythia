"""Shared pytest fixtures for the Pythia test suite."""

from __future__ import annotations

import pytest

from pythia.api import (
    BaseModel,
    EventBus,
    SimulationConfig,
    SimulationState,
)
from pythia.api.config import SimulationConfig


class NoOpModel(BaseModel):
    """A concrete model whose ``step`` returns the state unchanged.

    Used wherever the model behaviour is not the subject under test.
    """

    @property
    def name(self) -> str:
        return "no_op"

    def initialize(self, config: SimulationConfig) -> None:
        pass

    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        return 1.0, state

    def finalize(self, state: SimulationState) -> None:
        pass


@pytest.fixture()
def minimal_config() -> SimulationConfig:
    """A :class:`SimulationConfig` with small, reproducible defaults."""
    return SimulationConfig(max_time=10.0, random_seed=42)


@pytest.fixture()
def initial_state() -> SimulationState:
    """A zeroed-out :class:`SimulationState`."""
    return SimulationState()


@pytest.fixture()
def event_bus() -> EventBus:
    """A fresh :class:`EventBus` with no subscribers."""
    return EventBus()


@pytest.fixture()
def no_op_model() -> NoOpModel:
    """A :class:`BaseModel` whose step is a no-op."""
    return NoOpModel()
