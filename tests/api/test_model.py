"""Tests for pythia.api.model."""

from __future__ import annotations

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.exceptions import ConfigurationError
from pythia.api.model import BaseModel
from pythia.api.state import SimulationState


class TestBaseModelAbstract:
    def test_cannot_instantiate_without_abstract_methods(self) -> None:
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:
        class Partial(BaseModel):
            @property
            def name(self) -> str:
                return "partial"

            def initialize(self, config: SimulationConfig) -> None:
                pass

            def step(self, state: SimulationState) -> tuple[float, SimulationState]:
                return 1.0, state

            # finalize is missing

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


class TestBaseModelValidateConfig:
    """Validate that the base validate_config catches constraint violations."""

    class _ConcreteModel(BaseModel):
        @property
        def name(self) -> str:
            return "concrete"

        def initialize(self, config: SimulationConfig) -> None:
            self.validate_config(config)

        def step(self, state: SimulationState) -> tuple[float, SimulationState]:
            return 1.0, state

        def finalize(self, state: SimulationState) -> None:
            pass

    # These test the validate_config method directly (bypassing __post_init__)
    # by constructing a config, then monkeypatching the field.

    def test_valid_config_does_not_raise(self, minimal_config: SimulationConfig) -> None:
        model = self._ConcreteModel()
        # Should not raise
        model.validate_config(minimal_config)

    def test_invalid_max_time_raises(self, minimal_config: SimulationConfig) -> None:
        model = self._ConcreteModel()
        # Bypass frozen dataclass by creating a config via object.__setattr__
        import dataclasses
        bad = dataclasses.replace(minimal_config)
        object.__setattr__(bad, "max_time", -1.0)
        with pytest.raises(ConfigurationError, match="max_time"):
            model.validate_config(bad)


class TestBaseModelName:
    def test_name_is_accessible(self, no_op_model: BaseModel) -> None:
        assert no_op_model.name == "no_op"
