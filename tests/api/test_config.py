"""Tests for pythia.api.config."""

from __future__ import annotations

import pytest

from pythia.api.config import SimulationConfig
from pythia.api.exceptions import ConfigurationError


class TestSimulationConfigValidation:
    def test_valid_minimal(self) -> None:
        cfg = SimulationConfig(max_time=1.0)
        assert cfg.max_time == 1.0

    def test_zero_max_time_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_time"):
            SimulationConfig(max_time=0.0)

    def test_negative_max_time_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_time"):
            SimulationConfig(max_time=-5.0)

    def test_defaults(self) -> None:
        cfg = SimulationConfig(max_time=5.0)
        assert cfg.random_seed is None
        assert cfg.metadata == {}

    def test_frozen(self) -> None:
        cfg = SimulationConfig(max_time=5.0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_time = 99.0  # type: ignore[misc]


class TestSimulationConfigFromDict:
    def test_round_trip(self) -> None:
        cfg = SimulationConfig(max_time=20.0, random_seed=7)
        restored = SimulationConfig.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_missing_max_time_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_time"):
            SimulationConfig.from_dict({})

    def test_unknown_keys_go_to_metadata(self) -> None:
        cfg = SimulationConfig.from_dict({"max_time": 5.0, "custom_param": 42})
        assert cfg.metadata["custom_param"] == 42

    def test_explicit_metadata_wins_over_unknown_keys(self) -> None:
        cfg = SimulationConfig.from_dict(
            {"max_time": 5.0, "custom": "a", "metadata": {"custom": "b"}}
        )
        assert cfg.metadata["custom"] == "b"


class TestSimulationConfigToDict:
    def test_max_time_present(self) -> None:
        cfg = SimulationConfig(max_time=10.0)
        d = cfg.to_dict()
        assert d["max_time"] == 10.0

    def test_metadata_preserved(self) -> None:
        cfg = SimulationConfig(max_time=10.0, metadata={"k": "v"})
        assert cfg.to_dict()["metadata"] == {"k": "v"}
