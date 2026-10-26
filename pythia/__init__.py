"""Pythia — a modular simulation framework.

This package exposes the core public API. Import from here for stable,
version-guaranteed symbols. Internal submodule paths may change between
minor releases.

Examples:
    >>> from pythia import SimulationConfig, SimulationRunner
    >>> from pythia import BaseModel, SimulationState
"""

from pythia.api import (
    BaseModel,
    ConfigurationError,
    EventBus,
    EventKind,
    ModelError,
    PythiaError,
    RunnerError,
    SimulationConfig,
    SimulationEvent,
    SimulationRunner,
    SimulationState,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaseModel",
    "ConfigurationError",
    "EventBus",
    "EventKind",
    "ModelError",
    "PythiaError",
    "RunnerError",
    "SimulationConfig",
    "SimulationEvent",
    "SimulationRunner",
    "SimulationState",
]
