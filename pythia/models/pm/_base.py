"""Shared validation helpers for population models.

Internal module — not part of the public API.
"""

from __future__ import annotations

from pythia.api.config import SimulationConfig
from pythia.api.exceptions import ConfigurationError


def require_positive(config: SimulationConfig, key: str) -> float:
    """Return ``config.metadata[key]`` as a float, raising if absent or non-positive.

    Args:
        config: The simulation configuration to inspect.
        key: The metadata key to look up.

    Returns:
        The value cast to ``float``.

    Raises:
        ConfigurationError: If *key* is missing or its value is not positive.
    """
    if key not in config.metadata:
        raise ConfigurationError(f"population model requires '{key}' in config.metadata")
    value = float(config.metadata[key])
    if value <= 0.0:
        raise ConfigurationError(
            f"config.metadata['{key}'] must be positive, got {value!r}"
        )
    return value


def require_float(config: SimulationConfig, key: str) -> float:
    """Return ``config.metadata[key]`` as a float, raising if absent.

    Unlike :func:`require_positive`, this helper accepts any finite float
    (including zero and negatives), which is appropriate for rate parameters
    that can represent decay or extinction.

    Args:
        config: The simulation configuration to inspect.
        key: The metadata key to look up.

    Returns:
        The value cast to ``float``.

    Raises:
        ConfigurationError: If *key* is missing from ``config.metadata``.
    """
    if key not in config.metadata:
        raise ConfigurationError(f"population model requires '{key}' in config.metadata")
    return float(config.metadata[key])
