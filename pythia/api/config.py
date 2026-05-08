"""Simulation configuration.

:class:`SimulationConfig` is an immutable (frozen) dataclass that carries
all user-facing settings for a simulation run.  Because it is frozen, the
same instance can be shared safely across multiple runners or threads.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pythia.api.exceptions import ConfigurationError


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration for a single simulation run.

    Attributes:
        max_time: Upper bound on simulated time.  The runner stops as soon as
            ``state.elapsed_time`` reaches or exceeds this value.  Must be
            strictly positive.
        random_seed: Seed for any random-number generators used by the model.
            ``None`` means non-deterministic. Defaults to ``None``.
        metadata: Open-ended mapping for domain-specific settings that do not
            yet warrant a dedicated field.  Future modules read the keys they
            understand while ignoring others.

    Raises:
        ConfigurationError: If ``max_time`` violates its constraint.

    Example:
        >>> cfg = SimulationConfig(max_time=100.0, random_seed=42)
        >>> cfg.max_time
        100.0
    """

    max_time: float
    random_seed: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.max_time <= 0.0:
            raise ConfigurationError(
                f"max_time must be strictly positive, got {self.max_time!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Construct a :class:`SimulationConfig` from a plain dictionary.

        Useful for loading configuration from JSON or YAML files.

        Args:
            data: Mapping with keys matching the field names of
                :class:`SimulationConfig`.  Unknown keys are forwarded to
                ``metadata`` rather than raising an error.

        Returns:
            A new :class:`SimulationConfig` instance.

        Raises:
            ConfigurationError: If ``max_time`` is missing from *data*.

        Example:
            >>> cfg = SimulationConfig.from_dict({"max_time": 50.0})
            >>> cfg.max_time
            50.0
        """
        known_fields = {f.name for f in dataclasses.fields(cls)}
        known = {k: v for k, v in data.items() if k in known_fields}
        unknown = {k: v for k, v in data.items() if k not in known_fields}

        if "max_time" not in known:
            raise ConfigurationError("'max_time' is required in configuration data")

        # Merge unknown keys into metadata, letting explicit metadata win.
        merged_meta: dict[str, Any] = {**unknown, **known.get("metadata", {})}
        known["metadata"] = merged_meta

        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        """Serialise this configuration to a plain dictionary.

        Returns:
            A dictionary representation suitable for JSON or YAML serialisation.

        Example:
            >>> cfg = SimulationConfig(max_time=10.0)
            >>> cfg.to_dict()["max_time"]
            10.0
        """
        return dataclasses.asdict(self)
