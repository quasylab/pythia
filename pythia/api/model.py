"""Abstract base class for simulation models.

Every concrete model must subclass :class:`BaseModel` and implement the
three lifecycle methods: :meth:`initialize`, :meth:`step`, and
:meth:`finalize`.  The runner calls them in that order; models should not
call each other directly.

Example::

    from pythia.api import BaseModel, SimulationConfig, SimulationState

    class ConstantModel(BaseModel):
        \"\"\"A trivial model that never changes state.\"\"\"

        @property
        def name(self) -> str:
            return "constant"

        def initialize(self, config: SimulationConfig) -> None:
            self.validate_config(config)

        def step(self, state: SimulationState) -> tuple[float, SimulationState]:
            return 1.0, state

        def finalize(self, state: SimulationState) -> None:
            pass
"""

from __future__ import annotations

import abc

from pythia.api.config import SimulationConfig
from pythia.api.exceptions import ConfigurationError
from pythia.api.state import SimulationState


class BaseModel(abc.ABC):
    """Abstract base class that every simulation model must implement.

    Subclasses define the behaviour of a simulation by implementing
    :meth:`initialize`, :meth:`step`, and :meth:`finalize`.  The runner
    calls them in lifecycle order; models should be stateless between runs
    unless they explicitly store state set during :meth:`initialize`.

    The abstract :attr:`name` property is used in logging and event payloads.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this model.

        Used in log messages and event ``metadata`` payloads.  Should be
        short and unique within a project (e.g. ``"sir"``, ``"lorenz"``).
        """

    def setup(self, state: SimulationState) -> None:
        """Populate the initial simulation state before the first measure collection.

        Called by the runner after :meth:`initialize` and before measures are
        collected at ``t = 0``.  Override this method to write the model's
        starting data into ``state.data`` so that observers and data collectors
        see the correct initial values at time zero.

        The default implementation is a no-op; subclasses should override it
        when they need measures at ``t = 0`` to reflect the true initial state.

        Args:
            state: The empty :class:`~pythia.api.state.SimulationState` at
                ``elapsed_time = 0``.
        """

    @abc.abstractmethod
    def initialize(self, config: SimulationConfig) -> None:
        """Prepare the model for a new run.

        Called once by the runner before the step loop begins.  Use this
        method to seed random-number generators, pre-compute constants, or
        allocate data structures based on *config*.

        Subclasses should call :meth:`validate_config` here, then perform
        any domain-specific initialisation.

        Args:
            config: The immutable configuration for this run.

        Raises:
            ConfigurationError: If *config* is incompatible with this model.
        """

    @abc.abstractmethod
    def step(self, state: SimulationState) -> tuple[float, SimulationState]:
        """Advance the simulation by one tick.

        Called by the runner on every iteration until either the returned
        time step is ``0.0`` or
        :attr:`~pythia.api.config.SimulationConfig.max_time` is reached.

        Implementations may mutate *state* in place **or** return a new
        :class:`~pythia.api.state.SimulationState`.  The runner always uses
        the returned value.

        Args:
            state: The current simulation state.

        Returns:
            A pair ``(dt, state)`` where *dt* is the duration of the tick
            just computed.  Returning ``dt = 0.0`` signals that the
            simulation should terminate (e.g. no further transitions are
            possible).  May return the same *state* object (mutated in place)
            or a new instance.

        Raises:
            ModelError: If the model encounters an unrecoverable error.
        """

    @abc.abstractmethod
    def finalize(self, state: SimulationState) -> None:
        """Clean up after the run loop ends.

        Called once by the runner after the loop exits, regardless of whether
        the run completed normally or was halted by a zero time step.  Use
        this method to flush buffers, close resources, or compute summary
        statistics.

        Args:
            state: The final simulation state.
        """

    def validate_config(self, config: SimulationConfig) -> None:
        """Validate that *config* meets the requirements of this model.

        The base implementation checks the invariants guaranteed by
        :class:`~pythia.api.config.SimulationConfig` itself (which are already
        enforced at construction time).  Subclasses should call
        ``super().validate_config(config)`` and then add any domain-specific
        checks.

        Args:
            config: The configuration to validate.

        Raises:
            ConfigurationError: If any required field or metadata key is
                absent or has an invalid value.

        Example:
            >>> class MyModel(BaseModel):
            ...     def validate_config(self, config):
            ...         super().validate_config(config)
            ...         if "my_param" not in config.metadata:
            ...             raise ConfigurationError("'my_param' is required")
        """
        if config.max_time <= 0.0:
            raise ConfigurationError(
                f"max_time must be positive, got {config.max_time!r}"
            )
