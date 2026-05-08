"""Domain-specific exceptions for the Pythia simulation framework.

All exceptions inherit from :class:`PythiaError` so callers can catch
at whatever granularity they need::

    try:
        runner.run()
    except ConfigurationError:
        # handle bad config
        ...
    except PythiaError:
        # catch-all for any Pythia-level problem
        ...
"""


class PythiaError(Exception):
    """Root exception for all Pythia errors."""


class ConfigurationError(PythiaError):
    """Raised when a :class:`~pythia.api.config.SimulationConfig` is invalid.

    Examples of conditions that raise this error:

    - ``max_steps`` is not a positive integer.
    - ``time_step`` is zero or negative.
    - A model-required configuration key is absent from ``metadata``.
    """


class ModelError(PythiaError):
    """Raised when a model violates its contract at runtime.

    Examples of conditions that raise this error:

    - :meth:`~pythia.api.model.BaseModel.step` returns a value that is not
      a :class:`~pythia.api.state.SimulationState`.
    - A model raises an unexpected exception during ``step`` or
      ``initialize``; the runner wraps it in :class:`ModelError`.
    """


class RunnerError(PythiaError):
    """Raised when the :class:`~pythia.api.runner.SimulationRunner` itself fails.

    This covers orchestration-level problems that are not attributable to
    the model or its configuration, such as an invalid runner setup.
    """
