"""Simulator — repeated execution of a model over many independent runs.

A :class:`Simulator` wraps a :class:`~pythia.api.runner.SimulationRunner` and
runs the same model *N* times, collecting a fresh set of measurements for each
run.  All results are retained and can be queried afterwards.

When the base configuration contains a ``random_seed``, run *i* uses
``random_seed + i`` so every run is deterministic and independently
reproducible.  When ``random_seed`` is ``None`` each run is non-deterministic.

Example::

    from pythia.api import SimulationConfig
    from pythia.api.measure import Measure
    from pythia.api.simulator import Simulator

    measure = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
    config  = SimulationConfig(max_time=10.0, random_seed=0)
    sim     = Simulator(model, config, [measure])
    sim.run(n_runs=100)

    # All 100 per-run time-series for "t"
    all_series = sim.results_for("t")
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from pythia.api.collector import DataCollector
from pythia.api.config import SimulationConfig
from pythia.api.data_table import DataTable
from pythia.api.measure import Measure
from pythia.api.model import BaseModel
from pythia.api.runner import SimulationRunner


class Simulator:
    """Runs a model repeatedly and collects measurements from every run.

    Each call to :meth:`run` executes *n_runs* independent simulations, each
    with a fresh :class:`~pythia.api.collector.DataCollector`.  The results of
    all runs are stored and exposed via :attr:`results` and
    :meth:`results_for`.

    Args:
        model: The model to simulate.  The same instance is reused across
            runs; its :meth:`~pythia.api.model.BaseModel.initialize` method
            is called at the start of every run.
        config: Base configuration shared by all runs.  When
            :attr:`~pythia.api.config.SimulationConfig.random_seed` is set,
            run *i* uses ``random_seed + i`` to keep runs independent yet
            reproducible.
        measures: The :class:`~pythia.api.measure.Measure` objects to record
            in each run.  Must have unique names (enforced by
            :class:`~pythia.api.collector.DataCollector`).

    Example:
        >>> from pythia.api import SimulationConfig
        >>> from pythia.api.measure import Measure
        >>> from pythia.api.simulator import Simulator
        >>> m = Measure("t", lambda s: s.elapsed_time, frequency=1.0)
        >>> sim = Simulator(model, SimulationConfig(max_time=5.0, random_seed=0), [m])
        >>> sim.run(n_runs=10)
        >>> len(sim.results)
        10
    """

    def __init__(
        self,
        model: BaseModel,
        config: SimulationConfig,
        measures: Iterable[Measure],
    ) -> None:
        """Initialise the simulator."""
        self._model = model
        self._config = config
        self._measures: tuple[Measure, ...] = tuple(measures)
        self._results: list[DataTable] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, n_runs: int) -> None:
        """Execute *n_runs* independent simulations and store the results.

        Calling this method replaces any results from a previous call.

        Args:
            n_runs: Number of simulation runs to perform.  Must be a
                positive integer.

        Raises:
            ValueError: If *n_runs* is not a positive integer.
        """
        if n_runs < 1:
            raise ValueError(f"n_runs must be a positive integer, got {n_runs!r}")

        self._results = []
        for i in range(n_runs):
            run_config = self._config_for_run(i)
            collector = DataCollector(self._measures)
            runner = SimulationRunner(self._model, run_config, data_collector=collector)
            runner.run()
            simulation_result = collector.records
            if simulation_result is not None:
                self._results.append(simulation_result)

    @property
    def results(self) -> list[DataTable]:
        """All collected records, one :class:`~pythia.api.data_table.DataTable`
        per run (or ``None`` when no measures are registered).

        Each table has one column per measure and one row per scheduled sample
        point.  Returns an empty list if :meth:`run` has not been called.
        """
        return list(self._results)

    def results_for(self, name: str) -> Sequence[list[tuple[float, float]]]:
        """Return the ``(time, value)`` pairs for *name* across all runs.

        Args:
            name: The :attr:`~pythia.api.measure.Measure.name` of the measure.

        Returns:
            A list of ``[(time, value), ...]`` lists, one per run.

        Raises:
            KeyError: If *name* does not match any registered measure.
        """
        measure_names = {m.name for m in self._measures}
        if name not in measure_names:
            raise KeyError(f"No measure named {name!r}")
        result = []
        for table in self._results:
            if table is None:
                result.append([])
            else:
                result.append([(t, table.get(t)[name]) for t in table.times])
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config_for_run(self, index: int) -> SimulationConfig:
        """Return a config with a per-run seed derived from the base seed."""
        if self._config.random_seed is None:
            return self._config
        return dataclasses.replace(
            self._config, random_seed=self._config.random_seed + index
        )
