"""Utility subpackage for simulation handling and data visualisation.

``pythia.util`` collects helper classes and functions that sit above the core
``pythia.api`` layer and complement the domain-specific model modules.  Two
broad categories of utilities are provided:

* **Simulation handling** — tools that orchestrate multiple simulation runs,
  aggregate results, or post-process simulation output (e.g. multi-run
  estimators, batch runners).

* **Data visualisation** — helpers that transform collected
  :class:`~pythia.api.data_table.DataTable` or
  :class:`~pythia.api.data_tracer.DataTracer` data into forms suitable for
  plotting or further analysis.

Available classes:

- :class:`~pythia.util.result_plotter.ResultPlotter` — generate one
  Matplotlib figure per measure from multi-run simulation data
  (requires ``pip install 'pythia[visualization]'``).
- :class:`~pythia.util.data_statistics.DataStatistics` — compute mean,
  standard deviation, quartiles, min and max across a list of
  :class:`~pythia.api.data_table.DataTable` objects sharing the same labels
  and time points (requires ``pip install 'pythia[visualization]'``).

Import from the relevant submodule directly::

    from pythia.util.result_plotter import ResultPlotter
    from pythia.util.data_statistics import DataStatistics
"""

from pythia.util.data_statistics import DataStatistics
from pythia.util.result_plotter import ResultPlotter

__all__ = [
    "DataStatistics",
    "ResultPlotter",
]
