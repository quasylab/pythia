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

Import from the relevant submodule directly::

    from pythia.util.result_plotter import ResultPlotter
"""

from pythia.util.result_plotter import ResultPlotter

__all__ = [
    "ResultPlotter",
]
