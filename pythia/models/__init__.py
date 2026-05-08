"""Concrete simulation models for Pythia.

Each submodule provides one domain of models that implement
:class:`~pythia.api.model.BaseModel`.  Import from the submodule directly::

    from pythia.models.pm import PopulationModel, Rule
    from pythia.models.lio import LioModel
    from pythia.models.nba import NBAModel

Submodules:

- ``pm`` — Population Models (continuous-time stochastic population dynamics)
- ``lio`` — Language of Interacting Objects
- ``nba`` — Network of Broadcasting Agents
"""
