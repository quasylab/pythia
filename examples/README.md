# Pythia Examples

This folder collects self-contained example scripts for each of the
simulation model modules shipped with Pythia.

## Structure

```
examples/
├── pm/       # Population Models — Gillespie stochastic reaction rules
├── lio/      # Language of Interacting Objects
├── nba/      # Network of Broadcasting Agents
└── yoda/     # Yet anOther agent Description lAnguage
```

## Running an example

Install Pythia with its development dependencies, then run any script
directly with Python:

```bash
pip install -e ".[dev,visualization]"
python examples/pm/sir.py
```

Visualisation examples additionally require Matplotlib and NumPy, both
included in the `visualization` extra above.
