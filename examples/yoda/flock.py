"""Flocking — pythia.models.yoda example.

This YODA model is used to describe the behaviour of a set of agents moving
according to the Reynolds' model.

Agents move according three rules:
- Separation: an agent tends to change its direction when it is
  getting too close to other agents.
- Alignment: an agent steers towards average heading of neighbours.
- Cohesion: an agent moves towards the average position of its neighbours.

In the proposed yoda each agent is equipped with the following attributes:
- pos: a feature indicating the position of the agent;
- velocity: a two-dimensional vector indicating the velocity of the agent;

Observations are used to collect the following information:
- v_sep: a bidirectional vector indicating the direction to follow stay
  away from other agents;
- v_cmas: a bidirectional vector indicating the direction to reach the
  center of mass;
- v_aligment: a bidirectional vector indicating the average direction
  of the agents in the surrounding.

Parameters (defaults)
---------------------
N                   = 10     Number of agents

separation_distance = 10     Separation distance from other agents

cohesion_strength   = 100   Cohesion strength

Auxiliary Functions
---------------------

The following auxiliary functions are used in the YODA model:

- normalise: a function that nomarlises a velocity vector
-

Actions
------------
Each agent can perform the following actions:
- steer: the agent will select new direction.




Usage
-----
    python examples/yoda/flock.py

"""
from __future__ import annotations

import random
from collections.abc import Callable
from turtledemo import forest
from typing import Any
import numpy as np

from pythia import SimulationState, SimulationConfig
from pythia.api import Tracer, DataTracer
from pythia.models.yoda import Agent, AgentState, AgentDefinitions, Action, Store, SystemConfiguration, AgentStore, \
    AgentConfiguration, YodaModel
from pythia.util import TraceViewer

N = 100
cohesion_distance   = 1000
alignment_distance  = 10
separation_distance = 10
cohesion_strength   = 0.005
separation_strength = 0.1
alignment_strength  = 0.03
SPEED = 200

def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def centre_of_mass(pos: np.ndarray, positions: list[np.ndarray]) -> np.ndarray:
    elements = len(positions)
    if elements > 0:
        return pos-(np.sum(positions, 0)/elements)
    return np.zeros(2)

def separation(pos: np.ndarray, positions: list[np.ndarray]) -> np.ndarray:
    v_sep = np.zeros(2)
    for other in positions:
        v_sep -= other - pos
    return v_sep

def alignment(vel: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    elements = len(others)
    if elements > 0:
        return np.sum(others, 0)/elements
    return np.zeros(2)

def steer_function(rng: random.Random, s: Store, o: Store) -> Store:
    pos = s.get("pos",np.zeros(2))
    vel = s.get("velocity",np.zeros(2))
    v_sep = o.get("separation",np.zeros(2))
    cmas = o.get("cmas",np.zeros(2))
    v_alig = o.get("alignment",np.zeros(2))
    new_vel = vel
    new_vel += (pos - cmas)*cohesion_strength
    new_vel += v_sep*separation_strength
    new_vel += v_alig*alignment_strength
    return s.set("velocity", normalise(new_vel))

def next_to(pos: np.ndarray, a: AgentStore, dist: float) -> bool:
    if a.has("pos"):
        return np.linalg.norm(pos - a.get("pos")) <= dist
    return False

def extract_neighbours_positions(dist: float, store: AgentStore, conf: SystemConfiguration) -> list[np.ndarray]:
    if store.has("pos"):
        pos = store.get("pos")
        return conf.eval(lambda s: next_to(pos, s, dist), lambda a: a.agent_store.get("pos"))
    return []

def extract_neighbours_velocity(dist: float, store: AgentStore, conf: SystemConfiguration) -> list[np.ndarray]:
    if store.has("pos"):
        pos = store.get("pos")
        return conf.eval(lambda s: s.has("velocity") and next_to(pos, s, dist), lambda a: a.agent_store.get("velocity"))
    return []

def cmas_observation(store: AgentStore, conf: SystemConfiguration) -> np.ndarray:
    return centre_of_mass(store.get("pos"), extract_neighbours_positions(cohesion_distance, store, conf))

def separation_observation(store: AgentStore, conf: SystemConfiguration) -> np.ndarray:
    return separation(store.get("pos"),extract_neighbours_positions(separation_distance, store, conf))

def alignment_observation(store: AgentStore, conf: SystemConfiguration) -> np.ndarray:
    return alignment(store.get("velocity"),extract_neighbours_velocity(alignment_distance, store, conf))

def boid_observations(rng: random.Random, store: AgentStore, conf: SystemConfiguration) -> Store:
    return (store.observations
                .set("cmas",cmas_observation(store, conf))
                .set("separation",separation_observation(store, conf))
                .set("alignment",alignment_observation(store, conf))
            )

def boid_dynamics(rng: random.Random, store: AgentStore, dt: float) -> Store:
    pos = store.get("pos")
    vel = store.get("velocity")
    if vel is None:
        vel = np.zeros(2)
    return store.features.set("pos",pos+vel*SPEED*dt)

def create_agent(rng: random.Random, id: int, width: int =1000, height: int = 1000) -> AgentConfiguration:
    return AgentConfiguration(boid, flying, AgentStore(
        features=Store({"pos": np.array([rng.random()*width, rng.random()*height])}),
        state=Store({"id": id, "velocity": normalise(np.array([1-rng.random()*2, 1-rng.random()*2]))}),
        observations=Store({}))
    )

def generate_initial_state(rng: random.Random, size: int, width: int =100, height: int = 100) -> SystemConfiguration:
    return SystemConfiguration([create_agent(rng, id, width, height) for id in range(size)])

def extract_trace_info(s: SimulationState) -> dict[str, dict[str, float]]:
    conf: SystemConfiguration| None = s.data.get("yoda.configuration")
    if conf is not None and conf.elements is not None:
        return { f"Agent{a.agent_store.get("id")}": {"x": a.agent_store.get("pos")[0], "y":a.agent_store.get("pos")[1]} for a in conf.elements}
    return {}

steer = Action("steer", steer_function)

boid = Agent("Boid")
flying = AgentState("Flying")

agents = AgentDefinitions()
agents.add_transition(boid, flying, steer, flying, 1.0)
agents.add_observation(boid, boid_observations)
agents.add_dynamics(boid, boid_dynamics)

agents.set_default_action_weight(steer, lambda s: 1.0)

model = YodaModel(lambda rng: generate_initial_state(rng, N), agents, 0.5, .5)

tracer = Tracer(DataTracer(["x", "y"], extract_trace_info))
configuration = SimulationConfig(max_time=100)

tracer.run(model, configuration)

tables = tracer.data_tracer

viewer = TraceViewer(tables, lambda d: (d["x"], d["y"],"b"))
viewer.create_animation(0, 100, 0.1)
