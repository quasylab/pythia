"""Flocking — pythia.models.yoda example.

This YODA model is used to describe a set of agents that follow a
random walk. Each agent randomly select a direction that is followed
until a border of the arena is reached. At this point the agent stay
there for a random time and then another random direction is selected.

In the proposed yoda each agent is equipped with the following attributes:
- pos: a feature indicating the position of the agent;
- direction: a two-dimensional vector of magnitute 1 indicating the
  agent direction;
- speed: speed of the agent;
- border: an observation attribute that is true when the agent is at
  the border of the area.

Parameters (defaults)
---------------------
N      = 10     Number of agents

SPEED  = 1      Default speed of the agent

WIDTH  = 100    Width of the agents ares

HEIGHT = 100    Height of the agents ares

Auxiliary Functions
---------------------

The following auxiliary functions are used in the YODA model:

- direction: a function that given a random generator and the
  agent position randomly select a direction

- movement: given a position, a direction and a vector returns
  next position.

Actions
------------
Each agent can perform the following actions:
- steer:    used by an agent to select next agent direction.
- continue: the agent continues its movement.
- stop:     the agent stops its movement.


Usage
-----
    python examples/yoda/randomwalk.py

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

WIDTH = 1000.0
HEIGHT = 1000.0
N     = 200
SPEED = 10
eps   = 0.5

def robot_observations(rng: random.Random, s: AgentStore, conf: SystemConfiguration) -> Store:
    pos = s.features["pos"]
    x = pos[0]
    y = pos[1]
    border = x < eps or x > WIDTH - eps or y < eps or y > HEIGHT - eps
    return s.observations.set("border", border)

def choose_direction(rng: random.Random, dir: float) -> float:
    """
    This function is used by an agent to select a random direction
    that is the opposite of the one currently used.
    """
    return dir+np.pi+rng.random()*np.pi/2

def movement_function(pos: np.ndarray, dir: float, speed: float, dt: float) -> np.array:
    x: float = pos[0]+speed*dir*np.cos(dir)*dt
    y: float = pos[1]+speed*dir*np.sin(dir)*dt
    x = max(min(x, WIDTH), 0.0)
    y = max(min(y, HEIGHT), 0.0)
    return np.array([x, y])

def robot_dynamics(rng: random.Random, s: AgentStore, dt: float) -> Store:
    return s.features.set("pos",movement_function(s.get("pos"),s.get("dir"),s.get("speed"),dt))


robot = Agent("Robot")
walk = AgentState("Walk")
stop = AgentState("Stop")



wait_action  = Action("wait", lambda rng, s, o: s)
steer_action = Action("steer", lambda rng,s,o: s.set("dir", choose_direction(rng, s.get("dir",0.0))).set("speed", SPEED))
stop_action  = Action("stop", lambda rng, s, o: s.set("speed", 0.0))
continue_action = Action("continue", lambda rng, s, o: s)

agents = AgentDefinitions()
agents.add_transition(robot, stop, steer_action, walk, 1.0)
agents.add_transition(robot, stop, wait_action, stop, 1.0)
agents.add_transition(robot, walk, continue_action, walk, 1.0)
agents.add_transition(robot, walk, stop_action, stop, 1.0)

agents.add_dynamics(robot, robot_dynamics)
agents.add_observation(robot, robot_observations)

def weight_of_stop_action(s: AgentStore) -> float:
    if s.observations.get("border", False):
        return 1.0
    else:
        return 0.0

def weight_of_continue_action(s: AgentStore) -> float:
    if s.observations.get("border", False):
        return 0.0
    else:
        return 1.0

agents.set_default_action_weight(wait_action, lambda s: 1)
agents.set_default_action_weight(steer_action, lambda s: 1)
agents.set_default_action_weight(stop_action, lambda s: 1.0 if s.observations.get("border", False) else 0.0)
agents.set_default_action_weight(continue_action, lambda s: 0.0 if s.observations.get("border", False) else 1.0)

def create_agent_store(rng: random.Random, id: int) -> AgentStore:
    x = random.random()*WIDTH
    y = random.random()*HEIGHT
    return AgentStore(
        state=Store({"speed": 0, "dir": rng.random()*2*np.pi, "id": id}),
        features= Store({"pos":np.array([x,y])}),
        observations=Store({"border": False}),
    )

def create_agent(rng: random.Random, id: int) -> AgentConfiguration:
    return AgentConfiguration(robot, stop, create_agent_store(rng, id))

def generate_initial_state(rng: random.Random, size: int) -> SystemConfiguration:
    return SystemConfiguration([create_agent(rng, id) for id in range(size)])

def extract_trace_info(s: SimulationState) -> dict[str, dict[str, float]]:
    conf: SystemConfiguration| None = s.data.get("yoda.configuration")
    if conf is not None and conf.elements is not None:
        return { f"Agent{a.agent_store.get("id")}": {"x": a.agent_store.get("pos")[0], "y":a.agent_store.get("pos")[1], "w":(1.0 if a.agent_state == walk else 0.0)} for a in conf.elements}
    return {}

model = YodaModel(lambda rng: generate_initial_state(rng, N), agents, 0.5, 1.0)
tracer = Tracer(DataTracer(["x", "y", "w"], extract_trace_info))
configuration = SimulationConfig(max_time=100)

tracer.run(model, configuration)

tables = tracer.data_tracer

viewer = TraceViewer(tables, lambda d: (d["x"], d["y"],"r" if d["w"] == 0.0 else "b"))
ani = viewer.create_animation(0, 100, 0.1)
ani.save(filename="examples/yoda/randomwalk.gif", writer="pillow")
