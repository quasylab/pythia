"""Network of Broadcasting Agents (``nba``) submodule.

Provides simulation models based on the Network of Broadcasting Agents (NBA)
calculus, a process calculus in which a system is a population of agents that
interact via two mechanisms:

* **Point-to-point** channel communication — one agent sends (``c!``) and
  exactly one other agent receives (``c?``) on the same channel ``c``.
* **Broadcast** communication — one agent broadcasts (``c!!``) and all agents
  currently able to receive (``c??``) on channel ``c`` do so simultaneously.

All models use namespaced state keys (``"nba.*"``) to avoid collisions with
other modules.
"""

from pythia.models.nba.action import (
    Action,
    BroadcastReceive,
    BroadcastSend,
    Receive,
    Send,
)
from pythia.models.nba.agent import Agent
from pythia.models.nba.agent_definitions import AgentDefinitions
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment
from pythia.models.nba.nba_configuration import NBAConfiguration
from pythia.models.nba.nba_model import NBAModel

__all__ = [
    "Action",
    "Agent",
    "AgentDefinitions",
    "BroadcastReceive",
    "BroadcastSend",
    "Channel",
    "Environment",
    "NBAConfiguration",
    "NBAModel",
    "Receive",
    "Send",
]
