"""Environment — per-agent, per-channel rates and probabilities for NBA systems.

An :class:`Environment` stores four kinds of numeric parameters.  Each kind
has both an **agent-specific** table (indexed by ``(Agent, Channel)``) and a
**default** table (indexed by ``Channel`` alone).  The getters resolve values
with the following priority:

1. Agent-specific entry ``_agent_X[agent][channel]``.
2. Default entry ``_default_X[channel]``.
3. ``0.0`` — when neither table contains the channel.

The four parameter kinds are:

* **send rate** — the rate at which an agent executes a point-to-point send.
* **broadcast rate** — the rate at which an agent executes a broadcast send.
* **receive probability** — the probability that an agent receives a broadcast
  message (must be strictly between 0 and 1).
* **receive weight** — the weight used to derive the probability that an agent
  receives a point-to-point message.

Example::

    from pythia.models.nba.agent import Agent
    from pythia.models.nba.channel import Channel
    from pythia.models.nba.environment import Environment

    a = Agent("a")
    b = Agent("b")
    c = Channel("data")
    env = Environment()

    env.set_default_send_rate(c, 1.0)   # fallback for all agents
    env.set_send_rate(a, c, 2.5)        # overrides the default for agent a

    env.send_rate(a, c)   # 2.5  (agent-specific)
    env.send_rate(b, c)   # 1.0  (default)
"""

from __future__ import annotations

from pythia.models.nba.agent import Agent
from pythia.models.nba.channel import Channel


class Environment:
    """Per-agent, per-channel rates and probabilities for NBA interactions.

    Values are resolved with three-level priority: agent-specific entry,
    then channel default, then ``0.0``.

    Attributes:
        _agent_send_rate: ``Agent -> Channel -> float`` (positive).
        _agent_broadcast_rate: ``Agent -> Channel -> float`` (positive).
        _agent_receive_probability: ``Agent -> Channel -> float`` (in ``(0, 1)``).
        _agent_receive_weight: ``Agent -> Channel -> float`` (positive).
        _default_send_rate: ``Channel -> float`` (positive).
        _default_broadcast_rate: ``Channel -> float`` (positive).
        _default_receive_probability: ``Channel -> float`` (in ``(0, 1)``).
        _default_receive_weight: ``Channel -> float`` (positive).

    Examples:
        >>> from pythia.models.nba.agent import Agent
        >>> from pythia.models.nba.channel import Channel
        >>> a = Agent("a")
        >>> c = Channel("c")
        >>> env = Environment()
        >>> env.set_default_send_rate(c, 1.0)
        >>> env.send_rate(a, c)
        1.0
        >>> env.set_send_rate(a, c, 5.0)
        >>> env.send_rate(a, c)
        5.0
    """

    def __init__(self) -> None:
        """Initialise an environment with all maps empty."""
        self._agent_send_rate: dict[Agent, dict[Channel, float]] = {}
        self._agent_broadcast_rate: dict[Agent, dict[Channel, float]] = {}
        self._agent_receive_probability: dict[Agent, dict[Channel, float]] = {}
        self._agent_receive_weight: dict[Agent, dict[Channel, float]] = {}

        self._default_send_rate: dict[Channel, float] = {}
        self._default_broadcast_rate: dict[Channel, float] = {}
        self._default_receive_probability: dict[Channel, float] = {}
        self._default_receive_weight: dict[Channel, float] = {}

    # ------------------------------------------------------------------
    # Agent-specific setters
    # ------------------------------------------------------------------

    def set_send_rate(self, agent: Agent, channel: Channel, rate: float) -> None:
        """Set the send rate of *agent* on *channel* to *rate*.

        Args:
            agent: The agent whose send rate is being defined.
            channel: The channel on which the rate applies.
            rate: A strictly positive float.

        Raises:
            ValueError: If *rate* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_send_rate(a, c, 3.0)
            >>> env.send_rate(a, c)
            3.0
        """
        if rate <= 0.0:
            raise ValueError(f"Send rate must be strictly positive, got {rate!r}")
        self._agent_send_rate.setdefault(agent, {})[channel] = rate

    def set_broadcast_rate(self, agent: Agent, channel: Channel, rate: float) -> None:
        """Set the broadcast rate of *agent* on *channel* to *rate*.

        Args:
            agent: The agent whose broadcast rate is being defined.
            channel: The channel on which the rate applies.
            rate: A strictly positive float.

        Raises:
            ValueError: If *rate* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_broadcast_rate(a, c, 0.5)
            >>> env.broadcast_rate(a, c)
            0.5
        """
        if rate <= 0.0:
            raise ValueError(f"Broadcast rate must be strictly positive, got {rate!r}")
        self._agent_broadcast_rate.setdefault(agent, {})[channel] = rate

    def set_receive_probability(self, agent: Agent, channel: Channel, prob: float) -> None:
        """Set the broadcast-receive probability of *agent* on *channel* to *prob*.

        Args:
            agent: The agent whose receive probability is being defined.
            channel: The channel on which the probability applies.
            prob: A float strictly between 0 and 1 (exclusive).

        Raises:
            ValueError: If *prob* is not in the open interval ``(0, 1)``.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_receive_probability(a, c, 0.7)
            >>> env.receive_probability(a, c)
            0.7
        """
        if not (0.0 < prob < 1.0):
            raise ValueError(
                f"Receive probability must be strictly between 0 and 1, got {prob!r}"
            )
        self._agent_receive_probability.setdefault(agent, {})[channel] = prob

    def set_receive_weight(self, agent: Agent, channel: Channel, weight: float) -> None:
        """Set the receive weight of *agent* on *channel* to *weight*.

        Args:
            agent: The agent whose receive weight is being defined.
            channel: The channel on which the weight applies.
            weight: A strictly positive float.

        Raises:
            ValueError: If *weight* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_receive_weight(a, c, 2.0)
            >>> env.receive_weight(a, c)
            2.0
        """
        if weight <= 0.0:
            raise ValueError(f"Receive weight must be strictly positive, got {weight!r}")
        self._agent_receive_weight.setdefault(agent, {})[channel] = weight

    # ------------------------------------------------------------------
    # Default setters
    # ------------------------------------------------------------------

    def set_default_send_rate(self, channel: Channel, rate: float) -> None:
        """Set the default send rate for *channel*.

        Used as a fallback by :meth:`send_rate` when no agent-specific entry
        exists.

        Args:
            channel: The channel on which the default applies.
            rate: A strictly positive float.

        Raises:
            ValueError: If *rate* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_default_send_rate(c, 1.0)
            >>> env.send_rate(a, c)
            1.0
        """
        if rate <= 0.0:
            raise ValueError(f"Default send rate must be strictly positive, got {rate!r}")
        self._default_send_rate[channel] = rate

    def set_default_broadcast_rate(self, channel: Channel, rate: float) -> None:
        """Set the default broadcast rate for *channel*.

        Used as a fallback by :meth:`broadcast_rate` when no agent-specific
        entry exists.

        Args:
            channel: The channel on which the default applies.
            rate: A strictly positive float.

        Raises:
            ValueError: If *rate* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_default_broadcast_rate(c, 0.5)
            >>> env.broadcast_rate(a, c)
            0.5
        """
        if rate <= 0.0:
            raise ValueError(f"Default broadcast rate must be strictly positive, got {rate!r}")
        self._default_broadcast_rate[channel] = rate

    def set_default_receive_probability(self, channel: Channel, prob: float) -> None:
        """Set the default broadcast-receive probability for *channel*.

        Used as a fallback by :meth:`receive_probability` when no
        agent-specific entry exists.

        Args:
            channel: The channel on which the default applies.
            prob: A float strictly between 0 and 1 (exclusive).

        Raises:
            ValueError: If *prob* is not in the open interval ``(0, 1)``.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_default_receive_probability(c, 0.6)
            >>> env.receive_probability(a, c)
            0.6
        """
        if not (0.0 < prob < 1.0):
            raise ValueError(
                f"Default receive probability must be strictly between 0 and 1, got {prob!r}"
            )
        self._default_receive_probability[channel] = prob

    def set_default_receive_weight(self, channel: Channel, weight: float) -> None:
        """Set the default receive weight for *channel*.

        Used as a fallback by :meth:`receive_weight` when no agent-specific
        entry exists.

        Args:
            channel: The channel on which the default applies.
            weight: A strictly positive float.

        Raises:
            ValueError: If *weight* is not strictly positive.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, c = Agent("a"), Channel("c")
            >>> env = Environment()
            >>> env.set_default_receive_weight(c, 3.0)
            >>> env.receive_weight(a, c)
            3.0
        """
        if weight <= 0.0:
            raise ValueError(f"Default receive weight must be strictly positive, got {weight!r}")
        self._default_receive_weight[channel] = weight

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def send_rate(self, agent: Agent, channel: Channel) -> float:
        """Return the send rate of *agent* on *channel*.

        Resolution order: agent-specific entry, then channel default, then
        ``0.0``.

        Args:
            agent: The agent to query.
            channel: The channel to query.

        Returns:
            The resolved rate.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, b, c = Agent("a"), Agent("b"), Channel("c")
            >>> env = Environment()
            >>> env.send_rate(a, c)
            0.0
            >>> env.set_default_send_rate(c, 1.0)
            >>> env.send_rate(b, c)
            1.0
            >>> env.set_send_rate(a, c, 5.0)
            >>> env.send_rate(a, c)
            5.0
        """
        agent_val = self._agent_send_rate.get(agent, {}).get(channel)
        if agent_val is not None:
            return agent_val
        return self._default_send_rate.get(channel, 0.0)

    def broadcast_rate(self, agent: Agent, channel: Channel) -> float:
        """Return the broadcast rate of *agent* on *channel*.

        Resolution order: agent-specific entry, then channel default, then
        ``0.0``.

        Args:
            agent: The agent to query.
            channel: The channel to query.

        Returns:
            The resolved rate.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, b, c = Agent("a"), Agent("b"), Channel("c")
            >>> env = Environment()
            >>> env.broadcast_rate(a, c)
            0.0
            >>> env.set_default_broadcast_rate(c, 0.25)
            >>> env.broadcast_rate(b, c)
            0.25
            >>> env.set_broadcast_rate(a, c, 2.0)
            >>> env.broadcast_rate(a, c)
            2.0
        """
        agent_val = self._agent_broadcast_rate.get(agent, {}).get(channel)
        if agent_val is not None:
            return agent_val
        return self._default_broadcast_rate.get(channel, 0.0)

    def receive_probability(self, agent: Agent, channel: Channel) -> float:
        """Return the broadcast-receive probability of *agent* on *channel*.

        Resolution order: agent-specific entry, then channel default, then
        ``0.0``.

        Args:
            agent: The agent to query.
            channel: The channel to query.

        Returns:
            The resolved probability.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, b, c = Agent("a"), Agent("b"), Channel("c")
            >>> env = Environment()
            >>> env.receive_probability(a, c)
            0.0
            >>> env.set_default_receive_probability(c, 0.4)
            >>> env.receive_probability(b, c)
            0.4
            >>> env.set_receive_probability(a, c, 0.9)
            >>> env.receive_probability(a, c)
            0.9
        """
        agent_val = self._agent_receive_probability.get(agent, {}).get(channel)
        if agent_val is not None:
            return agent_val
        return self._default_receive_probability.get(channel, 0.0)

    def receive_weight(self, agent: Agent, channel: Channel) -> float:
        """Return the receive weight of *agent* on *channel*.

        Resolution order: agent-specific entry, then channel default, then
        ``0.0``.

        Args:
            agent: The agent to query.
            channel: The channel to query.

        Returns:
            The resolved weight.

        Examples:
            >>> from pythia.models.nba.agent import Agent
            >>> from pythia.models.nba.channel import Channel
            >>> a, b, c = Agent("a"), Agent("b"), Channel("c")
            >>> env = Environment()
            >>> env.receive_weight(a, c)
            0.0
            >>> env.set_default_receive_weight(c, 3.0)
            >>> env.receive_weight(b, c)
            3.0
            >>> env.set_receive_weight(a, c, 7.0)
            >>> env.receive_weight(a, c)
            7.0
        """
        agent_val = self._agent_receive_weight.get(agent, {}).get(channel)
        if agent_val is not None:
            return agent_val
        return self._default_receive_weight.get(channel, 0.0)
