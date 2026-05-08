"""AgentDefinitions — registry and transition table for NBA agents.

:class:`AgentDefinitions` has two responsibilities:

1. **Agent registry** — it acts as a factory and cache for
   :class:`~pythia.models.nba.agent.Agent` instances.  Requesting the same
   name twice always returns the *same* object.

2. **Transition table** — it owns the per-agent labeled transitions.  Each
   entry maps a ``(source, channel)`` pair to a target-weight distribution,
   separately for each action type: send, receive, broadcast send, and
   broadcast receive.

   All four tables (``_send_transitions``, ``_receive_transitions``,
   ``_broadcast_send_transitions``, ``_broadcast_receive_transitions``) map
   each source :class:`~pythia.models.nba.agent.Agent` to a
   ``dict[Channel, dict[Agent, float]]``, keying on the channel.  The action
   type is implicitly encoded by the table chosen.

Example::

    from pythia.models.nba.agent_definitions import AgentDefinitions
    from pythia.models.nba.channel import Channel

    c = Channel("data")
    defs = AgentDefinitions()
    idle = defs.get("idle")
    busy = defs.get("busy")

    defs.add_send_transition(idle, c, busy, 1.0)     # idle --c!--> busy
    defs.add_receive_transition(busy, c, idle, 1.0)  # busy --c?--> idle

    assert defs.get("idle") is idle        # same instance
    assert c in defs.enabled_send(idle)
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from pythia.models.nba.agent import Agent
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment

# Channel-based table: Agent -> Channel -> (Agent -> float)
_ChannelTable = dict[Agent, dict[Channel, dict[Agent, float]]]


class AgentDefinitions:
    """Registry and transition table for :class:`~pythia.models.nba.agent.Agent` instances.

    Each name maps to exactly one :class:`~pythia.models.nba.agent.Agent`
    instance (:meth:`get`).  Transitions are recorded via :meth:`send`,
    :meth:`receive`, :meth:`broadcast_send`, and :meth:`broadcast_receive`,
    and queried via :meth:`enabled_send`, :meth:`enabled_receive`,
    :meth:`enabled_broadcast`, and :meth:`exit_rate`.

    All four transition tables store ``Agent -> Channel -> dict[Agent, float]``;
    the action type is encoded by which table is used.

    Example:
        >>> from pythia.models.nba.channel import Channel
        >>> c = Channel("x")
        >>> defs = AgentDefinitions()
        >>> a = defs.get("a")
        >>> b = defs.get("b")
        >>> defs.add_send_transition(a, c, b, 1.0)
        >>> c in defs.enabled_send(a)
        True
        >>> defs.get("a") is a
        True
    """

    def __init__(self) -> None:
        """Initialise an empty registry and empty per-action transition tables."""
        self._agents: dict[str, Agent] = {}
        self._send_transitions: _ChannelTable = defaultdict(dict)
        self._receive_transitions: _ChannelTable = defaultdict(dict)
        self._broadcast_send_transitions: _ChannelTable = defaultdict(dict)
        self._broadcast_receive_transitions: _ChannelTable = defaultdict(dict)

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def get(self, name: str) -> Agent:
        """Return the :class:`~pythia.models.nba.agent.Agent` registered under *name*.

        If no agent with *name* exists yet, one is created, stored, and
        returned.  Subsequent calls with the same *name* always return the
        same instance.

        Args:
            name: The agent name.  Must be a non-empty string (enforced by
                :class:`~pythia.models.nba.agent.Agent` itself).

        Returns:
            The unique :class:`~pythia.models.nba.agent.Agent` for *name*.

        Raises:
            ValueError: If *name* is empty.

        Example:
            >>> defs = AgentDefinitions()
            >>> x = defs.get("x")
            >>> defs.get("x") is x
            True
        """
        if name not in self._agents:
            self._agents[name] = Agent(name)
        return self._agents[name]

    # ------------------------------------------------------------------
    # Transition table — one insertion method per action type
    # ------------------------------------------------------------------

    def _record(self, table: _ChannelTable, source: Agent, channel: Channel, target: Agent, w: float) -> None:
        """Accumulate *w* into *table* for the ``(source, channel, target)`` triple."""
        if w <= 0.0:
            raise ValueError(f"Transition weight must be strictly positive, got {w!r}")
        inner = table[source].setdefault(channel, {})
        inner[target] = inner.get(target, 0.0) + w

    def add_send_transition(self, source: Agent, channel: Channel, target: Agent, w: float) -> None:
        """Record a point-to-point send transition ``source --channel!--> target``.

        If no transition exists yet for ``(source, channel, target)`` it is
        created with weight *w*.  Otherwise *w* is added to the existing weight.

        Args:
            source: The agent state initiating the send.
            channel: The channel on which the send is performed.
            target: The agent state reached after the send.
            w: A strictly positive weight to add.

        Raises:
            ValueError: If *w* is not strictly positive.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("x")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> c in defs.enabled_send(a)
            True
        """
        self._record(self._send_transitions, source, channel, target, w)

    def add_receive_transition(self, source: Agent, channel: Channel, target: Agent, w: float) -> None:
        """Record a point-to-point receive transition ``source --channel?--> target``.

        If no transition exists yet for ``(source, channel, target)`` it is
        created with weight *w*.  Otherwise *w* is added to the existing weight.

        Args:
            source: The agent state initiating the receive.
            channel: The channel on which the receive is performed.
            target: The agent state reached after the receive.
            w: A strictly positive weight to add.

        Raises:
            ValueError: If *w* is not strictly positive.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("x")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_receive_transition(a, c, b, 2.0)
            >>> c in defs.enabled_receive(a)
            True
        """
        self._record(self._receive_transitions, source, channel, target, w)

    def add_broadcast_send_transition(self, source: Agent, channel: Channel, target: Agent, w: float) -> None:
        """Record a broadcast send transition ``source --channel!!--> target``.

        If no transition exists yet for ``(source, channel, target)`` it is
        created with weight *w*.  Otherwise *w* is added to the existing weight.

        Args:
            source: The agent state initiating the broadcast send.
            channel: The channel on which the broadcast is performed.
            target: The agent state reached after the broadcast send.
            w: A strictly positive weight to add.

        Raises:
            ValueError: If *w* is not strictly positive.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("x")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_send_transition(a, c, b, 3.0)
            >>> c in defs.enabled_broadcast(a)
            True
        """
        self._record(self._broadcast_send_transitions, source, channel, target, w)

    def add_broadcast_receive_transition(self, source: Agent, channel: Channel, target: Agent, w: float) -> None:
        """Record a broadcast receive transition ``source --channel??--> target``.

        If no transition exists yet for ``(source, channel, target)`` it is
        created with weight *w*.  Otherwise *w* is added to the existing weight.

        Args:
            source: The agent state initiating the broadcast receive.
            channel: The channel on which the broadcast receive is performed.
            target: The agent state reached after the broadcast receive.
            w: A strictly positive weight to add.

        Raises:
            ValueError: If *w* is not strictly positive.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("x")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_receive_transition(a, c, b, 4.0)
            >>> c in defs.enabled_receive(a)
            False
            >>> c in defs.enabled_broadcast(a)
            False
        """
        self._record(self._broadcast_receive_transitions, source, channel, target, w)

    # ------------------------------------------------------------------
    # Transition execution
    # ------------------------------------------------------------------

    def receive(self, agent: Agent, channel: Channel, rng: random.Random) -> Agent:
        """Randomly select the next agent state after a receive on *channel*.

        Samples from the outgoing receive transitions of *agent* on *channel*,
        with each candidate target weighted by its stored transition weight.

        Args:
            agent: The agent performing the receive.
            channel: The channel on which the receive is performed.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected next :class:`~pythia.models.nba.agent.Agent` state.

        Raises:
            ValueError: If *agent* has no receive transitions on *channel*.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_receive_transition(a, c, b, 1.0)
            >>> defs.receive(a, c, random.Random(0)) == b
            True
        """
        targets = self._receive_transitions.get(agent, {}).get(channel)
        if not targets:
            raise ValueError(
                f"Agent {agent!r} has no receive transitions on channel {channel!r}"
            )
        agents = list(targets.keys())
        weights = [targets[t] for t in agents]
        return rng.choices(agents, weights=weights, k=1)[0]

    def send(self, agent: Agent, channel: Channel, rng: random.Random) -> Agent:
        """Randomly select the next agent state after a send on *channel*.

        Samples from the outgoing send transitions of *agent* on *channel*,
        with each candidate target weighted by its stored transition weight.

        Args:
            agent: The agent performing the send.
            channel: The channel on which the send is performed.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected next :class:`~pythia.models.nba.agent.Agent` state.

        Raises:
            ValueError: If *agent* has no send transitions on *channel*.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> defs.send(a, c, random.Random(0)) == b
            True
        """
        targets = self._send_transitions.get(agent, {}).get(channel)
        if not targets:
            raise ValueError(
                f"Agent {agent!r} has no send transitions on channel {channel!r}"
            )
        agents = list(targets.keys())
        weights = [targets[t] for t in agents]
        return rng.choices(agents, weights=weights, k=1)[0]

    def broadcast_send(self, agent: Agent, channel: Channel, rng: random.Random) -> Agent:
        """Randomly select the next agent state after a broadcast send on *channel*.

        Samples from the outgoing broadcast-send transitions of *agent* on
        *channel*, with each candidate target weighted by its stored transition
        weight.

        Args:
            agent: The agent performing the broadcast send.
            channel: The channel on which the broadcast send is performed.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected next :class:`~pythia.models.nba.agent.Agent` state.

        Raises:
            ValueError: If *agent* has no broadcast-send transitions on *channel*.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_send_transition(a, c, b, 1.0)
            >>> defs.broadcast_send(a, c, random.Random(0)) == b
            True
        """
        targets = self._broadcast_send_transitions.get(agent, {}).get(channel)
        if not targets:
            raise ValueError(
                f"Agent {agent!r} has no broadcast-send transitions on channel {channel!r}"
            )
        agents = list(targets.keys())
        weights = [targets[t] for t in agents]
        return rng.choices(agents, weights=weights, k=1)[0]

    def select_send_channel(
        self,
        agent: Agent,
        env: Environment,
        enabled: Callable[[Channel], bool],
        rng: random.Random,
    ) -> Channel:
        """Randomly select an enabled send channel for *agent*, weighted by send rate.

        Considers only the channels on which *agent* has a send transition
        **and** for which ``enabled(channel)`` returns ``True``.  Among those,
        one channel is drawn at random with probability proportional to
        ``env.send_rate(agent, channel)``.

        Args:
            agent: The agent performing the send.
            env: The :class:`~pythia.models.nba.environment.Environment`
                providing per-agent, per-channel send rates.
            enabled: A predicate ``Channel -> bool`` indicating which send
                channels are currently enabled.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected :class:`~pythia.models.nba.channel.Channel`.

        Raises:
            ValueError: If *agent* has no enabled send channels.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> env = Environment()
            >>> env.set_send_rate(a, c, 2.0)
            >>> defs.select_send_channel(a, env, lambda ch: True, random.Random(0)) == c
            True
        """
        channels = list(self.enabled_send(agent, p=enabled))
        if not channels:
            raise ValueError(
                f"Agent {agent!r} has no enabled send channels"
            )
        weights = [env.send_rate(agent, ch) for ch in channels]
        return rng.choices(channels, weights=weights, k=1)[0]

    def select_broadcast_send_channel(
        self,
        agent: Agent,
        env: Environment,
        rng: random.Random,
    ) -> Channel:
        """Randomly select a broadcast-send channel for *agent*, weighted by broadcast rate.

        Considers all channels on which *agent* has a broadcast-send transition
        and draws one at random with probability proportional to
        ``env.broadcast_rate(agent, channel)``.

        Args:
            agent: The agent performing the broadcast send.
            env: The :class:`~pythia.models.nba.environment.Environment`
                providing per-agent, per-channel broadcast rates.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected :class:`~pythia.models.nba.channel.Channel`.

        Raises:
            ValueError: If *agent* has no broadcast-send transitions.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_send_transition(a, c, b, 1.0)
            >>> env = Environment()
            >>> env.set_broadcast_rate(a, c, 2.0)
            >>> defs.select_broadcast_send_channel(a, env, random.Random(0)) == c
            True
        """
        channels = list(self.enabled_broadcast(agent))
        if not channels:
            raise ValueError(f"Agent {agent!r} has no broadcast-send transitions")
        weights = [env.broadcast_rate(agent, ch) for ch in channels]
        return rng.choices(channels, weights=weights, k=1)[0]

    def broadcast_receive(self, agent: Agent, channel: Channel, rng: random.Random) -> Agent:
        """Randomly select the next agent state after a broadcast receive on *channel*.

        Samples from the outgoing broadcast-receive transitions of *agent* on
        *channel*, with each candidate target weighted by its stored transition
        weight.

        Args:
            agent: The agent performing the broadcast receive.
            channel: The channel on which the broadcast receive is performed.
            rng: A :class:`random.Random` instance used for the weighted
                selection.

        Returns:
            The selected next :class:`~pythia.models.nba.agent.Agent` state, or
            *agent* itself if it has no broadcast-receive transitions on *channel*.

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_receive_transition(a, c, b, 1.0)
            >>> defs.broadcast_receive(a, c, random.Random(0)) == b
            True
            >>> defs.broadcast_receive(b, c, random.Random(0)) == b
            True
        """
        targets = self._broadcast_receive_transitions.get(agent, {}).get(channel)
        if not targets:
            return agent
        agents = list(targets.keys())
        weights = [targets[t] for t in agents]
        return rng.choices(agents, weights=weights, k=1)[0]

    def step(
        self,
        agent: Agent,
        env: Environment,
        enabled: Callable[[Channel], bool],
        rng: random.Random,
    ) -> tuple[bool, Channel, Agent]:
        """Randomly execute one send or broadcast-send action for *agent*.

        Computes the total rate of enabled point-to-point send actions and the
        total rate of broadcast-send actions from the environment, then draws
        one action type proportional to those totals.  The selected action type
        then determines the channel (via :meth:`select_send_channel` or
        :meth:`select_broadcast_send_channel`) and the resulting agent state
        (via :meth:`send` or :meth:`broadcast_send`).

        Args:
            agent: The agent performing the step.
            env: The :class:`~pythia.models.nba.environment.Environment`
                providing per-agent, per-channel rates.
            enabled: A predicate ``Channel -> bool`` indicating which
                point-to-point send channels are currently enabled.
            rng: A :class:`random.Random` instance used for all random
                selections.

        Returns:
            A triple ``(is_broadcast, channel, next_agent)`` where
            *is_broadcast* is ``True`` when a broadcast-send was performed and
            ``False`` for a point-to-point send, *channel* is the channel used,
            and *next_agent* is the resulting agent state.

        Raises:
            ValueError: If *agent* has no enabled send or broadcast-send
                actions (total exit rate is zero).

        Example:
            >>> import random
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> env = Environment()
            >>> env.set_send_rate(a, c, 1.0)
            >>> is_broadcast, ch, next_a = defs.step(a, env, lambda ch: True, random.Random(0))
            >>> is_broadcast
            False
            >>> ch == c
            True
            >>> next_a == b
            True
        """
        send_channels = list(self.enabled_send(agent, p=enabled))
        broadcast_channels = list(self.enabled_broadcast(agent))

        send_total = sum(env.send_rate(agent, ch) for ch in send_channels)
        broadcast_total = sum(env.broadcast_rate(agent, ch) for ch in broadcast_channels)

        if send_total + broadcast_total == 0.0:
            raise ValueError(f"Agent {agent!r} has no enabled send or broadcast-send actions")

        is_broadcast = rng.choices(
            [False, True],
            weights=[send_total, broadcast_total],
            k=1,
        )[0]

        if is_broadcast:
            channel = self.select_broadcast_send_channel(agent, env, rng)
            next_agent = self.broadcast_send(agent, channel, rng)
        else:
            channel = self.select_send_channel(agent, env, enabled, rng)
            next_agent = self.send(agent, channel, rng)

        return is_broadcast, channel, next_agent

    # ------------------------------------------------------------------
    # Enabled-action queries
    # ------------------------------------------------------------------

    def is_receive_enabled(self, agent: Agent, channel: Channel) -> bool:
        """Return ``True`` if *agent* has a receive transition on *channel*.

        Args:
            agent: The agent state to query.
            channel: The channel to check.

        Returns:
            ``True`` when at least one transition
            ``agent --channel?--> target`` has been recorded, ``False``
            otherwise.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.is_receive_enabled(a, c)
            False
            >>> defs.add_receive_transition(a, c, b, 1.0)
            >>> defs.is_receive_enabled(a, c)
            True
            >>> defs.is_receive_enabled(a, d)
            False
        """
        return channel in self._receive_transitions.get(agent, {})

    def is_broadcast_receive_enabled(self, agent: Agent, channel: Channel) -> bool:
        """Return ``True`` if *agent* has a broadcast-receive transition on *channel*.

        Args:
            agent: The agent state to query.
            channel: The channel to check.

        Returns:
            ``True`` when at least one transition
            ``agent --channel??--> target`` has been recorded, ``False``
            otherwise.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.is_broadcast_receive_enabled(a, c)
            False
            >>> defs.add_broadcast_receive_transition(a, c, b, 1.0)
            >>> defs.is_broadcast_receive_enabled(a, c)
            True
            >>> defs.is_broadcast_receive_enabled(a, d)
            False
        """
        return channel in self._broadcast_receive_transitions.get(agent, {})

    def enabled_receive(self, source: Agent) -> set[Channel]:
        """Return the set of channels on which *source* can perform a receive.

        A channel is included when at least one transition
        ``source --channel?--> target`` has been recorded.

        Args:
            source: The agent state to query.

        Returns:
            A :class:`set` of :class:`~pythia.models.nba.channel.Channel`
            objects.  The set is empty when *source* has no receive
            transitions.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.enabled_receive(a)
            set()
            >>> defs.add_receive_transition(a, c, b, 1.0)
            >>> defs.add_receive_transition(a, d, b, 1.0)
            >>> defs.enabled_receive(a) == {c, d}
            True
        """
        return set(self._receive_transitions.get(source, {}))

    def enabled_send(
        self,
        source: Agent,
        p: Callable[[Channel], bool] = lambda _: True,
    ) -> set[Channel]:
        """Return the channels on which *source* can perform a send, filtered by *p*.

        A channel is included when at least one transition
        ``source --channel!--> target`` has been recorded **and** ``p(channel)``
        returns ``True``.  When *p* is omitted all recorded channels are
        returned.

        Args:
            source: The agent state to query.
            p: A predicate ``Channel -> bool``.  Only channels for which *p*
                returns ``True`` are included in the result.  Defaults to the
                predicate that accepts every channel.

        Returns:
            A :class:`set` of :class:`~pythia.models.nba.channel.Channel`
            objects.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.enabled_send(a)
            set()
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> defs.add_send_transition(a, d, b, 1.0)
            >>> defs.enabled_send(a) == {c, d}
            True
            >>> defs.enabled_send(a, p=lambda ch: ch.name == "c") == {c}
            True
        """
        return {ch for ch in self._send_transitions.get(source, {}) if p(ch)}

    def enabled_broadcast(self, source: Agent) -> set[Channel]:
        """Return the set of channels on which *source* can perform a broadcast send.

        A channel is included when at least one transition
        ``source --channel!!--> target`` has been recorded.

        Args:
            source: The agent state to query.

        Returns:
            A :class:`set` of :class:`~pythia.models.nba.channel.Channel`
            objects.  The set is empty when *source* has no broadcast-send
            transitions.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.enabled_broadcast(a)
            set()
            >>> defs.add_broadcast_send_transition(a, c, b, 1.0)
            >>> defs.add_broadcast_send_transition(a, d, b, 1.0)
            >>> defs.enabled_broadcast(a) == {c, d}
            True
        """
        return set(self._broadcast_send_transitions.get(source, {}))

    def exit_rate(
        self,
        agent: Agent,
        env: Environment,
        enabled: Callable[[Channel], bool],
    ) -> float:
        """Return the total exit rate for *agent* under the given environment.

        The exit rate is the sum of:

        * ``env.broadcast_rate(agent, c)`` for every channel *c* on which
          *agent* can perform a broadcast-send action, and
        * ``env.send_rate(agent, c)`` for every channel *c* on which *agent*
          can perform a (point-to-point) send action **and** ``enabled(c)``
          returns ``True``.

        Args:
            agent: The agent whose exit rate is computed.
            env: The :class:`~pythia.models.nba.environment.Environment`
                providing the per-agent, per-channel rates.
            enabled: A predicate ``Channel -> bool`` indicating which
                point-to-point send actions are currently enabled (typically
                obtained from
                :meth:`~pythia.models.nba.nba_configuration.NBAConfiguration.get_is_enabled_predicate`).

        Returns:
            A non-negative float representing the total exit rate.

        Example:
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c, d = Channel("c"), Channel("d")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> defs.add_broadcast_send_transition(a, d, b, 1.0)
            >>> env = Environment()
            >>> env.set_send_rate(a, c, 2.0)
            >>> env.set_broadcast_rate(a, d, 3.0)
            >>> defs.exit_rate(a, env, lambda ch: True)
            5.0
            >>> defs.exit_rate(a, env, lambda ch: False)
            3.0
        """
        total = 0.0
        for c in self.enabled_broadcast(agent):
            total += env.broadcast_rate(agent, c)
        for c in self.enabled_send(agent, p=enabled):
            total += env.send_rate(agent, c)
        return total

