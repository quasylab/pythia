"""NBAConfiguration — a population of NBA agents stored as a counting vector.

Rather than keeping one entry per individual, an :class:`NBAConfiguration`
stores each distinct :class:`~pythia.models.nba.agent.Agent` together with
the number of copies present in the population.

Example::

    from pythia.models.nba.agent import Agent
    from pythia.models.nba.nba_configuration import NBAConfiguration

    a = Agent("a")
    b = Agent("b")
    cfg = NBAConfiguration({a: 3, b: 1})

    cfg.size          # 4
    cfg.count(a)      # 3
    cfg.fraction(a)   # 0.75
"""

from __future__ import annotations

import random
from collections.abc import Callable

from pythia.models.nba.agent import Agent
from pythia.models.nba.agent_definitions import AgentDefinitions
from pythia.models.nba.channel import Channel
from pythia.models.nba.environment import Environment


class NBAConfiguration:
    """An immutable population of NBA agents stored as a counting vector.

    The population is represented as a mapping ``Agent -> int`` where each
    value is the number of copies of that agent present.  Every count must
    be a strictly positive integer; the mapping must be non-empty.

    Args:
        counts: A mapping from :class:`~pythia.models.nba.agent.Agent` to
            a strictly positive integer indicating how many copies of that
            agent are in the configuration.

    Raises:
        ValueError: If *counts* is empty or if any count is not a strictly
            positive integer.

    Example:
        >>> a = Agent("a")
        >>> b = Agent("b")
        >>> cfg = NBAConfiguration({a: 2, b: 1})
        >>> cfg.size
        3
        >>> cfg.count(a)
        2
        >>> cfg.fraction(b)
        0.3333333333333333
    """

    def __init__(self, counts: dict[Agent, int]) -> None:
        if not counts:
            raise ValueError("NBAConfiguration must contain at least one agent")
        for agent, n in counts.items():
            if not isinstance(n, int) or n <= 0:
                raise ValueError(
                    f"Count for agent {agent!r} must be a strictly positive integer, got {n!r}"
                )
        object.__setattr__(self, "_counts", dict(counts))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("NBAConfiguration is immutable")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def counts(self) -> dict[Agent, int]:
        """A copy of the agent→count mapping.

        Example:
            >>> a = Agent("a")
            >>> cfg = NBAConfiguration({a: 3})
            >>> cfg.counts == {a: 3}
            True
        """
        return dict(self._counts)

    @property
    def size(self) -> int:
        """The total number of agents (sum of all counts).

        Example:
            >>> a = Agent("a")
            >>> b = Agent("b")
            >>> NBAConfiguration({a: 2, b: 3}).size
            5
        """
        return sum(self._counts.values())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count(self, agent: Agent) -> int:
        """Return the number of copies of *agent* in the configuration.

        Args:
            agent: The agent to look up.

        Returns:
            The count for *agent*, or ``0`` if *agent* is not present.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> cfg = NBAConfiguration({a: 4})
            >>> cfg.count(a)
            4
            >>> cfg.count(b)
            0
        """
        return self._counts.get(agent, 0)

    def fraction(self, agent: Agent) -> float:
        """Return the fraction of agents equal to *agent*.

        Computed as ``count(agent) / size``.

        Args:
            agent: The agent whose fraction to compute.

        Returns:
            A float in ``[0.0, 1.0]``.

        Example:
            >>> a, b = Agent("a"), Agent("b")
            >>> cfg = NBAConfiguration({a: 1, b: 3})
            >>> cfg.fraction(a)
            0.25
            >>> cfg.fraction(b)
            0.75
        """
        return self.count(agent) / self.size

    def get_is_enabled_predicate(
        self,
        receivers: dict[Channel, set[Agent]],
        agent: Agent,
    ) -> Callable[[Channel], bool]:
        """Return a predicate that tests whether a send action is enabled for *agent*.

        A send action over channel *c* is enabled at *agent* when at least one
        receiver exists in the current configuration:

        * There is an agent *agt2* ≠ *agent* in ``receivers[c]``, **or**
        * *agent* is in ``receivers[c]`` and the number of copies of *agent*
          in the configuration is greater than 1 (so one copy can send while
          another receives).

        Args:
            receivers: A mapping from each :class:`~pythia.models.nba.channel.Channel`
                to the set of :class:`~pythia.models.nba.agent.Agent` types
                that can receive on it.  Typically obtained from
                :meth:`get_receivers`.
            agent: The agent for which send enablement is tested.

        Returns:
            A callable ``Channel -> bool``.

        Example:
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a = defs.get("a")
            >>> b = defs.get("b")
            >>> defs.add_receive_transition(b, c, a, 1.0)
            >>> cfg = NBAConfiguration({a: 1, b: 1})
            >>> receivers = cfg.get_receivers(defs)
            >>> pred_a = cfg.get_is_enabled_predicate(receivers, a)
            >>> pred_a(c)
            True
            >>> pred_b = cfg.get_is_enabled_predicate(receivers, b)
            >>> pred_b(c)
            False
        """
        def is_enabled(c: Channel) -> bool:
            receivers_of_c = receivers.get(c, set())
            if receivers_of_c - {agent}:
                return True
            return agent in receivers_of_c and self.count(agent) > 1

        return is_enabled

    def enabled_receive(self, adef: AgentDefinitions) -> dict[Agent, set[Channel]]:
        """Return the channels on which each agent in the configuration can receive.

        For every distinct agent type present, the result maps that agent to
        the set of channels returned by
        :meth:`~pythia.models.nba.agent_definitions.AgentDefinitions.enabled_receive`.

        Args:
            adef: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
                that owns the transition table.

        Returns:
            A ``dict`` mapping each :class:`~pythia.models.nba.agent.Agent` in
            the configuration to a :class:`set` of
            :class:`~pythia.models.nba.channel.Channel` objects.

        Example:
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a = defs.get("a")
            >>> b = defs.get("b")
            >>> defs.add_receive_transition(a, c, b, 1.0)
            >>> cfg = NBAConfiguration({a: 2, b: 1})
            >>> cfg.enabled_receive(defs) == {a: {c}, b: set()}
            True
        """
        return {agent: adef.enabled_receive(agent) for agent in self._counts}

    def get_receivers(self, adef: AgentDefinitions) -> dict[Channel, set[Agent]]:
        """Return a map from each channel to the agents that can receive on it.

        Iterates every distinct agent type in the configuration and collects
        the channels on which it can receive (via
        :meth:`~pythia.models.nba.agent_definitions.AgentDefinitions.enabled_receive`).
        The result is the inverse view: for each such channel, the set of
        agent types that have at least one receive transition on it.  Agent
        types with no receive transitions do not appear in the result.

        Args:
            adef: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
                that owns the transition table.

        Returns:
            A ``dict`` mapping each :class:`~pythia.models.nba.channel.Channel`
            to the :class:`set` of :class:`~pythia.models.nba.agent.Agent`
            types in the configuration that can receive on it.

        Example:
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a = defs.get("a")
            >>> b = defs.get("b")
            >>> defs.add_receive_transition(a, c, b, 1.0)
            >>> defs.add_receive_transition(b, c, a, 1.0)
            >>> cfg = NBAConfiguration({a: 1, b: 2})
            >>> cfg.get_receivers(defs) == {c: {a, b}}
            True
        """
        result: dict[Channel, set[Agent]] = {}
        for agent in self._counts:
            for channel in adef.enabled_receive(agent):
                result.setdefault(channel, set()).add(agent)
        return result

    def broadcast_send(
        self,
        rng: random.Random,
        adef: AgentDefinitions,
        source: Agent,
        c: Channel,
        target: Agent,
    ) -> NBAConfiguration:
        """Return the configuration produced by a broadcast-send action.

        Removes one instance of *source*, applies a broadcast-receive
        transition on channel *c* to every remaining agent (using
        :meth:`~pythia.models.nba.agent_definitions.AgentDefinitions.broadcast_receive`),
        then adds *target*.

        Args:
            rng: Random number generator used for the broadcast-receive
                transitions.
            adef: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
                that owns the broadcast-receive transition table.
            source: The agent performing the send (one instance is removed).
            c: The channel on which the send is performed.
            target: The agent state that *source* transitions into.

        Returns:
            A new :class:`NBAConfiguration` representing the successor state.

        Raises:
            ValueError: If *source* is not present in the configuration.

        Example:
            >>> import random
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_broadcast_receive_transition(b, c, a, 1.0)
            >>> cfg = NBAConfiguration({a: 1, b: 2})
            >>> new_cfg = cfg.broadcast_send(random.Random(0), defs, a, c, b)
            >>> new_cfg.count(a)
            2
            >>> new_cfg.count(b)
            1
        """
        if self.count(source) == 0:
            raise ValueError(f"Agent {source!r} is not present in the configuration")

        # Remove one instance of source.
        remaining: dict[Agent, int] = {}
        for agent, n in self._counts.items():
            if agent == source:
                if n > 1:
                    remaining[agent] = n - 1
            else:
                remaining[agent] = n

        # Apply broadcast_receive to every remaining individual.
        new_counts: dict[Agent, int] = {}
        for agent, n in remaining.items():
            for _ in range(n):
                next_agent = adef.broadcast_receive(agent, c, rng)
                new_counts[next_agent] = new_counts.get(next_agent, 0) + 1

        # Add the target agent.
        new_counts[target] = new_counts.get(target, 0) + 1

        return NBAConfiguration(new_counts)

    def send(
        self,
        rng: random.Random,
        adef: AgentDefinitions,
        env: Environment,
        source: Agent,
        c: Channel,
        target: Agent,
        receivers: set[Agent],
    ) -> NBAConfiguration:
        """Return the configuration produced by a point-to-point send action.

        Removes one instance of *source*, selects one receiving agent from
        *receivers* weighted by ``count(agent) * env.receive_weight(agent, c)``,
        performs a receive transition on that agent via
        :meth:`~pythia.models.nba.agent_definitions.AgentDefinitions.receive`,
        then adds *target*.

        The receiver is selected from the configuration that remains after
        *source* has been removed, so if *source* is also a receiver its
        effective weight is reduced accordingly.

        Args:
            rng: Random number generator used for both the receiver selection
                and the receive transition.
            adef: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
                that owns the receive transition table.
            env: The :class:`~pythia.models.nba.environment.Environment` that
                provides receive weights.
            source: The agent performing the send (one instance is removed).
            c: The channel on which the send is performed.
            target: The agent state that *source* transitions into.
            receivers: The set of agent types whose receive on *c* is enabled.

        Returns:
            A new :class:`NBAConfiguration` representing the successor state.

        Raises:
            ValueError: If *source* is not present in the configuration.
            ValueError: If no agent in *receivers* is present in the
                remaining configuration.

        Example:
            >>> import random
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_receive_transition(b, c, a, 1.0)
            >>> env = Environment()
            >>> env.set_receive_weight(b, c, 1.0)
            >>> cfg = NBAConfiguration({a: 1, b: 1})
            >>> new_cfg = cfg.send(random.Random(0), defs, env, a, c, a, {b})
            >>> new_cfg.count(a)
            2
            >>> new_cfg.count(b)
            0
        """
        if self.count(source) == 0:
            raise ValueError(f"Agent {source!r} is not present in the configuration")

        # Remove one instance of source.
        remaining: dict[Agent, int] = {}
        for agent, n in self._counts.items():
            if agent == source:
                if n > 1:
                    remaining[agent] = n - 1
            else:
                remaining[agent] = n

        # Select one receiver weighted by count * receive_weight.
        receiver_list = [a for a in receivers if remaining.get(a, 0) > 0]
        if not receiver_list:
            raise ValueError(
                "No agent in receivers is present in the remaining configuration"
            )
        weights = [remaining[a] * env.receive_weight(a, c) for a in receiver_list]
        selected = rng.choices(receiver_list, weights=weights, k=1)[0]

        # Perform the receive transition on the selected agent.
        next_selected = adef.receive(selected, c, rng)

        # Build new counts: replace one selected with next_selected, add target.
        new_counts: dict[Agent, int] = dict(remaining)
        if new_counts[selected] == 1:
            del new_counts[selected]
        else:
            new_counts[selected] -= 1
        new_counts[next_selected] = new_counts.get(next_selected, 0) + 1
        new_counts[target] = new_counts.get(target, 0) + 1

        return NBAConfiguration(new_counts)

    def step(
        self,
        rng: random.Random,
        adef: AgentDefinitions,
        env: Environment,
    ) -> tuple[float, NBAConfiguration] | None:
        """Randomly advance the configuration by one send action.

        Computes all enabled send actions, selects the acting agent
        proportionally to ``exit_rate(agent) * count(agent)``, invokes
        :meth:`~pythia.models.nba.agent_definitions.AgentDefinitions.step` to
        determine the action performed, then delegates to either
        :meth:`broadcast_send` or :meth:`send` depending on whether the chosen
        action is a broadcast-send or a point-to-point send.

        Args:
            rng: Random number generator used throughout the selection.
            adef: The :class:`~pythia.models.nba.agent_definitions.AgentDefinitions`
                that owns the transition and rate tables.
            env: The :class:`~pythia.models.nba.environment.Environment`
                providing per-agent, per-channel rates.

        Returns:
            ``None`` if no agent has any enabled send or broadcast-send action
            (total exit rate is zero).  Otherwise a pair ``(time, next_cfg)``
            where *time* is sampled from an exponential distribution with rate
            equal to the total weight and *next_cfg* is the successor
            :class:`NBAConfiguration`.

        Example:
            >>> import random
            >>> from pythia.models.nba.agent_definitions import AgentDefinitions
            >>> from pythia.models.nba.channel import Channel
            >>> from pythia.models.nba.environment import Environment
            >>> c = Channel("c")
            >>> defs = AgentDefinitions()
            >>> a, b = defs.get("a"), defs.get("b")
            >>> defs.add_send_transition(a, c, b, 1.0)
            >>> defs.add_receive_transition(b, c, b, 1.0)
            >>> env = Environment()
            >>> env.set_send_rate(a, c, 1.0)
            >>> env.set_receive_weight(b, c, 1.0)
            >>> cfg = NBAConfiguration({a: 1, b: 1})
            >>> time, new_cfg = cfg.step(random.Random(0), defs, env)
            >>> time > 0
            True
            >>> new_cfg.count(a)
            0
            >>> new_cfg.count(b)
            2
            >>> defs2 = AgentDefinitions()
            >>> x = defs2.get("x")
            >>> cfg2 = NBAConfiguration({x: 1})
            >>> cfg2.step(random.Random(0), defs2, Environment()) is None
            True
        """
        receivers = self.get_receivers(adef)

        agents = list(self._counts)
        enabled_preds = {
            agent: self.get_is_enabled_predicate(receivers, agent)
            for agent in agents
        }
        weights = [
            adef.exit_rate(agent, env, enabled_preds[agent]) * self._counts[agent]
            for agent in agents
        ]

        total = sum(weights)
        if total == 0.0:
            return None

        time = rng.expovariate(total)
        selected = rng.choices(agents, weights=weights, k=1)[0]
        is_broadcast, channel, next_agent = adef.step(
            selected, env, enabled_preds[selected], rng
        )

        if is_broadcast:
            return time, self.broadcast_send(rng, adef, selected, channel, next_agent)

        receivers_for_channel = receivers.get(channel, set())
        return time, self.send(rng, adef, env, selected, channel, next_agent, receivers_for_channel)

    def __repr__(self) -> str:
        return f"NBAConfiguration({self._counts!r})"
