"""SystemConfiguration — the runtime state of all agents in a YODA system.

A :class:`SystemConfiguration` holds an ordered list of
:class:`~pythia.models.yoda.agent_configuration.AgentConfiguration` objects,
one per agent currently operating in the system.  It is immutable: the list
of elements is fixed at construction time.

Example::

    from pythia.models.yoda.agent import Agent
    from pythia.models.yoda.agent_configuration import AgentConfiguration
    from pythia.models.yoda.agent_state import AgentState
    from pythia.models.yoda.agent_store import AgentStore
    from pythia.models.yoda.configuration import SystemConfiguration
    from pythia.models.yoda.store import Store

    a1 = AgentConfiguration(Agent("a"), AgentState("idle"), AgentStore(Store(), Store(), Store()))
    a2 = AgentConfiguration(Agent("b"), AgentState("busy"), AgentStore(Store(), Store(), Store()))

    cfg = SystemConfiguration([a1, a2])
    cfg.size          # 2
    cfg.get(0)        # a1
    cfg.get(1)        # a2
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pythia.models.yoda.agent_configuration import AgentConfiguration
from pythia.models.yoda.agent_store import AgentStore

# Preserve built-in names shadowed by same-named methods below.
builtins_min = min
builtins_max = max


class SystemConfiguration:
    """Immutable ordered collection of agent configurations.

    Represents the complete runtime state of all agents operating in a YODA
    system at a given point in time.

    Args:
        elements: The :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
            instances that make up the system, in order.

    Raises:
        ValueError: If *elements* is empty.

    Example:
        >>> from pythia.models.yoda.agent import Agent
        >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
        >>> from pythia.models.yoda.agent_state import AgentState
        >>> from pythia.models.yoda.agent_store import AgentStore
        >>> from pythia.models.yoda.store import Store
        >>> ac = AgentConfiguration(Agent("a"), AgentState("idle"), AgentStore(Store(), Store(), Store()))
        >>> cfg = SystemConfiguration([ac])
        >>> cfg.size()
        1
        >>> cfg.get(0) is ac
        True
    """

    def __init__(self, elements: Iterable[AgentConfiguration]) -> None:
        _elements: tuple[AgentConfiguration, ...] = tuple(elements)
        if not _elements:
            raise ValueError("SystemConfiguration must contain at least one agent configuration")
        object.__setattr__(self, "_elements", _elements)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SystemConfiguration is immutable")

    @property
    def elements(self) -> tuple[AgentConfiguration, ...]:
        """The agent configurations as an immutable tuple, in order.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> ac = AgentConfiguration(Agent("a"), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> SystemConfiguration([ac]).elements == (ac,)
            True
        """
        return self._elements  # type: ignore[return-value]

    def apply(
        self,
        f: Callable[[AgentConfiguration], AgentConfiguration],
    ) -> SystemConfiguration:
        """Return a new :class:`SystemConfiguration` with every element transformed by *f*.

        Applies *f* to each
        :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        in order and collects the results into a new :class:`SystemConfiguration`.

        Args:
            f: A callable that maps an
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
                to a new one.

        Returns:
            A new :class:`SystemConfiguration` of the same size.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> def _ac(name):
            ...     return AgentConfiguration(Agent(name), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> cfg  = SystemConfiguration([_ac("a"), _ac("b")])
            >>> busy = AgentState("busy")
            >>> cfg2 = cfg.apply(lambda ac: AgentConfiguration(ac.agent, busy, ac.agent_store))
            >>> cfg2.get(0).agent_state.name
            'busy'
            >>> cfg2.get(1).agent_state.name
            'busy'
        """
        return SystemConfiguration(f(self._elements[i])  # type: ignore[index]
                                   for i in range(len(self._elements)))  # type: ignore[arg-type]

    def remove(self, idx: int) -> SystemConfiguration:
        """Return a new :class:`SystemConfiguration` with the element at *idx* removed.

        The remaining :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
        objects are **not** copied; the new configuration holds the same
        references in the same order, minus the one at *idx*.

        Negative indices are supported with the same semantics as Python
        sequence indexing.

        Args:
            idx: Position of the element to remove.  Supports negative
                indices (e.g. ``-1`` removes the last element).

        Returns:
            A new :class:`SystemConfiguration` containing all elements except
            the one at *idx*.

        Raises:
            IndexError: If *idx* is out of range.
            ValueError: If the resulting configuration would be empty (i.e.
                the current configuration has exactly one element).

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> def _ac(name):
            ...     return AgentConfiguration(Agent(name), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> cfg = SystemConfiguration([_ac("a"), _ac("b"), _ac("c")])
            >>> cfg2 = cfg.remove(1)
            >>> cfg2.size()
            2
            >>> cfg2.get(0).agent.name
            'a'
            >>> cfg2.get(1).agent.name
            'c'
        """
        n = len(self._elements)  # type: ignore[arg-type]
        if not -n <= idx < n:
            raise IndexError(
                f"index {idx} out of range for configuration of size {n}"
            )
        if idx < 0:
            idx += n
        return SystemConfiguration(
            self._elements[i]  # type: ignore[index]
            for i in range(n)
            if i != idx
        )

    def size(self) -> int:
        """Return the number of agent configurations in this configuration.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> def _ac(name):
            ...     return AgentConfiguration(Agent(name), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> SystemConfiguration([_ac("a"), _ac("b")]).size()
            2
        """
        return len(self._elements)  # type: ignore[arg-type]

    def get(self, index: int) -> AgentConfiguration:
        """Return the agent configuration at position *index*.

        Args:
            index: Zero-based position in the list of agent configurations.

        Returns:
            The :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
            at *index*.

        Raises:
            IndexError: If *index* is out of range.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> ac = AgentConfiguration(Agent("a"), AgentState("s"), AgentStore(Store(), Store(), Store()))
            >>> SystemConfiguration([ac]).get(0) is ac
            True
        """
        return self._elements[index]  # type: ignore[index]

    def exists(
        self,
        predicate: Callable[[AgentConfiguration], bool],
        idx: int | None = None,
    ) -> bool:
        """Return ``True`` if any agent configuration satisfying *predicate* exists.

        When *idx* is ``None`` all elements are considered.  When *idx* is
        given, the element at that index is excluded and only the remaining
        elements are checked.

        Args:
            predicate: A callable that accepts an
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
                and returns a :class:`bool`.
            idx: Index to exclude from the search.  Defaults to ``None``
                (no exclusion).

        Returns:
            ``True`` when at least one qualifying element satisfies
            *predicate*; ``False`` otherwise.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> def _ac(name, state):
            ...     return AgentConfiguration(Agent(name), AgentState(state), AgentStore(Store(), Store(), Store()))
            >>> cfg = SystemConfiguration([_ac("a", "idle"), _ac("b", "busy"), _ac("c", "idle")])
            >>> cfg.exists(lambda ac: ac.agent_state == AgentState("idle"))
            True
            >>> cfg.exists(lambda ac: ac.agent_state == AgentState("idle"), idx=0)
            True
            >>> cfg.exists(lambda ac: ac.agent_state == AgentState("idle"), idx=2)
            False
        """
        return any(
            predicate(self._elements[i])  # type: ignore[index]
            for i in range(len(self._elements))  # type: ignore[arg-type]
            if idx is None or i != idx
        )

    def forall(
        self,
        predicate: Callable[[AgentConfiguration], bool],
        idx: int | None = None,
    ) -> bool:
        """Return ``True`` if every qualifying agent configuration satisfies *predicate*.

        When *idx* is ``None`` all elements are considered.  When *idx* is
        given, the element at that index is excluded and only the remaining
        elements are checked.

        Args:
            predicate: A callable that accepts an
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`
                and returns a :class:`bool`.
            idx: Index to exclude from the check.  Defaults to ``None``
                (no exclusion).

        Returns:
            ``True`` when all qualifying elements satisfy *predicate*,
            including vacuously when there are no qualifying elements;
            ``False`` otherwise.

        Example:
            >>> from pythia.models.yoda.agent import Agent
            >>> from pythia.models.yoda.agent_configuration import AgentConfiguration
            >>> from pythia.models.yoda.agent_state import AgentState
            >>> from pythia.models.yoda.agent_store import AgentStore
            >>> from pythia.models.yoda.store import Store
            >>> def _ac(name, state):
            ...     return AgentConfiguration(Agent(name), AgentState(state), AgentStore(Store(), Store(), Store()))
            >>> cfg = SystemConfiguration([_ac("a", "idle"), _ac("b", "idle"), _ac("c", "busy")])
            >>> cfg.forall(lambda ac: ac.agent_state == AgentState("idle"))
            False
            >>> cfg.forall(lambda ac: ac.agent_state == AgentState("idle"), idx=2)
            True
        """
        return all(
            predicate(self._elements[i])  # type: ignore[index]
            for i in range(len(self._elements))  # type: ignore[arg-type]
            if idx is None or i != idx
        )

    def _values(
        self,
        f: Callable[[AgentStore], float],
        pred: Callable[[AgentConfiguration], bool] | None,
        idx: int | None,
    ) -> list[float]:
        """Return ``f(ac.agent_store)`` for each qualifying element."""
        return [
            f(self._elements[i].agent_store)  # type: ignore[index]
            for i in range(len(self._elements))  # type: ignore[arg-type]
            if (idx is None or i != idx)
            and (pred is None or pred(self._elements[i]))  # type: ignore[index]
        ]

    def mean(
        self,
        f: Callable[[AgentStore], float],
        pred: Callable[[AgentConfiguration], bool] | None = None,
        idx: int | None = None,
        default: float | None = None,
    ) -> float | None:
        """Return the mean of ``f`` evaluated over qualifying agent stores.

        Qualifying elements are those not at position *idx* (when given) and
        satisfying *pred* (when given).  When no element qualifies, *default*
        is returned.

        Args:
            f: A function mapping an
                :class:`~pythia.models.yoda.agent_store.AgentStore` to a
                float.
            pred: Optional predicate on
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`.
                When ``None`` all elements (subject to *idx*) are included.
            idx: Optional index to exclude.  When ``None`` no element is
                excluded.
            default: Value returned when no element qualifies.  Defaults to
                ``None``.

        Returns:
            The arithmetic mean of the qualifying values, or *default* when
            no element qualifies.
        """
        values = self._values(f, pred, idx)
        if not values:
            return default
        return sum(values) / len(values)

    def min(
        self,
        f: Callable[[AgentStore], float],
        pred: Callable[[AgentConfiguration], bool] | None = None,
        idx: int | None = None,
        default: float | None = None,
    ) -> float | None:
        """Return the minimum of ``f`` evaluated over qualifying agent stores.

        Qualifying elements are those not at position *idx* (when given) and
        satisfying *pred* (when given).  When no element qualifies, *default*
        is returned.

        Args:
            f: A function mapping an
                :class:`~pythia.models.yoda.agent_store.AgentStore` to a
                float.
            pred: Optional predicate on
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`.
                When ``None`` all elements (subject to *idx*) are included.
            idx: Optional index to exclude.  When ``None`` no element is
                excluded.
            default: Value returned when no element qualifies.  Defaults to
                ``None``.

        Returns:
            The minimum of the qualifying values, or *default* when no
            element qualifies.
        """
        values = self._values(f, pred, idx)
        if not values:
            return default
        return builtins_min(values)

    def max(
        self,
        f: Callable[[AgentStore], float],
        pred: Callable[[AgentConfiguration], bool] | None = None,
        idx: int | None = None,
        default: float | None = None,
    ) -> float | None:
        """Return the maximum of ``f`` evaluated over qualifying agent stores.

        Qualifying elements are those not at position *idx* (when given) and
        satisfying *pred* (when given).  When no element qualifies, *default*
        is returned.

        Args:
            f: A function mapping an
                :class:`~pythia.models.yoda.agent_store.AgentStore` to a
                float.
            pred: Optional predicate on
                :class:`~pythia.models.yoda.agent_configuration.AgentConfiguration`.
                When ``None`` all elements (subject to *idx*) are included.
            idx: Optional index to exclude.  When ``None`` no element is
                excluded.
            default: Value returned when no element qualifies.  Defaults to
                ``None``.

        Returns:
            The maximum of the qualifying values, or *default* when no
            element qualifies.
        """
        values = self._values(f, pred, idx)
        if not values:
            return default
        return builtins_max(values)

    def __repr__(self) -> str:
        return f"SystemConfiguration({list(self._elements)!r})"  # type: ignore[arg-type]
