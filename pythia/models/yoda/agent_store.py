"""AgentStore — the layered state of a single agent in a YODA system.

An :class:`AgentStore` groups the three :class:`~pythia.models.yoda.store.Store`
layers that together describe an agent's state at a given point in time:

* **state** — private state variables owned by the agent.
* **features** — derived or computed attributes, typically set by the system.
* **observations** — values observed from the environment or other agents.

The :meth:`get` method provides unified attribute lookup across all three
layers, resolving in priority order: ``state`` → ``features`` → ``observations``.

Example::

    from pythia.models.yoda.store import Store
    from pythia.models.yoda.agent_store import AgentStore

    s = AgentStore(
        state=Store({"energy": 10}),
        features=Store({"energy": 5, "speed": 3}),
        observations=Store({"speed": 1, "heat": 42}),
    )

    s.get("energy")   # 10   (from state)
    s.get("speed")    # 3    (from features, shadowed in observations)
    s.get("heat")     # 42   (from observations)
    s.get("mass")     # None (absent in all layers)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from pythia.models.yoda.store import Store


class AgentStore:
    """Layered state of a single YODA agent.

    Combines three immutable :class:`~pythia.models.yoda.store.Store` layers
    into a single view.  Attribute lookup via :meth:`get` checks them in
    priority order: :attr:`state` first, then :attr:`features`, then
    :attr:`observations`.

    Args:
        observations: Values observed from the environment or other agents.
        features: Derived or computed attributes set by the system.
        state: Private state variables owned by the agent.

    Example:
        >>> from pythia.models.yoda.store import Store
        >>> s = AgentStore(
        ...     state=Store({"x": 1}),
        ...     features=Store({"x": 2, "y": 3}),
        ...     observations=Store({"y": 4, "z": 5}),
        ... )
        >>> s.get("x")
        1
        >>> s.get("y")
        3
        >>> s.get("z")
        5
        >>> s.get("missing") is None
        True
    """

    def __init__(
        self,
        observations: Store,
        features: Store,
        state: Store,
    ) -> None:
        object.__setattr__(self, "_observations", observations)
        object.__setattr__(self, "_features", features)
        object.__setattr__(self, "_state", state)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AgentStore is immutable")

    @property
    def state(self) -> Store:
        """The agent's private state variables.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store(), Store(), Store({"x": 1}))
            >>> s.state["x"]
            1
        """
        return self._state  # type: ignore[return-value]

    @property
    def features(self) -> Store:
        """The agent's derived or computed attributes.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store(), Store({"y": 2}), Store())
            >>> s.features["y"]
            2
        """
        return self._features  # type: ignore[return-value]

    @property
    def observations(self) -> Store:
        """The values observed from the environment or other agents.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store({"z": 3}), Store(), Store())
            >>> s.observations["z"]
            3
        """
        return self._observations  # type: ignore[return-value]

    def get(self, attribute: str) -> Any:
        """Return the value of *attribute* from the highest-priority layer.

        Checks :attr:`state` first, then :attr:`features`, then
        :attr:`observations`.  Returns ``None`` when the attribute is absent
        from all three layers.

        Args:
            attribute: The attribute name to look up.

        Returns:
            The resolved value, or ``None`` if the attribute is not defined
            in any layer.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(
            ...     state=Store({"a": 1}),
            ...     features=Store({"b": 2}),
            ...     observations=Store({"c": 3}),
            ... )
            >>> s.get("a")
            1
            >>> s.get("b")
            2
            >>> s.get("c")
            3
            >>> s.get("d") is None
            True
        """
        if attribute in self.state:
            return self.state[attribute]
        if attribute in self.features:
            return self.features[attribute]
        if attribute in self.observations:
            return self.observations[attribute]
        return None

    def set_state(self, state: Store) -> AgentStore:
        """Return a new :class:`AgentStore` with :attr:`state` replaced.

        All other fields are carried over unchanged.

        Args:
            state: The replacement :class:`~pythia.models.yoda.store.Store`.

        Returns:
            A new :class:`AgentStore` instance.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store(), Store(), Store({"x": 1}))
            >>> s2 = s.set_state(Store({"x": 99}))
            >>> s2.state["x"]
            99
            >>> s.state["x"]
            1
        """
        return AgentStore(
            observations=self.observations,
            features=self.features,
            state=state,
        )

    def set_features(self, features: Store) -> AgentStore:
        """Return a new :class:`AgentStore` with :attr:`features` replaced.

        All other fields are carried over unchanged.

        Args:
            features: The replacement :class:`~pythia.models.yoda.store.Store`.

        Returns:
            A new :class:`AgentStore` instance.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store(), Store({"y": 2}), Store())
            >>> s2 = s.set_features(Store({"y": 99}))
            >>> s2.features["y"]
            99
            >>> s.features["y"]
            2
        """
        return AgentStore(
            observations=self.observations,
            features=features,
            state=self.state,
        )

    def set_observations(self, observations: Store) -> AgentStore:
        """Return a new :class:`AgentStore` with :attr:`observations` replaced.

        All other fields are carried over unchanged.

        Args:
            observations: The replacement :class:`~pythia.models.yoda.store.Store`.

        Returns:
            A new :class:`AgentStore` instance.

        Example:
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store({"z": 3}), Store(), Store())
            >>> s2 = s.set_observations(Store({"z": 99}))
            >>> s2.observations["z"]
            99
            >>> s.observations["z"]
            3
        """
        return AgentStore(
            observations=observations,
            features=self.features,
            state=self.state,
        )

    def apply_dynamics(
        self,
        rng: random.Random,
        dt: float,
        dyn: Callable[[random.Random, AgentStore, float], Store],
    ) -> AgentStore:
        """Return a new :class:`AgentStore` with features updated by *dyn*.

        Calls ``dyn(rng, self, dt)`` and replaces :attr:`features` with the
        result.  :attr:`state` and :attr:`observations` are carried over
        unchanged.

        Args:
            rng: Random number generator forwarded to *dyn*.
            dt: Time step (strictly positive) forwarded to *dyn*.
            dyn: A callable ``(rng, s, dt) -> Store`` that receives this
                :class:`AgentStore` and computes the updated features.

        Returns:
            A new :class:`AgentStore` instance with updated features.

        Example:
            >>> import random
            >>> from pythia.models.yoda.store import Store
            >>> s = AgentStore(Store(), Store({"x": 0.0}), Store({"s": 1}))
            >>> s2 = s.apply_dynamics(
            ...     random.Random(0),
            ...     0.5,
            ...     lambda rng, s, dt: s.features.set("x", s.features.get("x", 0) + dt),
            ... )
            >>> s2.features["x"]
            0.5
            >>> s.features["x"]
            0.0
        """
        return self.set_features(dyn(rng, self, dt))

    def __repr__(self) -> str:
        return (
            f"AgentStore("
            f"state={self.state!r}, "
            f"features={self.features!r}, "
            f"observations={self.observations!r})"
        )
