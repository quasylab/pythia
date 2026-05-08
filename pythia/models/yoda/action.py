"""Action — a named transition in a YODA system.

An :class:`Action` encodes one possible behaviour that an agent can execute.
It combines:

* a **name** — a human-readable identifier.
* a **step function** — computes the next ``state`` store when the action
  fires, given a random generator, the agent's current ``state``, and its
  ``observations``.

Example::

    import random
    from pythia.models.yoda.action import Action
    from pythia.models.yoda.store import Store

    increment = Action(
        name="increment",
        step_function=lambda rng, state, observations: state.set(
            "count", state.get("count", 0) + 1
        ),
    )

    rng = random.Random(0)
    s   = Store({"count": 3})
    obs = Store()

    increment.step(rng, s, obs)     # Store({'count': 4})
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from pythia.models.yoda.store import Store


class Action:
    """A named agent transition for YODA simulations.

    An action is immutable after construction.  Its :meth:`step` method
    produces the next state store when the action fires.

    Args:
        name: A non-empty human-readable identifier for this action.
        step_function: A callable ``(rng, state, observations) -> Store``
            that returns the agent's next :attr:`state` store after the
            action fires.  The random generator *rng* is provided so that
            stochastic transitions can be expressed directly in the function.

    Raises:
        ValueError: If *name* is empty.

    Example:
        >>> import random
        >>> from pythia.models.yoda.store import Store
        >>> a = Action(
        ...     name="tick",
        ...     step_function=lambda rng, state, observations: state.set("t", 1),
        ... )
        >>> a.name
        'tick'
        >>> a.step(random.Random(0), Store(), Store())["t"]
        1
    """

    def __init__(
        self,
        name: str,
        step_function: Callable[[random.Random, Store, Store], Store],
    ) -> None:
        if not name:
            raise ValueError("Action name must be a non-empty string")
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_step_function", step_function)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Action is immutable")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The human-readable identifier of this action.

        Example:
            >>> Action("move", lambda rng, s, o: s).name
            'move'
        """
        return self._name  # type: ignore[return-value]

    @property
    def step_function(self) -> Callable[[random.Random, Store, Store], Store]:
        """The transition function ``(rng, state, observations) -> Store``.

        Example:
            >>> import random
            >>> a = Action("a", lambda rng, s, o: s.set("x", 9))
            >>> a.step_function(random.Random(0), Store(), Store())["x"]
            9
        """
        return self._step_function  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def step(self, rng: random.Random, state: Store, observations: Store) -> Store:
        """Return the next state store produced by firing this action.

        Delegates to :attr:`step_function`.

        Args:
            rng: Random number generator available to stochastic transitions.
            state: The agent's current private state.
            observations: The agent's current observations.

        Returns:
            A :class:`~pythia.models.yoda.store.Store` representing the
            agent's state after the action fires.

        Example:
            >>> import random
            >>> from pythia.models.yoda.store import Store
            >>> a = Action(
            ...     "reset",
            ...     step_function=lambda rng, state, observations: Store({"count": 0}),
            ... )
            >>> a.step(random.Random(0), Store({"count": 99}), Store())["count"]
            0
        """
        return self._step_function(rng, state, observations)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"Action({self._name!r})"  # type: ignore[attr-defined]
