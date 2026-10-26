"""Species descriptor for population models.

A :class:`Species` is an immutable value object that carries a species
identity — a :attr:`name`, a set of named :attr:`attributes`, and an
:attr:`index` that was assigned by a :class:`~pythia.models.pm.population.Population`
registry.

The index uniquely identifies the ``(name, attributes)`` pair *within the
registry that created it*: two :class:`Species` instances are considered equal
if and only if they share the same :attr:`index`.

.. note::
    :class:`Species` instances should be created via
    :meth:`~pythia.models.pm.population.Population.register`, not by calling
    the constructor directly.  Direct construction is supported for testing
    and advanced use cases, but the uniqueness invariant is only guaranteed
    when species are obtained through the same :class:`Population`.

Example::

    from pythia.models.pm import Species
    from pythia.models.pm.population import Population

    pop = Population()
    wolf   = pop.register("wolf",   {"diet": "carnivore", "max_age": 15})
    rabbit = pop.register("rabbit", {"diet": "herbivore", "max_age": 8})

    # Same content → same index
    wolf2 = pop.register("wolf", {"diet": "carnivore", "max_age": 15})
    assert wolf.index == wolf2.index

    # Different content → different index
    assert wolf.index != rabbit.index
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, eq=False)
class Species:
    """Immutable species descriptor used in population models.

    The :attr:`index` is assigned by a
    :class:`~pythia.models.pm.population.Population` registry and uniquely
    identifies the ``(name, attributes)`` pair within that registry.

    Equality and hashing are based solely on :attr:`index`, so two
    :class:`Species` instances are interchangeable as dict keys or set members
    whenever they carry the same index.

    Attributes:
        name: Human-readable species name (e.g. ``"wolf"``).
        attributes: Immutable mapping of species characteristics (e.g.
            ``{"diet": "carnivore"}``).
        index: Non-negative integer assigned by a
            :class:`~pythia.models.pm.population.Population`.  Uniquely
            identifies the ``(name, attributes)`` pair within that registry.

    Raises:
        ValueError: If *index* is negative.

    Examples:
        >>> from pythia.models.pm.population import Population
        >>> pop = Population()
        >>> wolf = pop.register("wolf", {"diet": "carnivore"})
        >>> wolf.name
        'wolf'
        >>> wolf.attributes["diet"]
        'carnivore'
        >>> wolf.index >= 0
        True
    """

    name: str
    attributes: Mapping[str, Any]
    index: int

    def __post_init__(self) -> None:
        """Freeze attributes and validate index."""
        if self.index < 0:
            raise ValueError(
                f"Species index must be non-negative, got {self.index!r}"
            )
        # Convert to a read-only proxy so the instance stays fully immutable
        # even if the caller mutates the original mapping afterwards.
        object.__setattr__(
            self, "attributes", types.MappingProxyType(dict(self.attributes))
        )

    # ------------------------------------------------------------------
    # Equality and hashing are index-based.
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Return ``True`` iff *other* is a :class:`Species` with the same index.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` when *other* is a :class:`Species` with an identical
            :attr:`index`; ``False`` for a :class:`Species` with a different
            index; ``NotImplemented`` for any other type.
        """
        if not isinstance(other, Species):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        """Return a hash derived solely from :attr:`index`.

        Returns:
            An integer hash value.
        """
        return hash(self.index)

    def __repr__(self) -> str:
        """Return a readable representation.

        Returns:
            A string of the form
            ``Species(index=0, name='wolf', attributes={'diet': 'carnivore'})``.
        """
        return (
            f"Species(index={self.index!r}, name={self.name!r}, "
            f"attributes={dict(self.attributes)!r})"
        )
