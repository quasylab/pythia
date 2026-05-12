"""Counting state for population models.

A :class:`CountingState` is an immutable mapping from species indices to
non-negative element counts.  It is the primary way to represent the
composition of a mixed population at a single point in time.

Only species with a strictly positive count are stored in
:attr:`counting_vector`; species absent from the mapping implicitly have a
count of zero.

Example::

    from pythia.models.pm.counting_state import CountingState
    from pythia.models.pm.population import Population

    pop = Population()
    wolf   = pop.register("wolf")
    rabbit = pop.register("rabbit")

    state = CountingState({wolf: 10, rabbit: 50})

    print(state.counting_vector)  # {0: 10, 1: 50}

    reserve = CountingState({wolf: 3, rabbit: 20})
    print(state.contains(reserve))  # True — enough of every species
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any

from pythia.models.pm.species import Species


class CountingState:
    """Immutable snapshot of species counts in a population.

    Maps each species — identified by its
    :attr:`~pythia.models.pm.species.Species.index` — to a non-negative
    integer count.  Species with a count of zero are not stored;
    any index absent from :attr:`counting_vector` implicitly has a count of
    zero.

    Two :class:`CountingState` instances are equal if and only if their
    :attr:`counting_vector` mappings are identical (same indices, same
    counts).

    Args:
        counts: A mapping from :class:`~pythia.models.pm.species.Species`
            instances to non-negative integer counts.  Entries with a count
            of zero are accepted but not stored.

    Raises:
        TypeError: If any count is not an :class:`int`.
        ValueError: If any count is negative.

    Example:
        >>> from pythia.models.pm.population import Population
        >>> pop = Population()
        >>> wolf   = pop.register("wolf")
        >>> rabbit = pop.register("rabbit")
        >>> state = CountingState({wolf: 10, rabbit: 50})
        >>> state.counting_vector[wolf.index]
        10
        >>> state.counting_vector[rabbit.index]
        50
    """

    def __init__(self, counts: Mapping[Species, int]) -> None:
        """Build a :class:`CountingState` from a species-to-count mapping."""
        vector: dict[int, int] = {}
        for species, count in counts.items():
            if not isinstance(count, int):
                raise TypeError(
                    f"counts must be integers, got {type(count).__name__!r} "
                    f"for species {species.name!r}"
                )
            if count < 0:
                raise ValueError(
                    f"counts must be non-negative, got {count!r} "
                    f"for species {species.name!r}"
                )
            if count > 0:
                vector[species.index] = count

        self._counting_vector: dict[int, int] = vector
        self._counting_vector_proxy: Mapping[int, int] = types.MappingProxyType(vector)
        self._size: int = sum(vector.values())

    # ------------------------------------------------------------------
    # Internal factory
    # ------------------------------------------------------------------

    @classmethod
    def _from_vector(cls, vector: dict[int, int]) -> CountingState:
        """Construct a :class:`CountingState` directly from an index-to-count dict.

        Bypasses the public constructor (which requires :class:`~pythia.models.pm.species.Species`
        objects) so that arithmetic methods can build results without needing
        to resolve indices back to species instances.  Zero entries are
        silently dropped.

        Args:
            vector: Mapping of species index to count.  May contain zeros;
                they will not be stored.

        Returns:
            A new :class:`CountingState` backed by the filtered *vector*.
        """
        instance = object.__new__(cls)
        filtered = {k: v for k, v in vector.items() if v > 0}
        instance._counting_vector = filtered
        instance._counting_vector_proxy = types.MappingProxyType(filtered)
        instance._size = sum(filtered.values())
        return instance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def counting_vector(self) -> Mapping[int, int]:
        """Immutable mapping from species index to positive element count.

        Only indices with a strictly positive count are present.  Any index
        absent from this mapping has an implicit count of zero.

        Returns:
            A read-only :class:`~types.MappingProxyType` view of the
            internal count dictionary.
        """
        return self._counting_vector_proxy

    @property
    def size(self) -> int:
        """Total number of items across all species in this state.

        Computed once at construction (and propagated through :meth:`add`
        and :meth:`remove`) so the value is available in O(1).

        Example:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> wolf   = pop.register("wolf")
            >>> rabbit = pop.register("rabbit")
            >>> CountingState({wolf: 3, rabbit: 7}).size
            10
            >>> CountingState({}).size
            0
        """
        return self._size

    def count(self, species: Species) -> int:
        """Return the number of elements of *species* in this state.

        Args:
            species: The :class:`~pythia.models.pm.species.Species` to look up.

        Returns:
            The non-negative integer count; ``0`` if *species* is absent.

        Example:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> wolf   = pop.register("wolf")
            >>> rabbit = pop.register("rabbit")
            >>> state = CountingState({wolf: 7, rabbit: 3})
            >>> state.count(wolf)
            7
            >>> state.count(rabbit)
            3
            >>> state.count(pop.register("bear"))
            0
        """
        return self._counting_vector.get(species.index, 0)

    def contains(self, other: CountingState) -> bool:
        """Return ``True`` if this state has at least as many elements as *other* for every species.

        For each index *i* that appears in *other*, the count in this state
        must be greater than or equal to the count in *other*.  Indices
        present in this state but absent from *other* are ignored.

        Args:
            other: The :class:`CountingState` to compare against.

        Returns:
            ``True`` when, for every index *i* in *other*,
            ``self.counting_vector.get(i, 0) >= other.counting_vector[i]``.

        Example:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> wolf = pop.register("wolf")
            >>> full  = CountingState({wolf: 10})
            >>> small = CountingState({wolf: 3})
            >>> full.contains(small)
            True
            >>> small.contains(full)
            False
            >>> full.contains(full)
            True
        """
        for index, count in other._counting_vector.items():
            if self._counting_vector.get(index, 0) < count:
                return False
        return True

    def add(self, other: CountingState) -> CountingState:
        """Return a new :class:`CountingState` with counts summed element-wise.

        For every index *i* present in either operand, the result contains
        the sum of the counts from *self* and *other*.  Indices absent from
        one operand contribute zero to the sum.

        The receiver and *other* are unchanged.

        Args:
            other: The :class:`CountingState` whose counts are added to this
                one.

        Returns:
            A new :class:`CountingState` where
            ``result.counting_vector[i] == self.counting_vector.get(i, 0)
            + other.counting_vector.get(i, 0)`` for every index *i* that
            appears in either operand.

        Example:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> wolf   = pop.register("wolf")
            >>> rabbit = pop.register("rabbit")
            >>> a = CountingState({wolf: 3, rabbit: 10})
            >>> b = CountingState({wolf: 2, rabbit: 5})
            >>> c = a.add(b)
            >>> c.counting_vector[wolf.index]
            5
            >>> c.counting_vector[rabbit.index]
            15
        """
        merged: dict[int, int] = dict(self._counting_vector)
        for index, count in other._counting_vector.items():
            merged[index] = merged.get(index, 0) + count
        return CountingState._from_vector(merged)

    def remove(self, other: CountingState) -> CountingState:
        """Return a new :class:`CountingState` with counts reduced element-wise.

        For every index *i*, the result contains
        ``max(0, self.counting_vector.get(i, 0) - other.counting_vector.get(i, 0))``.
        Indices present in *other* but absent from *self* produce a result of
        zero and are not stored.  Indices present in *self* but absent from
        *other* are carried over unchanged.

        The receiver and *other* are unchanged.

        Args:
            other: The :class:`CountingState` whose counts are subtracted from
                this one.

        Returns:
            A new :class:`CountingState` where each count is the non-negative
            difference between the corresponding counts of *self* and *other*.

        Example:
            >>> from pythia.models.pm.population import Population
            >>> pop = Population()
            >>> wolf   = pop.register("wolf")
            >>> rabbit = pop.register("rabbit")
            >>> a = CountingState({wolf: 5, rabbit: 10})
            >>> b = CountingState({wolf: 3, rabbit: 12})
            >>> c = a.remove(b)
            >>> c.counting_vector[wolf.index]
            2
            >>> rabbit.index in c.counting_vector
            False
        """
        result: dict[int, int] = {}
        all_indices = set(self._counting_vector) | set(other._counting_vector)
        for index in all_indices:
            value = max(0, self._counting_vector.get(index, 0) - other._counting_vector.get(index, 0))
            if value > 0:
                result[index] = value
        return CountingState._from_vector(result)

    # ------------------------------------------------------------------
    # Equality, hashing and representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Return ``True`` iff *other* has the same :attr:`counting_vector`.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` when *other* is a :class:`CountingState` with an
            identical index-to-count mapping; ``NotImplemented`` for any
            other type.
        """
        if not isinstance(other, CountingState):
            return NotImplemented
        return self._counting_vector == other._counting_vector

    def __hash__(self) -> int:
        """Return a hash derived from the :attr:`counting_vector` contents.

        Returns:
            An integer hash value consistent with :meth:`__eq__`.
        """
        return hash(frozenset(self._counting_vector.items()))

    def __repr__(self) -> str:
        """Return a readable representation.

        Returns:
            A string of the form ``CountingState({0: 10, 1: 50})``.
        """
        return f"CountingState({dict(self._counting_vector)!r})"
