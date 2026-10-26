"""Population registry for species management.

A :class:`Population` is a mutable registry that creates and tracks
:class:`~pythia.models.pm.species.Species` instances.  It owns the uniqueness
invariant: two species registered with the same ``(name, attributes)`` pair
always receive the same :attr:`~pythia.models.pm.species.Species.index`,
and no two distinct pairs share an index.

Indices are assigned sequentially from zero in registration order.  The
registry is local to each :class:`Population` instance, so species from
different populations are independent.

Example::

    from pythia.models.pm.population import Population

    pop = Population()

    wolf   = pop.register("wolf",   {"diet": "carnivore", "max_age": 15})
    rabbit = pop.register("rabbit", {"diet": "herbivore", "max_age": 8})

    # Re-registering the same content returns the existing species.
    wolf2 = pop.register("wolf", {"diet": "carnivore", "max_age": 15})
    assert wolf is wolf2

    # Lookup by index.
    assert pop[wolf.index] is wolf

    # Iteration yields all species in index order.
    for species in pop:
        print(species.index, species.name)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pythia.models.pm.species import Species


def _registry_key(
    name: str,
    attributes: Mapping[str, Any],
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Build a stable, hashable key from a ``(name, attributes)`` pair.

    Attribute items are sorted by key so that insertion order does not affect
    the key.

    Args:
        name: Species name.
        attributes: Species attributes mapping.

    Returns:
        A ``(name, sorted_items_tuple)`` pair suitable as a dict key.

    Raises:
        ValueError: If any attribute value is not hashable.
    """
    try:
        attrs_tuple = tuple(sorted(attributes.items()))
        hash(attrs_tuple)  # eagerly validate hashability
    except TypeError as exc:
        raise ValueError(
            f"Species attributes values must be hashable, "
            f"but {name!r} has an unhashable value: {exc}"
        ) from exc
    return (name, attrs_tuple)


class Population:
    """Mutable registry that creates and tracks :class:`~pythia.models.pm.species.Species`.

    Each :class:`Population` instance maintains its own independent index space.
    Indices start at zero and are assigned in registration order.

    The uniqueness invariant is guaranteed within a single :class:`Population`:
    two species have the same :attr:`~pythia.models.pm.species.Species.index`
    if and only if they have the same name and attributes.

    Examples:
        >>> pop = Population()
        >>> wolf = pop.register("wolf", {"diet": "carnivore"})
        >>> wolf2 = pop.register("wolf", {"diet": "carnivore"})
        >>> wolf is wolf2
        True
        >>> len(pop)
        1
        >>> pop[0] is wolf
        True
    """

    def __init__(self) -> None:
        """Initialise an empty population registry."""
        # Maps (name, sorted_attrs_tuple) → Species for O(1) deduplication.
        self._by_key: dict[tuple[str, tuple[tuple[str, Any], ...]], Species] = {}
        # Dense list indexed by species.index for O(1) lookup by index.
        self._by_index: list[Species] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Species:
        """Register a ``(name, attributes)`` pair and return its :class:`Species`.

        If the pair has already been registered, the existing
        :class:`Species` instance is returned unchanged.  Otherwise a new
        :class:`Species` is created with the next available index and stored
        in the registry.

        Args:
            name: Species name.
            attributes: Species attribute mapping.  Defaults to an empty
                mapping when ``None``.  All values must be hashable.

        Returns:
            The :class:`Species` corresponding to the given ``(name,
            attributes)`` pair.

        Raises:
            ValueError: If any value in *attributes* is not hashable.

        Examples:
            >>> pop = Population()
            >>> s = pop.register("fox", {"habitat": "forest"})
            >>> s.index
            0
            >>> pop.register("fox", {"habitat": "forest"}) is s
            True
        """
        if attributes is None:
            attributes = {}

        key = _registry_key(name, attributes)

        if key in self._by_key:
            return self._by_key[key]

        species = Species(
            name=name,
            attributes=attributes,
            index=len(self._by_index),
        )
        self._by_key[key] = species
        self._by_index.append(species)
        return species

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of distinct species in the registry.

        Returns:
            Number of registered species.

        Examples:
            >>> pop = Population()
            >>> _ = pop.register("deer")
            >>> len(pop)
            1
        """
        return len(self._by_index)

    def __iter__(self) -> Iterator[Species]:
        """Iterate over all species in index order.

        Yields:
            Each registered :class:`Species`, from index 0 upward.

        Examples:
            >>> pop = Population()
            >>> _ = pop.register("a")
            >>> _ = pop.register("b")
            >>> [s.name for s in pop] == ["a", "b"]
            True
        """
        return iter(self._by_index)

    def __contains__(self, item: object) -> bool:
        """Return ``True`` if *item* is a species registered in this population.

        Uses the species index for an O(1) lookup, so it correctly identifies
        the same species even if a different :class:`Species` instance was
        created with the same content.

        Args:
            item: Object to test.  Returns ``False`` for non-:class:`Species`
                objects.

        Returns:
            ``True`` when *item* is a :class:`Species` whose index falls
            within this population and whose content matches the entry at
            that index.

        Examples:
            >>> pop = Population()
            >>> wolf = pop.register("wolf")
            >>> wolf in pop
            True
            >>> from pythia.models.pm.species import Species
            >>> Species("wolf", {}, index=0) in pop
            True
        """
        if not isinstance(item, Species):
            return False
        idx = item.index
        return 0 <= idx < len(self._by_index) and self._by_index[idx] == item

    def __getitem__(self, index: int) -> Species:
        """Return the species registered at *index*.

        Args:
            index: Zero-based species index.

        Returns:
            The :class:`Species` registered at *index*.

        Raises:
            KeyError: If no species with *index* exists in this population.

        Examples:
            >>> pop = Population()
            >>> wolf = pop.register("wolf")
            >>> pop[0] is wolf
            True
        """
        if index < 0 or index >= len(self._by_index):
            raise KeyError(
                f"no species with index {index!r} in this population "
                f"(size={len(self._by_index)})"
            )
        return self._by_index[index]

    def get_by_name(self, name: str) -> list[Species]:
        """Return all species whose :attr:`~pythia.models.pm.species.Species.name` equals *name*.

        Args:
            name: The name to filter by.

        Returns:
            A list of matching :class:`Species` instances in index order.
            Empty if no species with *name* are registered.

        Examples:
            >>> pop = Population()
            >>> _ = pop.register("wolf", {"size": "large"})
            >>> _ = pop.register("wolf", {"size": "small"})
            >>> _ = pop.register("rabbit")
            >>> sizes = [s.attributes["size"] for s in pop.get_by_name("wolf")]
            >>> sizes == ["large", "small"]
            True
        """
        return [s for s in self._by_index if s.name == name]

    def __repr__(self) -> str:
        """Return a short representation showing the registry size.

        Returns:
            A string of the form ``Population(size=3)``.
        """
        return f"Population(size={len(self)!r})"
