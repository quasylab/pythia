"""Store — an immutable attribute-value mapping for YODA systems.

A :class:`Store` associates a fixed set of named attributes with arbitrary
values.  Once constructed it cannot be modified; any update produces a new
instance via :meth:`set`.

Example::

    from pythia.models.yoda.store import Store

    s = Store({"x": 1, "y": 2})

    s["x"]              # 1
    s.get("y")          # 2
    s.get("z", 0)       # 0  (default)
    "x" in s            # True
    s.attributes        # frozenset({"x", "y"})

    s2 = s.set("x", 10)
    s2["x"]             # 10
    s["x"]              # 1   (original unchanged)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Store:
    """An immutable mapping from attribute names to values.

    All attribute names must be non-empty strings.  Any Python object may be
    stored as a value.  After construction the store cannot be mutated; use
    :meth:`set` to obtain a new store with one attribute changed.

    Args:
        attributes: Mapping from attribute names to their initial values.
            Defaults to an empty mapping.

    Raises:
        ValueError: If any attribute name is an empty string.

    Examples:
        >>> s = Store({"a": 1, "b": "hello"})
        >>> s["a"]
        1
        >>> s["b"]
        'hello'
        >>> "a" in s
        True
        >>> "c" in s
        False
        >>> len(s)
        2
    """

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        data: dict[str, Any] = dict(attributes) if attributes else {}
        for key in data:
            if not key:
                raise ValueError("Attribute names must be non-empty strings")
        object.__setattr__(self, "_data", data)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Store is immutable")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def attributes(self) -> frozenset[str]:
        """The set of attribute names present in this store.

        Examples:
            >>> Store({"x": 1, "y": 2}).attributes == frozenset({"x", "y"})
            True
            >>> Store().attributes
            frozenset()
        """
        return frozenset(self._data)

    def __contains__(self, attribute: object) -> bool:
        """Return ``True`` if *attribute* is defined in this store.

        Examples:
            >>> s = Store({"x": 1})
            >>> "x" in s
            True
            >>> "y" in s
            False
        """
        return attribute in self._data

    def has(self, attribute: str) -> bool:
        """Return ``True`` if *attribute* has been registered in this store.

        Equivalent to ``attribute in self``.

        Args:
            attribute: The attribute name to look up.

        Returns:
            ``True`` when *attribute* is present; ``False`` otherwise.

        Examples:
            >>> s = Store({"x": 1})
            >>> s.has("x")
            True
            >>> s.has("y")
            False
            >>> Store().has("x")
            False
        """
        return attribute in self._data

    def __len__(self) -> int:
        """Return the number of attributes in this store.

        Examples:
            >>> len(Store({"a": 1, "b": 2}))
            2
            >>> len(Store())
            0
        """
        return len(self._data)

    def __getitem__(self, attribute: str) -> Any:
        """Return the value associated with *attribute*.

        Args:
            attribute: The attribute name to look up.

        Returns:
            The stored value.

        Raises:
            KeyError: If *attribute* is not defined in this store.

        Examples:
            >>> Store({"x": 42})["x"]
            42
        """
        return self._data[attribute]

    def get(self, attribute: str, default: Any = None) -> Any:
        """Return the value for *attribute*, or *default* if absent.

        Args:
            attribute: The attribute name to look up.
            default: Returned when *attribute* is not present.
                Defaults to ``None``.

        Returns:
            The stored value or *default*.

        Examples:
            >>> s = Store({"x": 1})
            >>> s.get("x")
            1
            >>> s.get("y", 99)
            99
            >>> s.get("z") is None
            True
        """
        return self._data.get(attribute, default)

    def test(self, attribute: str, predicate: Callable[[Any], bool]) -> bool:
        """Evaluate *predicate* on the value of *attribute*.

        Returns ``False`` immediately when *attribute* is not defined in
        this store, without calling *predicate*.

        Args:
            attribute: The attribute name to look up.
            predicate: A callable that accepts the stored value and returns
                a :class:`bool`.

        Returns:
            ``predicate(self[attribute])`` if *attribute* is present,
            ``False`` otherwise.

        Examples:
            >>> s = Store({"x": 4, "flag": True})
            >>> s.test("x", lambda v: v > 3)
            True
            >>> s.test("x", lambda v: v > 10)
            False
            >>> s.test("flag", bool)
            True
            >>> s.test("missing", lambda v: True)
            False
        """
        if attribute not in self._data:
            return False
        return predicate(self._data[attribute])

    # ------------------------------------------------------------------
    # Derived stores
    # ------------------------------------------------------------------

    def set(self, attribute: str, value: Any) -> Store:
        """Return a new store with *attribute* set to *value*.

        The current store is not modified.

        Args:
            attribute: The attribute name to set.  Must be non-empty.
            value: The new value.

        Returns:
            A new :class:`Store` with all existing attributes plus the
            updated *attribute*.

        Raises:
            ValueError: If *attribute* is an empty string.

        Examples:
            >>> s = Store({"x": 1})
            >>> s2 = s.set("x", 10)
            >>> s2["x"]
            10
            >>> s["x"]
            1
            >>> s3 = s.set("y", 2)
            >>> "y" in s3
            True
            >>> "y" in s
            False
        """
        if not attribute:
            raise ValueError("Attribute names must be non-empty strings")
        return Store({**self._data, attribute: value})

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Store):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"Store({self._data!r})"
