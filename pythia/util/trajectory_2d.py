"""Trajectory2D — time-ordered 2-D movement description built from a DataTable.

A :class:`Trajectory2D` wraps a sequence of ``(t, x, y, fmt)`` records that
describe where an object is at each observed time, together with a Matplotlib
format string controlling how the object should be rendered at that point.

The sequence is constructed from a :class:`~pythia.api.data_table.DataTable`
and a *trace function* ``ftrace`` that maps a row of the table
(a ``dict[str, float]``) to a ``(x, y, fmt)`` triple:

* ``x``   — horizontal coordinate.
* ``y``   — vertical coordinate.
* ``fmt`` — Matplotlib format string (e.g. ``"ro"``, ``"b^"``, ``"g--"``).

Records are ordered by ascending observation time, mirroring the order
returned by :attr:`~pythia.api.data_table.DataTable.times`.

Example::

    from pythia.api.data_table import DataTable
    from pythia.util.trajectory_2d import Trajectory2D

    table = DataTable(["px", "py"])
    table.add_row(0.0, {"px": 0.0, "py": 0.0})
    table.add_row(1.0, {"px": 1.0, "py": 2.0})
    table.add_row(2.0, {"px": 3.0, "py": 1.0})

    traj = Trajectory2D(table, lambda m: (m["px"], m["py"], "ro"))

    len(traj)           # 3
    traj.get(0)         # (0.0, 0.0, 0.0, 'ro')
    list(traj.times)    # [0.0, 1.0, 2.0]
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

from pythia.api.data_table import DataTable

_EPS = 1e-9   # tolerance for floating-point time comparisons


class Trajectory2D:
    """Time-ordered 2-D trajectory built from a :class:`~pythia.api.data_table.DataTable`.

    For every time *t* present in *table*, the row ``m = table.get(t)`` is
    passed to *ftrace*, which returns the spatial position ``(x, y)`` and the
    Matplotlib format string ``fmt`` for that instant.  The resulting
    ``(t, x, y, fmt)`` records are stored in ascending time order.

    Args:
        table: The source :class:`~pythia.api.data_table.DataTable`.  Every
            row is visited in the order given by
            :attr:`~pythia.api.data_table.DataTable.times`.
        ftrace: A callable ``dict[str, float] -> (x, y, fmt)`` that maps a
            table row to a 2-D position and a Matplotlib format string.

    Raises:
        ValueError: If *table* contains no rows.

    Examples:
        >>> from pythia.api.data_table import DataTable
        >>> table = DataTable(["px", "py"])
        >>> table.add_row(0.0, {"px": 0.0, "py": 0.0})
        >>> table.add_row(1.0, {"px": 1.0, "py": 2.0})
        >>> traj = Trajectory2D(table, lambda m: (m["px"], m["py"], "ro"))
        >>> len(traj)
        2
        >>> traj.get(0)
        (0.0, 0.0, 0.0, 'ro')
        >>> traj.get(1)
        (1.0, 1.0, 2.0, 'ro')
    """

    def __init__(
        self,
        table: DataTable,
        ftrace: Callable[[dict[str, float]], tuple[float, float, str]],
    ) -> None:
        if len(table) == 0:
            raise ValueError("Trajectory2D requires a non-empty DataTable")
        points: list[tuple[float, float, float, str]] = []
        for t in table.times:
            row = table.get(t)
            x, y, fmt = ftrace(row)
            points.append((t, x, y, fmt))
        object.__setattr__(self, "_points", tuple(points))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Trajectory2D is immutable")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, index: int) -> tuple[float, float, float, str]:
        """Return the ``(t, x, y, fmt)`` record at position *index*.

        Args:
            index: Zero-based position.  Negative indices are supported.

        Returns:
            A ``(t, x, y, fmt)`` tuple.

        Raises:
            IndexError: If *index* is out of range.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(5.0, {"v": 1.0})
            >>> traj = Trajectory2D(t, lambda m: (m["v"], 0.0, "b."))
            >>> traj.get(0)
            (5.0, 1.0, 0.0, 'b.')
        """
        return self._points[index]  # type: ignore[index]

    @property
    def times(self) -> tuple[float, ...]:
        """Observation times in ascending order.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(1.0, {"v": 0.0})
            >>> t.add_row(3.0, {"v": 0.0})
            >>> Trajectory2D(t, lambda m: (0.0, 0.0, ".")).times
            (1.0, 3.0)
        """
        return tuple(p[0] for p in self._points)  # type: ignore[index]

    @property
    def xs(self) -> tuple[float, ...]:
        """X-coordinates in time order.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["px"])
            >>> t.add_row(0.0, {"px": 3.0})
            >>> t.add_row(1.0, {"px": 7.0})
            >>> Trajectory2D(t, lambda m: (m["px"], 0.0, ".")).xs
            (3.0, 7.0)
        """
        return tuple(p[1] for p in self._points)  # type: ignore[index]

    @property
    def ys(self) -> tuple[float, ...]:
        """Y-coordinates in time order.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["py"])
            >>> t.add_row(0.0, {"py": 4.0})
            >>> t.add_row(1.0, {"py": 9.0})
            >>> Trajectory2D(t, lambda m: (0.0, m["py"], ".")).ys
            (4.0, 9.0)
        """
        return tuple(p[2] for p in self._points)  # type: ignore[index]

    @property
    def formats(self) -> tuple[str, ...]:
        """Matplotlib format strings in time order.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 1.0})
            >>> Trajectory2D(t, lambda m: (0.0, 0.0, "r^")).formats
            ('r^',)
        """
        return tuple(p[3] for p in self._points)  # type: ignore[index]

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_points(
        cls,
        points: tuple[tuple[float, float, float, str], ...],
    ) -> Trajectory2D:
        """Build a Trajectory2D directly from a pre-computed points tuple."""
        obj: Trajectory2D = object.__new__(cls)
        object.__setattr__(obj, "_points", points)
        return obj

    def _space_at(self, t: float) -> tuple[float, float, str]:
        """Return ``(x, y, fmt)`` space properties at time *t*.

        Resolution order:
        1. Exact match — a trajectory point within *_EPS* of *t*.
        2. Interpolation — *t* lies strictly between two consecutive points.
        3. Past end — *t* is beyond the last point; last point's properties
           are held constant.
        4. Before start — *t* is before the first point; first point's
           properties are used.
        """
        points: tuple = self._points  # type: ignore[assignment]

        # Case 1 — exact match
        for p in points:
            if abs(p[0] - t) < _EPS:
                return p[1], p[2], p[3]

        # Case 3 — past the last recorded time
        if t > points[-1][0]:
            last = points[-1]
            return last[1], last[2], last[3]

        # Case 2 — linear interpolation between two consecutive points
        for i in range(len(points) - 1):
            t1, x1, y1, fmt1 = points[i]
            t2, x2, y2, _ = points[i + 1]
            if t1 < t < t2:
                factor = (t - t1) / (t2 - t1)
                return x1 + (x2 - x1) * factor, y1 + (y2 - y1) * factor, fmt1

        # Case 4 — before the first recorded time (zero-order hold)
        first = points[0]
        return first[1], first[2], first[3]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalise(self, from_: float, to: float, dt: float) -> Trajectory2D:
        """Return a resampled :class:`Trajectory2D` on a uniform time grid.

        Generates every time point ``t = k * dt`` (integer *k*) satisfying
        ``from_ <= t <= to`` and computes the space properties ``(x, y, fmt)``
        at each such *t* by:

        1. **Exact match** — if a trajectory point at time *t* exists, its
           ``(x, y, fmt)`` are used directly.
        2. **Interpolation** — if *t* falls strictly between two consecutive
           trajectory points ``(t1, x1, y1, fmt1)`` and
           ``(t2, x2, y2, fmt2)``, the position is linearly interpolated and
           *fmt* is taken from the earlier point:
           ``x = x1 + (x2-x1)/(t2-t1)*(t-t1)``,
           ``y = y1 + (y2-y1)/(t2-t1)*(t-t1)``,
           ``fmt = fmt1``.
        3. **Past end** — if *t* exceeds the last trajectory point, the last
           point's ``(x, y, fmt)`` are held constant.

        Args:
            from_: Start of the sampling interval (inclusive).
            to: End of the sampling interval (inclusive).
            dt: Uniform time step (must be positive).

        Returns:
            A new :class:`Trajectory2D` containing all sampled points.

        Raises:
            ValueError: If no time point ``k * dt`` falls within
                ``[from_, to]``.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["px", "py"])
            >>> t.add_row(0.0, {"px": 0.0, "py": 0.0})
            >>> t.add_row(2.0, {"px": 2.0, "py": 4.0})
            >>> traj = Trajectory2D(t, lambda m: (m["px"], m["py"], "ro"))
            >>> n = traj.normalise(0.0, 2.0, 1.0)
            >>> len(n)
            3
            >>> n.get(1)
            (1.0, 1.0, 2.0, 'ro')
        """
        tpts: tuple = self._points  # type: ignore[assignment]
        n = len(tpts)

        k_start = math.ceil(from_ / dt - _EPS)
        points_list: list[tuple[float, float, float, str]] = []

        # Single left-to-right pass: trajectory pointer `i` only moves forward.
        # For each grid time t (ascending), advance `i` to the last trajectory
        # index whose time is <= t.  This gives O(n + m) total instead of O(n*m).
        i = 0
        k = k_start
        while True:
            t = k * dt
            if t > to + _EPS:
                break
            if t < from_ - _EPS:
                k += 1
                continue

            # Advance i to the last trajectory point with time <= t.
            while i + 1 < n and tpts[i + 1][0] <= t:
                i += 1

            t_i = tpts[i][0]

            if abs(t_i - t) < _EPS:
                # Case 1: exact match at trajectory point i.
                x, y, fmt = tpts[i][1], tpts[i][2], tpts[i][3]
            elif t_i < t:
                if i + 1 < n:
                    # Case 2: t is strictly between tpts[i] and tpts[i+1].
                    t1, x1, y1, fmt1 = tpts[i]
                    t2, x2, y2, _ = tpts[i + 1]
                    factor = (t - t1) / (t2 - t1)
                    x = x1 + (x2 - x1) * factor
                    y = y1 + (y2 - y1) * factor
                    fmt = fmt1
                else:
                    # Case 3: t is past the last trajectory point.
                    x, y, fmt = tpts[i][1], tpts[i][2], tpts[i][3]
            else:
                # Case 4: t is before the first trajectory point.
                x, y, fmt = tpts[0][1], tpts[0][2], tpts[0][3]

            points_list.append((t, x, y, fmt))
            k += 1

        if not points_list:
            raise ValueError(
                f"No time point k*{dt} falls within [{from_}, {to}]"
            )
        return Trajectory2D._from_points(tuple(points_list))

    # ------------------------------------------------------------------
    # Timing statistics
    # ------------------------------------------------------------------

    def min_step(self) -> float:
        """Return the minimum time difference between consecutive observations.

        Args:
            (none)

        Returns:
            The smallest ``t[i+1] - t[i]`` over all adjacent pairs.

        Raises:
            ValueError: If the trajectory contains fewer than two points.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 0.0})
            >>> t.add_row(1.0, {"v": 1.0})
            >>> t.add_row(3.0, {"v": 2.0})
            >>> Trajectory2D(t, lambda m: (0.0, 0.0, ".")).min_step()
            1.0
        """
        points: tuple = self._points  # type: ignore[assignment]
        if len(points) < 2:
            raise ValueError(
                "min_step requires at least two time points in the trajectory"
            )
        return min(
            points[i + 1][0] - points[i][0]
            for i in range(len(points) - 1)
        )

    def __len__(self) -> int:
        """Number of recorded time points.

        Examples:
            >>> from pythia.api.data_table import DataTable
            >>> t = DataTable(["v"])
            >>> t.add_row(0.0, {"v": 1.0})
            >>> t.add_row(1.0, {"v": 2.0})
            >>> len(Trajectory2D(t, lambda m: (0.0, 0.0, ".")))
            2
        """
        return len(self._points)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[tuple[float, float, float, str]]:
        """Iterate over ``(t, x, y, fmt)`` records in time order."""
        return iter(self._points)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"Trajectory2D(points={len(self._points)})"  # type: ignore[arg-type]
