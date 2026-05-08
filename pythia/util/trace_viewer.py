"""TraceViewer — animated 2-D visualisation of simulated traces.

A :class:`TraceViewer` takes a :class:`~pythia.api.data_tracer.DataTracer`
and a trace function, and produces a Matplotlib
:class:`~matplotlib.animation.FuncAnimation` that shows every traced object
moving through 2-D space over a specified time window.

For each named object in the tracer, a :class:`~pythia.util.trajectory_2d.Trajectory2D`
is constructed from its :class:`~pythia.api.data_table.DataTable` and normalised
onto the uniform grid ``[start, end]`` with step *dt* before the animation is
assembled.

Requires the ``matplotlib`` package (``pip install 'pythia[visualization]'``).

Example::

    from pythia.api.data_tracer import DataTracer
    from pythia.util.trace_viewer import TraceViewer

    tracer = DataTracer(
        labels=["px", "py"],
        trace_function=lambda s: {
            "agent": {"px": s.data["x"], "py": s.data["y"]}
        },
    )
    # ... run simulation feeding tracer ...

    viewer = TraceViewer(tracer, lambda m: (m["px"], m["py"], "ro"))
    anim = viewer.create_animation(start=0.0, end=10.0, dt=0.1)
    anim.save("trace.gif")
"""

from __future__ import annotations

from collections.abc import Callable

from pythia.api.data_tracer import DataTracer
from pythia.util.trajectory_2d import Trajectory2D

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TraceViewer requires matplotlib.  "
        "Install it with: pip install 'pythia[visualization]'"
    ) from exc


class TraceViewer:
    """Generate a Matplotlib animation from a :class:`~pythia.api.data_tracer.DataTracer`.

    For each named object in *tracer* a
    :class:`~pythia.util.trajectory_2d.Trajectory2D` is built using
    *ftrace*, normalised onto the uniform time grid ``[start, end, dt]``,
    and then animated frame by frame.

    Each animation frame corresponds to one grid time ``k * dt``.  At each
    frame every object is re-plotted using its current ``(x, y, fmt)`` space
    properties, so format strings that vary along the trajectory are correctly
    displayed.

    Args:
        tracer: The :class:`~pythia.api.data_tracer.DataTracer` whose named
            objects are to be animated.
        ftrace: A callable ``dict[str, float] -> (x, y, fmt)`` that maps a
            :class:`~pythia.api.data_table.DataTable` row to a 2-D position
            and a Matplotlib format string.
    """

    def __init__(
        self,
        tracer: DataTracer,
        ftrace: Callable[[dict[str, float]], tuple[float, float, str]],
    ) -> None:
        self._tracer = tracer
        self._ftrace = ftrace

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracer(self) -> DataTracer:
        """The :class:`~pythia.api.data_tracer.DataTracer` used by this viewer."""
        return self._tracer

    @property
    def ftrace(self) -> Callable[[dict[str, float]], tuple[float, float, str]]:
        """The trace function used to map table rows to 2-D positions."""
        return self._ftrace

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def create_animation(
        self,
        start: float,
        end: float,
        dt: float,
    ) -> FuncAnimation:
        """Build and return a :class:`~matplotlib.animation.FuncAnimation`.

        For every named object in :attr:`tracer`:

        1. Retrieve its :class:`~pythia.api.data_table.DataTable`.
        2. Construct a :class:`~pythia.util.trajectory_2d.Trajectory2D`
           using :attr:`ftrace`.
        3. Normalise it onto the grid ``[start, end]`` with step *dt*.

        The animation iterates over the normalised time points.  At each
        frame the axes are cleared and every object is plotted at its
        current ``(x, y, fmt)`` position, allowing per-object, per-frame
        format changes.  Axis limits are fixed to the bounding box of all
        trajectories with 10 % padding.

        Args:
            start: Start time of the animation window (≥ 0).
            end: End time of the animation window (> *start*).
            dt: Time step between frames (0 < *dt* < *end* − *start*).

        Returns:
            A :class:`~matplotlib.animation.FuncAnimation` instance.

        Raises:
            ValueError: If :attr:`tracer` contains no named objects, or if
                any trajectory cannot be normalised over ``[start, end]``.

        Example:
            >>> from pythia.api.data_table import DataTable
            >>> from pythia.api.data_tracer import DataTracer
            >>> from pythia.util.trace_viewer import TraceViewer
            >>> table = DataTable(["px", "py"])
            >>> table.add_row(0.0, {"px": 0.0, "py": 0.0})
            >>> table.add_row(2.0, {"px": 2.0, "py": 1.0})
            >>> tracer = DataTracer(["px", "py"], lambda s: {"obj": {"px": 0.0, "py": 0.0}})
            >>> tracer.trace.__doc__  # just check it exists
            ...
        """
        names = sorted(self._tracer.names)
        if not names:
            raise ValueError("DataTracer contains no named objects to animate")

        # Build and normalise one trajectory per object.
        trajectories: dict[str, Trajectory2D] = {}
        for name in names:
            table = self._tracer.get_table(name)
            raw = Trajectory2D(table, self._ftrace)
            trajectories[name] = raw.normalise(start, end, dt)

        # All normalised trajectories share the same number of frames.
        n_frames = len(next(iter(trajectories.values())))

        # Compute axis limits from the full bounding box of all trajectories.
        all_xs = [x for traj in trajectories.values() for x in traj.xs]
        all_ys = [y for traj in trajectories.values() for y in traj.ys]
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        x_pad = (x_max - x_min) * 0.1 or 1.0
        y_pad = (y_max - y_min) * 0.1 or 1.0
        x_lo, x_hi = x_min - x_pad, x_max + x_pad
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        fig, ax = plt.subplots()

        def update(frame: int) -> list:
            ax.clear()
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            artists = []
            for name, traj in trajectories.items():
                _, x, y, fmt = traj.get(frame)
                line, = ax.plot(x, y, fmt, label=name)
                artists.append(line)
            if len(trajectories) > 1:
                ax.legend(loc="upper right")
            return artists

        return FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=100,
            blit=False,
            repeat=True,
        )
