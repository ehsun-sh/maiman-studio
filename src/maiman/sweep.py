"""Parameter sweeps and repeated runs.

A single simulation is rarely the answer to an engineering question. "What is
the sensitivity?" and "how far can this reach?" are both curves, and producing
them by hand-writing a loop that mutates a graph is exactly where a stray value
gets left behind and quietly contaminates every later point. Sweeping is
first-class for that reason, and overrides are applied per run and rolled back.

Points are independent, which is what makes ``workers`` possible — and the
independence is the whole of the argument, because a sweep that gave different
answers depending on how many threads ran it would be worthless as a
measurement.

**Threads, with a graph each.** Not processes: measured on a nonlinear span
sweep the two are the same speed — 3.25x against 3.14x on twelve workers — and
threads cost nothing to start, need nothing to be picklable, and do not care
whether a component was defined in a notebook. What makes threads work here at
all is that the time goes into numpy's FFTs, which release the GIL; what stops
either of them scaling past about 3x is that a split step is memory-bound long
before it is core-bound.

The graph each worker holds is a deep copy, because a run *mutates*: overrides
are written onto the components and rolled back afterwards, so two threads
sharing one graph would overwrite each other's parameters and produce numbers
belonging to neither point. Copies are made per worker rather than per point —
they are not free — and results come back keyed by component **label**, which is
what makes a copy's results indistinguishable from the original's.
"""

from __future__ import annotations

import copy
import hashlib
import os
import queue
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeVar

import numpy as np

from .component import Component
from .graph import Graph, Results

T = TypeVar("T")

#: An axis is identified by the component (or its label) and a parameter name.
AxisKey = tuple[Component | str, str]


def _label(target: Component | str) -> str:
    return target if isinstance(target, str) else target.label


def derive_run_seed(base_seed: int, run_index: int) -> int:
    """A seed for repeated run ``run_index``, derived from ``base_seed``.

    Hashed rather than incremented. Adjacent seeds are not a problem for a
    modern generator, but deriving by hash keeps the same guarantee the
    per-block streams rely on — that seeds which look related are not — and it
    costs nothing.
    """
    digest = hashlib.blake2b(f"{base_seed}:{run_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class SweepPoint:
    """One combination of swept parameter values, with the runs taken there."""

    index: int
    values: dict[str, float | bool]
    """Swept values at this point, keyed ``"label.parameter"``."""

    runs: tuple[Results, ...]

    def __repr__(self) -> str:
        settings = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"SweepPoint({settings}; {len(self.runs)} run(s))"


@dataclass(frozen=True)
class SweepResult:
    """Every point of a sweep, in the order the axes were given."""

    points: tuple[SweepPoint, ...]
    axes: dict[str, tuple[float | bool, ...]]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Any:
        return iter(self.points)

    def axis(self, target: Component | str, parameter: str) -> np.ndarray:
        """The values one axis took, in sweep order."""
        key = f"{_label(target)}.{parameter}"
        if key not in self.axes:
            raise KeyError(f"{key!r} was not swept; axes were {sorted(self.axes)}")
        return np.array([point.values[key] for point in self.points], dtype=np.float64)

    def metric(self, component: Component, extract: Callable[[Any], float]) -> np.ndarray:
        """Pull one number out of every run, shaped ``(points, runs)``.

        ``extract`` receives whatever the component put on its output port::

            q = result.metric(analyzer, lambda m: m.q_factor)
            mean_q = q.mean(axis=1)
        """
        return np.array(
            [[float(extract(run[component])) for run in point.runs] for point in self.points],
            dtype=np.float64,
        )


def sweep(
    graph: Graph,
    axes: Mapping[AxisKey, Sequence[float | bool]],
    *,
    runs: int = 1,
    keep: list[Component] | None = None,
    fixed: Mapping[AxisKey, float | bool] | None = None,
    workers: int | None = 1,
) -> SweepResult:
    """Run ``graph`` once for every combination of the given parameter values.

    ``axes`` maps ``(component, parameter)`` to the values it should take; the
    sweep covers their cartesian product, varying the last axis fastest.

    ``runs`` repeats each point with independently derived seeds. That is what
    turns a noisy measurement into a distribution — a single BER estimate at a
    marginal operating point is one sample, not an answer.

    ``fixed`` applies the same override at every point, which saves editing the
    graph just to hold something constant for one study.

    ``workers`` runs that many points at a time, each on its own deep copy of the
    graph; ``None`` picks one per core, capped at the number of points. The
    default of 1 is sequential and copies nothing, because a sweep of four cheap
    points is slower to parallelise than to run.

    **The answer does not depend on it.** Every point is a function of its own
    overrides and its own derived seed, so the numbers are identical however many
    workers ran them, and they come back in sweep order regardless of the order
    they finished in. There is a test on exactly that.

    The graph is left exactly as it was found, including if a run raises — and
    above one worker it is never run at all, only copied.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if not axes:
        raise ValueError("a sweep needs at least one axis")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1 or None for one per core, got {workers}")

    keys = list(axes)
    named_axes = {f"{_label(t)}.{p}": tuple(axes[(t, p)]) for (t, p) in keys}
    for name, axis_values in named_axes.items():
        if not axis_values:
            raise ValueError(f"axis {name!r} has no values")

    combinations = list(product(*(axes[key] for key in keys)))

    def point(index: int, combination: tuple[float | bool, ...], on: Graph) -> SweepPoint:
        """One point of the sweep, run on whichever graph the worker was handed."""
        overrides: dict[AxisKey, float | bool] = dict(fixed or {})
        overrides.update(dict(zip(keys, combination, strict=True)))
        results = tuple(
            on.run(
                keep,
                overrides=overrides,
                # From the *original* context, not the copy's, so that adding a
                # worker cannot change which seeds a sweep draws. They are the
                # same object, and saying so here is cheaper than relying on it.
                seed=derive_run_seed(graph.ctx.seed, run_index) if runs > 1 else None,
            )
            for run_index in range(runs)
        )
        values = {
            f"{_label(target)}.{parameter}": value
            for (target, parameter), value in zip(keys, combination, strict=True)
        }
        return SweepPoint(index=index, values=values, runs=results)

    lanes = _lane_count(workers, len(combinations))
    if lanes == 1:
        points = [point(i, c, graph) for i, c in enumerate(combinations)]
    else:
        points = _run_in_parallel(graph, combinations, lanes, point)

    return SweepResult(points=tuple(points), axes=named_axes)


def _lane_count(workers: int | None, points: int) -> int:
    """How many points to have in flight. Never more than there are points."""
    if workers is None:
        workers = os.cpu_count() or 1
    return max(1, min(workers, points))


def _run_in_parallel(
    graph: Graph,
    combinations: list[tuple[float | bool, ...]],
    lanes: int,
    point: Callable[[int, tuple[float | bool, ...], Graph], SweepPoint],
) -> list[SweepPoint]:
    """Run the points across ``lanes`` threads, each with a graph of its own.

    The copies are handed out through a queue rather than indexed by worker.
    A thread pool makes no promise about which of its threads picks up which
    task, so binding a graph to a thread would be binding it to an assumption;
    taking one from a queue and putting it back binds it to the work instead.

    Deep copies, because :meth:`Graph.run` writes a point's overrides onto the
    components and rolls them back after. Two threads doing that to one graph
    would interleave, and the failure would not be an exception — it would be a
    curve with the right shape and the wrong numbers.
    """
    available: queue.SimpleQueue[Graph] = queue.SimpleQueue()
    for _ in range(lanes):
        available.put(copy.deepcopy(graph))

    def task(job: tuple[int, tuple[float | bool, ...]]) -> SweepPoint:
        index, combination = job
        borrowed = available.get()
        try:
            return point(index, combination, borrowed)
        finally:
            # Returned even if the run raised, so that one bad point does not
            # starve the workers still waiting behind it.
            available.put(borrowed)

    with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="maiman-sweep") as pool:
        # `map` yields in submission order, which is sweep order, whatever order
        # the points actually finished in — and re-raises the first failure.
        return list(pool.map(task, enumerate(combinations)))
