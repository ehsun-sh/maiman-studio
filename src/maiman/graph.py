"""The simulation graph and its execution engine.

Execution is **block-mode**: each component is invoked exactly once per run and
processes the entire time window in a single call. This is what commercial
system simulators do, and it makes every block a pure function of its inputs,
which in turn is what allows vectorised (and later GPU) implementations without
any change to the scheduler.

It is deliberately *not* a streaming, sample-at-a-time scheduler like GNU Radio.
Optical system simulation runs a fixed-length sequence and analyses the result;
streaming would buy nothing and complicate every block.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TypeVar

from .component import Component, Port, PortGroup, PortType
from .context import SimulationContext
from .signals import Signal

C = TypeVar("C", bound=Component)


@dataclass(frozen=True)
class Progress:
    """How far through a run the engine is, as it happens.

    ``fraction`` is what a bar is drawn from. It is deliberately *not* elapsed
    time over an estimate: a run has no estimate before it starts, and the one
    honest quantity available is how much of the work is behind it.

    Components are weighted equally, which is a lie in every graph and the least
    misleading one available. The alternative is a cost model per component, and
    a cost model that is wrong makes the bar run backwards — a fiber's cost
    depends on a step size chosen from the peak power it is about to see, which
    is not knowable until it has seen it. Equal weighting is at worst uneven;
    it is never wrong about the direction of travel.

    What rescues it in practice is ``within``. A component that spends real time
    reports its own fraction through :meth:`Component.report`, and the run's
    fraction advances smoothly across it instead of resting on one block for a
    minute. In a link with a thousand-kilometre span, that one block *is* the
    run, and without it the bar would sit still for the whole of it.
    """

    label: str
    """Label of the component now running."""

    index: int
    """Its position in the execution order, from 0."""

    total: int
    """How many nodes the run will execute.

    Nodes, not components, because a component that declares independent port
    groups is scheduled once per group — a circulator's routes do not depend on
    each other and are ordered separately. Such a block reports its own label
    more than once during a run, at different points in the order. Every other
    block is one node and the two counts are the same number.
    """

    within: float
    """How far through *this* component, 0 to 1. Zero unless it reports."""

    @property
    def fraction(self) -> float:
        """Fraction of the whole run behind this point, 0 to 1."""
        return (self.index + self.within) / self.total if self.total else 1.0

    def __str__(self) -> str:
        return f"{self.fraction * 100:5.1f}%  {self.label}"


class GraphError(Exception):
    """Invalid graph structure, detected before any component runs."""


class CycleError(GraphError):
    """The graph contains a feedback loop with no loop-control component.

    Cycles are not forbidden in principle — recirculating loops and optical
    feedback are real — but they need an explicit component that declares how
    many times the loop runs. Leaving the semantics implicit would make results
    depend on scheduler internals.
    """


class Results:
    """Signals produced by a run, addressed by component and port."""

    def __init__(self, values: dict[tuple[str, str], Signal]) -> None:
        self._values = values

    def port(self, component: Component, name: str) -> Signal:
        """The signal on one named output port."""
        key = (component.label, name)
        if key not in self._values:
            raise KeyError(
                f"no retained result for {component.label}.{name}; "
                f"intermediate signals are released once consumed — "
                f"pass keep=[{component.label!r}] to run() to retain it"
            )
        return self._values[key]

    def __getitem__(self, component: Component) -> Signal:
        """The signal on the component's only output port."""
        retained = [name for (label, name) in self._values if label == component.label]
        if not retained:
            raise KeyError(
                f"no retained result for {component.label}; "
                f"pass keep=[...] to run() to retain intermediate signals"
            )
        if len(retained) > 1:
            raise KeyError(
                f"{component.label} has {len(retained)} retained outputs "
                f"({sorted(retained)}); use results.port(component, name)"
            )
        return self._values[(component.label, retained[0])]

    def __contains__(self, component: Component) -> bool:
        return any(label == component.label for label, _ in self._values)

    def items(self) -> list[tuple[tuple[str, str], Signal]]:
        """Every retained result, as ``((label, port name), signal)`` pairs.

        Public because the session server needs it and the server is an ordinary
        client of this API — it gets no privileged access, so anything it can do
        is something a script can do. Returns a list rather than a view: callers
        iterate it while the results outlive the run, and a snapshot cannot be
        invalidated underneath them.
        """
        return sorted(self._values.items())

    def __repr__(self) -> str:
        keys = ", ".join(f"{label}.{name}" for label, name in sorted(self._values))
        return f"Results({keys})"


class Graph:
    """A directed graph of components, executed by :meth:`run`."""

    def __init__(self, ctx: SimulationContext) -> None:
        self.ctx = ctx
        self._components: list[Component] = []
        # (dst component label, dst port name) -> source Port
        self._edges: dict[tuple[str, str], Port] = {}

    def add(self, component: C) -> C:
        """Add a component and return it, so it can be assigned in one line.

        Components without an explicit label get one derived from their class and
        their position among components of that class in *this graph*. That makes
        labels a function of the graph alone, which matters because a component's
        label seeds its random stream: a process-global counter would make the
        same script produce different noise on every run.
        """
        if not component.has_explicit_label:
            same_kind = sum(1 for c in self._components if type(c) is type(component))
            component.label = f"{type(component).__name__}{same_kind + 1}"
        if any(c.label == component.label for c in self._components):
            raise GraphError(f"duplicate component label {component.label!r}")
        self._components.append(component)
        return component

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self._components)

    @property
    def edges(self) -> dict[tuple[str, str], Port]:
        """Connections, keyed by the destination ``(label, port name)``."""
        return dict(self._edges)

    def connect(self, src: Port | Component, dst: Port | Component) -> None:
        """Wire an output port to an input port.

        Components may be passed directly when they have exactly one output or
        one input respectively, which covers most of a linear link.
        """
        src_port = src.sole_output() if isinstance(src, Component) else src
        dst_port = dst.sole_input() if isinstance(dst, Component) else dst

        if src_port.name not in src_port.component.outputs:
            raise GraphError(f"{src_port} is not an output port")
        if dst_port.name not in dst_port.component.inputs:
            raise GraphError(f"{dst_port} is not an input port")
        if src_port.type is not dst_port.type:
            raise GraphError(
                f"cannot connect {src_port} ({src_port.type}) to "
                f"{dst_port} ({dst_port.type}): port types differ"
            )

        key = (dst_port.component.label, dst_port.name)
        if key in self._edges:
            raise GraphError(
                f"{dst_port} is already driven by {self._edges[key]}; "
                f"an input port takes exactly one connection"
            )
        for component in (src_port.component, dst_port.component):
            if component not in self._components:
                raise GraphError(f"{component.label} was not added to the graph")
        self._edges[key] = src_port

    def chain(self, *components: Component) -> None:
        """Connect a sequence of single-input/single-output components in order."""
        for upstream, downstream in pairwise(components):
            self.connect(upstream, downstream)

    # -- validation and scheduling ----------------------------------------

    def _validate(self) -> None:
        for component in self._components:
            for name in component.inputs:
                if (component.label, name) not in self._edges:
                    raise GraphError(f"{component.label}.{name} is not connected")
            # Settings that disagree with each other, reported before the first
            # block runs rather than when the offending one is reached.
            component.validate()

    def _topological_order(self) -> list[tuple[Component, PortGroup]]:
        """The run order, one entry per independent slice of a component.

        **The node is a port group, not a component.** For almost every block
        those are the same thing — one group holding every port — and this is
        the ordinary topological sort it has always been. They part company on a
        device whose routes do not touch: a circulator's ``out3`` is a function
        of ``in2`` alone, so a grating reflecting light back into port 2 is not
        a loop even though ``circulator -> grating -> circulator`` is a cycle in
        the component graph. Sorting groups sees that; sorting components cannot.

        See :class:`~maiman.component.PortGroup` for why a component is allowed
        to make that claim and what it costs if it makes it wrongly.
        """
        nodes: list[tuple[Component, PortGroup]] = []
        #: Which node produces a given output port, and which consumes an input.
        produced_by: dict[tuple[str, str], int] = {}
        consumed_by: dict[tuple[str, str], int] = {}
        for component in self._components:
            component.check_port_groups()
            for group in component.port_groups():
                index = len(nodes)
                nodes.append((component, group))
                for name in group.outputs:
                    produced_by[(component.label, name)] = index
                for name in group.inputs:
                    consumed_by[(component.label, name)] = index

        successors: dict[int, list[int]] = defaultdict(list)
        in_degree: dict[int, int] = dict.fromkeys(range(len(nodes)), 0)

        for destination, src_port in self._edges.items():
            consumer = consumed_by[destination]
            producer = produced_by[(src_port.component.label, src_port.name)]
            successors[producer].append(consumer)
            in_degree[consumer] += 1

        ready = deque(sorted(index for index, deg in in_degree.items() if deg == 0))
        order: list[tuple[Component, PortGroup]] = []
        while ready:
            index = ready.popleft()
            order.append(nodes[index])
            for successor in successors[index]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready.append(successor)

        if len(order) != len(nodes):
            unresolved = sorted(
                {nodes[index][0].label for index, deg in in_degree.items() if deg > 0}
            )
            raise CycleError(
                f"the graph contains a feedback loop involving {unresolved}. "
                f"Break it with an explicit loop-control component that declares "
                f"an iteration count."
            )
        return order

    # -- execution --------------------------------------------------------

    def _resolve_label(self, component: Component | str) -> str:
        if isinstance(component, str):
            return component
        return component.label

    @contextmanager
    def _applied(
        self, overrides: Mapping[tuple[Component | str, str], float | bool] | None
    ) -> Iterator[None]:
        """Temporarily apply parameter overrides, restoring them afterwards.

        A sweep must not leave the graph altered: the same graph object is reused
        for every point, so a leaked value would silently contaminate every later
        run. Restoration happens even if the run raises.
        """
        if not overrides:
            yield
            return

        by_label = {c.label: c for c in self._components}
        saved: list[tuple[Component, dict[str, float | bool]]] = []
        try:
            for (target, param), value in overrides.items():
                label = self._resolve_label(target)
                component = by_label.get(label)
                if component is None:
                    raise GraphError(f"no component labelled {label!r} in this graph")
                spec = component.param_specs().get(param)
                if spec is None:
                    raise GraphError(f"{label} has no parameter {param!r}")
                saved.append((component, dict(component._values)))
                component._values = {**component._values, param: spec.validate(value)}
            yield
        finally:
            for component, values in reversed(saved):
                component._values = values

    def run(
        self,
        keep: list[Component] | None = None,
        *,
        overrides: Mapping[tuple[Component | str, str], float | bool] | None = None,
        seed: int | None = None,
        progress: Callable[[Progress], None] | None = None,
    ) -> Results:
        """Execute every component once, in dependency order.

        Intermediate signals are released as soon as their last consumer has run,
        so peak memory is the width of the graph cut rather than the whole graph.
        Metric outputs, outputs of sink components, and anything named in ``keep``
        are retained and returned.

        ``overrides`` sets parameters for this run only, keyed by
        ``(component_or_label, parameter_name)``; the graph is left unchanged.
        ``seed`` replaces the context seed, which is how repeated runs draw
        independent noise from the same graph.

        ``progress`` is called as the run advances, with a :class:`Progress`
        describing where it has got to. It is called from the running thread,
        between components and from inside any component that reports, so it
        must be cheap and must not raise — an exception there aborts a run that
        was otherwise fine, which is a bad trade for a status bar.
        """
        with self._applied(overrides):
            return self._run(keep, seed, progress)

    def _run(
        self,
        keep: list[Component] | None,
        seed: int | None,
        progress: Callable[[Progress], None] | None = None,
    ) -> Results:
        ctx = self.ctx if seed is None else replace(self.ctx, seed=seed)
        self._validate()
        order = self._topological_order()

        consumers_remaining: dict[tuple[str, str], int] = defaultdict(int)
        for src_port in self._edges.values():
            consumers_remaining[(src_port.component.label, src_port.name)] += 1

        keep_labels = {c.label for c in (keep or [])}
        live: dict[tuple[str, str], Signal] = {}
        retained: dict[tuple[str, str], Signal] = {}

        total = len(order)
        for index, (component, group) in enumerate(order):
            inputs = {}
            for name in group.inputs:
                src = self._edges[(component.label, name)]
                inputs[name] = live[(src.component.label, src.name)]

            if progress is None:
                produced = component.run(ctx, inputs)
            else:
                # Announced before it runs, so the label on screen is the block
                # currently working rather than the one that has just finished.
                progress(Progress(component.label, index, total, 0.0))
                component._reporter = lambda within, _i=index, _c=component: progress(
                    Progress(_c.label, _i, total, within)
                )
                try:
                    produced = component.run(ctx, inputs)
                finally:
                    # Off again whatever happened. A reporter left installed
                    # would outlive the run that owns it and fire into a closure
                    # holding a graph nobody is using any more.
                    component._reporter = None

            # Exactly this group's outputs, which for a single-group component
            # -- almost all of them -- is exactly its outputs. A component that
            # splits itself is called once per group and answers for that group
            # alone; returning the other group's ports here would overwrite a
            # result computed from inputs this call was not even given.
            missing = set(group.outputs) - set(produced)
            if missing:
                raise GraphError(
                    f"{component.label}.run() did not return output(s) {sorted(missing)}"
                )
            extra = set(produced) - set(group.outputs)
            if extra:
                raise GraphError(
                    f"{component.label}.run() returned undeclared output(s) {sorted(extra)}"
                )

            # No downstream consumer means this is a sink: its results are what
            # the caller actually asked for, so they are always retained.
            is_sink = all(
                consumers_remaining[(component.label, name)] == 0 for name in group.outputs
            )
            for name, value in produced.items():
                key = (component.label, name)
                live[key] = value
                if (
                    component.outputs[name] is PortType.METRIC
                    or is_sink
                    or component.label in keep_labels
                ):
                    retained[key] = value

            # Release inputs whose last consumer has now run.
            for name in group.inputs:
                src_port = self._edges[(component.label, name)]
                src_key = (src_port.component.label, src_port.name)
                consumers_remaining[src_key] -= 1
                if consumers_remaining[src_key] == 0 and src_key not in retained:
                    live.pop(src_key, None)

        if progress is not None and order:
            # One final call at exactly 1.0. Without it the last thing a caller
            # sees is the last component starting, and a bar that stops at 90%
            # on every successful run teaches people to distrust it.
            progress(Progress(order[-1][0].label, total, total, 0.0))

        return Results(retained)

    def __repr__(self) -> str:
        return f"Graph({len(self._components)} components, {len(self._edges)} edges)"
