"""Loop control: the one block that lets a graph contain a cycle.

The scheduler orders a graph, and a cycle cannot be ordered. For most of what
this library models that is the truth rather than a limitation: a link goes one
way, and a cycle in its graph is a wiring mistake the engine is right to refuse.
Two things turned out not to be cycles at all -- a circulator's independent
routes, and its isolation leak -- and :class:`~maiman.component.PortGroup` says
so without any loop control.

Some things are cycles. Light reflected back out of a circulator's port 2
bounces into the grating on that port, which reflects it again: a cavity, whose
steady state is a geometric series nothing feed-forward can sum. This is the
explicit, declared way to run one.
"""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortGroup, PortType
from ..context import SimulationContext
from ..signals import OpticalSignal, Signal


class Feedback(Component):
    """Closes a loop by carrying what reaches it on one pass into the next.

    **Where it goes.** On any one wire of the cycle. ``out`` feeds the loop and
    ``in`` takes what comes back round, so the scheduler sees ``out`` as a source
    and ``in`` as a sink, and the graph has an order again.

    **What a pass is.** The whole graph is run ``passes`` times. On the first,
    ``out`` is dark -- nothing has been round yet. On every later pass it is
    whatever reached ``in`` on the pass before. That is a fixed-point iteration,
    and for a loop whose round-trip gain has magnitude below one it converges to
    the steady state geometrically: the error after ``n`` passes is of order the
    gain to the power ``n``. A circulator with 40 dB of return loss in front of a
    strong grating has a round-trip gain near 0.01, so a handful of passes is
    already exact; a loop with gain near one needs many, and one at or above one
    does not converge at all, and says so.

    **How to know it converged.** ``residual`` is the relative change in the
    returning field between the last two passes -- the norm of the difference
    over the norm of the field, across every band, with the noise power's relative
    change folded in. It is infinite after a single pass, because there is nothing
    yet to compare. Small and still falling means converged. Not falling means the
    loop gain is too high for the pass count, or for any pass count.

    **What it is not.** A pass is not a lap in time. Every signal here is a whole
    window in its own retarded frame, so a trip round the loop adds no delay and
    every lap lands on top of the last. That is right for a cavity short against
    the window -- a reflection between two parts centimetres apart -- and wrong for
    a recirculating loop a pulse goes round many times, where the laps should
    arrive one after another. For that, put a
    :class:`~maiman.components.DelayLine` in the loop as well: each pass is then a
    lap that arrives one loop-time after the last.

    Noise is drawn the same way on every pass, because every block seeds its
    generator from its own label, so the fixed point contains one realisation of
    it rather than a different one per lap.
    """

    display_name = "Feedback"
    category = "Signal Flow"

    passes = Param(
        8.0,
        unit="",
        min=1.0,
        max=10000.0,
        doc="Times the loop is run; the error falls as the loop gain to this power",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "residual": PortType.METRIC}

    def __init__(self, *, label: str | None = None, **params: float) -> None:
        super().__init__(label=label, **params)
        self._previous: OpticalSignal | None = None

    def feedback_passes(self) -> int:
        return int(self.passes)

    def reset_feedback(self) -> None:
        self._previous = None

    def port_groups(self) -> tuple[PortGroup, ...]:
        """A source half and a sink half, which is what removes the cycle.

        ``out`` depends on nothing *this pass* -- it is last pass's ``in`` -- so it
        is a group with no inputs, and the scheduler orders it first.
        """
        return (
            PortGroup(frozenset(), frozenset({"out"})),
            PortGroup(frozenset({"in"}), frozenset({"residual"})),
        )

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        # Called once per group on every pass. With nothing, this is the source
        # half and hands the loop what came back last time; with ``in``, it is the
        # sink half and remembers what came back this time.
        if "in" not in inputs:
            carried = self._previous if self._previous is not None else OpticalSignal()
            return {"out": carried}
        arrived = inputs["in"]
        if not isinstance(arrived, OpticalSignal):
            raise TypeError(
                f"{self.label}: Feedback carries an optical signal, got {type(arrived).__name__}"
            )
        residual = relative_change(self._previous, arrived)
        self._previous = arrived
        return {"residual": residual}


def relative_change(previous: OpticalSignal | None, current: OpticalSignal) -> float:
    """How much a returning signal moved since the last pass, relative to its size.

    The field difference is summed across bands matched by centre frequency; a
    band that appeared or vanished counts in full. The noise contributes its
    relative change in power. The larger of the two is returned, so neither can
    hide an unconverged other. Infinite when there is no previous pass.
    """
    if previous is None:
        return float("inf")

    before = {band.f0: band for band in previous.bands}
    difference = 0.0
    scale = 0.0
    for band in current.bands:
        ex = band.Ex.astype(np.complex128)
        ey = band.Ey.astype(np.complex128)
        power = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
        scale += power
        old = before.pop(band.f0, None)
        if old is None or old.Ex.shape != band.Ex.shape:
            difference += power
            continue
        difference += float(
            np.sum(
                np.abs(ex - old.Ex.astype(np.complex128)) ** 2
                + np.abs(ey - old.Ey.astype(np.complex128)) ** 2
            )
        )
    for old in before.values():
        difference += float(
            np.sum(
                np.abs(old.Ex.astype(np.complex128)) ** 2
                + np.abs(old.Ey.astype(np.complex128)) ** 2
            )
        )

    if scale > 0.0:
        field = float(np.sqrt(difference / scale))
    else:
        field = 0.0 if difference == 0.0 else float("inf")

    noise_now, noise_before = current.noise_power(), previous.noise_power()
    if noise_now > 0.0:
        noise = abs(noise_now - noise_before) / noise_now
    else:
        noise = 0.0 if noise_before == 0.0 else float("inf")

    return max(field, noise)
