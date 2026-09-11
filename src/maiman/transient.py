"""Erbium gain dynamics: what the amplifier does between steady states.

:class:`~maiman.components.EDFA` solves the gain an erbium amplifier *settles*
into. This module solves how it gets there, which is a different question on a
different time axis and is why it is an analysis rather than a block.

**The two axes are five orders of magnitude apart, and that is measured rather
than assumed.** The metastable level of Er3+ has a lifetime near 10 ms, and the
effective time constant under load is ``tau / (1 + P_out / P_sat)`` — for the
20 dB amplifier this repository ships, between 3.3 and 9.9 ms depending on how
hard it is driven. A simulation window is 4096 symbols at 32 GBd, which is
128 ns. The fastest transient here is **twenty-five thousand windows long**. A
per-sample block could not show any part of it: within one window the gain is a
constant, which is exactly what :class:`~maiman.components.EDFA` already
assumes, and it is right to.

So there is no ``metastable_lifetime`` parameter on the amplifier. A parameter
that cannot change the result of a run is a control the interface cannot back,
and this library does not ship those. The lifetime belongs to the analysis that
can use it, which is this one.

**The model is the dynamic form of the steady state already in the component**,
not a second model beside it. Writing the Saleh reservoir in terms of
``g = ln G`` gives::

    tau * dg/dt = ln G_0 - g - (e^g - 1) * P_in / P_sat

Setting ``dg/dt = 0`` returns ``G = G_0 * exp(-(G - 1) * P_in / P_sat)``, which
is the equation :meth:`~maiman.components.EDFA.effective_gain` solves by Newton.
The same ``G_0`` and the same ``P_sat``, with exactly one quantity added, so the
two cannot drift apart — and a test integrates the ODE to rest and checks it
lands on the component's own solver, two independent code paths meeting at 1e-9.

Linearising about that rest point gives the effective time constant::

    tau_eff = tau / (1 + P_out / P_sat)

which is why a saturated amplifier answers in a fraction of its lifetime: the
stimulated emission that drains the reservoir is itself proportional to how full
it is. That closed form is tested too, against a step response measured from the
integrator, to six figures.

**What this does not model.** One reservoir, so one gain for every wavelength.
A real erbium transient *tilts* the gain spectrum as the inversion changes, so a
surviving channel's excursion depends on where in the band it sits, and this
reports one number for all of them. The pump is implicit in ``G_0`` and does not
respond: a deployed amplifier has a control loop that pushes back on exactly the
excursion computed here, and what this gives is the uncontrolled case. Both are
real effects and both are absent rather than approximated.

Model references: A. A. M. Saleh, R. M. Jopson, J. D. Evankow and J. Aspell,
"Modeling of gain in erbium-doped fiber amplifiers", IEEE Photon. Technol. Lett.
2(10), 1990; Y. Sun, J. L. Zyskind and A. K. Srivastava, "Average inversion
level, modeling, and physics of erbium-doped fiber amplifiers", IEEE J. Sel.
Top. Quantum Electron. 3(4), 1997; A. Bononi and L. A. Rusch, "Doped-fiber
amplifier dynamics: a system perspective", J. Lightwave Technol. 16(5), 1998.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .components.amplifiers import EDFA
from .units import db_to_linear

#: Metastable (4I13/2) lifetime of Er3+ in silica, in seconds. Reported between
#: 8 and 12 ms across host compositions; 10 ms is the figure Desurvire and the
#: dynamics literature above use, and the one every number in this module's
#: docstrings was computed at.
METASTABLE_LIFETIME = 10e-3

#: Integration substeps per effective time constant. RK4 on this ODE is well
#: inside its accuracy at 32; the step actually used is the finer of this and the
#: caller's own grid, so a coarse output grid gets a correct curve rather than a
#: quietly wrong one. What holds this number honest is not the number itself but
#: the test that varies the *output* grid over a 200x range and requires the same
#: curve out of all of it: too few substeps and the coarse grids disagree.
SUBSTEPS_PER_CONSTANT = 32


@dataclass(frozen=True)
class GainTransient:
    """A gain excursion in time, and the two rest points it runs between."""

    times: np.ndarray
    """Time [s], the grid the caller asked for."""

    gain: np.ndarray
    """Linear power gain at each time."""

    input_power: np.ndarray
    """Total input power [W] at each time, as it was given."""

    lifetime: float
    """Metastable lifetime the integration used [s]."""

    substeps: int
    """Integration steps taken per output interval.

    Reported rather than hidden because it is the difference between a curve and
    a plausible-looking wrong one: the ODE is stiff against a coarse grid, and a
    caller asking for ten points across a millisecond gets the right answer only
    because the integrator refuses to take ten steps to produce them.
    """

    @property
    def gain_db(self) -> np.ndarray:
        """Gain in dB, which is the unit an excursion is quoted in."""
        return 10.0 * np.log10(self.gain)

    @property
    def output_power(self) -> np.ndarray:
        """Total output power [W] at each time."""
        return self.gain * self.input_power

    @property
    def excursion(self) -> float:
        """Largest departure from the starting gain [dB], signed.

        Signed because the two directions are different failures. Dropping
        channels raises the gain and can drive a receiver into overload; adding
        them lowers it and can starve one. Reporting an absolute value would make
        the two look alike.
        """
        start = self.gain_db[0]
        swing = self.gain_db - start
        return float(swing[np.argmax(np.abs(swing))])

    def settling_time(self, tolerance_db: float = 0.1) -> float:
        """Time until the gain is within ``tolerance_db`` of where it ends [s].

        Measured from the last time it was *outside* that band, not the first
        time it entered one, so a curve that crosses and comes back is not
        reported as settled at the crossing.
        """
        final = self.gain_db[-1]
        outside = np.nonzero(np.abs(self.gain_db - final) > tolerance_db)[0]
        if outside.size == 0:
            return 0.0
        return float(self.times[min(int(outside[-1]) + 1, self.times.size - 1)])

    def __repr__(self) -> str:
        return (
            f"GainTransient({self.gain_db[0]:.2f} -> {self.gain_db[-1]:.2f} dB, "
            f"excursion {self.excursion:+.2f} dB, "
            f"settles in {self.settling_time() * 1e3:.2f} ms)"
        )


def effective_time_constant(
    amplifier: EDFA, input_power: float, *, lifetime: float = METASTABLE_LIFETIME
) -> float:
    """``tau / (1 + P_out / P_sat)`` at this operating point [s].

    The rate the reservoir empties is proportional to how full it is, so an
    amplifier driven hard answers faster than its own lifetime. This is the
    linearisation of the ODE in this module about its rest point, and the number
    that says whether a transient matters on the timescale someone cares about.
    """
    small_signal = db_to_linear(amplifier.gain)
    if not amplifier.saturate or small_signal <= 2.0 or input_power <= 0.0:
        return lifetime
    saturation = amplifier.intrinsic_saturation_power(small_signal)
    if saturation <= 0.0:
        return lifetime
    output_power = amplifier.effective_gain(input_power) * input_power
    return lifetime / (1.0 + output_power / saturation)


def step_schedule(
    segments: Sequence[tuple[float, float]], *, points_per_segment: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(times, input_power)`` for a piecewise-constant input.

    ``segments`` is ``(duration [s], input power [W])`` in order. The common case
    this module exists for — an amplifier running settled, then a group of
    channels disappearing — is two segments.

    **The step is placed exactly on a sample**, and that is what makes the
    integration of it exact rather than nearly so. The input is treated as
    right-continuous: the power over an interval is the value at its *start*. So
    a segment's samples run from its own start time up to but not including the
    next segment's, and no interval ever straddles a discontinuity with one
    power standing in for two. Put the step between samples instead and the
    interval containing it is integrated at whichever power was picked, wrong for
    part of its length by an amount nothing reports.
    """
    if not segments:
        raise ValueError("a schedule needs at least one segment")
    if points_per_segment < 2:
        raise ValueError(f"a segment needs at least two points, got {points_per_segment}")

    times: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    start = 0.0
    for index, (duration, power) in enumerate(segments):
        if duration <= 0.0:
            raise ValueError(f"segment {index} has duration {duration}, which is not a duration")
        if power < 0.0:
            raise ValueError(f"segment {index} has input power {power}, which is not a power")
        # Closed at the left and open at the right, except for the last segment
        # which has to carry the final time. So each step lands on the first
        # sample of the segment it begins, and no interval straddles one.
        last = index == len(segments) - 1
        grid = np.linspace(start, start + duration, points_per_segment, endpoint=last)
        times.append(grid)
        powers.append(np.full(grid.shape, float(power)))
        start += duration
    return np.concatenate(times), np.concatenate(powers)


def gain_transient(
    amplifier: EDFA,
    times: np.ndarray,
    input_power: np.ndarray,
    *,
    lifetime: float = METASTABLE_LIFETIME,
    initial_gain: float | None = None,
) -> GainTransient:
    """Integrate the erbium reservoir across a changing input power.

    ``times`` is a strictly increasing grid [s] and ``input_power`` the total
    power into the amplifier at each of those times [W] — total, because
    saturation is driven by everything present, which is the whole reason
    dropping some channels changes the gain seen by the others.

    The run **starts settled**: the initial gain is the steady state at
    ``input_power[0]``, because the question this answers is what happens when a
    running amplifier is disturbed, not what happens when one is switched on.
    Pass ``initial_gain`` to start somewhere else — chaining one amplifier's
    output into the next, for instance, where the second is settled to a
    different point.

    **Chaining is exact rather than approximate.** There is no feedback from a
    later amplifier to an earlier one, so a span's amplifiers can be integrated
    in order, each taking the previous one's :attr:`GainTransient.output_power`
    on the same grid as its own input. That matters because a surviving channel's
    excursion *accumulates* down a chain, and one amplifier's 3 dB is not what
    takes a link out.

    An amplifier with ``saturate`` false has no dynamics to integrate — its gain
    does not depend on its input, so nothing relaxes — and the constant gain is
    returned rather than a flat line produced by integrating zero.
    """
    grid = np.asarray(times, dtype=np.float64)
    drive = np.asarray(input_power, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError(f"times must be 1-D, got {grid.ndim}-D")
    if drive.shape != grid.shape:
        raise ValueError(f"input_power must match times, got {drive.shape} against {grid.shape}")
    if grid.size < 2:
        raise ValueError("a transient needs at least two times")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if np.any(drive < 0.0):
        raise ValueError("input_power must not be negative")
    if lifetime <= 0.0:
        raise ValueError(f"lifetime must be positive, got {lifetime}")

    small_signal = db_to_linear(amplifier.gain)
    settled = (
        amplifier.effective_gain(float(drive[0])) if initial_gain is None else float(initial_gain)
    )
    if not amplifier.saturate or small_signal <= 2.0:
        return GainTransient(
            times=grid,
            gain=np.full(grid.shape, settled),
            input_power=drive,
            lifetime=lifetime,
            substeps=1,
        )

    saturation = amplifier.intrinsic_saturation_power(small_signal)
    log_small_signal = math.log(small_signal)

    # One substep count for the whole run, from the fastest constant any of these
    # powers implies. Per-interval adaptation would be cheaper and would make the
    # grid's own spacing a hidden parameter of the answer.
    fastest = min(
        effective_time_constant(amplifier, float(power), lifetime=lifetime)
        for power in (float(drive.max()), float(drive.min()))
    )
    coarsest = float(np.diff(grid).max())
    substeps = max(1, math.ceil(coarsest / (fastest / SUBSTEPS_PER_CONSTANT)))

    def slope(log_gain: float, power: float) -> float:
        return (
            log_small_signal - log_gain - (math.exp(log_gain) - 1.0) * power / saturation
        ) / lifetime

    gains = np.empty(grid.shape, dtype=np.float64)
    gains[0] = settled
    log_gain = math.log(settled)
    for index in range(1, grid.size):
        span = float(grid[index] - grid[index - 1])
        step = span / substeps
        # Right-continuous: the power over an interval is the one at its start,
        # which is exact for a step function whose steps land on samples and is
        # what `step_schedule` builds. Taking the end value instead would apply
        # the new power to the interval *before* the step.
        power = float(drive[index - 1])
        for _ in range(substeps):
            k1 = slope(log_gain, power)
            k2 = slope(log_gain + step * k1 / 2.0, power)
            k3 = slope(log_gain + step * k2 / 2.0, power)
            k4 = slope(log_gain + step * k3, power)
            log_gain += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        gains[index] = math.exp(log_gain)

    return GainTransient(
        times=grid,
        gain=gains,
        input_power=drive,
        lifetime=lifetime,
        substeps=substeps,
    )
