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

**The spectrum tilts, and** :func:`spectral_gain_transient` **shows by how much.**
One reservoir means one average inversion, and a real erbium transient moves
every wavelength's gain with it -- by a different number of decibels at each,
set by the fibre's own absorption and emission curves (:class:`ErbiumSpectrum`).
A surviving channel's excursion therefore depends on where in the band it sits,
and at the amplifier's centre wavelength it is exactly :func:`gain_transient`'s.

**Channels drain it at their own rate.** Stimulated emission empties the
reservoir once per photon, so a watt at one wavelength is not a watt at another:
:func:`saturating_power` weights each channel by its cross section and its photon
energy, and a comb sitting at the reference wavelength weighs exactly one, which
is the total :func:`gain_transient` was always driven with.

**And the pump answers back.** :func:`controlled_gain_transient` makes ``G_0`` a
state rather than a constant and integrates an integral control loop beside the
inversion, holding either the gain or the output power. That is what a deployed
amplifier does about the excursion the uncontrolled model computes -- and what it
cannot do past its pump's ceiling, which is reported rather than smoothed over.

**Its own noise is a load too.** With ``self_saturation`` set on the amplifier,
the ASE it emits from both ends drains the reservoir alongside the signal, a term
``b e^g`` beside ``(e^g - 1) P_in / P_sat``, in every integrator here as in the
static solve -- so a coil left in the dark rests below its small-signal gain, and
a drop to nothing lands there rather than on ``G_0``.

**What this does not model.** The loop is the ideal integral one, with no
detector noise, no delay and no dither; and the amplifier's ASE drains the
reservoir at the centre wavelength's rate, flat across the band, as it is emitted.

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
from .units import C_LIGHT, H_PLANCK, db_to_linear

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
    load = amplifier.self_saturation_load()
    if not amplifier.saturate or small_signal <= 2.0 or (input_power <= 0.0 and load <= 0.0):
        return lifetime
    saturation = amplifier.intrinsic_saturation_power(small_signal)
    if saturation <= 0.0:
        return lifetime
    # The amplifier's own ASE grows with the gain as the output does, so it
    # speeds the reservoir up by the same token: d/dg of b e^g is b e^g.
    gain = amplifier.effective_gain(input_power)
    output_power = gain * max(input_power, 0.0) + gain * load
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
    own = amplifier.self_saturation_load() / saturation
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
            log_small_signal
            - log_gain
            - (math.exp(log_gain) - 1.0) * power / saturation
            - own * math.exp(log_gain)
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


# --------------------------------------------------------------------------
# The gain spectrum, and how it tilts as the inversion moves
# --------------------------------------------------------------------------

#: Boltzmann's constant [J/K], for the McCumber relation.
BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class ErbiumSpectrum:
    """An erbium coil's gain spectrum at every inversion, from two measured curves.

    ``absorption_db`` is ``A(lambda)``, the small-signal loss of the whole coil
    with no pump [dB]; ``full_inversion_gain_db`` is ``G*(lambda)``, its gain
    with every ion excited [dB]. These are the Giles parameters, and they are
    what a doped-fibre datasheet publishes. For a homogeneously broadened medium
    at average inversion ``n`` the gain in decibels is linear in ``n``::

        G(lambda, n) = n (A + G*) - A

    which is all a gain spectrum can do as channels come and go: one number, the
    inversion, moves, and every wavelength follows it by its own slope
    ``A + G*``. The two curves come from a fibre, not from here: the shape of an
    erbium spectrum depends on the glass it is doped into, and a curve invented
    to look plausible would be a number nobody can back.
    """

    wavelengths: np.ndarray
    """Wavelengths the curves are sampled at [m], strictly increasing."""

    absorption_db: np.ndarray
    """``A(lambda)``: the unpumped coil's loss [dB], positive."""

    full_inversion_gain_db: np.ndarray
    """``G*(lambda)``: the fully inverted coil's gain [dB], positive."""

    def __post_init__(self) -> None:
        grid = np.asarray(self.wavelengths, dtype=np.float64)
        absorption = np.asarray(self.absorption_db, dtype=np.float64)
        emission = np.asarray(self.full_inversion_gain_db, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("a spectrum needs at least two wavelengths on a 1-D grid")
        if absorption.shape != grid.shape or emission.shape != grid.shape:
            raise ValueError("absorption and full-inversion gain must match the wavelength grid")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("wavelengths must be strictly increasing")
        if np.any(absorption < 0.0) or np.any(emission < 0.0):
            raise ValueError("absorption and full-inversion gain are magnitudes in dB, not signed")
        object.__setattr__(self, "wavelengths", grid)
        object.__setattr__(self, "absorption_db", absorption)
        object.__setattr__(self, "full_inversion_gain_db", emission)

    @classmethod
    def from_mccumber(
        cls,
        wavelengths: np.ndarray,
        absorption_db: np.ndarray,
        *,
        crossover_wavelength: float,
        temperature: float = 295.0,
    ) -> ErbiumSpectrum:
        """Derive the emission curve from the absorption curve, by McCumber.

        Absorption and emission cross sections of one transition are not
        independent: ``sigma_e / sigma_a = exp(h c (1/lambda_c - 1/lambda) / k T)``,
        with ``lambda_c`` the wavelength where they are equal. The coil's two
        curves are the cross sections times the same length and ion density, so
        the ratio carries over to them unchanged. Longward of the crossover
        emission wins, which is why an erbium amplifier's gain sits on the red
        side of its absorption peak.
        """
        grid = np.asarray(wavelengths, dtype=np.float64)
        absorption = np.asarray(absorption_db, dtype=np.float64)
        if crossover_wavelength <= 0.0 or temperature <= 0.0:
            raise ValueError("the crossover wavelength and temperature must be positive")
        ratio = np.exp(
            H_PLANCK
            * C_LIGHT
            * (1.0 / crossover_wavelength - 1.0 / grid)
            / (BOLTZMANN * temperature)
        )
        return cls(
            wavelengths=grid, absorption_db=absorption, full_inversion_gain_db=absorption * ratio
        )

    def _curves(self, wavelength: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
        low, high = float(self.wavelengths[0]), float(self.wavelengths[-1])
        if np.any(points < low) or np.any(points > high):
            raise ValueError(
                f"the spectrum covers {low * 1e9:.1f} to {high * 1e9:.1f} nm and will not "
                "extrapolate past what was measured"
            )
        return (
            np.interp(points, self.wavelengths, self.absorption_db),
            np.interp(points, self.wavelengths, self.full_inversion_gain_db),
        )

    def gain_db(self, wavelength: np.ndarray | float, inversion: np.ndarray | float) -> np.ndarray:
        """``n (A + G*) - A`` at each wavelength [dB]."""
        absorption, emission = self._curves(wavelength)
        return np.asarray(inversion, dtype=np.float64) * (absorption + emission) - absorption

    def inversion(self, wavelength: float, gain_db: np.ndarray | float) -> np.ndarray:
        """The average inversion that gives ``gain_db`` at ``wavelength``."""
        absorption, emission = self._curves(wavelength)
        return (np.asarray(gain_db, dtype=np.float64) + absorption[0]) / (
            absorption[0] + emission[0]
        )

    def tilt(self, wavelengths: np.ndarray | float, reference: float) -> np.ndarray:
        """Dynamic gain tilt: dB moved at each wavelength per dB moved at ``reference``.

        ``(A + G*)(lambda) / (A + G*)(lambda_ref)``, a ratio that does not depend
        on the inversion -- which is why a line system can characterise its
        amplifiers' transients with one curve measured once.
        """
        absorption, emission = self._curves(wavelengths)
        ref_absorption, ref_emission = self._curves(reference)
        return (absorption + emission) / (ref_absorption[0] + ref_emission[0])


@dataclass(frozen=True)
class SpectralTransient:
    """A gain transient seen at each of several wavelengths."""

    reference: GainTransient
    """The reservoir's own transient, at the amplifier's centre wavelength."""

    reference_wavelength: float
    """Where the reference transient's gain is quoted [m]."""

    wavelengths: np.ndarray
    """Channel wavelengths [m]."""

    inversion: np.ndarray
    """Average inversion at each time."""

    gain_db: np.ndarray
    """Gain [dB], shape ``(times, wavelengths)``."""

    @property
    def times(self) -> np.ndarray:
        return self.reference.times

    @property
    def excursions(self) -> np.ndarray:
        """Each channel's largest signed departure from its starting gain [dB]."""
        swing = self.gain_db - self.gain_db[0]
        peak = np.argmax(np.abs(swing), axis=0)
        return swing[peak, np.arange(swing.shape[1])]

    def __repr__(self) -> str:
        spread = ", ".join(
            f"{w * 1e9:.1f} nm {e:+.2f} dB"
            for w, e in zip(self.wavelengths, self.excursions, strict=True)
        )
        return f"SpectralTransient({spread})"


def spectral_gain_transient(
    amplifier: EDFA,
    spectrum: ErbiumSpectrum,
    times: np.ndarray,
    input_power: np.ndarray,
    wavelengths: Sequence[float] | np.ndarray,
    *,
    lifetime: float = METASTABLE_LIFETIME,
    initial_gain: float | None = None,
    channel_powers: np.ndarray | None = None,
) -> SpectralTransient:
    """The same transient as :func:`gain_transient`, spread across the spectrum.

    The reservoir is integrated exactly as :func:`gain_transient` integrates it,
    and its gain is taken to be the gain at the amplifier's
    ``center_wavelength``. That fixes the average inversion at every instant,
    and the inversion fixes the gain everywhere else through ``spectrum``. So at
    the centre wavelength this *is* the single-reservoir answer, and at any other
    wavelength the excursion is that answer times the spectrum's tilt there.

    **Driving it per channel.** Pass ``channel_powers`` -- ``(times, channels)``
    in watts -- and the reservoir is driven by :func:`saturating_power` instead of
    the total, so a channel drains the inversion by its own cross section and
    photon energy. Dropping the short-wavelength half of a comb then costs more
    than dropping the long-wavelength half of equal power, which a single total
    cannot express. Without it the total is used, exactly as before.

    Refused if the amplifier's small-signal gain at its centre wavelength needs
    more than full inversion by this spectrum, which would be an amplifier this
    fibre cannot be.
    """
    channels = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    reference_wavelength = amplifier.si("center_wavelength")
    _, emission = spectrum._curves(reference_wavelength)
    if amplifier.gain > float(emission[0]):
        where = reference_wavelength * 1e9
        raise ValueError(
            f"a {amplifier.gain:.1f} dB small-signal gain at {where:.1f} nm "
            f"needs more than full inversion from a spectrum that reaches "
            f"{float(emission[0]):.1f} dB there"
        )
    spectrum._curves(channels)  # refuse channels outside the measured band before integrating

    drive = np.asarray(input_power, dtype=np.float64)
    if channel_powers is not None:
        drive = saturating_power(spectrum, channels, channel_powers, reference_wavelength)
        if drive.shape != np.asarray(times, dtype=np.float64).shape:
            raise ValueError(
                f"channel_powers must carry one row per time, got {drive.shape[0]} rows for "
                f"{np.asarray(times).shape[0]} times"
            )
    reservoir = gain_transient(
        amplifier, times, drive, lifetime=lifetime, initial_gain=initial_gain
    )
    inversion = spectrum.inversion(reference_wavelength, reservoir.gain_db)
    gain_db = spectrum.gain_db(channels[None, :], inversion[:, None])
    return SpectralTransient(
        reference=reservoir,
        reference_wavelength=reference_wavelength,
        wavelengths=channels,
        inversion=inversion,
        gain_db=gain_db,
    )


# --------------------------------------------------------------------------
# Which channels drain the reservoir, and how hard
# --------------------------------------------------------------------------


def saturating_weights(
    spectrum: ErbiumSpectrum, wavelengths: np.ndarray, reference: float
) -> np.ndarray:
    """How hard a watt at each wavelength drains the inversion, against the reference.

    Stimulated emission empties the reservoir once per photon, so the rate a
    channel drains it at is its *photon* flux times the cross section it sees:
    ``(P / h nu) * sigma``. Relative to a watt at the reference wavelength that
    is ``tilt(lambda) * (lambda / lambda_ref)`` -- the same ``A + G*`` slope the
    gain tilt is built from, times the photon energy ratio.

    A channel at the reference wavelength weighs exactly one, so a comb that sits
    there reduces to the single number :func:`gain_transient` was always driven
    with, and nothing that was measured before this existed moves.
    """
    channels = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    return spectrum.tilt(channels, reference) * (channels / reference)


def saturating_power(
    spectrum: ErbiumSpectrum,
    wavelengths: np.ndarray,
    channel_powers: np.ndarray,
    reference: float,
) -> np.ndarray:
    """The weighted power the reservoir actually feels [W].

    ``channel_powers`` is ``(times, channels)`` in watts. Dropping a short
    wavelength therefore matters more than dropping the same power at a long one,
    which is what the single total could not express.
    """
    powers = np.atleast_2d(np.asarray(channel_powers, dtype=np.float64))
    weights = saturating_weights(spectrum, wavelengths, reference)
    if powers.shape[1] != weights.size:
        raise ValueError(
            f"channel_powers has {powers.shape[1]} channels and {weights.size} wavelengths "
            "were given"
        )
    return powers @ weights


# --------------------------------------------------------------------------
# The pump, and the loop that holds it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PumpControl:
    """A transient control loop: what it holds, how fast, and how far it can push.

    A deployed amplifier does not let its surviving channels rise by three
    decibels when a fibre is cut. It measures, and it moves the pump. This is
    that loop, as the integral controller it is in practice::

        d(ln G_0)/dt = (1 / tau_c) * (setpoint - measured)

    in nepers, with ``tau_c = 1 / (2 pi bandwidth)``. Integral action is not a
    detail: it is what makes the steady-state error exactly zero, so the gain
    comes back to where it was rather than near it.

    ``mode`` is ``"gain"`` -- hold the gain, which is what a line amplifier in a
    reconfigurable network does so that surviving channels do not move -- or
    ``"power"``, hold the total output power, which is what a booster does.
    ``setpoint`` is dB for gain and dBm for power; leave it ``None`` and the loop
    holds whatever the amplifier was settled at when the run started.

    ``max_pump_gain`` is the small-signal gain the pump can reach [dB]. A loop
    that runs into it stops correcting, which is a real failure of a real
    amplifier and is reported rather than smoothed over.
    """

    mode: str = "gain"
    bandwidth: float = 1e3
    setpoint: float | None = None
    max_pump_gain: float = 40.0
    min_pump_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("gain", "power"):
            raise ValueError(f"mode must be 'gain' or 'power', got {self.mode!r}")
        if self.bandwidth <= 0.0:
            raise ValueError(f"the loop bandwidth must be positive, got {self.bandwidth}")
        if self.max_pump_gain <= self.min_pump_gain:
            raise ValueError("the pump's ceiling must sit above its floor")

    @property
    def time_constant(self) -> float:
        """``1 / (2 pi bandwidth)`` [s]."""
        return 1.0 / (2.0 * math.pi * self.bandwidth)


@dataclass(frozen=True)
class ControlledTransient:
    """A transient with a pump loop acting on it, and what the pump had to do."""

    transient: GainTransient
    """The gain, power and timing, as :func:`gain_transient` reports them."""

    pump_gain_db: np.ndarray
    """The small-signal gain the loop called for at each time [dB]."""

    control: PumpControl
    setpoint: float
    """What the loop was holding: dB of gain, or dBm of output power."""

    @property
    def times(self) -> np.ndarray:
        return self.transient.times

    @property
    def gain_db(self) -> np.ndarray:
        return self.transient.gain_db

    @property
    def excursion(self) -> float:
        """The gain's largest signed departure [dB] -- what the loop is there to shrink."""
        return self.transient.excursion

    def _at_a_limit(self, pump: np.ndarray) -> np.ndarray:
        return (pump >= self.control.max_pump_gain - 1e-9) | (
            pump <= self.control.min_pump_gain + 1e-9
        )

    @property
    def pump_limited(self) -> bool:
        """Whether the run *ends* with the pump against a limit, still not correcting.

        This is the failure that matters: the loop has stopped and the setpoint is
        not being held. Separate from :attr:`pump_saturated`, because an integral
        loop slams its pump into the ceiling on any large step and then comes back
        -- reporting that as a limited amplifier would call a loop that worked a
        loop that failed.
        """
        return bool(self._at_a_limit(self.pump_gain_db)[-1])

    @property
    def pump_saturated(self) -> bool:
        """Whether the pump reached a limit at any point, transiently or not."""
        return bool(np.any(self._at_a_limit(self.pump_gain_db)))

    def __repr__(self) -> str:
        limited = (
            ", pump limited"
            if self.pump_limited
            else (", pump saturated on the way" if self.pump_saturated else "")
        )
        return (
            f"ControlledTransient({self.control.mode} at {self.setpoint:.2f}, "
            f"excursion {self.excursion:+.3f} dB, "
            f"pump {self.pump_gain_db.min():.2f} to {self.pump_gain_db.max():.2f} dB{limited})"
        )


def controlled_gain_transient(
    amplifier: EDFA,
    times: np.ndarray,
    input_power: np.ndarray,
    control: PumpControl,
    *,
    lifetime: float = METASTABLE_LIFETIME,
) -> ControlledTransient:
    """The same reservoir as :func:`gain_transient`, with a loop moving the pump.

    The pump enters the rate equation exactly where it always did, as the
    small-signal gain ``G_0``; the difference is that it is now a state rather
    than a constant, integrated alongside the inversion. Two timescales set what
    happens: the erbium's own ``tau / (1 + P_out / P_sat)``, and the loop's
    ``1 / (2 pi bandwidth)``. A loop faster than the erbium suppresses the
    excursion almost entirely; one slower than it watches it happen and cleans up
    afterwards.

    What the loop cannot do is exceed its pump. ``max_pump_gain`` is a real
    limit, and :attr:`ControlledTransient.pump_limited` says when it was reached.
    """
    grid = np.asarray(times, dtype=np.float64)
    drive = np.asarray(input_power, dtype=np.float64)
    if drive.shape != grid.shape:
        raise ValueError(f"input_power must match times, got {drive.shape} against {grid.shape}")
    if grid.size < 2:
        raise ValueError("a transient needs at least two times")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be strictly increasing")

    small_signal = db_to_linear(amplifier.gain)
    if not amplifier.saturate or small_signal <= 2.0:
        raise ValueError(
            "a control loop needs an amplifier that compresses; set saturate on the EDFA, "
            "since an unsaturated one holds its gain by construction and has nothing to correct"
        )

    if control.max_pump_gain < amplifier.gain:
        raise ValueError(
            f"the pump ceiling is {control.max_pump_gain:.1f} dB and the amplifier's own "
            f"small-signal gain is {amplifier.gain:.1f} dB, so the loop would start against "
            "its limit; raise max_pump_gain or lower the amplifier's gain"
        )
    saturation = amplifier.intrinsic_saturation_power(small_signal)
    own = amplifier.self_saturation_load() / saturation
    settled_gain = amplifier.effective_gain(float(drive[0]))
    if control.setpoint is None:
        setpoint = (
            10.0 * math.log10(settled_gain)
            if control.mode == "gain"
            else 10.0 * math.log10(settled_gain * float(drive[0]) * 1e3)
        )
    else:
        setpoint = float(control.setpoint)

    # One substep count for the run, from the faster of the two timescales.
    fastest = min(
        effective_time_constant(amplifier, float(power), lifetime=lifetime)
        for power in (float(drive.max()), float(drive.min()))
    )
    fastest = min(fastest, control.time_constant)
    coarsest = float(np.diff(grid).max())
    substeps = max(1, math.ceil(coarsest / (fastest / SUBSTEPS_PER_CONSTANT)))

    log_ceiling = math.log(db_to_linear(control.max_pump_gain))
    log_floor = math.log(db_to_linear(control.min_pump_gain))
    nepers = math.log(10.0) / 10.0  # one decibel, in the units the loop integrates

    def reservoir_slope(log_gain: float, log_pump: float, power: float) -> float:
        return (
            log_pump
            - log_gain
            - (math.exp(log_gain) - 1.0) * power / saturation
            - own * math.exp(log_gain)
        ) / lifetime

    def loop_slope(log_gain: float, power: float) -> float:
        if control.mode == "gain":
            measured = 10.0 * math.log10(math.exp(log_gain))
        else:
            output = math.exp(log_gain) * power
            measured = -math.inf if output <= 0.0 else 10.0 * math.log10(output * 1e3)
        error = setpoint - measured
        return nepers * error / control.time_constant

    gains = np.empty(grid.shape)
    pumps = np.empty(grid.shape)
    log_gain = math.log(settled_gain)
    log_pump = math.log(small_signal)
    gains[0], pumps[0] = settled_gain, amplifier.gain
    for index in range(1, grid.size):
        span = float(grid[index] - grid[index - 1])
        step = span / substeps
        power = float(drive[index - 1])
        for _ in range(substeps):
            # Two states, one Runge-Kutta: the inversion and the pump move
            # together, which is the whole point of a loop fast enough to matter.
            k1 = (reservoir_slope(log_gain, log_pump, power), loop_slope(log_gain, power))
            k2 = (
                reservoir_slope(
                    log_gain + 0.5 * step * k1[0], log_pump + 0.5 * step * k1[1], power
                ),
                loop_slope(log_gain + 0.5 * step * k1[0], power),
            )
            k3 = (
                reservoir_slope(
                    log_gain + 0.5 * step * k2[0], log_pump + 0.5 * step * k2[1], power
                ),
                loop_slope(log_gain + 0.5 * step * k2[0], power),
            )
            k4 = (
                reservoir_slope(log_gain + step * k3[0], log_pump + step * k3[1], power),
                loop_slope(log_gain + step * k3[0], power),
            )
            log_gain += step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            log_pump += step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            log_pump = min(max(log_pump, log_floor), log_ceiling)
        gains[index] = math.exp(log_gain)
        pumps[index] = 10.0 * math.log10(math.exp(log_pump))

    return ControlledTransient(
        transient=GainTransient(
            times=grid,
            gain=gains,
            input_power=drive,
            lifetime=lifetime,
            substeps=substeps,
        ),
        pump_gain_db=pumps,
        control=control,
        setpoint=setpoint,
    )
