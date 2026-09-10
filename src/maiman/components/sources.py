"""Optical sources."""

from __future__ import annotations

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, OpticalSignal, Signal
from ..units import C_LIGHT


class CWLaser(Component):
    """Continuous-wave laser, with the two noises a real one has.

    **Phase.** With a non-zero linewidth the phase performs a Wiener random walk,
    which is the standard Lorentzian-lineshape model: the phase increment per
    sample is drawn from ``N(0, 2*pi*linewidth*dt)``. Amplitude is untouched, so
    linewidth changes the spectrum without changing the average power — an
    invariant worth testing.

    **Intensity.** ``rin`` is the relative intensity noise, the single-sided
    spectral density of the fractional power fluctuation
    (``S_dP(f) / P_mean**2``, in 1/Hz, quoted in dB/Hz). A sampled window covers
    a one-sided bandwidth of ``fs/2``, so the fractional power fluctuation per
    sample is drawn from ``N(0, rin * fs / 2)`` and the field is the square root
    of what is left. See Agrawal, *Fiber-Optic Communication Systems*, 4th ed.,
    section 4.6.2.

    The two are drawn from **separate streams**, so changing one does not move
    the other's samples: an experiment that varies the linewidth and reads an
    intensity-noise-limited number back would otherwise be measuring both.

    What RIN buys that no other noise here does is a **floor that power cannot
    lift**. Shot and thermal noise fall behind the signal as the received power
    rises, which is why every sensitivity curve in this project keeps improving;
    intensity noise scales *with* the signal, so the ratio is fixed at
    ``1 / (rin * B)`` and more launch power buys exactly nothing. That is the
    same shape of ceiling that phase noise puts on a coherent link, and it is
    the reason a datasheet quotes RIN at all.

    ``rin = 0`` is a **sentinel meaning an ideal laser**, not a physical value:
    0 dB/Hz would be a fractional intensity variance of one per hertz of
    bandwidth, which is not a laser. Every real device is between about
    -110 dB/Hz (a cheap Fabry-Perot) and -165 dB/Hz (a good DFB), so the top of
    the scale is free to mean "off" and the default leaves it there — turning it
    on would move every existing result in this repository.
    """

    display_name = "CW Laser"
    category = "Optical Sources"

    power = Param(0.0, unit="dBm", doc="Average output power")
    wavelength = Param(1550.0, unit="nm", min=1200.0, max=1700.0, doc="Vacuum wavelength")
    linewidth = Param(
        0.0, unit="kHz", min=0.0, doc="Lorentzian FWHM linewidth; 0 disables phase noise"
    )
    rin = Param(
        0.0,
        unit="dB/Hz",
        max=0.0,
        doc="Relative intensity noise; 0 is the sentinel for an ideal laser",
    )

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        power_w = self.si("power")
        f0 = C_LIGHT / self.si("wavelength")
        amplitude = np.sqrt(power_w)

        n = ctx.num_samples
        if self.si("linewidth") > 0.0:
            sigma = np.sqrt(2.0 * np.pi * self.si("linewidth") * ctx.time_step)
            rng = ctx.rng("CWLaser", self.label, "phase_noise")
            increments = rng.normal(0.0, sigma, size=n)
            phase = np.cumsum(increments)
            phase -= phase[0]  # start at zero phase so runs are comparable
            Ex = amplitude * np.exp(1j * phase)
        else:
            Ex = np.full(n, amplitude, dtype=np.complex128)

        if self.rin < 0.0:
            # A sampled window carries a one-sided bandwidth of fs/2, so that is
            # the bandwidth the spectral density integrates over to give the
            # per-sample variance of the fractional power fluctuation.
            variance = self.si("rin") * ctx.sample_rate / 2.0
            rng = ctx.rng("CWLaser", self.label, "intensity_noise")
            fluctuation = rng.normal(0.0, np.sqrt(variance), size=n)
            # Power cannot be negative. At any RIN a laser has this clips
            # nothing — a good DFB at 512 GS/s sits 200 sigma away from it — and
            # where it does engage the model has already left the regime it was
            # fitted in, so the floor is the honest answer rather than a NaN.
            Ex = Ex * np.sqrt(np.maximum(1.0 + fluctuation, 0.0))

        band = Band(
            Ex=Ex.astype(ctx.complex_dtype),
            Ey=np.zeros(n, dtype=ctx.complex_dtype),
            f0=f0,
            fs=ctx.sample_rate,
        )
        return {"out": OpticalSignal(bands=(band,))}


class GaussianPulse(Component):
    """A single chirped Gaussian pulse, centred in the time window.

    Follows the standard form (Agrawal, *Nonlinear Fiber Optics*, eq. 3.2.1)::

        A(0, T) = sqrt(P0) * exp(-(1 + i*C) / 2 * (T / T0)**2)

    so the intensity envelope is ``P0 * exp(-(T/T0)**2)`` and ``width`` is T0, the
    1/e half-width of the *intensity* (``T_FWHM = 1.665 * T0``).

    This exists because dispersion has an exact analytical solution for a Gaussian
    input, which makes it the reference input for validating the fiber model.
    """

    display_name = "Gaussian Pulse"
    category = "Optical Sources"

    peak_power = Param(0.0, unit="dBm", doc="Peak power P0 (not average power)")
    width = Param(10.0, unit="ps", min=0.0, doc="T0, the 1/e intensity half-width")
    chirp = Param(0.0, doc="Linear chirp parameter C; sign matters against beta2")
    wavelength = Param(1550.0, unit="nm", min=1200.0, max=1700.0, doc="Vacuum wavelength")

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        t0 = self.si("width")
        if t0 <= 0.0:
            raise ValueError(f"{self.label}: width must be positive, got {self.width}")

        tau = (ctx.time_axis() - ctx.time_window / 2.0) / t0
        amplitude = np.sqrt(self.si("peak_power"))
        Ex = amplitude * np.exp(-(1.0 + 1j * self.chirp) * tau**2 / 2.0)

        band = Band(
            Ex=Ex.astype(ctx.complex_dtype),
            Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
            f0=C_LIGHT / self.si("wavelength"),
            fs=ctx.sample_rate,
        )
        return {"out": OpticalSignal(bands=(band,))}


class SechPulse(Component):
    """A hyperbolic-secant pulse, centred in the time window.

    ``A(0, T) = sqrt(P0) * sech(T / T0)``

    This shape exists here because it is the soliton shape. Launched into
    anomalous fiber at the peak power that makes ``gamma*P0*T0**2/|beta2| = 1``,
    it propagates without changing at all — the sharpest available check that
    dispersion and the Kerr effect are both right *and* have the right signs
    relative to each other. :func:`maiman.kernels.soliton_peak_power` computes
    that power.
    """

    display_name = "Sech Pulse"
    category = "Optical Sources"

    peak_power = Param(0.0, unit="dBm", doc="Peak power P0 (not average power)")
    width = Param(10.0, unit="ps", min=0.0, doc="T0, the soliton width parameter")
    wavelength = Param(1550.0, unit="nm", min=1200.0, max=1700.0, doc="Vacuum wavelength")

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        t0 = self.si("width")
        if t0 <= 0.0:
            raise ValueError(f"{self.label}: width must be positive, got {self.width}")

        tau = (ctx.time_axis() - ctx.time_window / 2.0) / t0
        Ex = np.sqrt(self.si("peak_power")) / np.cosh(tau)

        band = Band(
            Ex=Ex.astype(ctx.complex_dtype),
            Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
            f0=C_LIGHT / self.si("wavelength"),
            fs=ctx.sample_rate,
        )
        return {"out": OpticalSignal(bands=(band,))}
