"""Optical sources."""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..laser import (
    LaserParameters,
    MultimodeEnsemble,
    integrate_multimode,
    multimode_steady_state,
)
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


class FabryPerotLaser(Component):
    """Several longitudinal modes on one reservoir, and the partition between them.

    A Fabry-Perot laser does not emit one line. Its cavity has a comb of them,
    the gain picks a handful out around its peak, and they all feed from the same
    carrier reservoir -- so a photon one mode takes is a photon the others cannot
    have. The modes therefore fluctuate hard and *anti-correlate*: each line's
    own intensity noise is far above the sum's, and the sum is quiet. That is
    mode partition, and here it comes out of the multimode rate equations rather
    than from a partition coefficient: one carrier equation, ``modes`` photon
    equations with gain ``G (1 - (dlambda / gain_bandwidth)^2)``, and every
    Langevin force drawn against the one reservoir.

    **Why it matters, and where.** At the laser the partition cancels in the sum,
    which is why a power meter at the facet sees a quiet laser. Down a fibre the
    modes separate -- ``D L dlambda``, hundreds of picoseconds over a reach --
    and a detector adding them up no longer adds them at the same instant. What
    cancelled stops cancelling, and the residue is a noise floor that no amount
    of received power lifts. Set ``carry_walkoff`` on the spans between here and
    the detector and the link shows it; leave it off and the modes arrive
    together, as they left.

    ``power`` is what the datasheet quotes, and it is applied as a scale onto the
    deterministic steady state -- not onto the realized window, which would
    divide out the very fluctuation this block exists to produce. The mean of a
    given run therefore sits near the number asked for rather than exactly on it.

    ``modes`` is best left odd, which puts one line on the gain peak and the rest
    in pairs around it.
    """

    display_name = "Fabry-Perot Laser"
    category = "Optical Sources"

    power = Param(0.0, unit="dBm", doc="Average output power, summed over every mode")
    wavelength = Param(
        1310.0, unit="nm", min=1200.0, max=1700.0, doc="Where the gain peaks, in vacuum"
    )
    modes = Param(7.0, unit="", min=1.0, max=41.0, doc="Longitudinal modes the gain supports")
    mode_spacing = Param(
        1.1, unit="nm", min=0.001, max=50.0, doc="The cavity's free spectral range"
    )
    gain_bandwidth = Param(
        40.0, unit="nm", min=0.01, max=200.0, doc="Where the gain parabola falls to zero"
    )
    bias_current = Param(35.0, unit="mA", min=0.0, max=1000.0, doc="Drive current")
    spontaneous_coupling = Param(
        1e-4,
        unit="",
        min=0.0,
        max=1e-2,
        doc="beta: spontaneous emission into each mode, which is what feeds the side modes",
    )
    partition_noise = BoolParam(
        True, doc="Langevin forces on every mode, so they trade power; off is the steady state"
    )
    substeps = Param(
        16.0,
        unit="",
        min=1.0,
        max=1024.0,
        doc="Integration steps per sample",
        applies_when="partition_noise",
    )
    settle = Param(
        256.0,
        unit="",
        min=0.0,
        max=1e5,
        doc="Samples integrated and thrown away before the window opens",
        applies_when="partition_noise",
    )

    outputs = {"out": PortType.OPTICAL}

    def parameters(self) -> LaserParameters:
        """The rate equations' own parameters, at this block's wavelength."""
        return LaserParameters(
            wavelength=self.si("wavelength"),
            spontaneous_coupling=self.si("spontaneous_coupling"),
        )

    def mode_offsets(self) -> np.ndarray:
        """Each mode's wavelength from the gain peak [m]; an odd count sits one on it."""
        count = int(self.modes)
        return (np.arange(count) - (count - 1) / 2.0) * self.si("mode_spacing")

    def mode_wavelengths(self) -> np.ndarray:
        """Where each mode sits [m]."""
        return self.si("wavelength") + self.mode_offsets()

    def steady_state(self) -> np.ndarray:
        """The power each mode settles at with no noise [W], by the deterministic equations."""
        parameters = self.parameters()
        _, photons = multimode_steady_state(
            parameters,
            self.si("bias_current"),
            modes=int(self.modes),
            spacing=self.si("mode_spacing"),
            gain_bandwidth=self.si("gain_bandwidth"),
        )
        return np.asarray(parameters.power_from_photons(photons), dtype=np.float64)

    def scale(self) -> float:
        """What the modelled power is multiplied by to reach the quoted one."""
        settled = float(self.steady_state().sum())
        if settled <= 0.0:
            raise ValueError(
                f"{self.label}: this laser does not lase at {self.bias_current} mA, so there "
                "is no power to scale to the one asked for"
            )
        return self.si("power") / settled

    def solve(self, ctx: SimulationContext) -> MultimodeEnsemble:
        """Integrate the multimode rate equations across the window, unscaled."""
        return integrate_multimode(
            self.parameters(),
            self.si("bias_current"),
            ctx.sample_rate,
            ctx.num_samples,
            modes=int(self.modes),
            spacing=self.si("mode_spacing"),
            gain_bandwidth=self.si("gain_bandwidth"),
            realizations=1,
            rng=ctx.rng(type(self).__name__, self.label, "partition"),
            substeps=int(self.substeps),
            settle=int(self.settle),
        )

    def mode_power(self, ctx: SimulationContext) -> np.ndarray:
        """``(modes, samples)`` of emitted power [W], scaled to the quoted output."""
        if self.partition_noise:
            modelled = self.solve(ctx).mode_power[0]
        else:
            modelled = np.repeat(self.steady_state()[:, None], ctx.num_samples, axis=1)
        return np.asarray(modelled * self.scale(), dtype=np.float64)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        power = self.mode_power(ctx)
        zeros = np.zeros(ctx.num_samples, dtype=ctx.complex_dtype)
        bands = tuple(
            Band(
                Ex=np.sqrt(np.maximum(power[index], 0.0)).astype(ctx.complex_dtype),
                Ey=zeros,
                f0=C_LIGHT / wavelength,
                fs=ctx.sample_rate,
            )
            # Ascending in frequency, which is descending in wavelength: the
            # frame a span writes its walk-off against is then the shortest mode.
            for index, wavelength in reversed(list(enumerate(self.mode_wavelengths())))
        )
        return {"out": OpticalSignal(bands=bands)}


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
    chirp = Param(
        0.0,
        doc="Linear chirp parameter C; positive is an up-chirp, and its sign matters against beta2",
    )
    wavelength = Param(1550.0, unit="nm", min=1200.0, max=1700.0, doc="Vacuum wavelength")

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        t0 = self.si("width")
        if t0 <= 0.0:
            raise ValueError(f"{self.label}: width must be positive, got {self.width}")

        tau = (ctx.time_axis() - ctx.time_window / 2.0) / t0
        amplitude = np.sqrt(self.si("peak_power"))
        Ex = amplitude * np.exp(-(1.0 - 1j * self.chirp) * tau**2 / 2.0)

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
