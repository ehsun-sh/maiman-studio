"""Semiconductor laser rate equations: what a directly modulated laser does to its own light.

An external modulator leaves the laser alone and imposes data on a steady field.
Modulating the laser's *current* instead is cheaper and smaller -- it is how a
short-reach transceiver is built -- and it costs something an external modulator
does not: the carrier density has to move for the power to move, the refractive
index moves with it, and the optical frequency moves with that. The light is
chirped by the act of modulating it.

**The model.** Single-mode rate equations for the carrier density ``N`` and the
photon density ``S`` in the active region, with the phase driven by the carrier
density through the linewidth enhancement factor ``alpha``::

    dN/dt = I / (q V) - N / tau_n - G S
    dS/dt = Gamma G S - S / tau_p + Gamma beta N / tau_n
    dphi/dt = (alpha / 2) Gamma g0 (N - N_th)
    G = g0 (N - N_t) / (1 + epsilon S)

Everything a directly modulated laser is known for follows from these rather than
being declared beside them: the threshold current, the slope of the
light-current curve, the relaxation oscillation that rings after a step, and both
chirps -- the transient one that follows ``d(ln P)/dt`` and the adiabatic one
proportional to the power itself.

**What is not here.** One longitudinal mode, so no mode partition noise; no
spontaneous emission noise term, so the linewidth is what the chirp makes it and
not the Schawlow-Townes one; and no thermal drift of wavelength with bias.

Model references: G. P. Agrawal and N. K. Dutta, *Semiconductor Lasers*, ch. 6;
G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 3.5; T. L. Koch and
J. E. Bowers, "Nature of wavelength chirping in directly modulated semiconductor
lasers", Electron. Lett. 20(25), 1984.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import C_LIGHT, H_PLANCK

#: Electron charge [C].
ELECTRON_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class LaserParameters:
    """The active region, in SI units. The defaults are an ordinary 1310 nm DFB.

    These are the numbers a laser is described by in the literature rather than
    on a datasheet: a datasheet quotes a threshold current and a slope
    efficiency, which :meth:`threshold_current` and :meth:`slope_efficiency`
    compute from these, so fitting runs in that direction.
    """

    confinement: float = 0.3
    """``Gamma``: the fraction of the mode that overlaps the active region."""

    gain_slope: float = 2.1e-12
    """``g0 = v_g dg/dN`` [m^3/s]: group velocity times differential gain."""

    transparency_density: float = 1.0e24
    """``N_t`` [1/m^3]: the carrier density at which the medium stops absorbing."""

    carrier_lifetime: float = 1.0e-9
    """``tau_n`` [s]."""

    photon_lifetime: float = 2.5e-12
    """``tau_p`` [s]: the cavity's loss, and what sets the threshold gain."""

    gain_compression: float = 1.5e-23
    """``epsilon`` [m^3]: nonlinear gain compression, which damps the ringing."""

    spontaneous_coupling: float = 1.0e-4
    """``beta``: the spontaneous emission that lands in the lasing mode."""

    active_volume: float = 9.0e-17
    """``V`` [m^3]."""

    output_coupling: float = 0.2
    """Photons leaving the facet that reach the fibre, as a fraction."""

    wavelength: float = 1310e-9
    """Emission wavelength [m]."""

    linewidth_enhancement: float = 4.0
    """``alpha``: how far the index moves with the gain. 3 to 6 for InGaAsP."""

    def __post_init__(self) -> None:
        positive = (
            "confinement",
            "gain_slope",
            "transparency_density",
            "carrier_lifetime",
            "photon_lifetime",
            "active_volume",
            "wavelength",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.gain_compression < 0.0 or self.spontaneous_coupling < 0.0:
            raise ValueError("gain compression and spontaneous coupling cannot be negative")
        if not 0.0 < self.output_coupling <= 1.0:
            raise ValueError(f"output_coupling must be in (0, 1], got {self.output_coupling}")

    @property
    def photon_energy(self) -> float:
        """``h nu`` [J]."""
        return H_PLANCK * C_LIGHT / self.wavelength

    @property
    def threshold_density(self) -> float:
        """``N_th``: the carrier density whose gain exactly balances the cavity loss."""
        return self.transparency_density + 1.0 / (
            self.confinement * self.gain_slope * self.photon_lifetime
        )

    def threshold_current(self) -> float:
        """``I_th = q V N_th / tau_n`` [A].

        The current that holds the carrier density at threshold against
        recombination alone. Below it the stimulated term cannot sustain itself
        and what comes out is spontaneous emission; above it nearly every extra
        electron becomes a photon, which is the kink in a light-current curve.
        """
        return ELECTRON_CHARGE * self.active_volume * self.threshold_density / self.carrier_lifetime

    def slope_efficiency(self) -> float:
        """``dP/dI`` above threshold [W/A].

        ``eta h nu / q``: each electron past threshold makes one photon and
        ``output_coupling`` of them leave. It does not depend on any rate in the
        model, which makes it the cleanest check on the integration.
        """
        return self.output_coupling * self.photon_energy / ELECTRON_CHARGE

    def power_from_photons(self, photons: np.ndarray | float) -> np.ndarray:
        """Output power [W] for a photon density [1/m^3]."""
        return (
            np.asarray(photons, dtype=np.float64)
            * self.active_volume
            * self.photon_energy
            * self.output_coupling
            / (self.confinement * self.photon_lifetime)
        )

    def photons_from_power(self, power: float) -> float:
        """The inverse of :meth:`power_from_photons`."""
        return (
            power
            * self.confinement
            * self.photon_lifetime
            / (self.active_volume * self.photon_energy * self.output_coupling)
        )

    def steady_state(self, current: float) -> tuple[float, float]:
        """``(N, S)`` the laser settles at for a constant current.

        The carrier equation gives ``N`` for any ``S``, so substituting it into
        the photon equation leaves one scalar equation, bisected here. Spontaneous
        emission keeps ``S`` above zero below threshold, so there is exactly one
        root either side of it and no special case at the kink.
        """
        if current < 0.0:
            raise ValueError(f"current must not be negative, got {current}")
        pump = current / (ELECTRON_CHARGE * self.active_volume)

        def carriers_at(photons: float) -> float:
            gain = self.gain_slope / (1.0 + self.gain_compression * photons)
            return (pump + gain * self.transparency_density * photons) / (
                1.0 / self.carrier_lifetime + gain * photons
            )

        def balance(photons: float) -> float:
            carriers = carriers_at(photons)
            gain = (
                self.gain_slope
                * (carriers - self.transparency_density)
                / (1.0 + self.gain_compression * photons)
            )
            return (
                self.confinement * gain * photons
                - photons / self.photon_lifetime
                + self.confinement * self.spontaneous_coupling * carriers / self.carrier_lifetime
            )

        low, high = 0.0, 1.0
        while balance(high) > 0.0 and high < 1e30:
            high *= 10.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if balance(middle) > 0.0:
                low = middle
            else:
                high = middle
        photons = 0.5 * (low + high)
        return carriers_at(photons), photons

    def relaxation_frequency(self, current: float) -> float:
        """The frequency a step of current rings at [Hz].

        ``f_r = sqrt(g0 S0 / tau_p) / 2 pi`` at the photon density that current
        settles to: the exchange between the carrier and photon reservoirs. It is
        why a laser's modulation bandwidth is bought with bias current, and why a
        step response rings rather than simply rising.

        **The confinement factor is not in it**, and that is the algebra rather
        than an omission: linearising the pair of equations gives
        ``omega_r^2 = Gamma g0 S0 * G0``, and above threshold ``Gamma G0`` is
        ``1 / tau_p``, so the ``Gamma`` cancels. Leaving one in underestimates a
        1310 nm laser's ringing by ``sqrt(0.3)`` -- nearly a factor of two, which
        is what the measured step response caught.

        This is the undamped frequency. Gain compression damps the ringing, so a
        spectrum of the step response peaks slightly below this.
        """
        _, photons = self.steady_state(current)
        rate = self.gain_slope * photons / self.photon_lifetime
        return math.sqrt(max(rate, 0.0)) / (2.0 * math.pi)

    def adiabatic_chirp_factor(self) -> float:
        """``kappa`` [Hz/W]: the steady frequency offset per watt of output power.

        Gain compression means a brighter mode needs a higher carrier density to
        hold the same gain, and the index follows the carriers. In the steady
        state ``Gamma g0 (N - N_th) = Gamma g0 N_th epsilon S`` to first order in
        ``epsilon S``, so the offset is linear in power with this constant.
        """
        return (
            self.linewidth_enhancement
            * self.confinement
            * self.gain_slope
            * self.gain_compression
            * self.photons_from_power(1.0)
            * (self.threshold_density - self.transparency_density)
            / (4.0 * math.pi)
        )


@dataclass(frozen=True)
class LaserWaveform:
    """What the rate equations produced: power, phase, and the reservoirs behind them."""

    times: np.ndarray
    current: np.ndarray
    carriers: np.ndarray
    photons: np.ndarray
    phase: np.ndarray
    parameters: LaserParameters

    @property
    def power(self) -> np.ndarray:
        """Output power [W]."""
        return self.parameters.power_from_photons(self.photons)

    @property
    def field(self) -> np.ndarray:
        """Complex envelope [sqrt(W)]: ``sqrt(P) exp(i phi)``."""
        return np.sqrt(self.power) * np.exp(1j * self.phase)

    def instantaneous_frequency(self) -> np.ndarray:
        """Frequency offset from the unmodulated line [Hz], read off the phase."""
        step = float(self.times[1] - self.times[0])
        return np.gradient(self.phase, step) / (2.0 * math.pi)

    def extinction_ratio(self) -> float:
        """The high level over the low one [dB], from the settled ends of the power."""
        power = self.power
        high, low = float(np.max(power)), float(np.min(power))
        if low <= 0.0:
            return math.inf
        return 10.0 * math.log10(high / low)


def integrate_rate_equations(
    parameters: LaserParameters,
    current: np.ndarray,
    sample_rate: float,
    *,
    substeps: int = 16,
    initial: tuple[float, float] | None = None,
) -> LaserWaveform:
    """Integrate the rate equations across a drive current.

    ``current`` is one value per sample [A], held across its own sample -- the
    drive a digital-to-analogue converter actually produces. The integration
    takes ``substeps`` Runge-Kutta steps inside each sample, because the photon
    lifetime is picoseconds where a sample is tens of them: too few, and the
    ringing after a step is damped by the integrator rather than by the gain
    compression, which looks like a better laser than the parameters describe.

    The laser **starts settled** at the first current, so what is integrated is a
    modulated laser rather than one being switched on.
    """
    drive = np.asarray(current, dtype=np.float64)
    if drive.ndim != 1 or drive.size < 2:
        raise ValueError("the drive current must be a 1-D waveform of at least two samples")
    if np.any(drive < 0.0):
        raise ValueError("a laser cannot be driven with a negative current")
    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if substeps < 1:
        raise ValueError(f"substeps must be at least one, got {substeps}")

    carriers, photons = parameters.steady_state(float(drive[0])) if initial is None else initial
    volume_charge = ELECTRON_CHARGE * parameters.active_volume
    step = 1.0 / (sample_rate * substeps)
    threshold = parameters.threshold_density

    def derivatives(n: float, s: float, pump: float) -> tuple[float, float]:
        gain = (
            parameters.gain_slope
            * (n - parameters.transparency_density)
            / (1.0 + parameters.gain_compression * s)
        )
        carriers_rate = pump / volume_charge - n / parameters.carrier_lifetime - gain * s
        photons_rate = (
            parameters.confinement * gain * s
            - s / parameters.photon_lifetime
            + parameters.confinement
            * parameters.spontaneous_coupling
            * n
            / parameters.carrier_lifetime
        )
        return carriers_rate, photons_rate

    count = drive.size
    carrier_track = np.empty(count)
    photon_track = np.empty(count)
    phase_track = np.empty(count)
    phase = 0.0
    for index in range(count):
        carrier_track[index] = carriers
        photon_track[index] = photons
        phase_track[index] = phase
        pump = float(drive[index])
        for _ in range(substeps):
            k1 = derivatives(carriers, photons, pump)
            k2 = derivatives(carriers + 0.5 * step * k1[0], photons + 0.5 * step * k1[1], pump)
            k3 = derivatives(carriers + 0.5 * step * k2[0], photons + 0.5 * step * k2[1], pump)
            k4 = derivatives(carriers + step * k3[0], photons + step * k3[1], pump)
            carriers += step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            photons = max(photons + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0, 0.0)
            # The phase follows the carrier density's departure from threshold,
            # which is what makes the chirp a consequence of modulating rather
            # than a term added beside it.
            phase += (
                step
                * 0.5
                * parameters.linewidth_enhancement
                * parameters.confinement
                * parameters.gain_slope
                * (carriers - threshold)
            )

    return LaserWaveform(
        times=np.arange(count) / sample_rate,
        current=drive,
        carriers=carrier_track,
        photons=photon_track,
        phase=phase_track,
        parameters=parameters,
    )
