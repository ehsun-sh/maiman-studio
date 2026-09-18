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

**Noise, when asked for.** Spontaneous emission is random as well as a rate, and
with ``noise`` set the equations carry the Langevin forces that say so (Agrawal,
*Fiber-Optic Communication Systems*, section 3.5.4), in photon and carrier
*numbers* ``P = S V / Gamma`` and ``N V``::

    <F_P F_P> = 2 R_sp P      <F_N F_N> = 2 (R_sp P + N V / tau_n)
    <F_P F_N> = -2 R_sp P     <F_phi F_phi> = R_sp / (2 P)

with ``R_sp = beta N V / tau_n`` the spontaneous photons per second landing in
the mode. The phase noise alone would give the Schawlow-Townes linewidth
``R_sp / (4 pi P)``; the carrier noise, through ``alpha``, gives Henry's
``(1 + alpha^2)`` on top. Neither factor is written into the integration -- they
are what the tests measure coming out of it.

**Several modes.** :func:`integrate_multimode` gives a Fabry-Perot laser its
longitudinal modes, one carrier reservoir feeding all of them through a parabolic
gain curve. Their total is quiet and each one alone is not: the modes trade
power among themselves, which is mode partition, and fibre dispersion delays
each by its own amount so that what cancelled at the laser does not at the
receiver. :func:`dispersed_power` is that step.

**What is not here.** Thermal drift of wavelength with bias; and noise in the
pump current itself, which a quiet current source makes negligible.

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

    def spontaneous_rate(self, carriers: float) -> float:
        """``R_sp = beta N V / tau_n`` [1/s]: spontaneous photons per second into the mode."""
        return self.spontaneous_coupling * carriers * self.active_volume / self.carrier_lifetime

    def photon_number(self, photons: float) -> float:
        """Photons in the cavity, ``S V / Gamma``, for a photon density ``S`` [1/m^3]."""
        return photons * self.active_volume / self.confinement

    def schawlow_townes_linewidth(self, current: float) -> float:
        """``R_sp / (4 pi P)`` [Hz]: the linewidth spontaneous emission's phase noise alone gives.

        One spontaneous photon in ``P`` jolts the field's phase by ``1/sqrt(P)``
        in a random direction; the phase diffuses, and the line it draws is a
        Lorentzian this wide. It narrows as the laser gets brighter, which is the
        whole of Schawlow and Townes' result.
        """
        carriers, photons = self.steady_state(current)
        return self.spontaneous_rate(carriers) / (4.0 * math.pi * self.photon_number(photons))

    def linewidth(self, current: float) -> float:
        """Henry's ``(1 + alpha^2)`` times :meth:`schawlow_townes_linewidth` [Hz].

        Each spontaneous photon also moves the intensity, the gain restores it by
        moving the carrier density, and the index follows the carriers: a second
        phase kick ``alpha`` times the first. In a semiconductor laser, with
        ``alpha`` near 4, that is seventeen times the linewidth Schawlow and
        Townes would give it (C. H. Henry, IEEE J. Quantum Electron. 18(2), 1982).
        """
        return (1.0 + self.linewidth_enhancement**2) * self.schawlow_townes_linewidth(current)

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
    noise: bool = False,
    rng: np.random.Generator | None = None,
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

    With ``noise`` the Langevin forces in the module docstring are added after
    each Runge-Kutta step (Euler-Maruyama), drawn from ``rng``; without it the
    integration is deterministic and exactly what it was.
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
    if noise and rng is None:
        raise ValueError("noise needs a random generator: pass rng")

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
    kicks = (
        rng.standard_normal((count, substeps, 3))
        if noise and rng is not None
        else np.zeros((1, 1, 3))
    )
    for index in range(count):
        carrier_track[index] = carriers
        photon_track[index] = photons
        phase_track[index] = phase
        pump = float(drive[index])
        for sub in range(substeps):
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
            if noise:
                carriers, photons, phase = _langevin_kick(
                    parameters, carriers, photons, phase, step, kicks[index, sub]
                )

    return LaserWaveform(
        times=np.arange(count) / sample_rate,
        current=drive,
        carriers=carrier_track,
        photons=photon_track,
        phase=phase_track,
        parameters=parameters,
    )


def _langevin_kick(
    parameters: LaserParameters,
    carriers: float,
    photons: float,
    phase: float,
    step: float,
    kick: np.ndarray,
) -> tuple[float, float, float]:
    """One Euler-Maruyama step of the Langevin forces, in photon and carrier numbers.

    ``dN = -dP + sqrt(2 N V dt / tau_n) xi`` is the carrier force written so its
    correlation with the photon force, ``-2 R_sp P``, comes out by construction
    rather than through a factorisation.
    """
    mode_volume = parameters.active_volume / parameters.confinement
    number = max(photons * mode_volume, 1e-3)
    rate = parameters.spontaneous_rate(carriers)
    d_photons = math.sqrt(2.0 * rate * number * step) * float(kick[0])
    d_carriers = -d_photons + math.sqrt(
        2.0 * max(carriers, 0.0) * parameters.active_volume * step / parameters.carrier_lifetime
    ) * float(kick[1])
    d_phase = math.sqrt(rate * step / (2.0 * number)) * float(kick[2])
    return (
        carriers + d_carriers / parameters.active_volume,
        max(photons + d_photons / mode_volume, 0.0),
        phase + d_phase,
    )


@dataclass(frozen=True)
class LaserEnsemble:
    """Many independent runs of one noisy laser, held at one current."""

    times: np.ndarray
    carriers: np.ndarray
    """``(realizations, samples)`` [1/m^3]."""
    photons: np.ndarray
    """``(realizations, samples)`` [1/m^3]."""
    phase: np.ndarray
    """``(realizations, samples)`` [rad]."""
    parameters: LaserParameters

    @property
    def power(self) -> np.ndarray:
        return self.parameters.power_from_photons(self.photons)

    def phase_variance(self, lag: int) -> float:
        """The variance of ``phi(t + lag) - phi(t)`` over every run and every start [rad^2].

        The mean increment is taken out first: it is the steady frequency offset
        gain compression holds the carriers at, a line in a different place and
        not a wider one.
        """
        step = self.phase[:, lag:] - self.phase[:, :-lag]
        return float(np.var(step))

    def measured_linewidth(self, lag: int) -> float:
        """The Lorentzian width phase diffusion over ``lag`` samples implies [Hz].

        A phase that diffuses as ``2 pi dnu t`` draws a Lorentzian of full width
        ``dnu``. It holds for lags well past the relaxation oscillation, where the
        carriers have had time to answer each kick.
        """
        interval = float(self.times[lag] - self.times[0])
        return self.phase_variance(lag) / (2.0 * math.pi * interval)

    def relative_intensity_noise(self) -> tuple[np.ndarray, np.ndarray]:
        """``(frequencies [Hz], RIN [1/Hz])``, one-sided, averaged over the runs."""
        power = self.power
        mean = float(np.mean(power))
        fluctuation = power - power.mean(axis=1, keepdims=True)
        rate = 1.0 / float(self.times[1] - self.times[0])
        spectrum = np.abs(np.fft.rfft(fluctuation, axis=1)) ** 2
        density = 2.0 * spectrum.mean(axis=0) / (rate * power.shape[1]) / mean**2
        return np.fft.rfftfreq(power.shape[1], 1.0 / rate), density


def laser_ensemble(
    parameters: LaserParameters,
    current: float,
    sample_rate: float,
    samples: int,
    *,
    realizations: int,
    rng: np.random.Generator,
    substeps: int = 16,
) -> LaserEnsemble:
    """Run ``realizations`` copies of the noisy laser side by side at a constant current.

    The same equations and forces as :func:`integrate_rate_equations` with
    ``noise``, integrated as arrays so that a linewidth or a noise spectrum can be
    averaged over hundreds of runs in the time one long run would take.
    """
    if realizations < 1 or samples < 2:
        raise ValueError("need at least one realization and two samples")
    carriers0, photons0 = parameters.steady_state(current)
    carriers = np.full(realizations, carriers0)
    photons = np.full(realizations, photons0)
    phase = np.zeros(realizations)
    step = 1.0 / (sample_rate * substeps)
    pump = current / (ELECTRON_CHARGE * parameters.active_volume)
    threshold = parameters.threshold_density
    mode_volume = parameters.active_volume / parameters.confinement

    def rates(n: np.ndarray, s_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gain = (
            parameters.gain_slope
            * (n - parameters.transparency_density)
            / (1.0 + parameters.gain_compression * s_)
        )
        return (
            pump - n / parameters.carrier_lifetime - gain * s_,
            parameters.confinement * gain * s_
            - s_ / parameters.photon_lifetime
            + parameters.confinement
            * parameters.spontaneous_coupling
            * n
            / parameters.carrier_lifetime,
        )

    tracks = [np.empty((realizations, samples)) for _ in range(3)]
    for index in range(samples):
        tracks[0][:, index], tracks[1][:, index], tracks[2][:, index] = carriers, photons, phase
        for _ in range(substeps):
            k1 = rates(carriers, photons)
            k2 = rates(carriers + 0.5 * step * k1[0], photons + 0.5 * step * k1[1])
            k3 = rates(carriers + 0.5 * step * k2[0], photons + 0.5 * step * k2[1])
            k4 = rates(carriers + step * k3[0], photons + step * k3[1])
            carriers = carriers + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            photons = photons + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            phase = (
                phase
                + step
                * 0.5
                * parameters.linewidth_enhancement
                * parameters.confinement
                * parameters.gain_slope
                * (carriers - threshold)
            )

            kick = rng.standard_normal((3, realizations))
            number = np.maximum(photons * mode_volume, 1e-3)
            rate = (
                parameters.spontaneous_coupling
                * carriers
                * parameters.active_volume
                / parameters.carrier_lifetime
            )
            d_photons = np.sqrt(2.0 * rate * number * step) * kick[0]
            d_carriers = (
                -d_photons
                + np.sqrt(
                    2.0
                    * np.maximum(carriers, 0.0)
                    * parameters.active_volume
                    * step
                    / parameters.carrier_lifetime
                )
                * kick[1]
            )
            carriers = carriers + d_carriers / parameters.active_volume
            photons = np.maximum(photons + d_photons / mode_volume, 0.0)
            phase = phase + np.sqrt(rate * step / (2.0 * number)) * kick[2]

    return LaserEnsemble(
        times=np.arange(samples) / sample_rate,
        carriers=tracks[0],
        photons=tracks[1],
        phase=tracks[2],
        parameters=parameters,
    )


# --------------------------------------------------------------------------
# Several longitudinal modes, and the partition noise between them
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MultimodeEnsemble:
    """A Fabry-Perot laser's modes over many runs: one reservoir, several lines."""

    times: np.ndarray
    carriers: np.ndarray
    """``(realizations, samples)`` [1/m^3]."""
    photons: np.ndarray
    """``(realizations, modes, samples)`` [1/m^3]."""
    offsets: np.ndarray
    """Each mode's wavelength from the gain peak [m]."""
    parameters: LaserParameters

    @property
    def mode_power(self) -> np.ndarray:
        """``(realizations, modes, samples)`` [W]."""
        return self.parameters.power_from_photons(self.photons)

    @property
    def total_power(self) -> np.ndarray:
        return np.asarray(self.mode_power.sum(axis=1))


def _mode_gain_profile(
    modes: int, spacing: float, gain_bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    if modes < 1:
        raise ValueError(f"a laser has at least one mode, got {modes}")
    if spacing <= 0.0 or gain_bandwidth <= 0.0:
        raise ValueError("mode spacing and gain bandwidth must both be positive")
    offsets = (np.arange(modes) - (modes - 1) / 2.0) * spacing
    return offsets, 1.0 - (offsets / gain_bandwidth) ** 2


def multimode_steady_state(
    parameters: LaserParameters,
    current: float,
    *,
    modes: int,
    spacing: float,
    gain_bandwidth: float,
) -> tuple[float, np.ndarray]:
    """``(N, S_m)`` the deterministic multimode equations settle at.

    Gain compression is by the total, so the unknown is taken as the compressed
    gain variable ``x = (N - N_t) / (1 + epsilon S)``, which every mode shares.
    For a given ``x`` each mode obeys ``0 = Gamma g0 x p_m S_m - S_m / tau_p +
    Gamma beta N / tau_n``, so ``S_m = c_m N`` with ``c_m`` known; the definition
    of ``x`` then gives ``N = (N_t + x) / (1 - epsilon x sum c_m)``, and the
    carrier equation leaves one scalar equation in ``x``, bisected below the
    point where the strongest mode's gain would meet the loss. The mode at the
    gain peak takes most of the power and each neighbour less, by how far down
    the parabola it sits.
    """
    _, profile = _mode_gain_profile(modes, spacing, gain_bandwidth)
    pump = current / (ELECTRON_CHARGE * parameters.active_volume)
    spontaneous = (
        parameters.confinement * parameters.spontaneous_coupling / parameters.carrier_lifetime
    )
    g0, eps = parameters.gain_slope, parameters.gain_compression

    def state(x: float) -> tuple[float, np.ndarray] | None:
        per_carrier = spontaneous / (
            1.0 / parameters.photon_lifetime - parameters.confinement * g0 * x * profile
        )
        denominator = 1.0 - eps * x * float(per_carrier.sum())
        if denominator <= 0.0:
            return None
        n = (parameters.transparency_density + x) / denominator
        return n, per_carrier * n

    def balance(x: float) -> float:
        solved = state(x)
        if solved is None:
            return -math.inf
        n, each = solved
        return pump - n / parameters.carrier_lifetime - g0 * x * float(np.sum(profile * each))

    ceiling = 1.0 / (parameters.confinement * g0 * parameters.photon_lifetime * profile.max())
    low, high = 0.0, ceiling
    for _ in range(300):
        middle = 0.5 * (low + high)
        if balance(middle) > 0.0:
            low = middle
        else:
            high = middle
    solved = state(0.5 * (low + high))
    assert solved is not None
    return solved


def integrate_multimode(
    parameters: LaserParameters,
    current: float,
    sample_rate: float,
    samples: int,
    *,
    modes: int,
    spacing: float,
    gain_bandwidth: float,
    realizations: int,
    rng: np.random.Generator,
    substeps: int = 16,
    settle: int = 0,
) -> MultimodeEnsemble:
    """A Fabry-Perot laser's longitudinal modes at a constant current, with their noise.

    One carrier reservoir, ``modes`` photon populations ``spacing`` apart in
    wavelength, each with gain ``G (1 - (dlambda / gain_bandwidth)^2)`` and the
    same spontaneous coupling. Each mode has its own Langevin force and all of
    them are drawn against the one carrier force, so photons one mode gains the
    reservoir loses, and the others then lose too: mode partition, from the
    equations rather than from a partition coefficient.

    ``settle`` samples are integrated and discarded first, starting from the
    deterministic steady state.
    """
    offsets, profile = _mode_gain_profile(modes, spacing, gain_bandwidth)
    if realizations < 1 or samples < 2:
        raise ValueError("need at least one realization and two samples")
    n0, s0 = multimode_steady_state(
        parameters, current, modes=modes, spacing=spacing, gain_bandwidth=gain_bandwidth
    )
    carriers = np.full(realizations, n0)
    photons = np.tile(s0, (realizations, 1))
    step = 1.0 / (sample_rate * substeps)
    pump = current / (ELECTRON_CHARGE * parameters.active_volume)
    mode_volume = parameters.active_volume / parameters.confinement
    spontaneous = (
        parameters.confinement * parameters.spontaneous_coupling / parameters.carrier_lifetime
    )

    def rates(n: np.ndarray, s_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gain = (
            parameters.gain_slope
            * (n - parameters.transparency_density)[:, None]
            * profile[None, :]
            / (1.0 + parameters.gain_compression * s_.sum(axis=1, keepdims=True))
        )
        return (
            pump - n / parameters.carrier_lifetime - (gain * s_).sum(axis=1),
            parameters.confinement * gain * s_
            - s_ / parameters.photon_lifetime
            + spontaneous * n[:, None],
        )

    carrier_track = np.empty((realizations, samples))
    photon_track = np.empty((realizations, modes, samples))
    for index in range(-settle, samples):
        if index >= 0:
            carrier_track[:, index] = carriers
            photon_track[:, :, index] = photons
        for _ in range(substeps):
            k1 = rates(carriers, photons)
            k2 = rates(carriers + 0.5 * step * k1[0], photons + 0.5 * step * k1[1])
            k3 = rates(carriers + 0.5 * step * k2[0], photons + 0.5 * step * k2[1])
            k4 = rates(carriers + step * k3[0], photons + step * k3[1])
            carriers = carriers + step * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            photons = photons + step * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0

            rate = (
                parameters.spontaneous_coupling
                * carriers
                * parameters.active_volume
                / parameters.carrier_lifetime
            )
            number = np.maximum(photons * mode_volume, 1e-3)
            d_photons = np.sqrt(2.0 * rate[:, None] * number * step) * rng.standard_normal(
                number.shape
            )
            d_carriers = -d_photons.sum(axis=1) + np.sqrt(
                2.0
                * np.maximum(carriers, 0.0)
                * parameters.active_volume
                * step
                / parameters.carrier_lifetime
            ) * rng.standard_normal(realizations)
            carriers = carriers + d_carriers / parameters.active_volume
            photons = np.maximum(photons + d_photons / mode_volume, 0.0)

    return MultimodeEnsemble(
        times=np.arange(samples) / sample_rate,
        carriers=carrier_track,
        photons=photon_track,
        offsets=offsets,
        parameters=parameters,
    )


def dispersed_power(
    mode_power: np.ndarray, offsets: np.ndarray, sample_rate: float, *, dispersion: float
) -> np.ndarray:
    """What a detector sees after fibre dispersion: each mode delayed by its own amount, summed.

    ``dispersion`` is the accumulated ``D L`` [s/m]; the mode ``dlambda`` from the
    peak arrives ``D L dlambda`` late. The delay is applied to each mode's
    *power* by a Fourier phase ramp -- right while the mode's own bandwidth is
    far narrower than the spacing, which is what makes them modes -- and the
    delayed powers add, because distinct wavelengths do not interfere on a
    square-law detector averaged over their beat. At zero dispersion this is the
    total power, whose partition noise cancelled; with enough, the modes'
    fluctuations are decorrelated and nothing cancels.

    ``mode_power`` is ``(..., modes, samples)``, circular as every waveform here.
    """
    power = np.asarray(mode_power, dtype=np.float64)
    delays = dispersion * np.asarray(offsets, dtype=np.float64)
    frequencies = np.fft.fftfreq(power.shape[-1], 1.0 / sample_rate)
    ramp = np.exp(-2j * math.pi * frequencies[None, :] * delays[:, None])
    delayed = np.real(np.fft.ifft(np.fft.fft(power, axis=-1) * ramp, axis=-1))
    return np.asarray(delayed.sum(axis=-2))
