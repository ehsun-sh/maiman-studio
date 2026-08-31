"""Photodetectors.

Model reference: G. P. Agrawal, *Fiber-Optic Communication Systems*, ch. 4
(photodetectors, avalanche gain, receiver noise).
"""

from __future__ import annotations

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import ElectricalSignal, OpticalSignal, Signal
from ..units import K_BOLTZMANN, Q_ELECTRON


class PINPhotodiode(Component):
    """PIN photodiode: square-law detection with shot and thermal noise.

    The mean photocurrent is ``I = M * (R * P + I_dark)``, with ``R`` the
    responsivity [A/W] and ``M`` the multiplication gain — unity here, and
    overridden by :class:`APDPhotodiode`. Two noise sources are added, both
    white over the simulated bandwidth ``B = fs / 2``:

    * **Shot noise**, variance ``2 * q * I_primary * M**2 * F * B``. It scales
      with the instantaneous current, so it is generated per sample rather than
      as a single constant — bright samples really are noisier than dark ones,
      which is why an eye diagram's rails have different thicknesses.
    * **Thermal (Johnson) noise**, variance ``4 * k * T * B / R_load``,
      independent of the received power.

    Two simplifications worth stating plainly, both of which change results and
    neither of which is hidden by the interface:

    * Bands are detected incoherently — powers add. Beating between bands lands
      at their frequency separation, far above any realistic receiver bandwidth
      for the channel spacings this is used with, but it is genuinely absent
      rather than merely negligible.
    * Bands are detected incoherently, so a *band* never beats with another band.
      ASE does beat with the signal, and that is modelled — see below.

    **ASE beat noise.** A photodiode squares the field, so an ASE field arriving
    with the signal does not simply add its power: it beats. Writing
    ``i = R|E_s + E_n|**2`` gives a signal-spontaneous cross term and a
    spontaneous-spontaneous one, and on any amplified link the first dominates
    every other noise source in this class by orders of magnitude::

        var_sig_sp = 4 R**2 (P_x S_x + P_y S_y) B_e
        var_sp_sp  = 2 R**2 (S_x**2 + S_y**2) B_o B_e

    ``S`` is the one-sided ASE density per polarization at the signal's own
    frequency, which is why :meth:`OpticalSignal.noise_psd_at` exists — the
    integrated noise power is spread over the whole amplifier band and is the
    wrong number. The signal-spontaneous term is written per polarization so a
    signal on Y beats with Y's ASE and not with X's; ASE orthogonal to the signal
    contributes only to the spontaneous-spontaneous term, which is what a
    polarization filter in front of a receiver exploits.

    **The spontaneous-spontaneous term scales with the optical bandwidth, so
    something has to set it.** This diode integrates whatever ASE reaches it, and
    an unfiltered receiver on an eight-span link genuinely loses a factor of three
    in Q to spontaneous-spontaneous beating alone. That is a real result, not a
    modelling artefact, and it is why nobody builds one:
    :class:`~maiman.components.filters.OpticalFilter` belongs in front of this
    block on any amplified link. An earlier version of this class carried the
    passband as a parameter of its own, which was a stand-in for a component that
    did not exist yet; it does now, and two ways to express one piece of hardware
    would only drift apart.
    """

    display_name = "PIN Photodiode"
    category = "Receivers"

    responsivity = Param(0.8, unit="", min=0.0, doc="Responsivity R [A/W]")
    dark_current = Param(0.0, unit="", min=0.0, doc="Dark current [A]")
    load_resistance = Param(50.0, unit="", min=0.0, doc="Load resistance [ohm]")
    temperature = Param(300.0, unit="", min=0.0, doc="Receiver temperature [K]")
    shot_noise = BoolParam(True, doc="Add shot noise")
    thermal_noise = BoolParam(True, doc="Add thermal (Johnson) noise")
    ase_beat_noise = BoolParam(True, doc="Add signal-ASE and ASE-ASE beat noise")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.ELECTRICAL}

    def multiplication(self) -> float:
        """Avalanche gain. Unity for a PIN, which has no multiplication region."""
        return 1.0

    def excess_noise_factor(self) -> float:
        """Excess noise from the randomness of multiplication. Unity for a PIN."""
        return 1.0

    def noise_bandwidth(self, ctx: SimulationContext) -> float:
        """Effective one-sided noise bandwidth [Hz] of the sampled representation."""
        return ctx.sample_rate / 2.0

    def optical_noise_bandwidth(self, signal: OpticalSignal, frequency: float) -> float:
        """Optical bandwidth the ASE actually reaches the diode through [Hz].

        The width of the noise bin the signal sits in, which is whatever the last
        filter in the line left behind. A diode has no wavelength selectivity of
        its own, so this is a question about the link and not about the detector.
        """
        covering = [b.bandwidth for b in signal.noise if b.f_start <= frequency < b.f_end]
        return max(covering) if covering else 0.0

    def received_noise_power(self, signal: OpticalSignal, frequency: float) -> float:
        """ASE power arriving with the signal [W].

        Density at the signal's frequency times the width it survives over, which
        is the same pair of numbers the beat terms use. Deriving the mean power
        and the beat variance from one source is deliberate: an early version
        took them from different places — the whole amplifier band for the power
        and a narrow filter for the beat — and that is not a conservative
        approximation but two different receivers averaged together. It put 2.4x
        too much noise in a space.
        """
        psd_x, psd_y = signal.noise_psd_at(frequency)
        if psd_x == 0.0 and psd_y == 0.0:
            return signal.noise_power()
        return (psd_x + psd_y) * self.optical_noise_bandwidth(signal, frequency)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]

        power_x = np.zeros(ctx.num_samples, dtype=np.float64)
        power_y = np.zeros(ctx.num_samples, dtype=np.float64)
        for band in signal.bands:
            power_x += np.abs(band.Ex.astype(np.complex128)) ** 2
            power_y += np.abs(band.Ey.astype(np.complex128)) ** 2
        # The strongest band, not the first. A detector has no wavelength
        # selectivity, so "which channel is this" is decided by whatever survived
        # the last filter — and on a demultiplexed comb the first band in the list
        # is a suppressed neighbour a full channel spacing away. Reading the ASE
        # density there returns zero, because a filter has already clipped the
        # noise bins to its own passband, and the beat terms below then evaluate
        # to nothing at all. Silently: the link simply looks four times better
        # than its own OSNR allows.
        reference = (
            max(signal.bands, key=lambda b: b.average_power()).f0
            if signal.bands
            else (signal.noise[0].f_start if signal.noise else 0.0)
        )
        power = power_x + power_y + self.received_noise_power(signal, reference)

        gain = self.multiplication()
        primary = self.si("responsivity") * power + self.si("dark_current")
        current = gain * primary
        bandwidth = self.noise_bandwidth(ctx)

        if self.shot_noise:
            # Multiplication amplifies the signal by M and the shot noise power
            # by M**2 * F: that is exactly why an avalanche gain cannot be raised
            # without limit, and why an APD has an optimum gain rather than a
            # best one.
            shot_variance = (
                2.0
                * Q_ELECTRON
                * np.maximum(primary, 0.0)
                * gain**2
                * self.excess_noise_factor()
                * bandwidth
            )
            rng = ctx.rng(type(self).__name__, self.label, "shot")
            current = current + rng.normal(0.0, np.sqrt(shot_variance))

        if self.ase_beat_noise and signal.noise:
            responsivity = self.si("responsivity")
            psd_x, psd_y = signal.noise_psd_at(reference)
            optical = self.optical_noise_bandwidth(signal, reference)

            # Signal-spontaneous. Time-varying, because it rides on the
            # instantaneous power: a mark is noisier than a space, which is the
            # whole reason an amplified link's eye closes from the top.
            sig_sp = 4.0 * responsivity**2 * (power_x * psd_x + power_y * psd_y) * bandwidth
            # Spontaneous-spontaneous. Constant, and present even in a space.
            sp_sp = 2.0 * responsivity**2 * (psd_x**2 + psd_y**2) * optical * bandwidth

            variance = np.maximum(sig_sp + sp_sp, 0.0) * gain**2 * self.excess_noise_factor()
            rng = ctx.rng(type(self).__name__, self.label, "ase-beat")
            current = current + rng.normal(0.0, np.sqrt(variance))

        if self.thermal_noise and self.si("load_resistance") > 0.0:
            thermal_variance = (
                4.0 * K_BOLTZMANN * self.si("temperature") * bandwidth / self.si("load_resistance")
            )
            rng = ctx.rng(type(self).__name__, self.label, "thermal")
            current = current + rng.normal(0.0, np.sqrt(thermal_variance), size=ctx.num_samples)

        return {
            "out": ElectricalSignal(
                samples=current.astype(ctx.real_dtype), fs=ctx.sample_rate, unit="A"
            )
        }


class APDPhotodiode(PINPhotodiode):
    """Avalanche photodiode: internal gain, bought with excess noise.

    An APD multiplies the primary photocurrent by ``M`` before any electronics
    see it, which lifts a weak signal above the thermal noise floor of the load.
    The multiplication is a random cascade, though, so it amplifies shot noise
    by more than ``M**2``::

        F(M) = k*M + (2 - 1/M) * (1 - k)

    with ``k`` the ionisation coefficient ratio — a material property, around
    0.02 for silicon and 0.3 to 0.5 for InGaAs at 1550 nm. Lower is better.

    Because the signal grows as ``M`` while shot noise grows as ``M**2 * F`` and
    thermal noise does not grow at all, there is an **optimum gain**, not a best
    one. Below it the receiver is thermal-limited and more gain helps; above it
    the multiplication noise it creates outweighs what it buys.

    At ``M = 1`` the excess noise factor is 1 and this reduces exactly to
    :class:`PINPhotodiode` — asserted in the test suite rather than assumed.
    """

    display_name = "APD Photodiode"
    category = "Receivers"

    gain = Param(10.0, unit="", min=1.0, doc="Avalanche multiplication factor M")
    ionization_ratio = Param(
        0.3, unit="", min=0.0, max=1.0, doc="Ionisation coefficient ratio k; lower is quieter"
    )

    def multiplication(self) -> float:
        return self.gain

    def excess_noise_factor(self) -> float:
        """``F(M) = k*M + (2 - 1/M)(1 - k)``, which is 1 at M = 1 for any k."""
        m = self.gain
        k = self.ionization_ratio
        return k * m + (2.0 - 1.0 / m) * (1.0 - k)
