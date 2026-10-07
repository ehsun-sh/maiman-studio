"""Counting single photons, and turning the counts into a secret key.

Quantum key distribution sends a key in pulses so faint that most hold no
photon at all. An eavesdropper who measures a photon disturbs it, so the error
rate of what arrives bounds what anyone else can know, and privacy
amplification removes that much. What is left is a key nobody else holds.

The model here is the one every BB84 system is first designed with: Poisson
pulses from an attenuated laser, a lossy channel, and a gated detector with a
given efficiency and dark-count probability; the key rate is the
Gottesman-Lo-Lutkenhaus-Preskill bound with the single-photon terms an ideal
decoy-state method estimates exactly.
"""

from __future__ import annotations

import math

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import OpticalSignal, Readout, Signal
from ..units import C_LIGHT, H_PLANCK


def binary_entropy(p: float) -> float:
    """``H2(p)`` [bits]; zero at 0 and 1."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def bb84(
    mean_photons: float,
    transmittance: float,
    *,
    dark: float,
    misalignment: float,
    correction: float = 1.16,
    sifting: float = 0.5,
) -> dict[str, float]:
    """Gain, error rate and secure key per pulse of decoy-state BB84 (Lo, Ma and Chen, 2005).

    ``transmittance`` is the whole path, channel and detector efficiency. With
    ``eta`` that and ``Y0`` the dark-count probability per gate:

        Q   = Y0 + 1 - exp(-eta mu)                       detections per pulse
        E Q = Y0 / 2 + e_d (1 - exp(-eta mu))             of which wrong
        Y1  = Y0 + eta,   e1 = (Y0 / 2 + e_d eta) / Y1    single-photon pulses
        R   = q (Q1 (1 - H2(e1)) - f Q H2(E)),  Q1 = Y1 mu exp(-mu)

    ``q`` is the share of pulses whose bases agree and ``f`` how far error
    correction falls short of the Shannon limit.
    """
    eta, mu = transmittance, mean_photons
    clicks = 1.0 - math.exp(-eta * mu)
    gain = dark + clicks
    qber = (0.5 * dark + misalignment * clicks) / gain if gain > 0.0 else 0.5
    single_yield = dark + eta
    single_error = (0.5 * dark + misalignment * eta) / single_yield if single_yield else 0.5
    single_gain = single_yield * mu * math.exp(-mu)
    key = sifting * (
        single_gain * (1.0 - binary_entropy(single_error))
        - correction * gain * binary_entropy(qber)
    )
    return {
        "gain": gain,
        "qber": qber,
        "single_gain": single_gain,
        "single_error": single_error,
        "key_per_pulse": max(key, 0.0),
    }


class BB84Receiver(Component):
    """Bob's single-photon detectors, and the key the counts are worth.

    ``sent`` is what Alice launched and ``in`` what reaches Bob, both one pulse
    per symbol period. From their mean powers the block reads the mean photon
    number per pulse and the channel's transmittance; with the detector's own
    efficiency, dark counts and the optics' misalignment it reports the
    detection rate, the quantum bit error rate (QBER) and the secure key rate.

    The answer is the asymptotic one: an infinitely long key, and decoy states
    that pin the single-photon terms exactly. Real systems with finite keys and
    two or three decoy levels get somewhat less.
    """

    display_name = "BB84 Receiver"
    category = "Receivers"

    efficiency = Param(0.1, unit="", min=0.0, max=1.0, doc="Detector efficiency")
    dark_count = Param(
        1e-6, unit="", min=0.0, max=0.1, doc="Probability of a click with no photon, per gate"
    )
    misalignment = Param(
        0.015,
        unit="",
        min=0.0,
        max=0.5,
        doc="Share of photons the optics send to the wrong detector",
    )
    correction = Param(
        1.16, unit="", min=1.0, max=3.0, doc="Error correction over the Shannon limit"
    )

    inputs = {"in": PortType.OPTICAL, "sent": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        received: OpticalSignal = inputs["in"]
        sent: OpticalSignal = inputs["sent"]
        launched = sum(b.average_power() for b in sent.bands)
        arrived = sum(b.average_power() for b in received.bands)
        if launched <= 0.0:
            raise ValueError(f"{self.label}: nothing was sent")
        rate = ctx.bit_rate
        frequency = sent.bands[0].f0 if sent.bands else C_LIGHT / 1550e-9
        mean_photons = launched / (rate * H_PLANCK * frequency)
        channel = arrived / launched
        result = bb84(
            mean_photons,
            channel * self.efficiency,
            dark=self.dark_count,
            misalignment=self.misalignment,
            correction=self.correction,
        )
        values = {
            "mean_photons": mean_photons,
            "channel_db": -10.0 * math.log10(channel) if channel > 0.0 else math.inf,
            **result,
            "key_rate": result["key_per_pulse"] * rate,
        }
        key = values["key_rate"]
        return {
            "out": Readout(
                values=values,
                caption=f"QBER {values['qber'] * 100:.2f} %",
                summary=(
                    f"mu = {mean_photons:.3f} photons a pulse, {values['channel_db']:.1f} dB "
                    f"channel, {values['gain']:.2e} clicks a pulse, QBER "
                    f"{values['qber'] * 100:.2f} %, secure key {key / 1e3:.2f} kb/s"
                ),
            )
        }
