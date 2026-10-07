"""A beam through the open air between two telescopes.

A fibre guides its light; air does not. The beam leaves the transmitter's
aperture and spreads at its divergence angle, so a receiver of fixed size catches
a share that falls with the square of the distance. The air absorbs and
scatters some of it, a little on a clear day and almost all of it in fog. And
turbulence -- cells of warmer and cooler air drifting across the beam -- makes
the received power flicker, so a link that closes on average still drops out
some of the time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import NormalDist

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import OpticalSignal, Signal
from ..units import C_LIGHT


def geometric_loss_db(distance: float, divergence: float, tx: float, rx: float) -> float:
    """Power the receiver's aperture misses [dB], for a beam spreading at ``divergence``.

    The beam's diameter at the receiver is ``tx + divergence * distance`` (full
    angle, in radians, and metres throughout). A receiver of diameter ``rx``
    catches ``(rx / that)^2`` of it, and all of it once it is the larger.
    """
    spot = tx + divergence * distance
    return max(0.0, -20.0 * math.log10(rx / spot))


def rytov_variance(cn2: float, wavelength: float, distance: float) -> float:
    """The plane-wave Rytov variance ``1.23 Cn^2 k^(7/6) L^(11/6)``: the turbulence strength."""
    k = 2.0 * math.pi / wavelength
    return 1.23 * cn2 * k ** (7.0 / 6.0) * distance ** (11.0 / 6.0)


def fade_db(scintillation: float, probability: float) -> float:
    """How deep a fade is reached ``probability`` of the time [dB, positive].

    Weak turbulence leaves the received intensity log-normal: ``ln I`` is
    Gaussian with variance ``s^2 = ln(1 + sigma_I^2)`` and a mean of ``-s^2/2``,
    so the mean intensity is one. The power falls below ``I_p`` with probability
    ``p`` where ``ln I_p = -s^2/2 + s Phi^-1(p)``.
    """
    if probability <= 0.0 or scintillation <= 0.0:
        return 0.0
    s = math.sqrt(math.log1p(scintillation))
    return -10.0 / math.log(10.0) * (-(s**2) / 2.0 + s * NormalDist().inv_cdf(probability))


class FreeSpaceChannel(Component):
    """A free-space optical link: spreading, absorption and turbulence fading.

    **What it models.** The geometric loss of a beam spreading from its launch
    diameter at its divergence, against a receive aperture; a clear-air or fog
    attenuation in dB/km; and turbulence of strength ``cn2``, whose Rytov
    variance sets the scintillation index in the weak-turbulence limit. The
    flicker is far slower than any simulated window -- milliseconds, against
    nanoseconds -- so it is not drawn as a waveform. Instead ``outage`` picks
    one moment: 0 is the average channel, and 0.001 is the fade the link sits
    below 0.1 % of the time, which is what a margin is designed against.

    **What it does not.** No pointing error and no beam wander; scintillation is
    the weak-turbulence, point-receiver estimate, with no aperture averaging, so
    it is pessimistic for a large receiver. The light keeps its shape: no
    dispersion and no multipath at these distances.
    """

    display_name = "Free-Space Channel"
    category = "Free Space"

    distance = Param(1.0, unit="km", min=0.0, max=1000.0, doc="Between the two apertures")
    divergence = Param(1.0, unit="mrad", min=0.0, doc="Full angle the beam spreads at")
    tx_aperture = Param(2.5, unit="cm", min=0.0, doc="Beam diameter at the transmitter")
    rx_aperture = Param(10.0, unit="cm", min=0.0, doc="Diameter of the receiving lens")
    attenuation = Param(
        0.5,
        unit="dB/km",
        min=0.0,
        max=500.0,
        doc="What the air takes: about 0.5 clear, 10 in haze, 100 and more in fog",
    )
    cn2 = Param(
        1e-14,
        unit="",
        min=0.0,
        max=1e-11,
        doc="Turbulence strength Cn^2 [m^-2/3]: 1e-15 weak, 1e-13 strong",
    )
    outage = Param(
        0.0,
        unit="",
        min=0.0,
        max=0.5,
        doc="Share of the time the link is below this channel; 0 is the average channel",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def budget(self, wavelength: float = 1550e-9) -> dict[str, float]:
        """Each part of the loss [dB], and the turbulence that sets the fade."""
        distance = self.si("distance")
        geometric = geometric_loss_db(
            distance, self.si("divergence"), self.si("tx_aperture"), self.si("rx_aperture")
        )
        rytov = rytov_variance(self.cn2, wavelength, distance)
        return {
            "geometric": geometric,
            "atmospheric": self.attenuation * self.distance,
            "rytov": rytov,
            "fade": fade_db(rytov, self.outage),
        }

    def loss_db(self, wavelength: float = 1550e-9) -> float:
        parts = self.budget(wavelength)
        return parts["geometric"] + parts["atmospheric"] + parts["fade"]

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        bands = []
        for band in signal.bands:
            power = 10.0 ** (-self.loss_db(C_LIGHT / band.f0) / 10.0)
            bands.append(band.scale_amplitude(math.sqrt(power)))
        middle = 10.0 ** (-self.loss_db() / 10.0)
        noise = tuple(bin_.scale_power(middle) for bin_ in signal.noise)
        return {"out": replace(signal, bands=tuple(bands), noise=noise)}
