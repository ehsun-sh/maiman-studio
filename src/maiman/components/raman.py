"""Distributed Raman amplification: the span itself as the gain medium.

An EDFA makes up a span's loss at the end of it, after the signal has spent
all of it. A strong pump launched into the fibre gives gain *along* the span,
through the same stimulated Raman scattering that tilts a WDM comb, and the
signal never falls as low. The noise a span's amplification adds is set by how
weak the signal was where the gain was given, so gain spread over the last
tens of kilometres is quieter than the same gain lumped after them: the
equivalent noise figure of a counter-pumped span is often below zero dB.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..kernels import raman_gain
from ..signals import NoiseBin, OpticalSignal, Signal
from ..units import C_LIGHT, H_PLANCK, K_BOLTZMANN

#: Where the span's ASE is drawn: the erbium C-band, in bins this wide.
ASE_START_NM = 1528.0
ASE_STOP_NM = 1568.0
ASE_BIN = 25e9

#: Points the span is integrated over.
STEPS = 4000


class RamanAmplifiedSpan(Component):
    """A span of fibre with a Raman pump: loss, distributed gain, and the noise that comes with it.

    **What it models.** One pump at ``pump_wavelength`` with ``pump_power``,
    launched at the far end (counter-pumped, the usual choice) or at the near
    end. The pump is undepleted: it decays at ``pump_attenuation`` and is not
    drained by the signals, which holds while the signals are far weaker than it,
    as they are here. Each frequency ``f`` sees the gain coefficient of silica at
    its distance below the pump, the same measured shape
    :func:`maiman.kernels.raman_gain` gives the inter-channel SRS of ``Fiber``,
    scaled by ``raman_gain_slope``::

        dP/dz = (g(f_p - f) P_pump(z) - alpha) P

    so the on-off gain is ``exp(g P_0 L_eff,pump)``. Spontaneous Raman
    scattering adds noise all along the span, ``n_sp h nu g P_pump(z)`` per
    polarization per metre, with the thermal ``n_sp = 1 + 1/(exp(h Df / kT) -
    1)`` that the phonons set, and each slice of it is carried to the far end
    by the gain and loss after it. That noise is emitted as noise bins over the
    C-band.

    **What it does not.** No dispersion and no Kerr effect, no pump depletion,
    no double Rayleigh backscatter and no pump-to-signal noise transfer: it is
    the gain and noise budget of a span. Use a ``Fiber`` for the waveform.
    """

    display_name = "Raman-Amplified Span"
    category = "Amplifiers"

    length = Param(100.0, unit="km", min=0.0, max=500.0, doc="Span length")
    attenuation = Param(0.2, unit="dB/km", min=0.0, max=10.0, doc="Loss at the signal")
    pump_power = Param(500.0, unit="mW", min=0.0, max=5000.0, doc="Pump power launched")
    pump_wavelength = Param(1450.0, unit="nm", min=1300.0, max=1550.0, doc="Pump wavelength")
    pump_attenuation = Param(0.25, unit="dB/km", min=0.0, max=10.0, doc="Loss at the pump")
    counter_pumped = BoolParam(
        True, doc="Pump launched at the far end, against the signal; off is co-pumped"
    )
    raman_gain_slope = Param(
        0.028,
        unit="1/W/km/THz",
        min=0.0,
        doc="Raman gain slope C_R: 0.028 is SSMF, about 0.37 /W/km at the peak",
    )
    temperature = Param(300.0, unit="K", min=1.0, doc="Sets the phonons' share of the noise")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def _z(self) -> np.ndarray:
        return np.linspace(0.0, self.si("length"), STEPS + 1)

    def _pump(self, z: np.ndarray) -> np.ndarray:
        """Pump power along the span [W]."""
        alpha = self.pump_attenuation * math.log(10.0) / 10.0 / 1e3
        travelled = self.si("length") - z if self.counter_pumped else z
        return self.si("pump_power") * np.exp(-alpha * travelled)

    def gain_coefficient(self, frequency: float) -> float:
        """Raman gain coefficient at ``frequency`` from this pump [1/(W m)]."""
        separation = C_LIGHT / self.si("pump_wavelength") - frequency
        if separation <= 0.0:
            return 0.0
        return float(
            raman_gain(
                np.array([separation]), gain_slope=self.si("raman_gain_slope"), profile="silica"
            )[0]
        )

    def _profile(self, frequency: float) -> tuple[np.ndarray, np.ndarray]:
        """``ln`` of the power gain from the input to each point, and the pump there."""
        z = self._z()
        pump = self._pump(z)
        alpha = self.attenuation * math.log(10.0) / 10.0 / 1e3
        rate = self.gain_coefficient(frequency) * pump - alpha
        steps = np.diff(z) * (rate[1:] + rate[:-1]) / 2.0
        return np.concatenate(([0.0], np.cumsum(steps))), pump

    def on_off_gain_db(self, frequency: float) -> float:
        """Gain with the pump on over the gain with it off [dB]."""
        logs, _ = self._profile(frequency)
        alpha = self.attenuation * math.log(10.0) / 10.0 / 1e3
        return float(10.0 / math.log(10.0) * (logs[-1] + alpha * self.si("length")))

    def net_gain_db(self, frequency: float) -> float:
        """What the span does to a channel, end to end [dB]: on-off gain minus loss."""
        logs, _ = self._profile(frequency)
        return float(10.0 / math.log(10.0) * logs[-1])

    def ase_density(self, frequency: float) -> float:
        """Spontaneous Raman noise at the span's end, per polarization [W/Hz]."""
        logs, pump = self._profile(frequency)
        g = self.gain_coefficient(frequency)
        if g <= 0.0:
            return 0.0
        shift = C_LIGHT / self.si("pump_wavelength") - frequency
        thermal = 1.0 / math.expm1(H_PLANCK * shift / (K_BOLTZMANN * self.temperature))
        emitted = (1.0 + thermal) * H_PLANCK * frequency * g * pump
        carried = np.exp(logs[-1] - logs)
        z = self._z()
        integrand = emitted * carried
        return float(np.sum(np.diff(z) * (integrand[1:] + integrand[:-1]) / 2.0))

    def noise_figure_db(self, frequency: float) -> float:
        """Equivalent noise figure of the pumped span, referred to its end [dB].

        The noise figure a lumped amplifier of the on-off gain, placed after an
        unpumped span, would need to add the same noise: ``(2 S / h nu + 1) /
        G_on``. Below 0 dB when the gain is spread along the span.
        """
        on_off = 10.0 ** (self.on_off_gain_db(frequency) / 10.0)
        density = self.ase_density(frequency)
        return 10.0 * math.log10((2.0 * density / (H_PLANCK * frequency) + 1.0) / on_off)

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        bands = tuple(
            band.scale_amplitude(10.0 ** (self.net_gain_db(band.f0) / 20.0))
            for band in signal.bands
        )
        carried = tuple(
            bin_.scale_power(10.0 ** (self.net_gain_db((bin_.f_start + bin_.f_end) / 2) / 10.0))
            for bin_ in signal.noise
        )
        added = []
        if self.pump_power > 0.0:
            low, high = C_LIGHT / (ASE_STOP_NM * 1e-9), C_LIGHT / (ASE_START_NM * 1e-9)
            for start in np.arange(low, high, ASE_BIN):
                density = self.ase_density(start + ASE_BIN / 2)
                if density > 0.0:
                    added.append(NoiseBin(start, start + ASE_BIN, density, density))
        return {"out": replace(signal, bands=bands, noise=carried + tuple(added))}
