"""A whole amplified line in one block: many identical spans and their amplifiers.

A transoceanic cable is a hundred spans, each a length of fibre and the repeater
that makes up its loss. Drawn block by block that is two hundred blocks of the
same two things, so this block draws it once and says how many.

What a long line teaches is not in any one span. Each repeater adds its own
amplified spontaneous emission, so the noise grows with the number of spans and
the OSNR falls ten decibels for every factor of ten. And each repeater's gain is
a little higher at one end of the band than the other: a tenth of a decibel per
repeater is nothing, and a hundred repeaters of it is the edge channels ten
decibels apart. Undersea repeaters carry a gain-flattening filter for exactly
that reason, a passive filter cut to the inverse of the erbium's gain shape.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import NoiseBin, OpticalSignal, Signal
from ..units import C_LIGHT, H_PLANCK

#: Where the line's ASE is drawn: the erbium C-band, in bins this wide.
ASE_START_NM = 1528.0
ASE_STOP_NM = 1568.0
ASE_BIN = 25e9


class AmplifiedLine(Component):
    """``spans`` identical spans of fibre, each followed by an amplifier that makes up its loss.

    **What it models.** Each amplifier's gain is the span's loss plus a linear
    tilt across the band, ``tilt`` dB for every terahertz below ``centre``
    (erbium's gain rises towards longer wavelengths), so after ``n`` spans a
    channel ``df`` from the centre has gained ``-n * tilt * df`` dB. Each adds
    ASE of ``n_sp h nu (G - 1)`` per polarization, with ``n_sp = NF G / (2 (G -
    1))``, and every later span amplifies what earlier ones added by the same
    net gain as the signal. The ASE is emitted as noise bins over the whole
    C-band, so an OSA after the line shows its tilted floor.

    ``flatten`` puts a gain-flattening filter after each amplifier, cut to the
    inverse of its tilt, so every channel leaves with the power it came in with.
    The filter is not free: it has ``filter_loss`` of loss at its least
    attenuating point, which the amplifier is set that much higher to make up,
    and that costs a little OSNR.

    **What it does not.** No dispersion and no nonlinearity: the fields are only
    scaled, as if the line were dispersion-managed and run below the Kerr limit.
    So the budget is the OSNR, not the generalised SNR that adds nonlinear
    interference. Use ``Fiber`` and ``EDFA`` blocks for a link where the
    waveform matters.
    """

    display_name = "Amplified Line"
    category = "Amplifiers"

    spans = Param(10.0, unit="", min=1.0, max=1000.0, doc="Number of span-and-repeater pairs")
    span_length = Param(80.0, unit="km", min=0.0, max=500.0, doc="Fibre between repeaters")
    attenuation = Param(0.2, unit="dB/km", min=0.0, max=10.0, doc="Fibre loss")
    noise_figure = Param(5.0, unit="dB", min=0.0, doc="Each repeater's noise figure")
    tilt = Param(
        0.05,
        unit="dB/THz",
        doc="Each repeater's gain slope: how much more gain one terahertz lower down",
    )
    centre = Param(1550.0, unit="nm", doc="Where the gain is exactly the span loss")
    flatten = BoolParam(False, doc="A gain-flattening filter in every repeater")
    filter_loss = Param(
        1.0,
        unit="dB",
        min=0.0,
        doc="Loss of each flattening filter, made up by its amplifier",
        applies_when="flatten",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def count(self) -> int:
        return max(1, round(self.spans))

    def span_loss_db(self) -> float:
        return self.attenuation * self.span_length

    def length_km(self) -> float:
        return self.count() * self.span_length

    def tilt_db(self, frequency: float) -> float:
        """One repeater's gain above the span loss at ``frequency`` [dB]; filtered, zero."""
        if self.flatten:
            return 0.0
        return -self.tilt * (frequency - C_LIGHT / self.si("centre")) / 1e12

    def net_gain_db(self, frequency: float) -> float:
        """What the whole line does to a channel at ``frequency`` [dB]."""
        return self.count() * self.tilt_db(frequency)

    def ase_density(self, frequency: float) -> float:
        """ASE at the line's end, per polarization [W/Hz], at ``frequency``."""
        tilt = self.tilt_db(frequency)
        extra = self.filter_loss if self.flatten else 0.0
        gain = 10.0 ** ((self.span_loss_db() + extra + tilt) / 10.0)
        if gain <= 1.0:
            return 0.0
        nsp = 10.0 ** (self.noise_figure / 10.0) * gain / (2.0 * (gain - 1.0))
        emitted = nsp * H_PLANCK * frequency * (gain - 1.0) * 10.0 ** (-extra / 10.0)
        # Repeater k's noise crosses the n - k spans after it: a geometric sum.
        net = 10.0 ** (tilt / 10.0)
        n = self.count()
        carried = float(n) if net == 1.0 else (net**n - 1.0) / (net - 1.0)
        return emitted * carried

    def osnr_db(self, frequency: float, launch_w: float, reference: float = 12.5e9) -> float:
        """A channel's OSNR at the end of the line, for ``launch_w`` launched [dB]."""
        signal = launch_w * 10.0 ** (self.net_gain_db(frequency) / 10.0)
        return 10.0 * math.log10(signal / (2.0 * self.ase_density(frequency) * reference))

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
        low, high = C_LIGHT / (ASE_STOP_NM * 1e-9), C_LIGHT / (ASE_START_NM * 1e-9)
        edges = np.arange(low, high, ASE_BIN)
        added = []
        for start in edges:
            density = self.ase_density(start + ASE_BIN / 2)
            if density > 0.0:
                added.append(NoiseBin(start, start + ASE_BIN, density, density))
        return {"out": replace(signal, bands=bands, noise=carried + tuple(added))}
