"""Radio frequencies: a tone generator and the analyser that reads one.

Everything electrical in this library so far carried data: bits shaped into
pulses, read by an eye or a slicer. Analogue radio over fibre carries a radio
signal instead, and the question it asks is spectral -- how much of the tone
arrived, and what new tones the link made out of it. That needs a source of
clean sine waves and an instrument that plots power against frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import ElectricalSignal, Signal, freeze


def on_grid(frequency: float, ctx: SimulationContext) -> float:
    """``frequency`` moved to the nearest whole number of cycles in the window [Hz].

    A tone that does not fit the window a whole number of times has a step where
    the window wraps, and that step spreads its power over every frequency. On
    the grid, a tone lands in one bin of the analyser and nowhere else.
    """
    step = ctx.sample_rate / ctx.num_samples
    return round(frequency / step) * step


class RFTone(Component):
    """One or two sine waves on a DC level: a signal generator.

    Two tones of equal amplitude are the standard test of a link's linearity:
    anything that bends the signal mixes them, and the third-order products land
    at ``2 f1 - f2`` and ``2 f2 - f1``, right beside the tones where no filter
    can remove them. Set ``frequency2`` to 0 for a single tone.

    Each frequency is moved to the nearest whole number of cycles in the
    simulated window, so it is exact on the analyser; the change is at most half
    of one over the window.
    """

    display_name = "RF Tone"
    category = "Electrical Sources"

    frequency = Param(1.0, unit="GHz", min=0.0, doc="First tone")
    frequency2 = Param(0.0, unit="GHz", min=0.0, doc="Second tone; 0 for none")
    amplitude = Param(0.5, unit="V", min=0.0, doc="Peak voltage of each tone")
    offset = Param(0.0, unit="V", doc="DC level the tones ride on")

    outputs = {"out": PortType.ELECTRICAL}

    def tones(self, ctx: SimulationContext) -> list[float]:
        """The frequencies actually generated [Hz], on the window's grid."""
        wanted = [self.si("frequency"), self.si("frequency2")]
        return [on_grid(f, ctx) for f in wanted if f > 0.0]

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        t = np.arange(ctx.num_samples) / ctx.sample_rate
        wave = np.full(ctx.num_samples, self.si("offset"))
        for frequency in self.tones(ctx):
            wave = wave + self.si("amplitude") * np.cos(2.0 * math.pi * frequency * t)
        return {"out": ElectricalSignal(samples=wave.astype(ctx.real_dtype), fs=ctx.sample_rate)}


@dataclass(frozen=True)
class ElectricalSpectrum:
    """An electrical spectrum analyser's trace: power into a load against frequency.

    ``power_w`` is the power inside one resolution bandwidth centred on each
    frequency, which is what such an instrument displays: a tone reads its whole
    power whatever the resolution, and noise reads more the wider it is.
    """

    frequencies: np.ndarray
    """Display grid [Hz], from DC up."""

    power_w: np.ndarray
    """Power in one resolution bandwidth at each frequency [W]."""

    resolution_bandwidth: float
    """The instrument's resolution [Hz]."""

    def __post_init__(self) -> None:
        if self.frequencies.shape != self.power_w.shape:
            raise ValueError("frequencies and power_w must share a shape")
        object.__setattr__(self, "frequencies", freeze(self.frequencies))
        object.__setattr__(self, "power_w", freeze(self.power_w))

    def dbm_at(self, frequency: float) -> float:
        """The level the trace reads at the bin nearest ``frequency`` [dBm]."""
        index = int(np.argmin(np.abs(np.asarray(self.frequencies) - frequency)))
        return float(10.0 * np.log10(max(float(self.power_w[index]), 1e-30) / 1e-3))


class ElectricalSpectrumAnalyzer(Component):
    """Power against frequency of an electrical signal, into a matched load.

    The waveform is taken as a voltage across ``load`` or a current into it,
    whichever its unit says, and the one-sided power spectrum is computed from
    one FFT of the whole window, with no taper: tones from an :class:`RFTone`
    sit on the FFT's grid and land in single bins. Each displayed point is then
    the power summed over ``resolution_bandwidth`` around it. DC is left out of
    the display; it is a bias, not a signal.
    """

    display_name = "Electrical Spectrum Analyzer"
    category = "Measurements"

    resolution_bandwidth = Param(
        1.0, unit="MHz", min=0.0, doc="Width each point sums over; at least one FFT bin"
    )
    stop = Param(0.0, unit="GHz", min=0.0, doc="Highest frequency shown; 0 shows up to Nyquist")
    load = Param(50.0, unit="ohm", min=1e-3, doc="Load resistance")

    inputs = {"in": PortType.ELECTRICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: ElectricalSignal = inputs["in"]
        x = np.asarray(signal.samples, dtype=np.float64)
        n = x.size
        amplitude = np.fft.rfft(x) / n
        # One-sided: every bin but DC and Nyquist carries its mirror's power too.
        squared = np.abs(amplitude) ** 2
        squared[1:] *= 2.0
        if n % 2 == 0:
            squared[-1] /= 2.0
        load = self.load
        power = squared * load if signal.unit == "A" else squared / load
        frequencies = np.fft.rfftfreq(n, 1.0 / signal.fs)

        step = signal.fs / n
        width = max(1, round(self.si("resolution_bandwidth") / step))
        if width > 1:
            power = np.convolve(power, np.ones(width), mode="same")
        stop = self.si("stop")
        keep = frequencies > 0.0
        if stop > 0.0:
            keep &= frequencies <= stop
        return {
            "out": ElectricalSpectrum(
                frequencies=frequencies[keep],
                power_w=power[keep],
                resolution_bandwidth=width * step,
            )
        }
