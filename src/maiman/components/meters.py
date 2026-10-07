"""Measurement components."""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..analysis import OSNR_REFERENCE_BANDWIDTH, osnr
from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..signals import BandPower, OpticalSignal, PowerReading, ScopeTrace, Signal
from ..units import frequency_to_wavelength


class PowerMeter(Component):
    """Ideal optical power meter.

    Reports total power and a per-band breakdown. The breakdown matters as soon
    as more than one carrier is present: a single total is exactly what makes a
    WDM result impossible to interpret.
    """

    display_name = "Optical Power Meter"
    category = "Measurements"

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        bands = tuple(
            BandPower(
                f0=band.f0,
                wavelength_nm=frequency_to_wavelength(band.f0) * 1e9,
                power_w=band.average_power(),
            )
            for band in sorted(signal.bands, key=lambda b: b.f0)
        )
        return {
            "out": PowerReading(
                signal_power_w=signal.signal_power(),
                noise_power_w=signal.noise_power(),
                bands=bands,
            )
        }


class OSNRMeter(Component):
    """Measures optical signal-to-noise ratio in a reference bandwidth.

    OSNR is what actually predicts whether an amplified link will work, and it
    is only meaningful because noise is carried as a spectral density rather
    than mixed into the samples: the ratio depends on the noise in a 0.1 nm
    slice, not on however much of it the simulation happened to sample.
    """

    display_name = "OSNR Meter"
    category = "Measurements"

    reference_bandwidth = Param(
        OSNR_REFERENCE_BANDWIDTH / 1e9,
        unit="GHz",
        min=0.0,
        doc="Reference bandwidth; 12.5 GHz is 0.1 nm at 1550 nm",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        return {"out": osnr(signal, reference_bandwidth=self.si("reference_bandwidth"))}


class Oscilloscope(Component):
    """Optical power against time: one pulse, unfolded.

    What the eye diagram cannot show. An eye folds a waveform onto two symbol
    periods; a single pulse spreading in a fibre, compressing against its own
    chirp or holding its shape as a soliton has nothing to fold, and the
    question is what it looks like and how wide it is.

    The power is ``|Ex|**2 + |Ey|**2`` summed over every band on a common grid,
    as an ideal photodiode with unlimited bandwidth would see it; the beat
    between two bands oscillates at their spacing, terahertz apart, and a real
    detector averages it away. The trace is reduced to ``points`` by averaging,
    which keeps the energy honest; the peak, the FWHM and the RMS width are
    measured on the full-resolution samples before that, so they do not move
    when the display resolution does.

    The instantaneous frequency is read from the phase of the strongest band's
    stronger polarization, ``(1/2 pi) d(phi)/dt`` in the envelope's
    ``exp(+j 2 pi f0 t)`` convention, and only where the power is above a
    hundredth of the peak: in the dark the phase is noise and its derivative
    is larger noise.
    """

    display_name = "Optical Oscilloscope"
    category = "Measurements"

    span = Param(
        0.0, unit="ps", min=0.0, doc="Time shown around `centre`; 0 shows the whole window"
    )
    centre = Param(0.0, unit="ps", doc="Display centre, on the same time axis as the trace")
    decibels = BoolParam(False, doc="Draw the power in dBm, for a trace that falls over decades")
    from_start = BoolParam(
        False,
        doc="Count time from the start of the window, not its middle: an echo's round trip",
    )
    points = Param(1024.0, unit="", min=16.0, max=8192.0, doc="Points sent to the display")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        n = ctx.num_samples
        bands = [b for b in signal.bands if b.num_samples == n]
        if not bands:
            raise ValueError(f"{self.label}: no band on the simulation's time grid to display")

        dt = 1.0 / ctx.sample_rate
        t = np.arange(n) * dt - (0.0 if self.from_start else n * dt / 2.0)
        power = np.zeros(n)
        for band in bands:
            power += np.abs(np.asarray(band.Ex)) ** 2 + np.abs(np.asarray(band.Ey)) ** 2

        strongest = max(bands, key=lambda b: b.average_power())
        field_x = np.asarray(strongest.Ex)
        if np.sum(np.abs(field_x) ** 2) < np.sum(np.abs(np.asarray(strongest.Ey)) ** 2):
            field_x = np.asarray(strongest.Ey)
        phase = np.unwrap(np.angle(field_x))
        # The envelope multiplies exp(+j 2 pi f0 t), so a phase that grows with
        # time is a frequency above the carrier.
        chirp = np.gradient(phase, dt) / (2.0 * math.pi)

        span = self.si("span")
        centre = self.si("centre")
        if span > 0.0:
            window = (t >= centre - span / 2.0) & (t <= centre + span / 2.0)
            if not np.any(window):
                raise ValueError(f"{self.label}: the display window holds no samples")
        else:
            window = np.ones(n, dtype=bool)
        t, power, chirp = t[window], power[window], chirp[window]

        peak = float(power.max())
        energy = float(np.sum(power) * dt)
        if energy > 0.0:
            centroid = float(np.sum(t * power) * dt / energy)
            rms = float(np.sqrt(max(np.sum((t - centroid) ** 2 * power) * dt / energy, 0.0)))
        else:
            centroid = rms = 0.0
        chirp = np.where(power > 0.01 * peak, chirp, np.nan) if peak > 0.0 else chirp * np.nan

        points = min(int(self.points), t.size)
        edges = np.linspace(0, t.size, points + 1).astype(int)
        average = np.add.reduceat
        counts = np.diff(edges)
        shown_t = average(t, edges[:-1]) / counts
        shown_power = average(power, edges[:-1]) / counts
        with np.errstate(invalid="ignore"):
            shown_chirp = np.array(
                [
                    np.nanmean(chirp[a:b]) if np.any(np.isfinite(chirp[a:b])) else np.nan
                    for a, b in itertools.pairwise(edges)
                ]
            )

        return {
            "out": ScopeTrace(
                time=shown_t,
                power_w=shown_power,
                chirp_hz=shown_chirp,
                peak_power_w=peak,
                fwhm=_fwhm(t, power),
                rms_width=rms,
                centroid=centroid,
                energy_j=energy,
                decibels=bool(self.decibels),
            )
        }


def _fwhm(t: np.ndarray, power: np.ndarray) -> float:
    """Width at half the peak, interpolated at both crossings; NaN if a side never falls."""
    if t.size < 3:
        return float("nan")
    peak_index = int(np.argmax(power))
    half = power[peak_index] / 2.0
    if half <= 0.0:
        return float("nan")
    left = np.nonzero(power[:peak_index] < half)[0]
    right = np.nonzero(power[peak_index:] < half)[0]
    if not left.size or not right.size:
        return float("nan")
    i = int(left[-1])
    j = peak_index + int(right[0])

    def cross(a: int, b: int) -> float:
        p0, p1 = power[a], power[b]
        return float(t[a] + (half - p0) * (t[b] - t[a]) / (p1 - p0))

    return cross(j - 1, j) - cross(i, i + 1)
