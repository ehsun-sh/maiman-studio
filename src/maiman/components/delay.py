"""A delay line: the one block that moves light in time.

Every signal in this library is a whole time window reported in its own retarded
frame, so a span of fibre hands its field on without the group delay it took to
cross -- real receivers remove that delay in clock recovery, and nothing about a
single link depends on it. What does depend on it is a path that meets itself:
two arms of an interferometer of different lengths, and a recirculating loop,
where every lap has to arrive one loop-time after the last instead of on top of
it. This is the explicit way to say how much later.
"""

from __future__ import annotations

import math

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, OpticalSignal, Signal

#: How close ``delay * sample_rate`` has to be to a whole number for the delay to
#: be taken as a shift by whole samples, which is exact, rather than a phase ramp
#: in frequency, which rings on anything not band-limited.
WHOLE_SAMPLE_TOLERANCE = 1e-9


class DelayLine(Component):
    """Delays the field by a fixed time, carrier phase included.

    **What it does.** ``E(t) -> E(t - delay)``, on the field, not just the
    envelope. Under this engine's ``exp(+i omega t)`` convention the envelope of
    a band at ``f0`` becomes ``A(t - delay) * exp(-2 pi i f0 delay)``: the shape
    moves later, and the carrier picks up the phase that makes two paths of
    different length interfere the way they do in glass.

    **The window is periodic.** Sources draw a whole number of periods of their
    pattern, so the window repeats, and a delay moves the pattern round it: what
    leaves the end comes back in at the start. For a repeating PRBS that is what
    the hardware sees too. For a single pulse it means a delay longer than the
    window wraps -- choose a window longer than the longest delay that matters.

    **Whole samples or not.** A delay that is a whole number of samples is a
    shift, and exact. Any other delay is applied as a linear phase in frequency,
    which is exact for a band-limited field and rings on one with edges sharper
    than the sampling resolves -- an ideal NRZ transition, for instance.

    **What it is not.** It has no loss and no dispersion: put a ``Fiber`` or an
    ``Attenuator`` beside it for those. Noise passes unchanged, because a delay
    moves when noise arrives and not how much of it there is.

    With a :class:`~maiman.components.Feedback` on the same loop, each pass is a
    lap that arrives ``delay`` after the one before, which is what turns a pass
    count into a recirculating loop.
    """

    display_name = "Delay Line"
    category = "Passive"

    delay = Param(
        0.0, unit="ps", min=0.0, doc="Time the field is delayed by, carrier phase included"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal = inputs["in"]
        if not isinstance(signal, OpticalSignal):
            raise TypeError(
                f"{self.label}: a delay line carries an optical signal, got {type(signal).__name__}"
            )
        return {"out": delay_signal(signal, self.si("delay"))}


def delay_signal(signal: OpticalSignal, delay: float) -> OpticalSignal:
    """Every band of ``signal`` delayed by ``delay`` [s]; noise and history unchanged."""
    if delay < 0.0:
        raise ValueError(f"a delay line cannot advance a signal, got a delay of {delay} s")
    if delay == 0.0:
        return signal
    return OpticalSignal(
        bands=tuple(delay_band(band, delay) for band in signal.bands),
        noise=signal.noise,
        accumulated_gvd=signal.accumulated_gvd,
    )


def delay_band(band: Band, delay: float) -> Band:
    """One band delayed by ``delay`` [s], carrier phase included.

    The carrier's phase is taken modulo one cycle before it is exponentiated:
    ``f0 * delay`` is some hundred thousand cycles for a nanosecond at 1550 nm,
    and only its fractional part is physics.
    """
    carrier = np.exp(-2j * np.pi * math.fmod(band.f0 * delay, 1.0))
    shift = delay * band.fs
    whole = round(shift)

    if abs(shift - whole) <= WHOLE_SAMPLE_TOLERANCE * max(1.0, abs(shift)):
        ex = np.roll(band.Ex, whole) * carrier
        ey = np.roll(band.Ey, whole) * carrier
    else:
        frequencies = np.fft.fftfreq(band.num_samples, d=1.0 / band.fs)
        ramp = np.exp(-2j * np.pi * frequencies * delay) * carrier
        ex = np.fft.ifft(np.fft.fft(band.Ex.astype(np.complex128)) * ramp)
        ey = np.fft.ifft(np.fft.fft(band.Ey.astype(np.complex128)) * ramp)

    return Band(
        Ex=ex.astype(band.Ex.dtype),
        Ey=ey.astype(band.Ey.dtype),
        f0=band.f0,
        fs=band.fs,
    )
