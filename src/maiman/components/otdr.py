"""A fibre that scatters some of its light back, which is what an OTDR reads.

Rayleigh scattering is most of a fibre's loss at 1550 nm: the glass is frozen
with density fluctuations far smaller than a wavelength, and each one sends a
little of the light off in every direction. A fraction of that lands back inside
the core travelling the wrong way, and it arrives at the launch end later the
further out it was scattered. Send a short pulse in and listen, and the time the
echo arrives is a distance along the fibre.

That is an optical time-domain reflectometer, and the reason it exists is the
last property: when the fibre is broken under a road, the echo stops at the
break, and the time it stopped says how far to dig.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..signals import Band, OpticalSignal, Signal
from ..units import C_LIGHT


def backscatter_response(
    times: np.ndarray,
    *,
    length: float,
    attenuation: float,
    scattering: float,
    capture: float,
    group_index: float,
    splice_at: float = 0.0,
    splice_loss_db: float = 0.0,
    end_reflectance: float = 0.0,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """The power returned for each joule launched, against the time it returns.

    ``times`` [s] are measured from the launch. Light scattered at ``z`` is back
    after ``t = 2 z n_g / c``, and on the way out and back it has lost
    ``exp(-2 alpha z)``; a slice ``dz`` scatters ``alpha_s dz`` of it, of which
    ``capture`` is guided home. Per second of return time that is

        h(t) = capture * alpha_s * (v_g / 2) * exp(-2 alpha z)        [1/s]

    so a pulse of energy ``E`` much shorter than the fibre returns ``E h(t)``
    watts. ``attenuation`` and ``scattering`` are power coefficients [1/m]. A
    splice ``splice_at`` metres out costs its loss twice, once each way.

    The fibre ends at ``length``. If ``end_reflectance`` is not zero the end is a
    mirror, and that is returned separately as ``(time, fraction)``: it is not a
    density but one echo, the launched pulse back again, ``fraction`` as strong.
    """
    speed = C_LIGHT / group_index
    z = speed * times / 2.0
    inside = (z >= 0.0) & (z < length)
    splice = 10.0 ** (-2.0 * splice_loss_db / 10.0)
    beyond_splice = (splice_at > 0.0) & (z >= splice_at)
    density = capture * scattering * speed / 2.0 * np.exp(-2.0 * attenuation * z)
    density = np.where(beyond_splice, density * splice, density)
    density = np.where(inside, density, 0.0)
    echo = None
    if end_reflectance > 0.0:
        survived = math.exp(-2.0 * attenuation * length)
        if 0.0 < splice_at < length:
            survived *= splice
        echo = (2.0 * length / speed, end_reflectance * survived)
    return density, echo


class BackscatterFiber(Component):
    """A fibre seen from its launch end: what comes back, against time.

    **What it models.** Rayleigh backscatter as a continuum, a fusion splice as a
    step, and the end of the glass as a Fresnel reflection, each one an echo of
    whatever was launched. The input is the probe; ``backscatter`` is the light
    returning to the launch end, with the moment of launch at the start of the
    window, so the trace's time from the start is the round trip. ``out`` is what
    reaches the far end: nothing, if it is broken.

    A pulse is launched where the input is brightest; its whole envelope is then
    convolved with the fibre's response, so a long pulse smears the trace and
    hides a splice the way it does on hardware, ``v_g tau / 2`` wide.

    **What it does not.** The echo is incoherent: real Rayleigh backscatter from
    a coherent source is speckle, which an OTDR averages away with a broad source
    and many shots, and this is that average. There is no detector, so no noise
    floor and no dynamic range: the trace falls forever, where a real one sinks
    into noise. No dispersion and no nonlinearity; use a ``Fiber`` for those.
    The trace is as long as the simulated window and no longer: a fibre whose
    round trip does not fit is drawn up to the window's edge, and nothing past it
    is shown. Lengthen the sequence or lower the symbol rate to see further.
    """

    display_name = "Fiber with Backscatter"
    category = "Fiber"

    length = Param(25.0, unit="km", min=0.0, max=500.0, doc="Length of glass the light can reach")
    attenuation = Param(0.2, unit="dB/km", min=0.0, max=10.0, doc="Total loss, one way")
    scattering = Param(
        0.17,
        unit="dB/km",
        min=0.0,
        max=10.0,
        doc="The part of the loss that is Rayleigh scattering",
    )
    capture = Param(
        0.0017,
        unit="",
        min=0.0,
        max=1.0,
        doc="Fraction of the scattered light the core guides back: (NA / n)^2 / 4.55",
    )
    group_index = Param(1.468, unit="", min=1.0, max=4.0, doc="Sets how far a microsecond is")
    splice_at = Param(0.0, unit="km", min=0.0, max=500.0, doc="Where a splice sits; 0 for none")
    splice_loss = Param(0.0, unit="dB", min=0.0, max=20.0, doc="Loss of the splice, one way")
    end_reflectance = Param(
        -14.7,
        unit="dB",
        max=0.0,
        doc="Fresnel reflection where the glass ends: -14.7 dB is a clean cleave in air",
    )
    break_at = Param(
        0.0, unit="km", min=0.0, max=500.0, doc="Where the fibre is broken; 0 if it is whole"
    )
    break_reflectance = Param(
        -30.0,
        unit="dB",
        max=0.0,
        doc="What a broken face reflects, if `break_at` is set: rougher than a cleave",
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"backscatter": PortType.OPTICAL, "out": PortType.OPTICAL}

    def broken(self) -> bool:
        return 0.0 < self.break_at < self.length

    def reach(self) -> float:
        """Where the glass ends [m]: the break, if there is one."""
        return self.si("break_at") if self.broken() else self.si("length")

    def round_trip(self) -> float:
        """Time for light to reach the end and come back [s]."""
        return 2.0 * self.reach() * self.group_index / C_LIGHT

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        n = ctx.num_samples
        bands = [b for b in signal.bands if b.num_samples == n]
        if not bands:
            raise ValueError(f"{self.label}: no band on the simulation's time grid to launch")
        dt = 1.0 / ctx.sample_rate

        power = np.zeros(n)
        for band in bands:
            power += np.abs(np.asarray(band.Ex)) ** 2 + np.abs(np.asarray(band.Ey)) ** 2
        # The pulse in the middle of a buffer, so its leading edge -- earlier
        # than the launch, which is its peak -- has room before it.
        middle = n // 2
        centred = np.roll(power, middle - int(np.argmax(power)))

        to_per_m = math.log(10.0) / 10.0 / 1e3
        density, echo = backscatter_response(
            np.arange(n) * dt,
            length=self.reach(),
            attenuation=self.attenuation * to_per_m,
            scattering=self.scattering * to_per_m,
            capture=self.capture,
            group_index=self.group_index,
            splice_at=self.si("splice_at"),
            splice_loss_db=self.splice_loss,
            end_reflectance=10.0
            ** ((self.break_reflectance if self.broken() else self.end_reflectance) / 10.0),
        )
        kernel = density * dt
        if echo is not None and round(echo[0] / dt) < n:
            kernel[round(echo[0] / dt)] += echo[1]
        # A linear convolution, padded so nothing wraps; sample ``middle + k`` of
        # it is ``k`` samples after the launch.
        returned = np.fft.irfft(np.fft.rfft(centred, 2 * n) * np.fft.rfft(kernel, 2 * n), 2 * n)
        returned = np.clip(returned[middle : middle + n], 0.0, None)

        brightest = max(bands, key=lambda b: b.average_power())
        back = Band(
            Ex=np.sqrt(returned).astype(ctx.complex_dtype),
            Ey=np.zeros(n, dtype=ctx.complex_dtype),
            f0=brightest.f0,
            fs=brightest.fs,
        )

        # A broken fibre delivers nothing; a whole one its loss and the splice's.
        spliced = self.splice_loss if 0.0 < self.splice_at < self.length else 0.0
        through = (
            0.0 if self.broken() else 10.0 ** (-(self.attenuation * self.length + spliced) / 20.0)
        )
        forward = tuple(
            Band(Ex=b.Ex * through, Ey=b.Ey * through, f0=b.f0, fs=b.fs) for b in signal.bands
        )
        # Whatever the light carried in -- its delays, its Kerr and mixing
        # histories -- it carries back out, both ways.
        return {
            "backscatter": replace(signal, bands=(back,)),
            "out": replace(signal, bands=forward),
        }
