"""Filtering, electrical and optical."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..component import BoolParam, Component, Param, PortType
from ..context import SimulationContext
from ..kernels import (
    gaussian_noise_bandwidth,
    lowpass_filter,
    super_gaussian_noise_bandwidth,
    super_gaussian_response,
)
from ..signals import ElectricalSignal, NoiseBin, OpticalSignal, OpticalSpectrum, Signal
from ..units import db_to_linear, wavelength_to_frequency


class ElectricalFilter(Component):
    """Gaussian low-pass filter for a received waveform.

    Every real receiver band-limits before deciding, and without it a simulation
    is not merely optimistic but meaningless: the noise a detector adds spans the
    whole simulated bandwidth ``fs/2``, which is set by the oversampling factor
    rather than by anything physical. Filtering to a receiver bandwidth is what
    makes a Q-factor or a BER correspond to a real link.

    Standard practice is a fourth-order Bessel at roughly 0.7 times the symbol
    rate. The Gaussian shape used here is close in effect and has a closed-form
    noise bandwidth, ``B_n = B * sqrt(pi / 4 ln2) ~ 1.0645 * B``, which makes the
    effect on noise checkable exactly rather than approximately.

    The filter is zero-phase, so there is no group delay to compensate; see
    :func:`maiman.kernels.lowpass_filter`.
    """

    display_name = "Electrical Filter"
    category = "Electrical"

    bandwidth = Param(7.0, unit="GHz", min=0.0, doc="3 dB power bandwidth")

    inputs = {"in": PortType.ELECTRICAL}
    outputs = {"out": PortType.ELECTRICAL}

    def noise_bandwidth(self) -> float:
        """Equivalent noise bandwidth [Hz]."""
        return gaussian_noise_bandwidth(self.si("bandwidth"))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        waveform: ElectricalSignal = inputs["in"]
        filtered = lowpass_filter(waveform.samples, waveform.fs, self.si("bandwidth"))
        return {
            "out": ElectricalSignal(
                samples=filtered.astype(ctx.real_dtype), fs=waveform.fs, unit=waveform.unit
            )
        }


class OpticalFilter(Component):
    """Wavelength-selective band-pass: the demultiplexer, and the ASE gate.

    Two jobs, and the second is the one that surprises people.

    **Selecting a channel.** Bands whose centre falls outside the passband are
    suppressed, which is what turns a multi-carrier signal back into one channel.
    Because every band carries its own centre frequency this is a real
    wavelength-selective operation rather than an index lookup: a filter tuned
    between two channels attenuates both, and one tuned slightly off centre clips
    the edge of the spectrum it is trying to pass. Crosstalk from a neighbour is
    therefore a number this model produces rather than an assumption it makes.

    **Gating the ASE.** An amplifier emits spontaneous emission across its whole
    gain band — 4 THz by default — and every hertz of it reaches the photodiode
    and beats there unless something stops it. On an eight-span link that costs
    about a factor of three in Q against a filtered receiver, and no receiver is
    built without one. This block is what makes that a property of a component in
    the graph rather than a parameter tucked inside the detector.

    **How the noise bins are cut.** The passband's *equivalent noise bandwidth*
    is used rather than its shape: what survives keeps the peak transmission as
    its density and is clipped to ``B_n`` wide. That makes both quantities a
    detector asks for exact — the density at the signal's own frequency, which
    sets signal-spontaneous beating, and the integrated power, which sets the
    spontaneous-spontaneous term — while treating the passband as flat. It is the
    same idealisation :class:`ElectricalFilter` already makes, for the same
    reason: it can be checked by arithmetic instead of eyeballed.

    ``order`` selects the shape. 1 is a Gaussian, the response of a thin-film
    filter; 3 to 5 approximates the flat top of a wavelength-selective switch.
    The 3 dB width means the same thing at every order — only the skirts change.

    ``extinction`` floors the response, because a super-Gaussian's skirts fall
    away without limit and real hardware's do not. At order 3 a 50 GHz passband is
    exp(-2838) one channel spacing away, which underflows to exactly zero and
    makes a neighbour's rejection report as infinite. A wavelength-selective
    switch specifies 30 to 50 dB, and in a chain of them it is that floor rather
    than the skirt that sets the crosstalk which accumulates.
    """

    display_name = "Optical Filter"
    category = "Passive"

    center_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Centre of the passband"
    )
    bandwidth = Param(50.0, unit="GHz", min=0.0, doc="3 dB full width")
    order = Param(3.0, unit="", min=1.0, max=10.0, doc="Super-Gaussian order; 1 is Gaussian")
    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Loss at peak transmission")
    extinction = Param(
        40.0, unit="dB", min=0.0, doc="Out-of-band rejection floor; 0 disables the floor"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def center_frequency(self) -> float:
        """Centre of the passband [Hz]."""
        return wavelength_to_frequency(self.si("center_wavelength"))

    def noise_bandwidth(self) -> float:
        """Equivalent noise bandwidth of the passband [Hz]."""
        return super_gaussian_noise_bandwidth(self.si("bandwidth"), int(self.order))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        centre = self.center_frequency()
        width = self.si("bandwidth")
        order = int(self.order)
        # Raw dB, not si(): the unit machinery already turns a dB parameter into a
        # linear ratio, so converting again turns a declared 0 dB into 0.79x. It did.
        transmission = db_to_linear(-self.insertion_loss)

        # A super-Gaussian's skirts fall away without limit, which no real filter
        # does: a third-order 50 GHz passband is exp(-2838) at 100 GHz off centre,
        # so a neighbouring channel does not merely become small but underflows to
        # exactly zero and its rejection reports as infinite. Real hardware
        # specifies a finite floor — 30 to 50 dB for a wavelength-selective switch
        # — and it matters, because in a long chain the crosstalk that accumulates
        # is the floor rather than the skirt.
        floor = db_to_linear(-self.extinction) if self.extinction > 0.0 else 0.0

        bands = []
        for band in signal.bands:
            # Each band is filtered on its own grid, offset by how far its centre
            # sits from the filter's, so a band far outside the passband is
            # attenuated by the same response rather than special-cased away.
            offsets = np.fft.fftfreq(band.Ex.shape[0], d=1.0 / band.fs)
            shape = super_gaussian_response(offsets + (band.f0 - centre), width, order) ** 2
            response = np.sqrt(np.maximum(shape, floor))
            response = response * np.sqrt(transmission)
            filtered_x = np.fft.ifft(np.fft.fft(band.Ex.astype(np.complex128)) * response)
            filtered_y = np.fft.ifft(np.fft.fft(band.Ey.astype(np.complex128)) * response)
            bands.append(
                replace(
                    band,
                    Ex=filtered_x.astype(band.Ex.dtype),
                    Ey=filtered_y.astype(band.Ey.dtype),
                )
            )

        noise_bandwidth = self.noise_bandwidth()
        low = centre - noise_bandwidth / 2.0
        high = centre + noise_bandwidth / 2.0
        noise = []
        for bin_ in signal.noise:
            start, end = max(bin_.f_start, low), min(bin_.f_end, high)
            if end <= start:
                continue  # entirely outside the passband
            noise.append(
                NoiseBin(
                    f_start=start,
                    f_end=end,
                    psd_x=bin_.psd_x * transmission,
                    psd_y=bin_.psd_y * transmission,
                )
            )

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(noise),
                accumulated_gvd=signal.accumulated_gvd,
            )
        }


class OpticalSpectrumAnalyzer(Component):
    """Power spectral density against wavelength — the OSA.

    The signal model has carried independently sampled bands plus spectral noise
    bins since the first commit, and until now nothing could look at that
    structure directly. A power meter integrates it to one number and an OSNR
    meter to a ratio; this shows the shape, which is how anyone actually finds a
    tilted amplifier, a misaligned filter, or a channel quietly squeezed by
    passing through six of them.

    Bands and noise bins are rendered onto one grid, because that is what an
    instrument shows: it cannot tell you which parts of its own trace the
    simulator happened to sample. Each band's periodogram is placed at its own
    centre frequency and each noise bin contributes its density across its span.

    ``resolution_bandwidth`` is the instrument's, and it is not cosmetic. An OSA
    reports power *per resolution bandwidth*, so widening it raises the ASE trace
    while leaving the signal — narrower than either setting — where it is. That
    asymmetry is the whole reason OSNR has to be quoted in a stated reference
    bandwidth, and watching the two respond differently to one knob is the
    clearest demonstration of it available.

    **Full span is the default, because an instrument you have to aim is no use
    for finding something.** A real OSA powers up sweeping its whole range; you
    narrow it once you know where to look. This one used to start at 1550 nm
    with a 1000 GHz window and drop everything outside — so a laser at 1560 nm
    produced an empty trace, and the only way to see it was to already know its
    wavelength and type it in here as well. That is backwards: the wavelength is
    usually the thing being measured.

    With ``auto_span`` on, the window is whatever the signal actually occupies —
    every band's sampled bandwidth and every noise bin, plus a small margin —
    so the trace shows what is there rather than what you guessed. Turn it off
    and ``center_wavelength`` and ``span`` mean what they always did, which is
    what you want once you are looking at a known channel and care about
    resolution rather than coverage.

    Full span is not free and the instrument says so rather than hiding it:
    ``points`` is fixed, so a wider window is a coarser trace. That is true of a
    real OSA at full span too, and it is why the narrow mode still exists.
    """

    display_name = "Optical Spectrum Analyzer"
    category = "Measurements"

    auto_span = BoolParam(
        True, doc="Sweep whatever the signal occupies, instead of a window you set"
    )
    center_wavelength = Param(
        1550.0,
        unit="nm",
        min=1200.0,
        max=1700.0,
        doc="Centre of the span, when not automatic",
        applies_when="!auto_span",
    )
    span = Param(
        1000.0,
        unit="GHz",
        min=1.0,
        doc="Displayed frequency span, when not automatic",
        applies_when="!auto_span",
    )
    points = Param(1024.0, unit="", min=16.0, max=16384.0, doc="Trace points")
    resolution_bandwidth = Param(
        12.5, unit="GHz", min=0.001, doc="Instrument resolution; 12.5 GHz is 0.1 nm at 1550 nm"
    )

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.METRIC}

    #: Margin left either side of the occupied range in automatic mode, as a
    #: fraction of it. A trace that ends exactly on the outermost sample looks
    #: like one that was cut off there.
    AUTO_MARGIN = 0.05

    def occupied_range(self, signal: OpticalSignal) -> tuple[float, float] | None:
        """Lowest and highest frequency the signal has anything at [Hz].

        Everything the instrument could possibly see: each band covers its own
        sample rate about its centre frequency, and each noise bin its declared
        extent. ``None`` when there is nothing at all to look at, which is a
        real case — an unwired analyser, or one downstream of a block that
        emitted an empty signal — and the declared window is the only sensible
        answer to it.
        """
        low, high = float("inf"), float("-inf")
        for band in signal.bands:
            low = min(low, band.f0 - band.fs / 2.0)
            high = max(high, band.f0 + band.fs / 2.0)
        for bin_ in signal.noise:
            low = min(low, bin_.f_start)
            high = max(high, bin_.f_end)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None
        return low, high

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        points = int(self.points)

        window = self.occupied_range(signal) if self.auto_span else None
        if window is None:
            centre = wavelength_to_frequency(self.si("center_wavelength"))
            span = self.si("span")
            start, stop = centre - span / 2.0, centre + span / 2.0
        else:
            low, high = window
            margin = (high - low) * self.AUTO_MARGIN
            start, stop = low - margin, high + margin
            span = stop - start

        frequencies = np.linspace(start, stop, points)
        power = np.zeros(points, dtype=np.float64)
        step = (stop - start) / max(points - 1, 1)

        for band in signal.bands:
            # Periodogram normalised so that summing it returns the band's average
            # power. Accumulated as power *per display bin* rather than as a
            # density, because a carrier and a noise floor have to respond to the
            # resolution setting differently and a density cannot express that.
            length = band.Ex.shape[0]
            spectrum = (
                np.abs(np.fft.fft(band.Ex.astype(np.complex128))) ** 2
                + np.abs(np.fft.fft(band.Ey.astype(np.complex128))) ** 2
            ) / float(length) ** 2
            absolute = band.f0 + np.fft.fftfreq(length, d=1.0 / band.fs)
            index = np.rint((absolute - frequencies[0]) / step).astype(int)
            inside = (index >= 0) & (index < points)
            np.add.at(power, index[inside], spectrum[inside])

        for bin_ in signal.noise:
            # A bin is a flat density, so each display cell collects its own share.
            covered = (frequencies >= bin_.f_start) & (frequencies < bin_.f_end)
            power[covered] += (bin_.psd_x + bin_.psd_y) * step

        return {
            "out": OpticalSpectrum(
                frequencies=frequencies,
                power_w=power,
                resolution_bandwidth=self.si("resolution_bandwidth"),
            )
        }
