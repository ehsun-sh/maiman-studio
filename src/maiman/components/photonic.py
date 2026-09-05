"""Integrated-photonic blocks: waveguide, directional coupler, ring resonator.

**How a circuit joins a link.** The top-level simulation is dataflow — every
block is a pure function of the waveform on its inputs — and a photonic circuit
is not, because light in a ring reaches steady state rather than passing through.
The two meet at a scattering matrix. :mod:`maiman.circuit` solves the circuit
exactly, in the frequency domain, at whatever frequencies it is asked about; the
blocks here ask about the frequencies the incoming band actually occupies and
multiply its spectrum by the answer. Which is to say a solved PIC is a transfer
function, and this library already knows what to do with one of those.

That is the whole architecture: the circuit stays a circuit, the link stays a
link, and neither has to know how the other is executed.

**One response for both polarizations.** These are strip-waveguide models, and a
strip waveguide is birefringent enough that TE and TM see different indices
entirely — a real ring resonates at two sets of wavelengths. Modelling that needs
a polarization-resolved scattering matrix, which is a second index on every
device, and it is not here. What is here applies the TE response to both axes,
so a device fed unpolarized light returns unpolarized light with the wrong answer
on one axis. Launch into one axis until that changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from ..circuit import SMatrix
from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..kernels import dispersion_to_beta2
from ..photonics import (
    SILICON_STRIP_NEFF,
    SILICON_STRIP_NGROUP,
    directional_coupler,
    free_spectral_range,
    resonance_linewidth,
    ring_resonator,
    straight_waveguide,
)
from ..signals import Band, NoiseBin, OpticalSignal, Signal, joined_accumulated_gvd
from ..units import C_LIGHT, frequency_to_wavelength, wavelength_to_frequency

#: Fewest points a noise bin is averaged over. A bin is flat and the response is
#: not, so what survives is the *mean* power transmission across it, and that mean
#: has to be integrated rather than read at a point.
MIN_NOISE_POINTS = 4096

#: Most. At one point per 8 kHz of a 700 GHz period this resolves a resonance
#: down to about 90 MHz — a loaded Q of two million, which is a very good ring —
#: and costs 0.35 s. Past that the integral is quietly under-resolved rather than
#: wrong by a little, so the ceiling is stated here instead of being discovered.
MAX_NOISE_POINTS = 65536

#: Samples across the narrowest feature. Measured: the mean is converged to four
#: digits at 2.7 and unchanged from 5 upward, so 8 is comfortable rather than
#: hopeful.
POINTS_PER_LINEWIDTH = 8


def average_power_response(
    bin_: NoiseBin,
    response: Callable[[np.ndarray], np.ndarray],
    *,
    period: float | None = None,
    resolution: float | None = None,
) -> float:
    """Mean power transmission across one noise bin.

    **Reading a resonator at one frequency would be worse than useless.** ASE
    spans terahertz and a ring's comb repeats every few hundred gigahertz, so the
    value at the bin's centre is whatever that frequency happened to land on — a
    resonance or the flat between two — and the answer swings by the full
    extinction depending on a parameter nobody thought was involved. What an
    amplifier's noise meets is the average.

    Two things make that average trustworthy, and both are about *resolution*.

    ``period`` shrinks the span that has to be covered. The mean of a periodic
    response over many periods is its mean over one, so a comb is integrated over
    a single free spectral range at the bin's centre — which, at a fixed budget
    of points, is as many times finer as the bin was periods wide.

    ``resolution`` sets the budget. It is the narrowest feature the response has
    — a resonance's linewidth — and without it a fixed point count silently fails
    on a high-Q device. A ring 116 MHz wide inside a 714 GHz period, averaged at
    4096 points, steps 174 MHz at a time and strides over its own resonance:
    against a converged 2.44e-4 it returns 3.13e-4 when the window happens to sit
    on the resonance and 2.14e-4 when it sits on a real amplifier's band instead.
    **Which way it is wrong depends on where the bin was**, which is what makes an
    under-resolved integral worse than a merely inaccurate one. Sampling the whole
    4 THz bin at the same budget returns 4.81e-4, 97 % high. All three are
    measured in ``tests/test_photonics.py``.
    """
    span = bin_.f_end - bin_.f_start
    width = period if (period is not None and 0.0 < period < span) else span

    points = MIN_NOISE_POINTS
    if resolution is not None and resolution > 0.0:
        wanted = int(np.ceil(POINTS_PER_LINEWIDTH * width / resolution))
        points = int(np.clip(wanted, MIN_NOISE_POINTS, MAX_NOISE_POINTS))

    offsets = np.linspace(-0.5, 0.5, points, endpoint=False) * width
    centre = 0.5 * (bin_.f_start + bin_.f_end)
    return float(np.mean(np.abs(response(centre + offsets)) ** 2))


def apply_response(
    signal: OpticalSignal,
    response: Callable[[np.ndarray], np.ndarray],
    *,
    period: float | None = None,
    resolution: float | None = None,
    accumulated_gvd: float | None = None,
) -> OpticalSignal:
    """Multiply every band's spectrum by ``response`` and scale the noise by its mean power.

    ``response`` is called with *absolute* optical frequencies, so a device knows
    where in the spectrum it is being asked about — which is what makes a ring
    resonate at the wavelengths it should rather than at the ones the sampling
    grid happens to start at.
    """
    bands: list[Band] = []
    for band in signal.bands:
        absolute = band.f0 + np.fft.fftfreq(band.num_samples, d=1.0 / band.fs)
        transfer = response(absolute)
        bands.append(
            replace(
                band,
                Ex=np.fft.ifft(np.fft.fft(band.Ex.astype(np.complex128)) * transfer).astype(
                    band.Ex.dtype
                ),
                Ey=np.fft.ifft(np.fft.fft(band.Ey.astype(np.complex128)) * transfer).astype(
                    band.Ey.dtype
                ),
            )
        )

    noise: list[NoiseBin] = []
    for bin_ in signal.noise:
        factor = average_power_response(bin_, response, period=period, resolution=resolution)
        noise.append(replace(bin_, psd_x=bin_.psd_x * factor, psd_y=bin_.psd_y * factor))

    return OpticalSignal(
        bands=tuple(bands),
        noise=tuple(noise),
        accumulated_gvd=(signal.accumulated_gvd if accumulated_gvd is None else accumulated_gvd),
    )


def port_response(
    matrix_for: Callable[[np.ndarray], SMatrix], output: str, input_: str
) -> Callable[[np.ndarray], np.ndarray]:
    """The transfer function from one port of a solved circuit to another."""

    def response(frequencies: np.ndarray) -> np.ndarray:
        return matrix_for(frequencies).transmission(output, input_)

    return response


def solve_once(build: Callable[[np.ndarray], SMatrix]) -> Callable[[np.ndarray], SMatrix]:
    """Wrap a circuit builder so one grid is solved once, not once per output port.

    A block with two outputs asks its circuit the same question twice — the
    scattering matrix already contains every port's answer — and for a ring that
    question is a linear solve per frequency. Keyed on the grid's bytes rather
    than on its identity, because the caller builds a fresh array each time and
    an identity check would never hit.
    """
    cache: dict[bytes, SMatrix] = {}

    def solved(frequencies: np.ndarray) -> SMatrix:
        key = np.ascontiguousarray(frequencies, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = build(frequencies)
        return cache[key]

    return solved


class _Photonic(Component):
    """Shared parameters of the integrated blocks: the waveguide they are made of.

    Not a block itself. The three devices below all reduce to lengths of the same
    waveguide, so the indices, the loss and the dispersion are declared once here
    rather than three times with three chances to disagree.
    """

    abstract = True
    category = "Photonic IC"

    reference_wavelength = Param(
        1550.0, unit="nm", min=1200.0, max=1700.0, doc="Wavelength the indices are quoted at"
    )
    n_eff = Param(
        SILICON_STRIP_NEFF, unit="", min=1.0, max=5.0, doc="Effective index: sets the phase"
    )
    n_group = Param(
        SILICON_STRIP_NGROUP,
        unit="",
        min=1.0,
        max=10.0,
        doc="Group index: sets the delay, and a ring's free spectral range",
    )
    propagation_loss = Param(
        2.0, unit="dB/cm", min=0.0, doc="Waveguide loss; 2 dB/cm is a typical silicon strip"
    )
    dispersion = Param(
        0.0, unit="ps/nm/km", doc="Waveguide dispersion D; negligible below a millimetre"
    )

    def reference_frequency(self) -> float:
        """The frequency the indices are quoted at [Hz]."""
        return wavelength_to_frequency(self.si("reference_wavelength"))

    def _waveguide_kwargs(self) -> dict[str, Any]:
        return {
            "n_eff": self.n_eff,
            "n_group": self.n_group,
            "reference_frequency": self.reference_frequency(),
            "dispersion": self.si("dispersion"),
            "loss_db_per_m": self.si("propagation_loss"),
        }


class Waveguide(_Photonic):
    """A length of on-chip waveguide: delay, phase and loss.

    The building block everything else on a die is made of, and on its own it is
    a delay line. At a group index of 4.20 a millimetre of silicon holds 14 ps —
    which is why an optical buffer is a spiral occupying square millimetres and
    still measured in nanoseconds.

    **It is the one block here that touches** ``accumulated_gvd``, because it is
    the one with an unambiguous length. A resonator's group delay is a strong
    function of frequency and calling any single number "the dispersion it added"
    would be a fiction; a straight waveguide's ``beta2 * L`` is exactly what the
    four-wave-mixing bookkeeping means by the term.
    """

    display_name = "Waveguide"

    length = Param(1000.0, unit="um", min=0.0, doc="Physical length")

    inputs = {"in": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL}

    def group_delay(self) -> float:
        """Time to traverse the waveguide [s]."""
        return self.n_group * self.si("length") / C_LIGHT

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        length = self.si("length")
        settings = self._waveguide_kwargs()

        def response(frequencies: np.ndarray) -> np.ndarray:
            return straight_waveguide(frequencies, length=length, **settings).transmission(
                "out", "in"
            )

        beta2 = dispersion_to_beta2(
            self.si("dispersion"), frequency_to_wavelength(self.reference_frequency())
        )
        return {
            "out": apply_response(
                signal,
                response,
                accumulated_gvd=signal.accumulated_gvd + beta2 * length,
            )
        }


class DirectionalCoupler(_Photonic):
    """Two waveguides run close enough to trade power. The 2x2 splitter of a PIC.

    ``coupling`` is the *power* fraction that crosses over, so 0.5 is a 3 dB
    coupler and 0.0 is two waveguides passing without noticing each other. The
    cross path carries a quarter-wave of phase relative to the straight one,
    which is not a convention but the condition for the device to conserve
    energy; see :func:`maiman.photonics.directional_coupler`.

    The second input is off by default because most uses drive one. Turn it on
    (``DirectionalCoupler(second_input=True)``) to build an interferometer, where
    both inputs carrying something is the entire point.
    """

    display_name = "Directional Coupler"

    coupling = Param(0.5, unit="", min=0.0, max=1.0, doc="Power fraction crossing over")
    insertion_loss = Param(0.0, unit="dB", min=0.0, doc="Excess loss through the coupler")

    outputs = {"out1": PortType.OPTICAL, "out2": PortType.OPTICAL}

    def __init__(
        self, second_input: bool = False, *, label: str | None = None, **params: float
    ) -> None:
        super().__init__(label=label, **params)
        self.second_input = second_input
        self.inputs = {"in1": PortType.OPTICAL}
        if second_input:
            self.inputs["in2"] = PortType.OPTICAL

    def structural_config(self) -> dict[str, Any]:
        return {"second_input": self.second_input}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        matrix_for = self._matrix_factory()
        outputs: dict[str, Signal] = {}
        for out in ("out1", "out2"):
            contributions = [
                apply_response(inputs[in_], port_response(matrix_for, out, in_))
                for in_ in self.inputs
            ]
            outputs[out] = _sum_signals(contributions, where=f"{self.label}.{out}")
        return outputs

    def _matrix_factory(self) -> Callable[[np.ndarray], SMatrix]:
        coupling = self.coupling
        # Raw dB: si() on a dB parameter returns a linear power ratio, and the
        # model wants the decibels themselves.
        loss = self.insertion_loss
        return solve_once(
            lambda f: directional_coupler(f, coupling=coupling, insertion_loss_db=loss)
        )


class RingResonator(_Photonic):
    """A loop of waveguide coupled to a bus: the filter, the delay, the modulator.

    The device an integrated photonic circuit is mostly made of. On resonance the
    circulating field builds up until what leaks back out balances what comes in;
    off resonance the loop is dark. That makes it a notch filter on the through
    port, a band-pass on the drop port, a delay line with a group delay a
    finesse-fold longer than its length, and — because silicon's index moves with
    carrier density — the basis of a modulator that fits in ten micrometres.

    **Three numbers set the behaviour.** The circumference sets the free spectral
    range through the *group* index, ``c / (n_g * L)``: 714 GHz for a 100 um
    silicon ring. The round-trip loss and the coupling together set the depth and
    the width, and the case worth knowing is **critical coupling** — when the
    coupling exactly matches the round-trip loss, the through port goes to zero,
    because the field coupled back out of the ring cancels the field that stayed
    on the bus. A 100 um ring at 3 dB/cm needs kappa = 0.0069 for that, and the
    notch measures 66 dB deep on a grid fine enough to find its floor.

    Under-coupling and over-coupling give the *same* notch depth on either side
    of that point, so a measured extinction does not identify a ring; the drop
    port and the linewidth do.

    ``drop_coupling`` adds the second bus and turns the notch into a channel
    filter. The ``add`` input, which is what makes it an add-drop multiplexer
    rather than a drop filter, is off unless asked for
    (``RingResonator(add_port=True)``).
    """

    display_name = "Ring Resonator"

    length = Param(100.0, unit="um", min=0.1, doc="Round-trip circumference")
    coupling = Param(0.05, unit="", min=0.0, max=1.0, doc="Power coupling to the input bus")
    drop_coupling = Param(
        0.0, unit="", min=0.0, max=1.0, doc="Power coupling to the drop bus; 0 is an all-pass ring"
    )

    outputs = {"through": PortType.OPTICAL, "drop": PortType.OPTICAL}

    def __init__(
        self, add_port: bool = False, *, label: str | None = None, **params: float
    ) -> None:
        super().__init__(label=label, **params)
        self.add_port = add_port
        self.inputs = {"in": PortType.OPTICAL}
        if add_port:
            self.inputs["add"] = PortType.OPTICAL

    def structural_config(self) -> dict[str, Any]:
        return {"add_port": self.add_port}

    def free_spectral_range(self) -> float:
        """Spacing between resonances [Hz]."""
        return free_spectral_range(self.si("length"), self.n_group)

    def linewidth(self) -> float:
        """Full width at half depth of one resonance [Hz]."""
        return resonance_linewidth(
            self.si("length"),
            self.n_group,
            coupling=self.coupling,
            drop_coupling=self.drop_coupling,
            loss_db_per_m=self.si("propagation_loss"),
        )

    def finesse(self) -> float:
        """Free spectral range divided by linewidth: how many resonances fit in one."""
        width = self.linewidth()
        return float("inf") if width == 0.0 else self.free_spectral_range() / width

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        matrix_for = self._matrix_factory()
        # The comb repeats every free spectral range and its narrowest feature is
        # one resonance: the first says how far the noise average has to reach,
        # the second how finely. See `average_power_response`.
        period = self.free_spectral_range()
        resolution = self.linewidth()
        outputs: dict[str, Signal] = {}
        for out in ("through", "drop"):
            contributions = [
                apply_response(
                    inputs[in_],
                    port_response(matrix_for, out, in_),
                    period=period,
                    resolution=resolution,
                )
                for in_ in self.inputs
            ]
            outputs[out] = _sum_signals(contributions, where=f"{self.label}.{out}")
        return outputs

    def _matrix_factory(self) -> Callable[[np.ndarray], SMatrix]:
        settings = self._waveguide_kwargs()
        length = self.si("length")
        coupling, drop = self.coupling, self.drop_coupling
        return solve_once(
            lambda f: ring_resonator(
                f, length=length, coupling=coupling, drop_coupling=drop, **settings
            )
        )


def _sum_signals(signals: list[OpticalSignal], *, where: str) -> OpticalSignal:
    """Add two paths that arrive at the same port.

    **Bands add as fields and noise adds as power**, which is the same rule the
    rest of the engine follows and is not a detail: two coherent paths through an
    interferometer cancel, two independent spontaneous-emission densities do not.
    Adding the noise as fields would make a Mach-Zehnder appear to null its own
    amplifier's ASE, which is exactly the sort of result that looks like a
    discovery.
    """
    if len(signals) == 1:
        return signals[0]

    by_centre: dict[float, Band] = {}
    for signal in signals:
        for band in signal.bands:
            existing = by_centre.get(band.f0)
            if existing is None:
                by_centre[band.f0] = band
                continue
            if existing.num_samples != band.num_samples or existing.fs != band.fs:
                raise ValueError(
                    f"{where}: two paths carry a band at {band.f0 / 1e12:.6f} THz on "
                    f"different grids ({existing.num_samples} at {existing.fs:g} Hz "
                    f"and {band.num_samples} at {band.fs:g} Hz); they cannot be added"
                )
            by_centre[band.f0] = replace(
                existing, Ex=existing.Ex + band.Ex, Ey=existing.Ey + band.Ey
            )

    noise: list[NoiseBin] = []
    for signal in signals:
        noise.extend(signal.noise)

    return OpticalSignal(
        bands=tuple(by_centre[f0] for f0 in sorted(by_centre)),
        noise=tuple(noise),
        accumulated_gvd=joined_accumulated_gvd(signals, where=where),
    )
