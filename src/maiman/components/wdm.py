"""The two ends of a wavelength-division link: a multiplexer and a demultiplexer.

The roadmap has said "DWDM MUX/DEMUX with crosstalk" since Phase 3 and what was
actually here was the pieces: :class:`~maiman.components.Combiner` merges bands
and says in its own docstring that it is a WDM multiplexer, and
:class:`~maiman.components.OpticalFilter` says in its first line that it is the
demultiplexer. Both are true, and building an eight-channel demux out of them
takes a splitter and eight filters whose centres somebody has to work out.

These are that, in one block, on a grid.

**What they add over wiring it by hand is not convenience.** It is that the
channel plan comes from :mod:`maiman.grid` rather than from arithmetic repeated
per project, and that the crosstalk between neighbours is then a number the model
produces — the filter skirts and the extinction floor at the actual channel
spacing — rather than something each project reinvents a different way.

**They are routers, not splitters, and that is the modelling choice worth
stating.** A broadcast-and-select demux really does divide power ``N`` ways and
costs ``10 log10(N)``: 9 dB at eight channels, 15 at thirty-two, which is why
nobody builds a large one that way. An arrayed-waveguide grating *routes* instead
— each channel leaves by its own port, and the loss is 3 to 5 dB whatever ``N``
is. The second is what a line system uses, so it is what ``insertion_loss``
means here, and it does not grow with the channel count. Model the other by
putting a :class:`~maiman.components.Splitter` in front of filters, which is
exactly what that device is.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..grid import channel_frequencies
from ..signals import (
    Band,
    NoiseBin,
    OpticalSignal,
    Signal,
    joined_accumulated_gvd,
    joined_nonlinear_history,
    joined_walkoff,
)
from ..units import frequency_to_wavelength, wavelength_to_frequency
from .filters import apply_passband


class _Grid(Component):
    """The channel plan both devices share, declared once.

    Not a block itself. The multiplexer and the demultiplexer are the same grid
    facing opposite directions, and two copies of "where is channel k" is two
    chances for a mux and the demux at the far end of one link to disagree about
    what they are carrying.
    """

    abstract = True
    category = "Passive"

    #: Set by each subclass in ``__init__``, because the port count is what it
    #: decides and the grid is what it is a count of.
    channels: int = 0

    first_wavelength = Param(
        1550.0,
        unit="nm",
        min=1200.0,
        max=1700.0,
        doc="Channel 0. The rest are stepped from it up in frequency, down in wavelength",
    )
    spacing = Param(
        100.0, unit="GHz", min=1.0, doc="Channel spacing, in frequency because the grid is"
    )
    wavelength_spacing = Param(
        0.0,
        unit="nm",
        min=0.0,
        doc="A coarse grid's step in wavelength (CWDM: 20 nm), replacing `spacing`; 0 is off",
    )
    bandwidth = Param(50.0, unit="GHz", min=0.0, doc="3 dB full width of each channel's passband")
    order = Param(3.0, unit="", min=1.0, max=10.0, doc="Super-Gaussian order; 3 to 5 is an AWG")
    insertion_loss = Param(
        4.0,
        unit="dB",
        min=0.0,
        doc="Loss per channel. An AWG routes, so this does not grow with the count",
    )
    extinction = Param(
        40.0, unit="dB", min=0.0, doc="Out-of-band rejection floor; 0 disables the floor"
    )

    def channel_frequencies(self) -> tuple[float, ...]:
        """Centre of every channel [Hz], ascending from ``first_wavelength``.

        DWDM is uniform in frequency and CWDM in wavelength (see :mod:`maiman.grid`),
        so a coarse grid is stepped in nanometres when ``wavelength_spacing`` is
        set: channel ``k`` sits at ``first_wavelength - k * wavelength_spacing``,
        still ascending in frequency. A fixed frequency step cannot land on the
        coarse grid: 3550 GHz from 1331 nm puts the next channel at 1310.35 nm,
        0.65 nm off its standard centre.
        """
        step = self.si("wavelength_spacing")
        if step > 0.0:
            first = self.si("first_wavelength")
            wavelengths = [first - index * step for index in range(self.channels)]
            if wavelengths[-1] <= 0.0:
                raise ValueError(
                    f"{self.label}: {self.channels} channels {self.wavelength_spacing} nm "
                    f"apart run below zero wavelength from {self.first_wavelength} nm"
                )
            return tuple(wavelength_to_frequency(w) for w in wavelengths)
        return channel_frequencies(
            self.channels,
            spacing=self.si("spacing"),
            anchor=wavelength_to_frequency(self.si("first_wavelength")),
        )

    def channel_wavelengths(self) -> tuple[float, ...]:
        """The same channels as wavelengths [m] — **descending**, as a frequency grid is."""
        return tuple(frequency_to_wavelength(f) for f in self.channel_frequencies())

    def _passband(self, signal: OpticalSignal, index: int) -> OpticalSignal:
        return apply_passband(
            signal,
            centre=self.channel_frequencies()[index],
            bandwidth=self.si("bandwidth"),
            order=int(self.order),
            insertion_loss_db=self.insertion_loss,
            extinction_db=self.extinction,
        )


class Demultiplexer(_Grid):
    """One fibre in, one channel out of each port. The receive end of a WDM link.

    Every output carries the same filter as :class:`~maiman.components.OpticalFilter`
    — literally the same function — tuned to its own channel. So a neighbour's
    leakage into a port is the super-Gaussian's skirt at the declared spacing,
    floored by ``extinction``, and it is a number this produces rather than an
    assumption it makes: narrow the spacing and the crosstalk rises on its own.

    A band that falls between two channels is attenuated by both and appears on
    both, which is the right answer and is what a misconfigured plan looks like
    from the receive end.
    """

    display_name = "Demultiplexer"

    inputs = {"in": PortType.OPTICAL}

    def __init__(self, channels: int = 4, *, label: str | None = None, **params: float) -> None:
        if channels < 1:
            raise ValueError(f"a demultiplexer needs at least one channel, got {channels}")
        super().__init__(label=label, **params)
        self.channels = channels
        self.outputs = {f"out{index}": PortType.OPTICAL for index in range(channels)}

    def structural_config(self) -> dict[str, Any]:
        return {"channels": self.channels}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        signal: OpticalSignal = inputs["in"]
        return {f"out{index}": self._passband(signal, index) for index in range(self.channels)}


class Multiplexer(_Grid):
    """One channel into each port, one fibre out. The transmit end, and a filter.

    The complement of the demultiplexer and **not** the same thing as
    :class:`~maiman.components.Combiner`, which merges whatever it is given. This
    filters each input to its own channel first, which is what a real mux does and
    is the reason a transmitter tuned to the wrong channel does not simply appear
    on the line: it lands on a port whose passband rejects it, and the rejection
    is the same number the demux at the far end would apply.

    Two inputs carrying a band at the same centre frequency are refused for the
    reason :class:`~maiman.components.Combiner` refuses them — co-located carriers
    interfere and have to be added as fields on a common grid, which is a
    different operation with different physics.
    """

    display_name = "Multiplexer"

    outputs = {"out": PortType.OPTICAL}

    def __init__(self, channels: int = 4, *, label: str | None = None, **params: float) -> None:
        if channels < 1:
            raise ValueError(f"a multiplexer needs at least one channel, got {channels}")
        super().__init__(label=label, **params)
        self.channels = channels
        self.inputs = {f"in{index}": PortType.OPTICAL for index in range(channels)}

    def structural_config(self) -> dict[str, Any]:
        return {"channels": self.channels}

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        filtered = [self._passband(inputs[f"in{index}"], index) for index in range(self.channels)]

        bands: list[Band] = []
        noise: list[NoiseBin] = []
        seen: dict[float, int] = {}
        for index, signal in enumerate(filtered):
            for band in signal.bands:
                if band.f0 in seen:
                    raise ValueError(
                        f"{self.label}: inputs 'in{seen[band.f0]}' and 'in{index}' both carry "
                        f"a band centred at {band.f0 / 1e12:.6f} THz. Co-located carriers must "
                        f"be added coherently on a common grid, not multiplexed."
                    )
                seen[band.f0] = index
                bands.append(band)
            noise.extend(signal.noise)

        return {
            "out": OpticalSignal(
                bands=tuple(bands),
                noise=tuple(noise),
                accumulated_gvd=joined_accumulated_gvd(filtered, where=self.label),
                walkoff=joined_walkoff(filtered, where=self.label),
                nonlinear_history=joined_nonlinear_history(filtered, where=self.label),
            )
        }


class WavelengthSelectiveSwitch(_Grid):
    """A ROADM's add/drop degree: one channel off the line, a new one on in its place.

    ``in`` is the line arriving at the node, ``add`` the local transmitter. Every
    channel of the line but ``dropped`` leaves by ``out`` through its own
    passband (express); the ``dropped`` channel leaves by ``drop`` through its
    passband, and whatever arrives on ``add`` in that slot takes its place on
    ``out``.

    **The switch does not block perfectly.** ``isolation`` is how far below its
    arrival the dropped channel still reaches ``out``, and there it lands on the
    added channel's own frequency: in-band crosstalk, which no filter
    downstream can remove. The two are added as fields, so the leak beats with
    the new channel at the receiver -- the reason a node's isolation is
    specified at 35 dB or more, far beyond a demultiplexer's neighbour
    rejection.

    The passband of every port is the demultiplexer's, on the same grid, so a
    channel off the plan is attenuated by the skirt rather than special-cased.
    """

    display_name = "Wavelength-Selective Switch"

    inputs = {"in": PortType.OPTICAL, "add": PortType.OPTICAL}
    outputs = {"out": PortType.OPTICAL, "drop": PortType.OPTICAL}

    dropped = Param(1.0, unit="", min=0.0, doc="The channel dropped and re-added, counted from 0")
    isolation = Param(
        35.0,
        unit="dB",
        min=0.0,
        doc="How far below its arrival the dropped channel still leaks to `out`",
    )

    def __init__(self, channels: int = 4, *, label: str | None = None, **params: float) -> None:
        if channels < 1:
            raise ValueError(f"a switch needs at least one channel, got {channels}")
        super().__init__(label=label, **params)
        self.channels = channels
        if int(self.dropped) != self.dropped or self.dropped >= channels:
            raise ValueError(
                f"{self.label}: dropped={self.dropped:g} is not one of channels 0 to {channels - 1}"
            )

    def structural_config(self) -> dict[str, Any]:
        return {"channels": self.channels}

    def _channel_of(self, f0: float) -> int:
        centres = self.channel_frequencies()
        return min(range(self.channels), key=lambda index: abs(centres[index] - f0))

    def run(self, ctx: SimulationContext, inputs: dict[str, Signal]) -> dict[str, Signal]:
        line: OpticalSignal = inputs["in"]
        add: OpticalSignal = inputs["add"]
        dropped = int(self.dropped)

        def only(signal: OpticalSignal, bands: list[Band], noise: bool) -> OpticalSignal:
            return replace(signal, bands=tuple(bands), noise=signal.noise if noise else ())

        # Express: each line channel through its own passband. The noise is cut
        # by every express passband in turn, so the dropped slot's ASE is gone.
        express: list[OpticalSignal] = []
        leaked: list[Band] = []
        for index in range(self.channels):
            if index == dropped:
                continue
            mine = [b for b in line.bands if self._channel_of(b.f0) == index]
            express.append(self._passband(only(line, mine, noise=True), index))
        leak = 10.0 ** (-self.isolation / 20.0)
        for band in line.bands:
            if self._channel_of(band.f0) == dropped:
                leaked.append(band.scale_amplitude(leak))

        added = self._passband(add, dropped)

        bands: dict[float, Band] = {}
        noise: list[NoiseBin] = []
        for signal in express:
            bands.update((b.f0, b) for b in signal.bands)
            noise.extend(signal.noise)
        noise.extend(added.noise)
        for band in (*leaked, *added.bands):
            here = bands.get(band.f0)
            if here is None:
                bands[band.f0] = band
            elif here.fs == band.fs and here.num_samples == band.num_samples:
                # The same carrier frequency by two paths: added as fields.
                bands[band.f0] = Band(
                    Ex=here.Ex + band.Ex, Ey=here.Ey + band.Ey, f0=band.f0, fs=band.fs
                )
            else:
                raise ValueError(
                    f"{self.label}: two carriers at {band.f0 / 1e12:.6f} THz on different "
                    f"sample grids cannot be added as fields"
                )

        joined = [line, add]
        return {
            "out": OpticalSignal(
                bands=tuple(sorted(bands.values(), key=lambda b: b.f0)),
                noise=tuple(noise),
                accumulated_gvd=joined_accumulated_gvd(joined, where=self.label),
                walkoff=joined_walkoff(joined, where=self.label),
                nonlinear_history=joined_nonlinear_history(joined, where=self.label),
            ),
            "drop": self._passband(line, dropped),
        }
