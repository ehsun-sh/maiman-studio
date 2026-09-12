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

from typing import Any

from ..component import Component, Param, PortType
from ..context import SimulationContext
from ..grid import channel_frequencies
from ..signals import Band, NoiseBin, OpticalSignal, Signal, joined_accumulated_gvd
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
        doc="Channel 0. The rest are stepped from it in frequency",
    )
    spacing = Param(
        100.0, unit="GHz", min=1.0, doc="Channel spacing, in frequency because the grid is"
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
        """Centre of every channel [Hz], ascending from ``first_wavelength``."""
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
            )
        }
