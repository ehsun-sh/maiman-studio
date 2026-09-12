"""The two ITU grids, and a channel plan that produces its own crosstalk.

The roadmap has said "DWDM MUX/DEMUX with crosstalk" since Phase 3, and what was
actually here was the pieces — a combiner whose docstring calls it a multiplexer,
a filter whose first line calls it a demultiplexer. Building an eight-channel
demux out of those takes a splitter and eight filters whose centres somebody has
to work out, and *that arithmetic* was the missing part.

Four things are worth reading off the output.

**DWDM is a frequency grid and CWDM is a wavelength grid**, and that is the whole
difference between them. It has a consequence people trip over: the dense grid's
channels are not evenly spaced in wavelength, so building a "100 GHz plan" by
stepping 0.8 nm drifts off ITU by most of a channel across the C band.

**The 20 nm of the coarse grid is a specification, not a round number.** An
uncooled DFB walks about 0.1 nm per kelvin, so 70 kelvin of case temperature is
7 nm of drift before manufacturing spread. 20 nm spacing is what makes a laser
with no cooler, no wavelength locker and no control loop a legal channel — which
is why CWDM optics cost a fraction of DWDM optics.

**The crosstalk is produced, not declared.** Each demultiplexer port is the same
super-Gaussian the single-filter block is — literally the same function — so
narrowing the spacing raises the neighbour's leakage on its own.

**And an arrayed-waveguide grating routes rather than splits.** Its loss does not
grow with the channel count, which is the whole reason nobody builds a large
demux by broadcasting into N filters.
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.components import CWLaser, Demultiplexer, Multiplexer, PowerMeter
from maiman.grid import (
    DWDM_ANCHOR,
    cwdm_frequencies,
    cwdm_wavelengths,
    dwdm_frequencies,
    dwdm_wavelengths,
    wavelength_spacing,
)
from maiman.units import wavelength_to_frequency


def the_two_grids() -> None:
    """One uniform in frequency, one uniform in wavelength, neither in both."""
    print("1. G.694.1 — the dense grid, anchored at 193.1 THz")
    frequencies = dwdm_frequencies(5, spacing=100e9, first=-2)
    wavelengths = dwdm_wavelengths(5, spacing=100e9, first=-2)
    print(f"     {'frequency':>13}  {'wavelength':>13}  {'step':>10}")
    previous = None
    for frequency, wavelength in zip(frequencies, wavelengths, strict=True):
        column = "" if previous is None else f"{(previous - wavelength) * 1e9:9.4f}nm"
        print(f"     {frequency / 1e12:12.4f}THz  {wavelength * 1e9:12.4f}nm  {column:>10}")
        previous = wavelength
    print(f"     the anchor is {DWDM_ANCHOR / 1e12:.1f} THz = 1552.5244 nm, and the")
    print("     wavelength steps are not equal. They cannot be: the grid is in hertz.")
    print()
    print("     what 100 GHz is worth, across the C band:")
    for nanometres in (1530.0, 1550.0, 1565.0):
        step = wavelength_spacing(wavelength_to_frequency(nanometres * 1e-9), 100e9)
        print(f"       {nanometres:7.1f} nm  ->  {step * 1e9:.4f} nm")
    print("     stepping a flat 0.8 nm loses most of a channel over that range.\n")

    print("2. G.694.2 — the coarse grid, and why 20 nm")
    coarse = cwdm_wavelengths()
    coarse_f = cwdm_frequencies()
    steps = np.abs(np.diff(np.array(coarse_f)))
    print(f"     {len(coarse)} channels, {coarse[0] * 1e9:.0f} nm to {coarse[-1] * 1e9:.0f} nm")
    print(f"     wavelength step   exactly {(coarse[1] - coarse[0]) * 1e9:.1f} nm, every time")
    print(
        f"     frequency step    {steps.min() / 1e12:.3f} to {steps.max() / 1e12:.3f} THz "
        f"— a {steps.max() / steps.min() - 1:.0%} spread"
    )
    print("     an uncooled DFB walks ~0.1 nm/K, so 70 K of case temperature is 7 nm")
    print("     of drift. 20 nm is what makes a laser with no cooler a legal channel.\n")


def build(
    channels: int,
    *,
    spacing: float,
    bandwidth: float = 50.0,
    loss: float = 4.0,
    extinction: float = 40.0,
) -> tuple[Graph, Multiplexer, list[PowerMeter]]:
    """A transmitter per channel, a mux, a demux, and a meter on every port."""
    settings = {
        "first_wavelength": 1550.0,
        "spacing": spacing,
        "bandwidth": bandwidth,
        "insertion_loss": loss,
        "extinction": extinction,
    }
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64)
    graph = Graph(ctx)
    mux = Multiplexer(channels, label="mux", **settings)
    demux = Demultiplexer(channels, label="demux", **settings)
    graph.add(mux)
    graph.add(demux)

    for index, wavelength in enumerate(mux.channel_wavelengths()):
        laser = graph.add(CWLaser(power=0.0, wavelength=wavelength * 1e9, label=f"tx{index}"))
        graph.connect(laser, mux[f"in{index}"])
    graph.connect(mux, demux["in"])

    meters = [graph.add(PowerMeter(label=f"rx{index}")) for index in range(channels)]
    for index, meter in enumerate(meters):
        graph.connect(demux[f"out{index}"], meter["in"])
    return graph, mux, meters


def the_round_trip() -> None:
    """Eight channels out and eight channels back, with the budget."""
    print("3. Eight channels through a mux and a demux, 100 GHz apart")
    graph, mux, meters = build(8, spacing=100.0)
    results = graph.run()
    wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]

    print(f"     {'port':>5}  {'channel':>11}  {'kept':>9}  {'worst neighbour':>17}")
    for index, meter in enumerate(meters):
        reading = results[meter]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[index]))
        others = [b.power_dbm for b in reading.bands if b is not wanted]
        worst = max(others) - wanted.power_dbm if others else float("-inf")
        print(
            f"     {index:5d}  {wavelengths[index]:10.4f}nm  {wanted.power_dbm:8.3f}dB  "
            f"{worst:16.2f}dB"
        )
    print("     kept = 0 dBm launched, 4 dB through the mux, 4 dB through the demux.")
    print("     the neighbours sit on the 40 dB extinction floor: at this spacing")
    print("     the filter's own skirt is already below it.\n")


def crosstalk_against_spacing() -> None:
    """The number the block exists to produce, and which of two things sets it."""
    print("4. What actually sets the crosstalk")
    print(f"     {'spacing':>9}  {'passband':>9}  {'with floor':>12}  {'skirt alone':>13}")
    for spacing, bandwidth in (
        (200.0, 50.0),
        (100.0, 50.0),
        (50.0, 50.0),
        (40.0, 50.0),
        (25.0, 50.0),
    ):
        row = []
        for extinction in (40.0, 0.0):
            graph, mux, meters = build(
                4, spacing=spacing, bandwidth=bandwidth, extinction=extinction
            )
            results = graph.run()
            wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
            reading = results[meters[1]]
            wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[1]))
            others = [b.power_dbm for b in reading.bands if b is not wanted]
            row.append(max(others) - wanted.power_dbm)
        print(f"     {spacing:8.0f}G  {bandwidth:8.0f}G  {row[0]:11.2f}dB  {row[1]:12.2f}dB")
    print("     Nothing here was declared. The last column is the super-Gaussian's")
    print("     own skirt, and -inf is not a rejection of infinity: it is the field")
    print("     underflowing complex64, which happens past about -600 dB. The first")
    print("     column is what a real system sees, because at any sane plan the")
    print("     skirt is already below the extinction floor, and it is the floor")
    print("     that accumulates down a chain of these. At 25 GHz with a 50 GHz")
    print("     passband the neighbour is inside the channel and no floor is")
    print("     involved at all.")
    print()


def routing_is_not_splitting() -> None:
    """Why an AWG and a splitter-plus-filters are different devices."""
    print("5. An AWG routes; it does not divide power N ways")
    print(f"     {'channels':>9}  {'kept':>9}  {'a splitter would be':>21}")
    for channels in (2, 4, 8, 16, 32):
        graph, mux, meters = build(channels, spacing=100.0)
        results = graph.run()
        wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
        reading = results[meters[0]]
        wanted = min(reading.bands, key=lambda b: abs(b.wavelength_nm - wavelengths[0]))
        broadcast = -8.0 - 10.0 * np.log10(channels)
        print(f"     {channels:9d}  {wanted.power_dbm:8.3f}dB  {broadcast:20.2f}dB")
    print("     15 dB of split at 32 channels is why nobody builds a large demux by")
    print("     broadcasting into filters. Model that one with a Splitter in front.\n")


def main() -> None:
    the_two_grids()
    the_round_trip()
    crosstalk_against_spacing()
    routing_is_not_splitting()


if __name__ == "__main__":
    main()
