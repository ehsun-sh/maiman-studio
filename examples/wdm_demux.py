"""Four channels on a grid, amplified, then demultiplexed.

The signal model has carried independently sampled bands since the first commit
precisely so that this would be possible without a sample rate no machine can
afford: four carriers 100 GHz apart span 300 GHz, but each is sampled only across
its own 160 GHz and never on a common grid. Put those two lasers 6 THz apart
instead and nothing about the run changes.

Three things are worth reading off the output.

**The demultiplexer is wavelength-selective, not an index lookup.** Each filter
is tuned to one channel and rejects its neighbours by the response at their
offset — a number the model produces rather than an assumption it makes. Tune one
between two channels and it attenuates both.

**A filter is also the ASE gate.** The amplifier emits spontaneous emission
across four terahertz and every hertz of it reaches a photodiode and beats there.
The final table is what that costs: the same link, the same OSNR, and a factor of
three in Q depending on whether anything filtered it first.

**OSNR does not see it.** The reference bandwidth is fixed at 12.5 GHz, so
cutting ASE outside that band changes the noise power by orders of magnitude and
the OSNR figure not at all. That is worth seeing once: the number everyone quotes
is blind to the single cheapest improvement available to a receiver.
"""

from __future__ import annotations

import math

import numpy as np

from oosim import Graph, SimulationContext
from oosim.component import Component
from oosim.components import (
    EDFA,
    BERAnalyzer,
    Combiner,
    CWLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    OpticalFilter,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from oosim.kernels import super_gaussian_response
from oosim.units import C_LIGHT, wavelength_to_frequency

ANCHOR = 1550.0  # nm — channel 0
SPACING = 100e9  # Hz
CHANNELS = 4
BIT_RATE = 10e9
FILTER_WIDTH = 50.0  # GHz
OSNR_REFERENCE = 12.5e9
RECEIVER_BANDWIDTH = 7.0  # GHz


def channel_wavelength(index: int) -> float:
    """Wavelength of channel ``index`` on the grid [nm]."""
    return C_LIGHT / (wavelength_to_frequency(ANCHOR * 1e-9) + index * SPACING) * 1e9


def q_from_osnr(osnr_db: float) -> float:
    """Textbook Q for NRZ-OOK limited by ASE beat noise."""
    ratio = 10.0 ** (osnr_db / 10.0)
    noise_bandwidth = RECEIVER_BANDWIDTH * 1e9 * math.sqrt(math.pi / (4.0 * math.log(2.0)))
    return (
        2.0
        * math.sqrt(OSNR_REFERENCE / noise_bandwidth)
        * ratio
        / (1.0 + math.sqrt(1.0 + 4.0 * ratio))
    )


def build(select: int | None, *, filtered: bool = True) -> tuple[Graph, dict[str, object]]:
    """A four-channel comb through one amplifier, optionally demultiplexed.

    ``select`` picks which channel the receiver is tuned to; ``None`` leaves the
    whole comb on the detector, which is what a link without a demultiplexer
    would do and is included because the difference is the point.
    """
    ctx = SimulationContext(bit_rate=BIT_RATE, samples_per_symbol=16, sequence_length=2048, seed=17)
    graph = Graph(ctx)
    combiner = graph.add(Combiner(CHANNELS, label="mux"))

    # Only the channel under test is modulated; the neighbours are unmodulated
    # carriers at the same power. That keeps the crosstalk number about the
    # filter's response rather than about someone else's data pattern.
    prbs = graph.add(PRBSGenerator(order=15.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs["out"], driver["in"])

    for index in range(CHANNELS):
        laser = graph.add(
            CWLaser(power=0.0, wavelength=channel_wavelength(index), label=f"ch{index}")
        )
        if index == (select or 0):
            modulator = graph.add(MachZehnderModulator(v_pi=4.0, label=f"mzm{index}"))
            graph.connect(laser, modulator["optical_in"])
            graph.connect(driver, modulator["electrical_in"])
            graph.connect(modulator, combiner[f"in{index}"])
        else:
            graph.connect(laser, combiner[f"in{index}"])

    # Four spans, each amplified back to transparency. Without real loss to make
    # up there is no ASE worth speaking of and the last table has nothing to show.
    node: Component = combiner
    for span in range(4):
        fiber = graph.add(Fiber(length=80.0, attenuation=0.2, dispersion=0.0, label=f"f{span}"))
        amplifier = graph.add(EDFA(gain=16.0, noise_figure=6.0, label=f"edfa{span}"))
        graph.connect(node, fiber["in"])
        graph.connect(fiber, amplifier["in"])
        node = amplifier

    if filtered and select is not None:
        demux = graph.add(
            OpticalFilter(
                center_wavelength=channel_wavelength(select),
                bandwidth=FILTER_WIDTH,
                order=3.0,
                label="demux",
            )
        )
        graph.connect(node, demux["in"])
        node = demux

    osa = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=channel_wavelength(1),
            span=800.0,
            points=4096,
            label="osa",
        )
    )
    meter = graph.add(PowerMeter(label="pm"))
    osnr = graph.add(OSNRMeter(label="osnr"))
    detector = graph.add(PINPhotodiode(label="pin"))
    receiver = graph.add(ElectricalFilter(bandwidth=RECEIVER_BANDWIDTH, label="lpf"))
    analyzer = graph.add(BERAnalyzer(label="ber"))
    graph.connect(node, osa["in"])
    graph.connect(node, meter["in"])
    graph.connect(node, osnr["in"])
    graph.connect(node, detector["in"])
    graph.connect(detector, receiver["in"])
    graph.connect(receiver, analyzer["in"])
    graph.connect(prbs["out"], analyzer["reference"])

    return graph, {"osa": osa, "pm": meter, "osnr": osnr, "ber": analyzer}


def main() -> None:
    print(f"{CHANNELS} channels on a {SPACING / 1e9:.0f} GHz grid from {ANCHOR:.1f} nm")
    print(f"four amplified 80 km spans, then a {FILTER_WIDTH:.0f} GHz third-order demultiplexer\n")

    # The shape itself, before any link is involved. At a 100 GHz spacing every
    # sensible order buries the neighbour far below any real component's
    # extinction, so what limits crosstalk on this grid is the floor and not the
    # skirt — which is why the floor is a parameter rather than an idealisation.
    print("  Passband shape: transmission against offset from centre")
    print(f"  {'offset':>10}  {'order 1':>10}  {'order 3':>10}  {'order 5':>10}")
    print("  " + "-" * 46)
    for offset_ghz in (0.0, 12.5, 25.0, 31.25, 37.5, 50.0):
        row = []
        for order in (1, 2 + 1, 5):
            t = super_gaussian_response(np.array([offset_ghz * 1e9]), FILTER_WIDTH * 1e9, order)[0]
            db = 10.0 * math.log10(max(t**2, 1e-30))
            row.append(f"{db:9.1f} " if db > -300 else "     ---  ")
        print(f"  {offset_ghz:8.1f}GHz  " + " ".join(row))
    print(f"  a neighbour sits {SPACING / 1e9:.0f} GHz out, past every column above —")
    print("  so the extinction floor, not the shape, is what it lands on.")
    print()

    print("  Demultiplexer selectivity — power in each channel after the filter")
    print("  tuned to      ch0       ch1       ch2       ch3      worst rejection")
    print("  " + "-" * 74)
    for select in range(CHANNELS):
        graph, ports = build(select)
        reading = graph.run()[ports["pm"]]  # type: ignore[index]
        by_frequency = sorted(reading.bands, key=lambda b: b.wavelength_nm, reverse=True)
        powers = [band.power_dbm for band in by_frequency]
        others = [p for i, p in enumerate(powers) if i != select]
        print(
            f"  ch{select}      "
            + "".join(f"{p:9.1f} " for p in powers)
            + f"   {powers[select] - max(others):6.1f} dB"
        )

    print("\n  What the filter is worth at the receiver")
    print(f"  {'link':>28}  {'OSNR':>9}  {'ASE power':>11}  {'Q':>7}  {'vs OSNR limit':>14}")
    print("  " + "-" * 78)
    for filtered, label in ((False, "no demultiplexer"), (True, f"{FILTER_WIDTH:.0f} GHz demux")):
        graph, ports = build(1, filtered=filtered)
        results = graph.run()
        osnr_db = float(results[ports["osnr"]])  # type: ignore[index]
        reading = results[ports["pm"]]  # type: ignore[index]
        q = results[ports["ber"]].q_factor  # type: ignore[index]
        print(
            f"  {label:>28}  {osnr_db:6.2f} dB  {reading.noise_power_w * 1e3:8.4f} mW  "
            f"{q:7.2f}  {q / q_from_osnr(osnr_db):13.2f}x"
        )

    print("\n  The OSNR figure is identical either way. It is quoted in a fixed")
    print("  12.5 GHz reference bandwidth, so it cannot see ASE removed outside it.")

    graph, ports = build(1)
    spectrum = graph.run()[ports["osa"]]  # type: ignore[index]
    peak_frequency, peak_power = spectrum.peak()
    print(
        f"\n  OSA: {len(spectrum.frequencies)} points, peak at "
        f"{C_LIGHT / peak_frequency * 1e9:.3f} nm, "
        f"{10.0 * math.log10(peak_power * 1e3):.2f} dBm per "
        f"{spectrum.resolution_bandwidth / 1e9:.1f} GHz"
    )


if __name__ == "__main__":
    main()
