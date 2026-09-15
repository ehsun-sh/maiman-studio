"""An eight-channel DWDM link, end to end: the studio's DWDM template.

Eight 10 Gb/s NRZ transmitters on the ITU 100 GHz grid, multiplexed, launched
through a booster into 80 km of standard fibre with the Kerr effect on,
compensated by a spool of dispersion-compensating fibre, amplified, watched on an
analyser and an OSNR meter, demultiplexed, and one channel received on a
photodiode and counted.

Every block is one the palette has, and every number the template shows comes
from running this graph. It is a script so that the project the studio ships is
generated rather than drawn; ``python dwdm_link.py`` rewrites
``dwdm_link.maiman`` beside it and prints what the link does.

Two settings are chosen for a template that has to run while someone watches.
Eight samples per symbol rather than sixteen, because an NRZ channel at 10 Gb/s
needs no more, and cross-phase modulation stepped at sixteen takes three times
as long for the same answer to a percent. And 512 symbols, enough to count
errors on a link with margin and short enough to run in seconds. Switch the
span's ``cross_phase_modulation`` off and the same link runs in a fifth of a
second, without the impairment a DWDM link is mostly about.
"""

from __future__ import annotations

import time
from pathlib import Path

from maiman import Graph, SimulationContext
from maiman.components import (
    EDFA,
    BERAnalyzer,
    CWLaser,
    Demultiplexer,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    Multiplexer,
    NRZDriver,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
)
from maiman.project import save
from maiman.signals import EyeMeasurement, PowerReading
from maiman.units import frequency_to_wavelength

PROJECT = Path(__file__).parent / "dwdm_link.maiman"

CHANNELS = 8
SPACING_GHZ = 100.0
#: ITU-T G.694.1's anchor, 193.1 THz, as channel 0. The rest step up in frequency.
FIRST_NM = frequency_to_wavelength(193.1e12) * 1e9
#: The channel the receiver is on: one in the middle, with a neighbour each side.
DROPPED = 3

SPAN_KM = 80.0
D_SPAN = 17.0
#: 13.6 km at -100 ps/nm/km takes the span's 1360 ps/nm back out.
DCF_KM = SPAN_KM * D_SPAN / 100.0

LAYOUT: dict[str, dict[str, float]] = {
    "prbs": {"x": 30.0, "y": 830.0},
    "drv": {"x": 168.0, "y": 830.0},
    **{f"ch{i}": {"x": 30.0, "y": 30.0 + i * 100.0} for i in range(CHANNELS)},
    **{f"mzm{i}": {"x": 306.0, "y": 30.0 + i * 100.0} for i in range(CHANNELS)},
    "mux": {"x": 444.0, "y": 380.0},
    "booster": {"x": 582.0, "y": 380.0},
    "span": {"x": 720.0, "y": 380.0},
    "dcf": {"x": 858.0, "y": 380.0},
    "preamp": {"x": 996.0, "y": 380.0},
    "osa": {"x": 1134.0, "y": 200.0},
    "osnr": {"x": 1134.0, "y": 290.0},
    "demux": {"x": 1134.0, "y": 470.0},
    "drop_power": {"x": 1272.0, "y": 380.0},
    "pin": {"x": 1272.0, "y": 560.0},
    "lpf": {"x": 1410.0, "y": 560.0},
    "ber": {"x": 1548.0, "y": 560.0},
}


def build() -> Graph:
    """The link the template opens, exactly."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=7)
    graph = Graph(ctx)

    # One pattern drives every modulator, as in the four-channel WDM project:
    # independent data would be eight more blocks and would not change what a
    # channel's width or its neighbours' crosstalk is set by.
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])

    mux = graph.add(
        Multiplexer(CHANNELS, first_wavelength=FIRST_NM, spacing=SPACING_GHZ, label="mux")
    )
    # Tuned from the multiplexer's own grid, so a laser cannot sit off its port.
    for index, wavelength in enumerate(mux.channel_wavelengths()):
        laser = graph.add(CWLaser(power=0.0, wavelength=wavelength * 1e9, label=f"ch{index}"))
        modulator = graph.add(MachZehnderModulator(v_pi=4.0, label=f"mzm{index}"))
        graph.connect(laser, modulator["optical_in"])
        graph.connect(driver, modulator["electrical_in"])
        graph.connect(modulator, mux[f"in{index}"])

    booster = graph.add(EDFA(gain=10.0, noise_figure=5.0, label="booster"))
    span = graph.add(
        Fiber(
            length=SPAN_KM,
            attenuation=0.2,
            dispersion=D_SPAN,
            nonlinearity=1.3,
            label="span",
        )
    )
    dcf = graph.add(Fiber(length=DCF_KM, attenuation=0.5, dispersion=-100.0, label="dcf"))
    # 16 dB of span and 6.8 dB of spool, made back up.
    preamp = graph.add(EDFA(gain=23.0, noise_figure=5.0, label="preamp"))
    graph.connect(mux, booster["in"])
    graph.connect(booster, span["in"])
    graph.connect(span["out"], dcf["in"])
    graph.connect(dcf["out"], preamp["in"])

    wavelengths = [w * 1e9 for w in mux.channel_wavelengths()]
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            center_wavelength=(wavelengths[0] + wavelengths[-1]) / 2.0,
            span=1200.0,
            points=1024.0,
            resolution_bandwidth=12.5,
            label="osa",
        )
    )
    osnr = graph.add(OSNRMeter(label="osnr"))
    graph.connect(preamp, osa["in"])
    graph.connect(preamp, osnr["in"])

    demux = graph.add(
        Demultiplexer(CHANNELS, first_wavelength=FIRST_NM, spacing=SPACING_GHZ, label="demux")
    )
    graph.connect(preamp, demux["in"])
    power = graph.add(PowerMeter(label="drop_power"))
    pin = graph.add(PINPhotodiode(responsivity=0.8, label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    ber = graph.add(BERAnalyzer(label="ber"))
    graph.connect(demux[f"out{DROPPED}"], power["in"])
    graph.connect(demux[f"out{DROPPED}"], pin["in"])
    graph.connect(pin, lpf["in"])
    graph.connect(lpf, ber["in"])
    graph.connect(prbs, ber["reference"])
    return graph


def write() -> Graph:
    """Build the template and save it where the studio's export reads it."""
    graph = build()
    save(graph, PROJECT, ui=LAYOUT)
    return graph


def main() -> None:
    graph = write()
    start = time.perf_counter()
    results = graph.run()
    seconds = time.perf_counter() - start
    by_label = {label: value for (label, _), value in results.items()}

    drop = by_label["drop_power"]
    eye = by_label["ber"]
    assert isinstance(drop, PowerReading)
    assert isinstance(eye, EyeMeasurement)
    kept = max(band.power_dbm for band in drop.bands)
    leaked = sorted(band.power_dbm for band in drop.bands)[-2]

    plan = f"{CHANNELS} x 10 Gb/s on {SPACING_GHZ:.0f} GHz from {FIRST_NM:.3f} nm"
    print(f"{plan}, ran in {seconds:.1f} s")
    print(f"   OSNR after the preamp   {by_label['osnr']:.2f} dB / 0.1 nm")
    print(f"   channel {DROPPED} at the demux    {kept:.2f} dBm")
    print(f"   strongest neighbour     {leaked:.2f} dBm ({kept - leaked:.1f} dB below)")
    print(f"   Q                       {eye.q_factor:.2f}")
    print(f"   errors counted          {eye.errors}/{eye.bits_evaluated}")
    print(f"   written to {PROJECT.name}")


if __name__ == "__main__":
    main()
