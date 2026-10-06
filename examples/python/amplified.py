"""Phase 3 of the example roadmap: amplified and WDM links.

Each example is a studio project with its lesson: one amplifier read on a
spectrum, the OSNR a chain of them leaves, the launch power between noise and
the Kerr effect, four-wave mixing on a fibre with no dispersion, and a ROADM
node that drops one wavelength and adds another in its place.

``python amplified.py`` writes every project into ``examples/maiman/`` and
prints what each measures; ``python amplified.py roadm`` does one.
``tests/test_amplified.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path

import lessons
import numpy as np
from direct_detection import Layout, at, ook_transmitter, receive, with_eyes

from maiman import Graph, SimulationContext, sweep
from maiman.component import Component
from maiman.components import (
    EDFA,
    CWLaser,
    Demultiplexer,
    Fiber,
    Multiplexer,
    NRZDriver,
    OpticalSpectrumAnalyzer,
    OSNRMeter,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
    Splitter,
    WavelengthSelectiveSwitch,
)
from maiman.project import save
from maiman.signals import EyeMeasurement, OpticalSpectrum, PowerReading
from maiman.units import C_LIGHT, wavelength_to_frequency

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"

SPAN_KM = 80.0
SPAN_LOSS_DB = 0.2 * SPAN_KM


def spectrum(
    label: str, centre: float = 1550.0, span: float = 600.0, points: float = 1024.0
) -> OpticalSpectrumAnalyzer:
    """An OSA at a fixed window, so the noise floor sits where the lesson says it does."""
    return OpticalSpectrumAnalyzer(
        auto_span=False,
        center_wavelength=centre,
        span=span,
        points=points,
        resolution_bandwidth=12.5,
        label=label,
    )


# ---------------------------------------------------------------------------
# 3.1  One EDFA, before and after

EDFA_LAYOUT: Layout = {
    "laser": at(0, 1),
    "pm_in": at(1, 0),
    "osa_in": at(1, 2),
    "edfa": at(2, 1),
    "pm_out": at(3, 0),
    "osnr": at(3, 1),
    "osa_out": at(3, 2),
}

EDFA_INPUT_DBM = -20.0
EDFA_GAIN_DB = 20.0
EDFA_NF_DB = 5.0


def edfa_basics() -> Graph:
    """A weak CW line into one amplifier, read on meters and spectra at both sides."""
    graph = Graph(SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256))
    laser = graph.add(CWLaser(power=EDFA_INPUT_DBM, wavelength=1550.0, label="laser"))
    edfa = graph.add(EDFA(gain=EDFA_GAIN_DB, noise_figure=EDFA_NF_DB, label="edfa"))
    graph.connect(laser, edfa["in"])
    for side, source in (("in", laser), ("out", edfa)):
        graph.connect(source, graph.add(PowerMeter(label=f"pm_{side}"))["in"])
        graph.connect(source, graph.add(spectrum(f"osa_{side}", span=4000.0))["in"])
    graph.connect(edfa, graph.add(OSNRMeter(label="osnr"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 3.2  A chain of amplified spans

CHAIN_SPANS = 8
#: The spans after which the OSNR is read: each is double the one before.
CHAIN_TAPS = (1, 2, 4, 8)

CHAIN_LAYOUT: Layout = {
    "laser": at(0, 0),
    **{
        f"{part}{n}": at(2 * ((n - 1) % 4) + 1 + offset, 0 if n <= 4 else 2.5)
        for n in range(1, CHAIN_SPANS + 1)
        for offset, part in ((0, "span"), (1, "edfa"))
    },
    **{f"osnr{n}": at(2 * ((n - 1) % 4) + 2, 1 if n <= 4 else 3.5) for n in CHAIN_TAPS},
    "osa1": at(2, 1.8),
    "osa8": at(8, 1.5),
}


def amplified_chain() -> Graph:
    """Eight 80 km spans, each amplified back to its launch power, read as it goes."""
    graph = Graph(SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256))
    previous: Component = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    for n in range(1, CHAIN_SPANS + 1):
        span = graph.add(Fiber(length=SPAN_KM, attenuation=0.2, label=f"span{n}"))
        edfa = graph.add(EDFA(gain=SPAN_LOSS_DB, noise_figure=5.0, label=f"edfa{n}"))
        graph.connect(previous, span["in"])
        graph.connect(span["out"], edfa["in"])
        if n in CHAIN_TAPS:
            graph.connect(edfa, graph.add(OSNRMeter(label=f"osnr{n}"))["in"])
        previous = edfa
        if n in (1, CHAIN_SPANS):
            graph.connect(edfa, graph.add(spectrum(f"osa{n}", span=4000.0))["in"])
    return graph


# ---------------------------------------------------------------------------
# 3.5  Optimum launch power

LAUNCH_SPANS = 5
#: Dispersion-compensating fibre that undoes one span: 80 x 17 = 13.6 x 100 ps/nm.
DCF_KM = 13.6
#: Each amplifier makes up a span and its spool: 16 + 6.8 dB.
LINE_GAIN_DB = SPAN_LOSS_DB + 0.5 * DCF_KM
#: The booster gains the lesson's table sweeps [dB].
BOOSTER_SWEEP = [float(g) for g in range(0, 16, 3)]

LAUNCH_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "drv": at(1, 0),
    "tx": at(0, 1),
    "mzm": at(1, 1),
    "booster": at(2, 1),
    "pm_launch": at(2, 0),
    **{
        f"{part}{n}": at(3 + 3 * ((n - 1) % 3) + offset, 1 if n <= 3 else 3.2)
        for n in range(1, LAUNCH_SPANS + 1)
        for offset, part in ((0, "span"), (1, "dcf"), (2, "edfa"))
    },
    "osnr": at(9, 3.2),
    "pin": at(9, 4.2),
    "lpf": at(10, 4.2),
    "ber": at(11, 4.2),
}


def launch_power() -> Graph:
    """10 Gb/s OOK over five compensated spans, launched by a booster the lesson sweeps."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=7)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=9.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    _, mzm = ook_transmitter(graph)
    graph.connect(driver, mzm["electrical_in"])
    booster = graph.add(EDFA(gain=6.0, noise_figure=5.0, label="booster"))
    graph.connect(mzm, booster["in"])
    graph.connect(booster, graph.add(PowerMeter(label="pm_launch"))["in"])
    previous: Component = booster
    for n in range(1, LAUNCH_SPANS + 1):
        span = graph.add(
            Fiber(
                length=SPAN_KM,
                attenuation=0.2,
                dispersion=17.0,
                nonlinearity=1.3,
                label=f"span{n}",
            )
        )
        dcf = graph.add(Fiber(length=DCF_KM, attenuation=0.5, dispersion=-100.0, label=f"dcf{n}"))
        edfa = graph.add(EDFA(gain=LINE_GAIN_DB, noise_figure=5.0, label=f"edfa{n}"))
        graph.connect(previous, span["in"])
        graph.connect(span["out"], dcf["in"])
        graph.connect(dcf["out"], edfa["in"])
        previous = edfa
    graph.connect(previous, graph.add(OSNRMeter(label="osnr"))["in"])
    pin = graph.add(PINPhotodiode(label="pin"))
    receive(graph, previous, prbs, detector=pin, bandwidth_ghz=7.0)
    return graph


@cache
def launch_table() -> tuple[tuple[float, float, float, float], ...]:
    """(booster gain dB, launch dBm, OSNR dB, Q) at every gain of the sweep."""
    graph = launch_power()
    by = {c.label: c for c in graph.components}
    result = sweep(graph, {(by["booster"], "gain"): BOOSTER_SWEEP})
    rows = []
    for gain, point in zip(BOOSTER_SWEEP, result.points, strict=True):
        run = point.runs[0]
        launch = run[by["pm_launch"]]
        assert isinstance(launch, PowerReading)
        rows.append((gain, launch.power_dbm, float(run[by["osnr"]]), run[by["ber"]].q_factor))
    return tuple(rows)


# ---------------------------------------------------------------------------
# 3.6  Four-wave mixing on a zero-dispersion fibre

FWM_CHANNELS = 4
FWM_FIRST_NM = 1550.0
FWM_SPACING_GHZ = 100.0
FWM_CHANNEL_DBM = 6.0
#: G.653's zero at 1550 nm, against G.655's few ps/(nm km) of non-zero dispersion.
FWM_FIBRES = (("dsf", 0.0), ("nzdsf", 4.0))

FWM_LAYOUT: Layout = {
    **{f"ch{i}": at(0, i) for i in range(FWM_CHANNELS)},
    "mux": at(1, 1.5),
    "split": at(2, 1.5),
    "fibre_dsf": at(3, 0.5),
    "osa_dsf": at(4, 0.5),
    "fibre_nzdsf": at(3, 2.5),
    "osa_nzdsf": at(4, 2.5),
}


def fwm_slot(index: float) -> float:
    """The wavelength [nm] of slot ``index`` on the grid; 0 to 3 are the channels."""
    f0 = wavelength_to_frequency(FWM_FIRST_NM * 1e-9)
    return float(C_LIGHT / (f0 + index * FWM_SPACING_GHZ * 1e9) * 1e9)


def fwm_dsf() -> Graph:
    """Four CW channels at +6 dBm through 80 km of G.653 and of G.655, side by side."""
    graph = Graph(SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256))
    mux = graph.add(
        Multiplexer(
            FWM_CHANNELS,
            first_wavelength=FWM_FIRST_NM,
            spacing=FWM_SPACING_GHZ,
            insertion_loss=0.0,
            label="mux",
        )
    )
    for index, wavelength in enumerate(mux.channel_wavelengths()):
        laser = graph.add(
            CWLaser(power=FWM_CHANNEL_DBM, wavelength=wavelength * 1e9, label=f"ch{index}")
        )
        graph.connect(laser, mux[f"in{index}"])
    split = graph.add(Splitter(2, label="split"))
    graph.connect(mux, split["in"])
    for port, (name, dispersion) in enumerate(FWM_FIBRES):
        fibre = graph.add(
            Fiber(
                length=SPAN_KM,
                attenuation=0.2,
                dispersion=dispersion,
                nonlinearity=1.3,
                # Keep products down to 100 dB below the channels, so the
                # suppressed ones on G.655 are drawn rather than discarded.
                mixing_floor=100.0,
                label=f"fibre_{name}",
            )
        )
        graph.connect(split[f"out{port}"], fibre["in"])
        osa = graph.add(spectrum(f"osa_{name}", centre=fwm_slot(1.5), span=1000.0, points=2048.0))
        graph.connect(fibre["out"], osa["in"])
    return graph


def slot_levels(reading: object) -> dict[int, float]:
    """The peak [dBm per 0.1 nm] in each grid slot from -3 to 6, read off a spectrum."""
    assert isinstance(reading, OpticalSpectrum)
    wavelengths = np.asarray(reading.wavelengths_nm)
    levels = np.asarray(reading.power_dbm())
    out = {}
    for index in range(-3, FWM_CHANNELS + 3):
        nearest = int(np.argmin(np.abs(wavelengths - fwm_slot(index))))
        out[index] = float(levels[max(nearest - 3, 0) : nearest + 4].max())
    return out


# ---------------------------------------------------------------------------
# 3.9  A ROADM node: drop one wavelength, add another in its place

ROADM_CHANNELS = 4
ROADM_DROPPED = 1
ROADM_GRID = {"first_wavelength": 1550.0, "spacing": 100.0}
#: The added transmitter's power, set so it leaves the node level with the
#: express channels, which have come through a span and an amplifier.
ROADM_ADD_DBM = 2.0

ROADM_LAYOUT: Layout = {
    "prbs": at(0, 4.5),
    "drv": at(1, 4.5),
    **{f"tx{i}": at(0, i) for i in range(ROADM_CHANNELS)},
    **{f"mzm{i}": at(1, i) for i in range(ROADM_CHANNELS)},
    "mux_a": at(2, 1.5),
    "span_a": at(3, 1.5),
    "edfa_a": at(4, 1.5),
    "osa_a": at(4, 0),
    "wss": at(5, 1.5),
    "pin_drop": at(6, 3.5),
    "lpf_drop": at(7, 3.5),
    "ber_drop": at(8, 3.5),
    "prbs_add": at(5, 5),
    "drv_add": at(6, 5),
    "tx_add": at(5, 6),
    "mzm_add": at(6, 6),
    "osa_b": at(8, 0),
    "span_b": at(8, 1.5),
    "edfa_b": at(9, 1.5),
    "demux_b": at(10, 1.5),
    "pin_add": at(11, 3.5),
    "lpf_add": at(12, 3.5),
    "ber_add": at(13, 3.5),
}


def roadm() -> Graph:
    """Four channels into a node that expresses three, drops one and adds a new one there."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=512, seed=9)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=9.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    mux_a = graph.add(Multiplexer(ROADM_CHANNELS, label="mux_a", **ROADM_GRID))
    for index, wavelength in enumerate(mux_a.channel_wavelengths()):
        _, mzm = ook_transmitter(graph, wavelength=wavelength * 1e9, suffix=str(index))
        graph.connect(driver, mzm["electrical_in"])
        graph.connect(mzm, mux_a[f"in{index}"])

    def line(name: str, source: object) -> EDFA:
        span = graph.add(Fiber(length=SPAN_KM, attenuation=0.2, label=f"span_{name}"))
        edfa = graph.add(EDFA(gain=SPAN_LOSS_DB + 6.0, noise_figure=5.0, label=f"edfa_{name}"))
        graph.connect(source, span["in"])  # type: ignore[arg-type]
        graph.connect(span["out"], edfa["in"])
        return edfa

    edfa_a = line("a", mux_a)
    graph.connect(edfa_a, graph.add(spectrum("osa_a", centre=1548.8))["in"])
    wss = graph.add(
        WavelengthSelectiveSwitch(
            ROADM_CHANNELS, dropped=float(ROADM_DROPPED), label="wss", **ROADM_GRID
        )
    )
    graph.connect(edfa_a, wss["in"])
    receive(
        graph,
        wss["drop"],
        prbs,
        detector=graph.add(PINPhotodiode(label="pin_drop")),
        bandwidth_ghz=7.0,
        suffix="_drop",
    )

    # The added channel carries its own data, so the far receiver can tell it
    # from whatever of the dropped one leaked through the switch.
    prbs_add = graph.add(PRBSGenerator(order=11.0, label="prbs_add"))
    driver_add = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv_add"))
    graph.connect(prbs_add, driver_add["in"])
    wavelength = mux_a.channel_wavelengths()[ROADM_DROPPED] * 1e9
    _, mzm_add = ook_transmitter(graph, wavelength=wavelength, power=ROADM_ADD_DBM, suffix="_add")
    graph.connect(driver_add, mzm_add["electrical_in"])
    graph.connect(mzm_add, wss["add"])

    edfa_b = line("b", wss["out"])
    graph.connect(wss["out"], graph.add(spectrum("osa_b", centre=1548.8))["in"])
    demux_b = graph.add(Demultiplexer(ROADM_CHANNELS, label="demux_b", **ROADM_GRID))
    graph.connect(edfa_b, demux_b["in"])
    receive(
        graph,
        demux_b[f"out{ROADM_DROPPED}"],
        prbs_add,
        detector=graph.add(PINPhotodiode(label="pin_add")),
        bandwidth_ghz=7.0,
        suffix="_add",
    )
    return graph


# ---------------------------------------------------------------------------

#: key -> (builder, layout). The key names the project file and its lesson.
EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "edfa_basics": (edfa_basics, EDFA_LAYOUT),
    "amplified_chain": (amplified_chain, CHAIN_LAYOUT),
    "launch_power": (launch_power, with_eyes(LAUNCH_LAYOUT)),
    "fwm_dsf": (fwm_dsf, FWM_LAYOUT),
    "roadm": (roadm, with_eyes(ROADM_LAYOUT)),
}


def project_path(key: str) -> Path:
    return PROJECTS / f"{key}.maiman"


def write(key: str) -> Graph:
    build, layout = EXAMPLES[key]
    graph = build()
    save(
        graph,
        project_path(key),
        ui=layout,
        notes=lessons.NOTES[key],
        annotations=lessons.marks(key, layout),
    )
    return graph


def results_by_label(graph: Graph) -> dict[str, object]:
    return {label: value for (label, _), value in graph.run().items()}


def report(key: str, graph: Graph) -> None:
    if key == "launch_power":
        print("   booster   launch    OSNR      Q")
        for gain, launch, osnr, q in launch_table():
            print(f"   {gain:5.1f} dB {launch:6.2f} dBm {osnr:6.2f} dB {q:7.2f}")
    results = results_by_label(graph)
    for label, value in results.items():
        if isinstance(value, EyeMeasurement):
            print(f"   {label:14s} Q {value.q_factor:8.2f}  errors {value.errors}")
        elif isinstance(value, PowerReading):
            print(f"   {label:14s} {value.power_dbm:7.2f} dBm")
        elif isinstance(value, float):
            print(f"   {label:14s} {value:7.2f} dB")
        elif isinstance(value, OpticalSpectrum) and key == "fwm_dsf":
            levels = slot_levels(value)
            print(f"   {label:14s} " + "  ".join(f"{k:+d}:{v:6.1f}" for k, v in levels.items()))


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
