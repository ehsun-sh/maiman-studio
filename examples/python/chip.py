"""Phase 5 of the example roadmap: photonic integrated circuits.

Each example is a studio project with its lesson. Most of them light the chip
the way a laboratory does: with the amplified spontaneous emission of an EDFA
that has nothing to amplify, a source four terahertz wide, and an optical
spectrum analyser on every output. The device's transmission is then written
on the noise, and the S-matrix tab draws the same device without any noise at
all.

``python chip.py`` writes every project into ``examples/maiman/`` and prints
what each measures; ``python chip.py ring`` does one.
``tests/test_chip.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import lessons
import numpy as np
from direct_detection import Layout, at

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    MMI,
    CoupledWaveguides,
    CWLaser,
    EdgeCoupler,
    GratingCoupler,
    MachZehnderInterferometer,
    OpticalSpectrumAnalyzer,
    PowerMeter,
    RingResonator,
    Splitter,
    Waveguide,
)
from maiman.pdk import load_pdk
from maiman.project import save
from maiman.signals import OpticalSpectrum, PowerReading
from maiman.units import C_LIGHT

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"
KIT = Path(__file__).resolve().parent / "silicon_220nm.pdk.json"


def context() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)


def white_light(graph: Graph, span: float = 2000.0, resolution: float = 1.0) -> Component:
    """An EDFA with almost nothing at its input: four terahertz of ASE.

    With an OSA on it, ``osa_in``, as the reference every transmission is
    divided by: the ASE is not flat, and a lab measures a device against what
    went into it.
    """
    seed = graph.add(CWLaser(power=-100.0, wavelength=1550.0, label="seed"))
    source = graph.add(EDFA(gain=30.0, noise_figure=5.0, label="ase"))
    graph.connect(seed, source["in"])
    graph.connect(source, graph.add(spectrum("osa_in", span, resolution))["in"])
    return source


def spectrum(label: str, span: float = 2000.0, resolution: float = 1.0) -> OpticalSpectrumAnalyzer:
    """An OSA on 1550 nm, ``span`` GHz wide, fine enough to resolve a ring."""
    return OpticalSpectrumAnalyzer(
        auto_span=False,
        center_wavelength=1550.0,
        span=span,
        points=2048.0,
        resolution_bandwidth=resolution,
        label=label,
    )


# ---------------------------------------------------------------------------
# 5.1  A directional coupler's split ratio against its length

DELTA_N = 0.04
#: All the power has crossed at lambda / (2 delta_n): 19.375 um at 1550 nm.
COUPLING_LENGTH_UM = 1.55 / (2.0 * DELTA_N)
COUPLER_LENGTHS = [0.0, 4.84, 9.69, 14.53, 19.38, 29.06, 38.75]

COUPLER_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "dc": at(2, 1),
    "pm_bar": at(3, 0),
    "osa_bar": at(4, 0),
    "pm_cross": at(3, 2),
    "osa_cross": at(4, 2),
}


def coupler_length(length_um: float = COUPLING_LENGTH_UM / 2.0) -> Graph:
    graph = Graph(context())
    source = white_light(graph, span=4000.0)
    coupler = graph.add(CoupledWaveguides(length=round(length_um, 2), delta_n=DELTA_N, label="dc"))
    graph.connect(source, coupler["in1"])
    for port, name in (("out1", "bar"), ("out2", "cross")):
        graph.connect(coupler[port], graph.add(PowerMeter(label=f"pm_{name}"))["in"])
        graph.connect(coupler[port], graph.add(spectrum(f"osa_{name}", span=4000.0))["in"])
    return graph


def cross_fraction(length_um: float) -> float:
    """The share of the white light that leaves by the cross port."""
    results = results_by_label(coupler_length(length_um))
    bar, cross = results["pm_bar"], results["pm_cross"]
    assert isinstance(bar, PowerReading) and isinstance(cross, PowerReading)
    return cross.power_w / (bar.power_w + cross.power_w)


# ---------------------------------------------------------------------------
# 5.2  An MZI interleaver

MZI_DELTA_L_UM = 200.0
MZI_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "mzi": at(2, 1),
    "osa_bar": at(3, 0),
    "osa_cross": at(3, 2),
}


def mzi() -> Graph:
    graph = Graph(context())
    source = white_light(graph)
    device = graph.add(
        MachZehnderInterferometer(
            arm_length=100.0, length_difference=MZI_DELTA_L_UM, propagation_loss=0.0, label="mzi"
        )
    )
    graph.connect(source, device["in1"])
    graph.connect(device["out1"], graph.add(spectrum("osa_bar"))["in"])
    graph.connect(device["out2"], graph.add(spectrum("osa_cross"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 5.3  A ring resonator filter

RING_UM = 100.0
RING_COUPLING = 0.05
RING_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "ring": at(2, 1),
    "osa_through": at(3, 0),
    "osa_drop": at(3, 2),
}


def ring() -> Graph:
    graph = Graph(context())
    source = white_light(graph)
    device = graph.add(
        RingResonator(
            length=RING_UM, coupling=RING_COUPLING, drop_coupling=RING_COUPLING, label="ring"
        )
    )
    graph.connect(source, device["in"])
    graph.connect(device["through"], graph.add(spectrum("osa_through"))["in"])
    graph.connect(device["drop"], graph.add(spectrum("osa_drop"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 5.4  A birefringent ring

BIREF_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "sp": at(2, 1),
    "ring_plain": at(3, 0),
    "ring_biref": at(3, 2),
    "osa_plain": at(4, 0),
    "osa_biref": at(4, 2),
}


def birefringent_ring() -> Graph:
    """The same ring twice under unpolarised light: one mode, then both."""
    graph = Graph(context())
    source = white_light(graph)
    split = graph.add(Splitter(2, label="sp"))
    graph.connect(source, split["in"])
    for index, (name, both) in enumerate((("plain", False), ("biref", True))):
        device = graph.add(
            RingResonator(
                length=RING_UM,
                coupling=RING_COUPLING,
                drop_coupling=RING_COUPLING,
                birefringent=both,
                label=f"ring_{name}",
            )
        )
        graph.connect(split[f"out{index}"], device["in"])
        graph.connect(device["drop"], graph.add(spectrum(f"osa_{name}"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 5.5  On and off a chip

COUPLERS_LAYOUT: Layout = {
    "tx": at(0, 0),
    "edge_in": at(1, 0),
    "wg_edge": at(2, 0),
    "edge_out": at(3, 0),
    "pm_edge": at(4, 0),
    "seed": at(0, 2),
    "ase": at(1, 2),
    "gc_in": at(2, 2),
    "wg_gc": at(3, 2),
    "gc_out": at(4, 2),
    "pm_gc": at(5, 1.5),
    "osa_gc": at(5, 2.5),
    "pm_ase": at(0, 3.3),
    "osa_in": at(1, 3.3),
}


def chip_couplers() -> Graph:
    graph = Graph(context())
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
    edge_in = graph.add(EdgeCoupler(label="edge_in"))
    guide = graph.add(Waveguide(length=5000.0, propagation_loss=2.0, label="wg_edge"))
    edge_out = graph.add(EdgeCoupler(label="edge_out"))
    graph.chain(laser, edge_in, guide, edge_out)
    graph.connect(edge_out, graph.add(PowerMeter(label="pm_edge"))["in"])
    source = white_light(graph, span=8000.0, resolution=12.5)
    graph.connect(source, graph.add(PowerMeter(label="pm_ase"))["in"])
    gc_in = graph.add(GratingCoupler(label="gc_in"))
    guide = graph.add(Waveguide(length=5000.0, propagation_loss=2.0, label="wg_gc"))
    gc_out = graph.add(GratingCoupler(label="gc_out"))
    graph.connect(source, gc_in["in"])
    graph.chain(gc_in, guide, gc_out)
    graph.connect(gc_out, graph.add(PowerMeter(label="pm_gc"))["in"])
    graph.connect(gc_out, graph.add(spectrum("osa_gc", span=8000.0, resolution=12.5))["in"])
    return graph


# ---------------------------------------------------------------------------
# 5.6  A splitter from a foundry kit

PDK_LAYOUT: Layout = {
    "tx": at(0, 2.65),
    "sp": at(1, 2.65),
    "ideal": at(2, 1.05),
    "kit": at(2, 4.25),
    **{f"pm_ideal{k}": at(3, (k - 1) * 0.7) for k in range(1, 5)},
    **{f"pm_kit{k}": at(3, 3.2 + (k - 1) * 0.7) for k in range(1, 5)},
}


def pdk_splitter() -> Graph:
    """A 1 x 4 MMI as the textbook draws it, beside the kit's."""
    graph = Graph(context())
    laser = graph.add(CWLaser(power=3.0, wavelength=1550.0, label="tx"))
    split = graph.add(Splitter(2, label="sp"))
    graph.connect(laser, split["in"])
    ideal = graph.add(MMI(4, excess_loss=0.0, imbalance=0.0, label="ideal"))
    kit = graph.add(load_pdk(KIT).make("mmi_1x4", label="kit"))
    for index, (name, device) in enumerate((("ideal", ideal), ("kit", kit))):
        graph.connect(split[f"out{index}"], device["in1"])
        for k in range(1, 5):
            graph.connect(device[f"out{k}"], graph.add(PowerMeter(label=f"pm_{name}{k}"))["in"])
    return graph


# ---------------------------------------------------------------------------


def transmission(reading: object, reference: object) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies [Hz] and the trace over the reference, in dB, where the source has light."""
    assert isinstance(reading, OpticalSpectrum) and isinstance(reference, OpticalSpectrum)
    lit = reference.power_per_resolution() > 1e-3 * reference.power_per_resolution().max()
    ratio = reading.power_per_resolution()[lit] / reference.power_per_resolution()[lit]
    return np.asarray(reading.frequencies)[lit], 10.0 * np.log10(ratio)


def peaks(reading: object, reference: object, *, dips: bool = False) -> np.ndarray:
    """Frequencies [Hz] of the comb's maxima (or minima) in a transmission."""
    frequencies, db = transmission(reading, reference)
    trace = -db if dips else db
    level = trace.min() + 0.5 * (trace.max() - trace.min())
    inner = np.arange(1, trace.size - 1)
    found = inner[(trace[1:-1] > trace[:-2]) & (trace[1:-1] >= trace[2:]) & (trace[1:-1] > level)]
    return frequencies[found]


def spacing_ghz(reading: object, reference: object, *, dips: bool = False) -> float:
    """The mean spacing of the comb [GHz]."""
    found = np.sort(peaks(reading, reference, dips=dips))
    return float(np.mean(np.diff(found))) / 1e9


def fsr_ghz(n_group: float, length_um: float) -> float:
    """``c / (n_g L)`` [GHz]."""
    return C_LIGHT / (n_group * length_um * 1e-6) / 1e9


EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "coupler_length": (coupler_length, COUPLER_LAYOUT),
    "mzi": (mzi, MZI_LAYOUT),
    "ring": (ring, RING_LAYOUT),
    "birefringent_ring": (birefringent_ring, BIREF_LAYOUT),
    "chip_couplers": (chip_couplers, COUPLERS_LAYOUT),
    "pdk_splitter": (pdk_splitter, PDK_LAYOUT),
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
    if key == "coupler_length":
        print("   length     cross    sin^2(pi dn L / lambda)")
        for length in COUPLER_LENGTHS:
            ideal = np.sin(np.pi * DELTA_N * length / 1.55) ** 2
            print(f"   {length:6.2f} um  {cross_fraction(length):6.3f}   {ideal:6.3f}")
    results = results_by_label(graph)
    for label, value in results.items():
        if isinstance(value, PowerReading):
            print(f"   {label:12s} {value.power_dbm:8.3f} dBm")
        elif isinstance(value, OpticalSpectrum) and label != "osa_in":
            dips = label == "osa_through"
            found = peaks(value, results["osa_in"], dips=dips)
            _, db = transmission(value, results["osa_in"])
            spacing = np.mean(np.diff(np.sort(found))) / 1e9 if found.size > 1 else float("nan")
            print(
                f"   {label:12s} {found.size:3d} {'dips' if dips else 'peaks'}"
                f"  spacing {spacing:8.2f} GHz  from {db.min():7.2f} to {db.max():6.2f} dB"
            )


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
