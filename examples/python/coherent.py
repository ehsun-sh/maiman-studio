"""Phase 4 of the example roadmap: coherent links.

Each example is a studio project with its lesson: QPSK back to back with the
noise turned up, two tributaries on one wavelength through a rotated channel,
a receiver acquiring a carrier 20 GHz away, the 400ZR and 800ZR reference
designs, a pulse going round a recirculating loop, and 16-QAM with its inner
points sent more often than its outer ones.

``python coherent.py`` writes every project into ``examples/maiman/`` and
prints what each measures; ``python coherent.py pcs`` does one.
``tests/test_coherent.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

import lessons
import numpy as np
import reference_rates
from direct_detection import Layout, at

from maiman import Graph, SimulationContext
from maiman.component import Component, Port
from maiman.components import (
    EDFA,
    Attenuator,
    ButterflyEqualizer,
    CarrierRecovery,
    CoarseFrequencyRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    ConstellationDiagram,
    CWLaser,
    DelayLine,
    DirectionalCoupler,
    DualPolarizationReceiver,
    Feedback,
    FrequencyRecovery,
    GaussianPulse,
    IQDriver,
    IQModulator,
    IQSampler,
    OpticalSpectrumAnalyzer,
    Oscilloscope,
    OSNRMeter,
    PCSMapper,
    PolarizationCombiner,
    PolarizationRotator,
    PRBSGenerator,
    QAMMapper,
    Splitter,
    TimingRecovery,
)
from maiman.project import save
from maiman.signals import ConstellationMeasurement
from maiman.units import C_LIGHT

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"

SYMBOL_RATE = 32e9
V_PI = 4.0
ROLL_OFF = 0.2


def context(sequence_length: int = 8192, samples_per_symbol: int = 4) -> SimulationContext:
    return SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=samples_per_symbol,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )


def noise_loaded(
    graph: Graph,
    source: Component,
    pad_db: float,
    suffix: str = "",
) -> Component:
    """A variable attenuator and an amplifier that undoes it, as a bench loads noise.

    The power comes back and the amplifier's noise does not go away, so the
    attenuation is an OSNR knob that leaves everything after it untouched.
    """
    pad = graph.add(Attenuator(attenuation=pad_db, label=f"voa{suffix}"))
    loader = graph.add(EDFA(gain=pad_db, noise_figure=5.0, label=f"ase{suffix}"))
    graph.connect(source, pad["in"])
    graph.connect(pad, loader["in"])
    return loader


def single_pol_receiver(
    graph: Graph, signal: Component, reference: Port, suffix: str = ""
) -> ConstellationAnalyzer:
    """LO, 90-degree hybrid, sampler, analyser and a constellation to look at."""
    lo = graph.add(CWLaser(power=10.0, wavelength=1550.0, label=f"lo{suffix}"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label=f"rx{suffix}"))
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=ROLL_OFF, label=f"smp{suffix}"))
    analyzer = graph.add(ConstellationAnalyzer(label=f"vsa{suffix}"))
    diagram = graph.add(ConstellationDiagram(label=f"const{suffix}"))
    graph.connect(signal, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(reference, sampler["reference"])
    graph.connect(sampler["out"], analyzer["in"])
    graph.connect(reference, analyzer["reference"])
    graph.connect(sampler["out"], diagram["in"])
    return analyzer


def spectrum(label: str) -> OpticalSpectrumAnalyzer:
    return OpticalSpectrumAnalyzer(
        auto_span=False,
        center_wavelength=1550.0,
        span=1.6,
        points=512.0,
        resolution_bandwidth=12.5,
        label=label,
    )


# ---------------------------------------------------------------------------
# 4.1  QPSK back to back

QPSK_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "map": at(1, 0),
    "drv": at(2, 0),
    "tx": at(2, 1),
    "mod": at(3, 1),
    "voa": at(4, 1),
    "ase": at(5, 1),
    "lo": at(6, 0),
    "rx": at(7, 1),
    "smp": at(8, 1),
    "osnr": at(6, 3),
    "osa": at(7, 3),
    "vsa": at(9, 0),
    "const": at(9, 2),
}

#: The attenuation the project opens with: an OSNR where the cloud is plainly
#: a cloud and still nowhere near an error.
QPSK_PAD_DB = 35.0
QPSK_PADS = [26.0, 29.0, 32.0, 35.0, 38.0]
TX_DBM = -10.0


def qpsk_b2b(pad_db: float = QPSK_PAD_DB) -> Graph:
    graph = Graph(context())
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=2.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=2.0, label="map"))
    driver = graph.add(
        IQDriver(v_pi=V_PI, predistort=True, pulse_shaping=True, roll_off=ROLL_OFF, label="drv")
    )
    laser = graph.add(CWLaser(power=TX_DBM + 10.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    graph.chain(prbs, mapper, driver)
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    loaded = noise_loaded(graph, modulator, pad_db)
    graph.connect(loaded, graph.add(OSNRMeter(label="osnr"))["in"])
    graph.connect(loaded, graph.add(spectrum("osa"))["in"])
    single_pol_receiver(graph, loaded, mapper["out"])
    return graph


def qpsk_table() -> tuple[tuple[float, float, float, float], ...]:
    """(pad, OSNR, SNR, EVM %) for each attenuation the lesson tabulates."""
    rows = []
    for pad in QPSK_PADS:
        results = results_by_label(qpsk_b2b(pad))
        measured = results["vsa"]
        assert isinstance(measured, ConstellationMeasurement)
        osnr = results["osnr"]
        assert isinstance(osnr, float)
        rows.append((pad, osnr, measured.snr_db, measured.evm * 100))
    return tuple(rows)


# ---------------------------------------------------------------------------
# 4.8  Probabilistic shaping

PCS_LAYOUT: Layout = {
    "prbs_u": at(0, 0),
    "map_u": at(1, 0),
    "drv_u": at(2, 0),
    "tx_u": at(2, 1),
    "mod_u": at(3, 1),
    "voa_u": at(4, 1),
    "ase_u": at(5, 1),
    "lo_u": at(5, 0),
    "rx_u": at(6, 1),
    "smp_u": at(7, 1),
    "vsa_u": at(8, 0),
    "const_u": at(8, 1),
    "prbs_s": at(0, 3),
    "map_s": at(1, 3),
    "drv_s": at(2, 3),
    "tx_s": at(2, 4),
    "mod_s": at(3, 4),
    "voa_s": at(4, 4),
    "ase_s": at(5, 4),
    "lo_s": at(5, 3),
    "rx_s": at(6, 4),
    "smp_s": at(7, 4),
    "vsa_s": at(8, 3),
    "const_s": at(8, 4),
}

PCS_PAD_DB = 36.0
PCS_ENTROPY = 3.7
PCS_ENTROPIES = [4.0, 3.85, 3.7, 3.5, 3.28]


def peak_backoff_db(entropy: float) -> float:
    """How much further below its peak a shaped alphabet sits than the uniform one.

    The driver scales the outermost point to the modulator's full swing, so a
    shaped signal, its outer points rarer and further out at the same mean
    power, leaves the modulator weaker by this much. The project gives it back
    at the laser, so both branches launch the same mean power into the noise.
    """
    shaped = PCSMapper(alphabet=16.0, entropy=entropy).constellation()
    uniform = PCSMapper(alphabet=16.0, entropy=4.0).constellation()
    return 10.0 * math.log10(float(np.max(np.abs(shaped) ** 2) / np.max(np.abs(uniform) ** 2)))


def pcs(pad_db: float = PCS_PAD_DB, entropy: float = PCS_ENTROPY) -> Graph:
    """Uniform and shaped 16-QAM through the same noise, side by side."""
    graph = Graph(context())
    for suffix in ("_u", "_s"):
        if suffix == "_u":
            prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=4.0, label="prbs_u"))
            mapper: Component = graph.add(QAMMapper(bits_per_symbol=4.0, label="map_u"))
        else:
            prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=8.0, label="prbs_s"))
            mapper = graph.add(PCSMapper(alphabet=16.0, entropy=entropy, label="map_s"))
        driver = graph.add(
            IQDriver(
                v_pi=V_PI,
                predistort=True,
                pulse_shaping=True,
                roll_off=ROLL_OFF,
                label=f"drv{suffix}",
            )
        )
        launch = TX_DBM + 10.0 + (peak_backoff_db(entropy) if suffix == "_s" else 0.0)
        laser = graph.add(CWLaser(power=round(launch, 2), wavelength=1550.0, label=f"tx{suffix}"))
        modulator = graph.add(IQModulator(v_pi=V_PI, label=f"mod{suffix}"))
        graph.chain(prbs, mapper, driver)
        graph.connect(laser, modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        loaded = noise_loaded(graph, modulator, pad_db, suffix)
        single_pol_receiver(graph, loaded, mapper["out"], suffix)
    return graph


def pcs_table() -> tuple[tuple[float, float, float, float], ...]:
    """(entropy, back-off, SNR, MI) of the shaped branch at the project's noise."""
    rows = []
    for entropy in PCS_ENTROPIES:
        measured = results_by_label(pcs(entropy=entropy))["vsa_s"]
        assert isinstance(measured, ConstellationMeasurement)
        rows.append(
            (entropy, peak_backoff_db(entropy), measured.snr_db, measured.mutual_information)
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# 4.4  Two polarisations through a rotated channel

DUALPOL_LAYOUT: Layout = {
    "prbs_x": at(0, 0),
    "map_x": at(1, 0),
    "drv_x": at(2, 0),
    "mod_x": at(3, 1),
    "tx": at(0, 2),
    "sp": at(1, 2),
    "prbs_y": at(0, 4),
    "map_y": at(1, 4),
    "drv_y": at(2, 4),
    "mod_y": at(3, 3),
    "pbc": at(4, 2),
    "rot": at(5, 2),
    "rx": at(6, 2),
    "lo": at(6, 3),
    "smp_x": at(7, 1),
    "smp_y": at(7, 3),
    "const_raw": at(7, 4.3),
    "eq": at(8, 2),
    "cr_x": at(9, 1),
    "cr_y": at(9, 3),
    "vsa_x": at(10, 0),
    "const_x": at(10, 1),
    "vsa_y": at(10, 3),
    "const_y": at(10, 4),
}

ROTATION = 30.0


def dualpol(rotation: float = ROTATION) -> Graph:
    graph = Graph(context(sequence_length=4096))
    laser = graph.add(CWLaser(power=0.0, linewidth=100.0, label="tx"))
    splitter = graph.add(Splitter(2, label="sp"))
    graph.connect(laser, splitter["in"])
    mappers: dict[str, Component] = {}
    modulators: dict[str, Component] = {}
    for index, axis in enumerate(("x", "y")):
        prbs = graph.add(
            PRBSGenerator(
                order=23.0 if axis == "x" else 15.0, bits_per_symbol=4.0, label=f"prbs_{axis}"
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label=f"map_{axis}"))
        driver = graph.add(IQDriver(label=f"drv_{axis}"))
        modulator = graph.add(IQModulator(label=f"mod_{axis}"))
        graph.chain(prbs, mapper, driver)
        graph.connect(splitter[f"out{index}"], modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        mappers[axis], modulators[axis] = mapper, modulator
    combiner = graph.add(PolarizationCombiner(label="pbc"))
    graph.connect(modulators["x"], combiner["x"])
    graph.connect(modulators["y"], combiner["y"])
    rotator = graph.add(PolarizationRotator(angle=rotation, phase=25.0, label="rot"))
    graph.connect(combiner, rotator["in"])
    lo = graph.add(CWLaser(power=13.0, linewidth=100.0, label="lo"))
    receiver = graph.add(DualPolarizationReceiver(label="rx"))
    graph.connect(rotator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    samplers: dict[str, Component] = {}
    for axis in ("x", "y"):
        sampler = graph.add(IQSampler(label=f"smp_{axis}"))
        graph.connect(receiver[f"{axis}i"], sampler["i"])
        graph.connect(receiver[f"{axis}q"], sampler["q"])
        graph.connect(mappers[axis]["out"], sampler["reference"])
        samplers[axis] = sampler
    # What the receiver's x branch holds before anything separates it.
    graph.connect(samplers["x"]["out"], graph.add(ConstellationDiagram(label="const_raw"))["in"])
    equalizer = graph.add(ButterflyEqualizer(label="eq"))
    graph.connect(samplers["x"]["out"], equalizer["x"])
    graph.connect(samplers["y"]["out"], equalizer["y"])
    for axis in ("x", "y"):
        recovery = graph.add(CarrierRecovery(label=f"cr_{axis}"))
        graph.connect(equalizer[f"{axis}_out"], recovery["in"])
        analyzer = graph.add(ConstellationAnalyzer(ignore_edges=128.0, label=f"vsa_{axis}"))
        graph.connect(recovery["out"], analyzer["in"])
        graph.connect(mappers[axis]["out"], analyzer["reference"])
        graph.connect(recovery["out"], graph.add(ConstellationDiagram(label=f"const_{axis}"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 4.5  Acquiring a carrier far from the LO

ACQ_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "map": at(1, 0),
    "drv": at(2, 0),
    "tx": at(2, 1),
    "mod": at(3, 1),
    "lo": at(4, 0),
    "rx": at(4, 1),
    "acq": at(5, 1),
    "tr": at(6, 1),
    "smp": at(7, 1),
    "fo": at(8, 1),
    "vsa": at(9, 0),
    "const": at(9, 2),
}

ACQ_OFFSET = 20e9
#: Where the fourth-power estimate folds: symbol_rate / 8 for square QAM.
FINE_LIMIT = SYMBOL_RATE / 8.0


def acquisition(offset: float = ACQ_OFFSET) -> Graph:
    graph = Graph(
        SimulationContext(bit_rate=SYMBOL_RATE, samples_per_symbol=16, sequence_length=2048, seed=7)
    )
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label="map"))
    driver = graph.add(
        IQDriver(v_pi=V_PI, predistort=True, pulse_shaping=True, roll_off=ROLL_OFF, label="drv")
    )
    laser = graph.add(CWLaser(power=2.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    detuned = C_LIGHT / (C_LIGHT / 1550e-9 - offset) * 1e9
    lo = graph.add(CWLaser(power=10.0, wavelength=detuned, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    acquire = graph.add(CoarseFrequencyRecovery(label="acq"))
    timing = graph.add(TimingRecovery(label="tr"))
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=ROLL_OFF, label="smp"))
    fine = graph.add(FrequencyRecovery(label="fo"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))
    diagram = graph.add(ConstellationDiagram(label="const"))
    graph.chain(prbs, mapper, driver)
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], acquire["i"])
    graph.connect(receiver["q"], acquire["q"])
    graph.connect(acquire["i"], timing["i"])
    graph.connect(acquire["q"], timing["q"])
    graph.connect(timing["i"], sampler["i"])
    graph.connect(timing["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], fine["in"])
    graph.connect(fine["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    graph.connect(fine["out"], diagram["in"])
    return graph


# ---------------------------------------------------------------------------
# 4.6  400ZR and 800ZR


ZR = {
    "zr400": reference_rates.CONFIGURATIONS["400G DP-16QAM"],
    "zr800": reference_rates.CONFIGURATIONS["800G DP-16QAM"],
}


def zr(key: str) -> Callable[[], Graph]:
    rate, bits, _ = ZR[key]

    def build() -> Graph:
        return reference_rates.build(rate, bits, span_km=reference_rates.SPAN_KM)[0]

    return build


ZR_LAYOUT: Layout = {
    "prbs_x": at(0, 0),
    "map_x": at(1, 0),
    "drv_x": at(2, 0),
    "prbs_y": at(0, 4),
    "map_y": at(1, 4),
    "drv_y": at(2, 4),
    "tx": at(1, 2),
    "pbs": at(2, 2),
    "mod_x": at(3, 1),
    "mod_y": at(3, 3),
    "pbc": at(4, 2),
    "fib": at(5, 2),
    "edfa": at(6, 2),
    "osnr": at(7, 0),
    "rx": at(7, 2),
    "lo": at(7, 3),
    "cdc_x": at(8, 1),
    "cdc_y": at(8, 3),
    "smp_x": at(9, 1),
    "smp_y": at(9, 3),
    "eq": at(10, 2),
    "cr_x": at(11, 1),
    "cr_y": at(11, 3),
    "vsa_x": at(12, 1),
    "vsa_y": at(12, 3),
}


# ---------------------------------------------------------------------------
# 4.7  A recirculating loop

LOOP_LAYOUT: Layout = {
    "pulse": at(0, 1),
    "coupler": at(1, 1),
    "delay": at(2, 0),
    "loss": at(3, 0),
    "loop": at(3, 2),
    "scope": at(4, 1),
}

LOOP_PS = 800.0
LOOP_LOSS_DB = 1.0
LAPS = 4


def loop() -> Graph:
    graph = Graph(SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64))
    pulse = graph.add(GaussianPulse(width=10.0, label="pulse"))
    coupler = graph.add(DirectionalCoupler(True, coupling=0.5, label="coupler"))
    delay = graph.add(DelayLine(delay=LOOP_PS, label="delay"))
    loss = graph.add(Attenuator(attenuation=LOOP_LOSS_DB, label="loss"))
    feedback = graph.add(Feedback(passes=float(LAPS), label="loop"))
    scope = graph.add(Oscilloscope(points=2048.0, label="scope"))
    graph.connect(pulse, coupler["in1"])
    graph.connect(coupler["out2"], delay["in"])
    graph.connect(delay["out"], loss["in"])
    graph.connect(loss["out"], feedback["in"])
    graph.connect(feedback["out"], coupler["in2"])
    graph.connect(coupler["out1"], scope["in"])
    return graph


# ---------------------------------------------------------------------------

EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "qpsk_b2b": (qpsk_b2b, QPSK_LAYOUT),
    "dualpol": (dualpol, DUALPOL_LAYOUT),
    "acquisition": (acquisition, ACQ_LAYOUT),
    "zr400": (zr("zr400"), ZR_LAYOUT),
    "zr800": (zr("zr800"), ZR_LAYOUT),
    "loop": (loop, LOOP_LAYOUT),
    "pcs": (pcs, PCS_LAYOUT),
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
    if key == "qpsk_b2b":
        print("   pad      OSNR      SNR     EVM")
        for pad, osnr, snr, evm in qpsk_table():
            print(f"   {pad:4.0f} dB {osnr:6.2f} dB {snr:6.2f} dB {evm:5.2f} %")
    if key == "pcs":
        print("   entropy  back-off   SNR      MI")
        for entropy, backoff, snr, mi in pcs_table():
            print(f"   {entropy:5.2f}   {backoff:5.2f} dB {snr:6.2f} dB  {mi:5.3f}")
    for label, value in results_by_label(graph).items():
        if isinstance(value, ConstellationMeasurement):
            print(
                f"   {label:8s} EVM {value.evm * 100:6.2f} %  SNR {value.snr_db:6.2f} dB"
                f"  errors {value.symbol_errors:5d}  H {value.entropy:5.3f}"
                f"  MI {value.mutual_information:5.3f}"
            )
        elif isinstance(value, float):
            print(f"   {label:8s} {value:7.2f} dB")


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
