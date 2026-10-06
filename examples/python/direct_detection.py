"""Phase 2 of the example roadmap: direct-detection links.

The links most fibre in the world still carries, each one a studio project with
its lesson: what an avalanche photodiode buys a receiver, what a directly
modulated laser's chirp costs it, why a Fabry-Perot laser does not reach far,
how a PAM4 lane gets its eye back, a four-lane CWDM link in the O-band, and the
downstream of a passive optical network.

``python direct_detection.py`` writes every project into ``examples/maiman/`` and
prints what each measures; ``python direct_detection.py gpon`` does one.
``tests/test_direct_detection.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path

import dml_reach
import lessons
import numpy as np

from maiman import Graph, SimulationContext, sweep
from maiman.component import Component
from maiman.components import (
    APDPhotodiode,
    Attenuator,
    BERAnalyzer,
    CWLaser,
    Demultiplexer,
    ElectricalFilter,
    EyeDiagram,
    FabryPerotLaser,
    FFEDFEEqualizer,
    Fiber,
    MachZehnderModulator,
    Multiplexer,
    NRZDriver,
    OpticalSpectrumAnalyzer,
    Oscilloscope,
    PAM4Driver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
    Splitter,
)
from maiman.project import save
from maiman.signals import EyeMeasurement, PAMMeasurement, PowerReading, ScopeTrace

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"

Layout = dict[str, dict[str, float]]


def col(i: float) -> float:
    """The x of the i-th column on the canvas."""
    return 30.0 + 150.0 * i


def row(j: float) -> float:
    """The y of the j-th row on the canvas."""
    return 40.0 + 100.0 * j


def at(i: float, j: float) -> dict[str, float]:
    return {"x": col(i), "y": row(j)}


def ook_transmitter(
    graph: Graph, *, wavelength: float = 1550.0, power: float = 0.0, suffix: str = ""
) -> tuple[Component, Component]:
    """A CW laser on a Mach-Zehnder, the way every OOK example here is driven.

    Returns ``(laser, modulator)``; the caller connects a driver to the
    modulator's electrical input. A one is the low voltage, because the
    push-pull modulator transmits fully at zero volts.
    """
    laser = graph.add(CWLaser(power=power, wavelength=wavelength, label=f"tx{suffix}"))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0, label=f"mzm{suffix}"))
    graph.connect(laser, modulator["optical_in"])
    return laser, modulator


def receive(
    graph: Graph,
    source: object,
    reference: Component,
    *,
    detector: Component,
    bandwidth_ghz: float,
    suffix: str = "",
) -> BERAnalyzer:
    """Detector, filter and error counter, the receiver every example here shares."""
    lpf = graph.add(ElectricalFilter(bandwidth=bandwidth_ghz, label=f"lpf{suffix}"))
    ber = graph.add(BERAnalyzer(label=f"ber{suffix}"))
    graph.connect(source, detector["in"])  # type: ignore[arg-type]
    graph.connect(detector, lpf["in"])
    graph.connect(lpf, ber["in"])
    graph.connect(reference, ber["reference"])
    return ber


# ---------------------------------------------------------------------------
# 2.2  Receiver sensitivity: PIN against APD

SENSITIVITY_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "drv": at(1, 0),
    "tx": at(0, 1.5),
    "mzm": at(1, 1.5),
    "att": at(2, 1.5),
    "split": at(3, 1.5),
    "pm_rx": at(4, 0),
    "pin": at(4, 1.5),
    "lpf_pin": at(5, 1.5),
    "ber_pin": at(6, 1.5),
    "apd": at(4, 3),
    "lpf_apd": at(5, 3),
    "ber_apd": at(6, 3),
}

#: InGaAs at 1550 nm, near the top of its usual 0.3 to 0.5.
APD_K = 0.5
#: The attenuations the sensitivity table sweeps [dB].
SENSITIVITY_SWEEP = [float(a) for a in range(4, 31, 2)]


def receiver_sensitivity() -> Graph:
    """10 Gb/s OOK through a variable attenuator into a PIN and an APD side by side."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=2048, seed=3)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    _, mzm = ook_transmitter(graph)
    graph.connect(driver, mzm["electrical_in"])
    att = graph.add(Attenuator(attenuation=20.0, label="att"))
    # Three equal copies: one for the meter, one for each receiver. The meter
    # then reads exactly what each photodiode is offered.
    split = graph.add(Splitter(3, label="split"))
    graph.chain(mzm, att, split)
    graph.connect(split["out0"], graph.add(PowerMeter(label="pm_rx"))["in"])
    pin = graph.add(PINPhotodiode(label="pin"))
    apd = graph.add(APDPhotodiode(gain=10.0, ionization_ratio=APD_K, label="apd"))
    receive(graph, split["out1"], prbs, detector=pin, bandwidth_ghz=7.0, suffix="_pin")
    receive(graph, split["out2"], prbs, detector=apd, bandwidth_ghz=7.0, suffix="_apd")
    return graph


def sensitivity_table(graph: Graph) -> list[tuple[float, float, float]]:
    """(received dBm, PIN Q, APD Q) at every attenuation of the sweep."""
    by = {c.label: c for c in graph.components}
    table = []
    for point in sweep(graph, {(by["att"], "attenuation"): SENSITIVITY_SWEEP}):
        run = point.runs[0]
        table.append(
            (
                power_dbm(run[by["pm_rx"]]),
                eye(run[by["ber_pin"]]).q_factor,
                eye(run[by["ber_apd"]]).q_factor,
            )
        )
    return table


def sensitivity_at(q_target: float, powers: list[float], qs: list[float]) -> float:
    """Received power [dBm] where Q falls through ``q_target``, interpolated in log Q."""
    for (p0, q0), (p1, q1) in zip(
        zip(powers, qs, strict=True), zip(powers[1:], qs[1:], strict=True), strict=True
    ):
        if q0 >= q_target > q1:
            f = (math.log(q0) - math.log(q_target)) / (math.log(q0) - math.log(q1))
            return p0 + f * (p1 - p0)
    raise ValueError(f"Q never crosses {q_target} on this sweep")


#: The gains the optimum-gain table sweeps, at a fixed received power.
APD_GAINS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0]


def apd_gain_table(graph: Graph, attenuation: float = 20.0) -> list[tuple[float, float]]:
    by = {c.label: c for c in graph.components}
    curve = sweep(
        graph, {(by["apd"], "gain"): APD_GAINS}, fixed={(by["att"], "attenuation"): attenuation}
    )
    return [
        (gain, eye(point.runs[0][by["ber_apd"]]).q_factor)
        for gain, point in zip(APD_GAINS, curve, strict=True)
    ]


# ---------------------------------------------------------------------------
# 2.3  The reach of a directly modulated laser

DML_LAYOUT: Layout = {
    "prbs": at(0, 1.5),
    "drv_dml": at(1, 0),
    "dml": at(2, 0),
    "fibre_dml": at(3, 0),
    "pin_dml": at(4, 0),
    "lpf_dml": at(5, 0),
    "ber_dml": at(6, 0),
    "chirp": at(3, 1),
    "drv_ext": at(1, 3),
    "cw": at(1, 2),
    "mzm": at(2, 3),
    "fibre_ext": at(3, 3),
    "pin_ext": at(4, 3),
    "lpf_ext": at(5, 3),
    "ber_ext": at(6, 3),
}

DML_SPAN_KM = 20.0


@cache
def matched_external() -> tuple[float, float]:
    """(CW power [dBm], extinction [dB]) that give the MZM the DML's own average and ER."""
    return round(dml_reach.external_power_dbm(), 2), round(dml_reach.external_extinction(), 2)


def dml_link() -> Graph:
    """The same pattern from a DML and from a CW laser on an MZM, through 20 km each."""
    graph = Graph(dml_reach.context())
    prbs = graph.add(PRBSGenerator(order=7.0, label="prbs"))
    power, extinction = matched_external()

    # More current is more light, so a one is the high voltage.
    drv_dml = graph.add(NRZDriver(v_low=0.0, v_high=1.0, label="drv_dml"))
    dml = graph.add(dml_reach.laser())
    graph.connect(prbs, drv_dml["in"])
    graph.connect(drv_dml, dml["in"])

    drv_ext = graph.add(NRZDriver(v_low=1.0, v_high=0.0, label="drv_ext"))
    cw = graph.add(CWLaser(power=power, wavelength=dml_reach.WAVELENGTH_NM, label="cw"))
    mzm = graph.add(MachZehnderModulator(v_pi=1.0, extinction_ratio=extinction, label="mzm"))
    graph.connect(prbs, drv_ext["in"])
    graph.connect(cw, mzm["optical_in"])
    graph.connect(drv_ext, mzm["electrical_in"])

    # A nanosecond of the DML's output: ten bits, enough to see each edge chirp.
    graph.connect(dml, graph.add(Oscilloscope(span=1000.0, label="chirp"))["in"])

    for source, name in ((dml, "dml"), (mzm, "ext")):
        fibre = graph.add(
            Fiber(length=DML_SPAN_KM, attenuation=0.2, dispersion=17.0, label=f"fibre_{name}")
        )
        graph.connect(source, fibre["in"])
        receive(
            graph,
            fibre["out"],
            prbs,
            detector=graph.add(PINPhotodiode(label=f"pin_{name}")),
            bandwidth_ghz=7.5,
            suffix=f"_{name}",
        )
    return graph


# ---------------------------------------------------------------------------
# 2.4  Laser noise: a Fabry-Perot laser's partition against a DFB

PARTITION_LAYOUT: Layout = {
    "prbs": at(0, 3),
    "drv": at(1, 3),
    **{
        f"{part}_{name}": at(i, j)
        for j, name in ((0, "dfb"), (1, "fp_steady"), (2, "fp"))
        for i, part in ((2, "tx"), (3, "mzm"), (4, "fibre"), (5, "pin"), (6, "lpf"), (7, "ber"))
    },
}

PARTITION_KM = 3.0
PARTITION_RATE = 2.5e9


def mode_partition() -> Graph:
    """One pattern on three lasers, through 3 km each: a DFB, and a Fabry-Perot twice.

    The Fabry-Perot laser runs once with its Langevin forces off -- its seven
    modes at their steady powers, still spread by the fibre -- and once with
    them on, trading power between modes. The difference between those two
    receivers is mode partition noise and nothing else.
    """
    ctx = SimulationContext(
        bit_rate=PARTITION_RATE, samples_per_symbol=8, sequence_length=512, seed=5
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=9.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    sources: dict[str, Component] = {
        "dfb": CWLaser(power=0.0, wavelength=1550.0, label="tx_dfb"),
        "fp_steady": FabryPerotLaser(
            power=0.0, wavelength=1550.0, partition_noise=False, label="tx_fp_steady"
        ),
        "fp": FabryPerotLaser(power=0.0, wavelength=1550.0, label="tx_fp"),
    }
    for name, source in sources.items():
        graph.add(source)
        mzm = graph.add(MachZehnderModulator(v_pi=4.0, label=f"mzm_{name}"))
        graph.connect(source, mzm["optical_in"])
        graph.connect(driver, mzm["electrical_in"])
        # The walk-off is what turns partition into noise: each mode's own
        # delay is carried to the detector instead of cancelling in its frame.
        fibre = graph.add(
            Fiber(
                length=PARTITION_KM,
                attenuation=0.2,
                dispersion=17.0,
                nonlinearity=0.0,
                four_wave_mixing=False,
                carry_walkoff=True,
                label=f"fibre_{name}",
            )
        )
        graph.connect(mzm, fibre["in"])
        receive(
            graph,
            fibre["out"],
            prbs,
            detector=graph.add(PINPhotodiode(label=f"pin_{name}")),
            bandwidth_ghz=0.7 * PARTITION_RATE / 1e9,
            suffix=f"_{name}",
        )
    return graph


# ---------------------------------------------------------------------------
# 2.5  A 53 Gb/s PAM4 lane with an equaliser

PAM4_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "pam4": at(1, 0),
    "tx": at(1, 1),
    "mzm": at(2, 0.5),
    "pin": at(3, 0.5),
    "lpf": at(4, 0.5),
    "eye": at(5, -0.5),
    "eq_off": at(5, 0.5),
    "eq": at(5, 1.5),
}

PAM4_RATE = 26.5625e9
PAM4_RECEIVER_GHZ = 7.0
#: Twice what ``pam4_lane.py`` uses, so the eye has 32 columns across its two
#: symbols rather than 16 -- an eye drawn from 16 sampling instants is a bar chart.
PAM4_SAMPLES = 16


def pam4_lane() -> Graph:
    """One 200G-class lane: PAM4 at 26.5625 GBd into a 7 GHz receiver, with and without DSP."""
    ctx = SimulationContext(
        bit_rate=PAM4_RATE, samples_per_symbol=PAM4_SAMPLES, sequence_length=4096
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=2.0, label="prbs"))
    driver = graph.add(PAM4Driver(v_low=3.2, v_high=0.8, predistort=True, label="pam4"))
    laser = graph.add(CWLaser(power=0.0, wavelength=1310.0, label="tx"))
    mzm = graph.add(MachZehnderModulator(v_pi=4.0, label="mzm"))
    pin = graph.add(PINPhotodiode(label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=PAM4_RECEIVER_GHZ, label="lpf"))
    graph.connect(prbs, driver["in"])
    graph.connect(laser, mzm["optical_in"])
    graph.connect(driver, mzm["electrical_in"])
    graph.chain(mzm, pin, lpf)
    graph.connect(lpf, graph.add(EyeDiagram(label="eye"))["in"])
    # One tap and no feedback is a plain slicer at the best instant: the
    # receiver with no equalisation, measured the same way as the one with it.
    for label, ffe, dfe in (("eq_off", 1.0, 0.0), ("eq", 9.0, 2.0)):
        eq = graph.add(FFEDFEEqualizer(ffe_taps=ffe, dfe_taps=dfe, label=label))
        graph.connect(lpf, eq["in"])
        graph.connect(prbs, eq["reference"])
    return graph


# ---------------------------------------------------------------------------
# 2.6  CWDM4: four lanes on the coarse grid in the O-band

CWDM_LAYOUT: Layout = {
    "prbs": at(0, 4),
    "drv": at(1, 4),
    **{f"tx{i}": at(0, i) for i in range(4)},
    **{f"mzm{i}": at(1, i) for i in range(4)},
    "mux": at(2, 1.5),
    "fibre": at(3, 1.5),
    "osa": at(4, 0),
    "demux": at(4, 1.5),
    **{f"pin{i}": at(5, i) for i in range(4)},
    **{f"lpf{i}": at(6, i) for i in range(4)},
    **{f"ber{i}": at(7, i) for i in range(4)},
}

CWDM_RATE = 25.78125e9
CWDM_KM = 10.0
#: Standard fibre's zero-dispersion wavelength, and its slope there.
ZERO_DISPERSION_NM = 1310.0
DISPERSION_SLOPE = 0.092  # ps/(nm^2 km)


def cwdm4() -> Graph:
    """100G CWDM4/LR4-style: four 25G NRZ lanes, 1331 to 1271 nm, over 10 km, no amplifier."""
    ctx = SimulationContext(bit_rate=CWDM_RATE, samples_per_symbol=8, sequence_length=1024, seed=4)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    grid = {
        "first_wavelength": 1331.0,
        "wavelength_spacing": 20.0,
        # 13 nm of flat top in a 20 nm slot, as G.694.2 leaves room for
        # uncooled lasers to drift.
        "bandwidth": 2300.0,
        "order": 4.0,
        "insertion_loss": 2.0,
    }
    mux = graph.add(Multiplexer(4, label="mux", **grid))
    demux = graph.add(Demultiplexer(4, label="demux", **grid))
    for index, wavelength in enumerate(mux.channel_wavelengths()):
        _, mzm = ook_transmitter(graph, wavelength=wavelength * 1e9, suffix=str(index))
        graph.connect(driver, mzm["electrical_in"])
        graph.connect(mzm, mux[f"in{index}"])
    fibre = graph.add(
        Fiber(
            length=CWDM_KM,
            attenuation=0.35,
            dispersion=0.0,
            reference_wavelength=ZERO_DISPERSION_NM,
            dispersion_slope=DISPERSION_SLOPE,
            label="fibre",
        )
    )
    graph.connect(mux, fibre["in"])
    graph.connect(fibre["out"], demux["in"])
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            auto_span=False,
            center_wavelength=1301.0,
            span=12000.0,
            points=2048.0,
            resolution_bandwidth=12.5,
            label="osa",
        )
    )
    graph.connect(fibre["out"], osa["in"])
    for index in range(4):
        receive(
            graph,
            demux[f"out{index}"],
            prbs,
            detector=graph.add(PINPhotodiode(label=f"pin{index}")),
            bandwidth_ghz=0.75 * CWDM_RATE / 1e9,
            suffix=str(index),
        )
    return graph


def lane_dispersion(wavelength_nm: float) -> float:
    """D at a lane [ps/(nm km)], linear in the slope about the zero."""
    return DISPERSION_SLOPE * (wavelength_nm - ZERO_DISPERSION_NM)


# ---------------------------------------------------------------------------
# 2.7  GPON downstream

GPON_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "drv": at(1, 0),
    "olt": at(0, 1),
    "mzm": at(1, 1),
    "feeder": at(2, 1),
    "split32": at(3, 1),
    "pm_onu": at(4, 0),
    "apd": at(4, 1),
    "lpf": at(5, 1),
    "ber": at(6, 1),
}

GPON_RATE = 2.48832e9
GPON_FEEDER_KM = 20.0


def gpon() -> Graph:
    """2.488 Gb/s at 1490 nm from the OLT, 20 km of feeder, a 1:32 split, an APD at an ONU."""
    ctx = SimulationContext(bit_rate=GPON_RATE, samples_per_symbol=8, sequence_length=2048, seed=6)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    laser = graph.add(CWLaser(power=3.0, wavelength=1490.0, label="olt"))
    mzm = graph.add(MachZehnderModulator(v_pi=4.0, label="mzm"))
    graph.connect(laser, mzm["optical_in"])
    graph.connect(driver, mzm["electrical_in"])
    feeder = graph.add(
        Fiber(length=GPON_FEEDER_KM, attenuation=0.25, dispersion=13.0, label="feeder")
    )
    # 15.05 dB of ideal split plus the excess a real 1:32 PLC splitter adds.
    split = graph.add(Splitter(32, excess_loss=1.5, label="split32"))
    graph.connect(mzm, feeder["in"])
    graph.connect(feeder["out"], split["in"])
    graph.connect(split["out0"], graph.add(PowerMeter(label="pm_onu"))["in"])
    apd = graph.add(APDPhotodiode(gain=10.0, ionization_ratio=APD_K, label="apd"))
    receive(graph, split["out1"], prbs, detector=apd, bandwidth_ghz=1.9)
    return graph


# ---------------------------------------------------------------------------

#: key -> (builder, layout). The key names the project file and its lesson.
EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "receiver_sensitivity": (receiver_sensitivity, SENSITIVITY_LAYOUT),
    "dml_reach": (dml_link, DML_LAYOUT),
    "mode_partition": (mode_partition, PARTITION_LAYOUT),
    "pam4_lane": (pam4_lane, PAM4_LAYOUT),
    "cwdm4": (cwdm4, CWDM_LAYOUT),
    "gpon": (gpon, GPON_LAYOUT),
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


def power_dbm(value: object) -> float:
    assert isinstance(value, PowerReading)
    return value.power_dbm


def eye(value: object) -> EyeMeasurement:
    assert isinstance(value, EyeMeasurement)
    return value


def pam(value: object) -> PAMMeasurement:
    assert isinstance(value, PAMMeasurement)
    return value


def report(key: str, graph: Graph) -> None:
    if key == "receiver_sensitivity":
        table = sensitivity_table(graph)
        print("   received    PIN Q    APD Q")
        for p, q_pin, q_apd in table:
            print(f"   {p:7.2f} dBm {q_pin:7.2f} {q_apd:8.2f}")
        powers = [p for p, _, _ in table]
        pin = sensitivity_at(6.0, powers, [q for _, q, _ in table])
        apd = sensitivity_at(6.0, powers, [q for _, _, q in table])
        print(f"   sensitivity at Q = 6: PIN {pin:.1f} dBm, APD {apd:.1f} dBm")
        for gain, q in apd_gain_table(graph):
            print(f"   APD gain {gain:4.0f}  Q {q:6.2f}")
        return

    results = results_by_label(graph)
    for label, value in results.items():
        if isinstance(value, EyeMeasurement):
            print(f"   {label:14s} Q {value.q_factor:8.2f}  errors {value.errors}")
        elif isinstance(value, PAMMeasurement):
            print(
                f"   {label:14s} SNR {value.snr_db:6.2f} dB  "
                f"symbol errors {value.symbol_errors}/{value.symbols_evaluated}"
            )
        elif isinstance(value, PowerReading):
            print(f"   {label:14s} {value.power_dbm:7.2f} dBm")
        elif isinstance(value, ScopeTrace):
            lit = np.isfinite(value.chirp_hz)
            print(
                f"   {label:14s} chirp {np.nanmin(value.chirp_hz[lit]) / 1e9:+.1f} to "
                f"{np.nanmax(value.chirp_hz[lit]) / 1e9:+.1f} GHz"
            )


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
