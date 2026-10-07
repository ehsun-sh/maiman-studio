"""Phase 7 of the example roadmap: real-world systems.

Each example is a system an operator or a laboratory would recognise, built
from the blocks the earlier phases introduced and the few this phase adds: a
metro ring of ROADM nodes, a transoceanic cable, a 400ZR pluggable on a DWDM
line, radio carried on light, a laser link through open air, a LiDAR that
reads range off a beat note, and a quantum key sent one photon at a time.

``python systems.py`` writes every project into ``examples/maiman/`` and prints
what each measures; ``python systems.py rof`` does one.
``tests/test_systems.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

import coherent
import lessons
import numpy as np
import reference_rates
from chip import results_by_label, transmission
from direct_detection import Layout, at, ook_transmitter, receive
from sensing import spectrum

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    AmplifiedLine,
    Attenuator,
    BB84Receiver,
    Combiner,
    CWLaser,
    DelayLine,
    Demultiplexer,
    DirectionalCoupler,
    ElectricalSpectrumAnalyzer,
    Fiber,
    FreeSpaceChannel,
    GaussianPulse,
    MachZehnderModulator,
    Multiplexer,
    NRZDriver,
    PINPhotodiode,
    PowerMeter,
    PRBSGenerator,
    RFTone,
    Splitter,
    SweptLaser,
    WavelengthSelectiveSwitch,
)
from maiman.project import save
from maiman.signals import OpticalSignal
from maiman.units import C_LIGHT

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"


def nm_at(frequency: float) -> float:
    return C_LIGHT / frequency * 1e9


# ---------------------------------------------------------------------------
# 7.1  A metro ring of ROADM nodes: the passband narrows

NODES = 8
WSS_BANDWIDTH, WSS_ORDER = 50.0, 3.0
PROBE_PS = 4.0
F_CHANNEL = C_LIGHT / 1550e-9

RING_LAYOUT: Layout = {
    "probe": at(0, 1),
    "osa_in": at(0, 2.5),
    "add": at(0, -0.5),
    **{f"node{k}": at(k, 1) for k in range(1, NODES + 1)},
    "osa_1": at(1.5, 2.5),
    f"osa_{NODES}": at(NODES + 1, 1),
}


def ring_context() -> SimulationContext:
    """400 GHz of sampled bandwidth: room for the probe's whole spectrum."""
    return SimulationContext(bit_rate=25e9, samples_per_symbol=16, sequence_length=64, seed=1)


def ring_osa(label: str) -> Component:
    return spectrum(label, centre=1550.0, span=300.0, resolution=1.0)


def metro_ring() -> Graph:
    """A 4 ps pulse expressed through eight switches; node 1 and node 8 are watched."""
    graph = Graph(ring_context())
    probe = graph.add(GaussianPulse(peak_power=20.0, width=PROBE_PS, label="probe"))
    graph.connect(probe, graph.add(ring_osa("osa_in"))["in"])
    # Every node drops channel 1 and adds its own on the same slot.
    add = graph.add(CWLaser(power=-40.0, wavelength=nm_at(F_CHANNEL + 100e9), label="add"))
    previous = probe["out"]
    for k in range(1, NODES + 1):
        node = graph.add(
            WavelengthSelectiveSwitch(4, bandwidth=WSS_BANDWIDTH, order=WSS_ORDER, label=f"node{k}")
        )
        graph.connect(previous, node["in"])
        graph.connect(add, node["add"])
        previous = node["out"]
        if k in (1, NODES):
            graph.connect(node["out"], graph.add(ring_osa(f"osa_{k}"))["in"])
    return graph


def passband_ghz(reading: object, reference: object) -> tuple[float, float]:
    """The channel's -3 dB width [GHz] and its peak transmission [dB]."""
    frequencies, db = transmission(reading, reference)
    near = np.abs(frequencies - F_CHANNEL) < 60e9
    top = float(db[near].max())
    inside = frequencies[near][db[near] > top - 3.0]
    return float(inside.max() - inside.min()) / 1e9, top


def cascade_width_ghz(
    nodes: int, bandwidth: float = WSS_BANDWIDTH, order: float = WSS_ORDER
) -> float:
    """Super-Gaussian passbands in cascade: ``B N^(-1/2m)``."""
    return bandwidth * nodes ** (-1.0 / (2.0 * order))


# ---------------------------------------------------------------------------
# 7.2  A transoceanic segment: 6000 km and a hundred repeaters

CABLE_NM = tuple(1532.0 + 4.0 * index for index in range(8))


def cable(flatten: bool = False) -> AmplifiedLine:
    """100 spans of 60 km of 0.16 dB/km fibre, repeaters of 4.5 dB noise figure."""
    return AmplifiedLine(
        spans=100.0,
        span_length=60.0,
        attenuation=0.16,
        noise_figure=4.5,
        flatten=flatten,
        label="cable",
    )


CABLE_LAYOUT: Layout = {
    **{f"ch{index}": at(0, index * 0.6 - 0.6) for index in range(len(CABLE_NM))},
    "mux": at(1, 1.5),
    "cable": at(2, 1.5),
    "osa": at(3, 1.5),
}


def cable_context() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)


def submarine(flatten: bool = False) -> Graph:
    """Eight 0 dBm channels across the C-band, through 100 spans of 60 km."""
    graph = Graph(cable_context())
    mux = graph.add(Combiner(len(CABLE_NM), label="mux"))
    for index, wavelength in enumerate(CABLE_NM):
        laser = graph.add(CWLaser(power=0.0, wavelength=wavelength, label=f"ch{index}"))
        graph.connect(laser, mux[f"in{index}"])
    line = graph.add(cable(flatten))
    graph.connect(mux, line["in"])
    osa = OpticalSpectrumAnalyzerWide("osa")
    graph.connect(line, graph.add(osa)["in"])
    return graph


def OpticalSpectrumAnalyzerWide(label: str) -> Component:  # noqa: N802 - reads as a block
    """The whole C-band at the 0.1 nm an OSNR is quoted in."""
    return spectrum(label, centre=1547.0, span=5000.0, resolution=12.5)


def channel_table(signal: object) -> list[tuple[float, float, float]]:
    """(wavelength nm, power dBm, OSNR dB in 0.1 nm) of every channel of an optical signal."""
    assert isinstance(signal, OpticalSignal)
    rows = []
    for band in signal.bands:
        power = band.average_power()
        noise = signal.noise_power_in(band.f0, 12.5e9)
        rows.append(
            (
                nm_at(band.f0),
                10.0 * math.log10(power / 1e-3),
                10.0 * math.log10(power / noise),
            )
        )
    return rows


def cable_channels(flatten: bool = False) -> list[tuple[float, float, float]]:
    graph = submarine(flatten)
    line = next(c for c in graph.components if c.label == "cable")
    return channel_table(graph.run(keep=[line]).port(line, "out"))


# ---------------------------------------------------------------------------
# 7.3  400ZR on an open DWDM line

ZR_SPACING = 75e9
ZR_NEIGHBOUR_DBM = -13.8  # what the pluggable itself puts out
ZR_SPAN_KM = 80.0
ZR_BOOST_DB = 18.0


def zr_grid(**extra: float) -> dict[str, float]:
    """The 75 GHz three-channel grid the mux and demux share, ZR in the middle."""
    return {
        "first_wavelength": nm_at(F_CHANNEL - ZR_SPACING),
        "spacing": ZR_SPACING / 1e9,
        "bandwidth": ZR_SPACING / 1e9,
        "order": 4.0,
        **extra,
    }


def zr_line(
    boost_db: float = ZR_BOOST_DB, demux: bool = True
) -> Callable[[Graph, Component], Component]:
    def line(graph: Graph, tx: Component) -> Component:
        mux = graph.add(Multiplexer(3, label="mux", **zr_grid()))
        for index, offset in ((0, -ZR_SPACING), (2, ZR_SPACING)):
            neighbour = CWLaser(
                power=ZR_NEIGHBOUR_DBM, wavelength=nm_at(F_CHANNEL + offset), label=f"nb{index}"
            )
            graph.connect(graph.add(neighbour), mux[f"in{index}"])
        graph.connect(tx, mux["in1"])
        boost = graph.add(EDFA(gain=boost_db, noise_figure=5.0, label="boost"))
        fibre = graph.add(Fiber(length=ZR_SPAN_KM, attenuation=0.2, dispersion=17.0, label="fib"))
        pre = graph.add(EDFA(gain=0.2 * ZR_SPAN_KM, noise_figure=5.0, label="pre"))
        graph.chain(mux, boost, fibre, pre)
        graph.connect(pre, graph.add(spectrum("osa", span=400.0, resolution=1.0))["in"])
        if not demux:
            return pre
        split = graph.add(Demultiplexer(3, label="demux", **zr_grid()))
        graph.connect(pre, split["in"])
        return split["out1"]  # type: ignore[return-value]

    return line


def zr_dwdm(boost_db: float = ZR_BOOST_DB, demux: bool = True) -> Graph:
    rate, bits, _ = reference_rates.CONFIGURATIONS["400G DP-16QAM"]
    return reference_rates.build(rate, bits, span_km=ZR_SPAN_KM, line=zr_line(boost_db, demux))[0]


def _zr_layout() -> Layout:
    layout: Layout = {}
    for key, place in coherent.ZR_LAYOUT.items():
        if key in ("fib", "edfa"):
            continue
        column = (place["x"] - 30.0) / 150.0
        row = (place["y"] - 40.0) / 100.0
        layout[key] = at(column + 3 if column >= 7 else column, row)
    layout.update(
        {
            "nb0": at(4, 0.5),
            "nb2": at(4, 3.5),
            "mux": at(5, 2),
            "boost": at(6, 2),
            "fib": at(7, 2),
            "pre": at(8, 2),
            "osa": at(8, 0.5),
            "demux": at(9, 2),
        }
    )
    return layout


ZR_DWDM_LAYOUT = _zr_layout()


def zr_evm(results: dict[str, object]) -> float:
    """EVM averaged over the two polarizations [%]."""
    return float(np.mean([results[f"vsa_{axis}"].evm for axis in "xy"])) * 100.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 7.4  Radio over fibre: two tones and their intermodulation

ROF_TONES = (1.0, 1.1)  # GHz
ROF_AMPLITUDE = 0.5  # V peak, each tone
ROF_VPI = 4.0

ROF_LAYOUT: Layout = {
    "rf": at(0, 0),
    "tx": at(0, 2),
    "mzm": at(1, 1),
    "fib": at(2, 1),
    "pd": at(3, 1),
    "esa": at(4, 1),
}


def rof_context() -> SimulationContext:
    """A microsecond of window: 1 MHz between analyser bins."""
    return SimulationContext(bit_rate=1e9, samples_per_symbol=16, sequence_length=1000, seed=1)


def rof(amplitude: float = ROF_AMPLITUDE) -> Graph:
    """Two tones on a quadrature-biased modulator, 10 km of fibre and a photodiode."""
    graph = Graph(rof_context())
    tone = graph.add(
        RFTone(frequency=ROF_TONES[0], frequency2=ROF_TONES[1], amplitude=amplitude, label="rf")
    )
    laser = graph.add(CWLaser(power=10.0, label="tx"))
    modulator = graph.add(MachZehnderModulator(v_pi=ROF_VPI, v_bias=-ROF_VPI / 2, label="mzm"))
    fibre = graph.add(Fiber(length=10.0, label="fib"))
    detector = graph.add(PINPhotodiode(label="pd"))
    esa = graph.add(ElectricalSpectrumAnalyzer(stop=3.0, label="esa"))
    graph.connect(laser, modulator["optical_in"])
    graph.connect(tone, modulator["electrical_in"])
    graph.chain(modulator, fibre, detector, esa)
    return graph


def bessel_j(order: int, x: float) -> float:
    """``J_n(x)`` from its integral, so the theory needs nothing beyond numpy."""
    t = np.linspace(0.0, math.pi, 20001)
    return float(np.trapezoid(np.cos(order * t - x * np.sin(t)), t) / math.pi)


def carrier_to_intermod_db(amplitude: float) -> float:
    """Carrier over third-order product for a quadrature-biased MZM [dB]: ``20 log J0(b)/J2(b)``.

    The carrier is ``J1(b) J0(b)`` and the product at ``2 f1 - f2`` is
    ``J2(b) J1(b)``; the ratio leaves ``J0 / J2``. ``b`` is the phase swing of
    one tone across the push-pull modulator, ``pi A / V_pi``.
    """
    swing = math.pi * amplitude / ROF_VPI
    return 20.0 * math.log10(bessel_j(0, swing) / bessel_j(2, swing))


def rof_levels(reading: object) -> dict[str, float]:
    """Carrier, third-order products and second harmonic [dBm]."""
    f1, f2 = (f * 1e9 for f in ROF_TONES)
    return {
        "carrier": reading.dbm_at(f1),  # type: ignore[attr-defined]
        "imd_low": reading.dbm_at(2 * f1 - f2),  # type: ignore[attr-defined]
        "imd_high": reading.dbm_at(2 * f2 - f1),  # type: ignore[attr-defined]
        "harmonic": reading.dbm_at(2 * f1),  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# 7.5  A laser link through open air

FSO_LAYOUT: Layout = {
    "prbs": at(0, 0),
    "drv": at(1, 0),
    "tx": at(0, 1.5),
    "mzm": at(1, 1.5),
    "air": at(2, 1.5),
    "pm_rx": at(3, 0),
    "pin": at(3, 1.5),
    "lpf": at(4, 1.5),
    "ber": at(5, 1.5),
    "eye": at(5, 0),
}


def free_space(**channel: float) -> Graph:
    """10 Gb/s OOK at 10 dBm across a kilometre of air into a PIN."""
    graph = Graph(
        SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=2048, seed=3)
    )
    prbs = graph.add(PRBSGenerator(order=11.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    graph.connect(prbs, driver["in"])
    _, modulator = ook_transmitter(graph, power=10.0)
    graph.connect(driver, modulator["electrical_in"])
    air = graph.add(FreeSpaceChannel(label="air", **channel))
    graph.connect(modulator, air["in"])
    graph.connect(air, graph.add(PowerMeter(label="pm_rx"))["in"])
    receive(graph, air, prbs, detector=graph.add(PINPhotodiode(label="pin")), bandwidth_ghz=7.0)
    return graph


# ---------------------------------------------------------------------------
# 7.6  FMCW LiDAR: range from a beat note

LIDAR_SWEEP_GHZ = 1.0
LIDAR_RANGE_M = 30.0
LIDAR_ECHO_DB = 30.0

LIDAR_LAYOUT: Layout = {
    "tx": at(0, 1),
    "tap": at(1, 1),
    "target": at(2, 2),
    "echo": at(3, 2),
    "mix": at(4, 1),
    "pd": at(5, 1),
    "esa": at(6, 1),
}


def lidar_context() -> SimulationContext:
    """One microsecond sweep, read in 1 MHz bins: 15 cm of range each."""
    return SimulationContext(bit_rate=1e9, samples_per_symbol=4, sequence_length=1000, seed=1)


def lidar(range_m: float = LIDAR_RANGE_M) -> Graph:
    """A swept laser split into a local copy and a round trip to a target."""
    graph = Graph(lidar_context())
    laser = graph.add(SweptLaser(power=10.0, sweep=LIDAR_SWEEP_GHZ, label="tx"))
    tap = graph.add(Splitter(2, label="tap"))
    target = graph.add(DelayLine(delay=round_trip_ps(range_m), label="target"))
    echo = graph.add(Attenuator(attenuation=LIDAR_ECHO_DB, label="echo"))
    mix = graph.add(DirectionalCoupler(True, coupling=0.5, label="mix"))
    detector = graph.add(PINPhotodiode(label="pd"))
    esa = graph.add(ElectricalSpectrumAnalyzer(stop=0.5, label="esa"))
    graph.connect(laser, tap["in"])
    graph.connect(tap["out0"], mix["in1"])
    graph.connect(tap["out1"], target["in"])
    graph.connect(target, echo["in"])
    graph.connect(echo, mix["in2"])
    graph.connect(mix["out1"], detector["in"])
    graph.connect(detector, esa["in"])
    return graph


def round_trip_ps(range_m: float) -> float:
    return 2.0 * range_m / C_LIGHT * 1e12


def beat_hz(reading: object) -> float:
    """The strongest line on the analyser [Hz]."""
    power = np.asarray(reading.power_w)  # type: ignore[attr-defined]
    return float(np.asarray(reading.frequencies)[int(np.argmax(power))])  # type: ignore[attr-defined]


def range_from_beat(beat: float) -> float:
    """``R = c f_b T / 2B``: the beat is the sweep rate times the round trip."""
    ctx = lidar_context()
    rate = LIDAR_SWEEP_GHZ * 1e9 / ctx.time_window
    return C_LIGHT * beat / (2.0 * rate)


# ---------------------------------------------------------------------------
# 7.7  BB84 with decoy states

QKD_MU = 0.5
QKD_KM = 50.0
QKD_RATE = 1e9

QKD_LAYOUT: Layout = {
    "alice": at(0, 1),
    "attenuate": at(1, 1),
    "fib": at(2, 1),
    "bob": at(3, 1),
}


def mu_attenuation_db(mu: float = QKD_MU, rate: float = QKD_RATE) -> float:
    """What brings 0 dBm at 1550 nm down to ``mu`` photons a pulse."""
    photon = 6.62607015e-34 * C_LIGHT / 1550e-9
    return -10.0 * math.log10(mu * photon * rate / 1e-3)


def qkd(km: float = QKD_KM) -> Graph:
    graph = Graph(
        SimulationContext(bit_rate=QKD_RATE, samples_per_symbol=4, sequence_length=256, seed=1)
    )
    alice = graph.add(CWLaser(power=0.0, label="alice"))
    attenuate = graph.add(Attenuator(attenuation=round(mu_attenuation_db(), 2), label="attenuate"))
    fibre = graph.add(Fiber(length=km, attenuation=0.2, label="fib"))
    bob = graph.add(BB84Receiver(label="bob"))
    graph.chain(alice, attenuate, fibre)
    graph.connect(fibre, bob["in"])
    graph.connect(attenuate, bob["sent"])
    return graph


# ---------------------------------------------------------------------------

EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "metro_ring": (metro_ring, RING_LAYOUT),
    "submarine": (submarine, CABLE_LAYOUT),
    "zr_dwdm": (zr_dwdm, ZR_DWDM_LAYOUT),
    "rof": (rof, ROF_LAYOUT),
    "free_space": (free_space, FSO_LAYOUT),
    "lidar": (lidar, LIDAR_LAYOUT),
    "qkd": (qkd, QKD_LAYOUT),
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


def report(key: str, graph: Graph) -> None:
    if key == "submarine":
        for flatten in (False, True):
            print(f"   flatten={flatten}")
            for wavelength, power, osnr in cable_channels(flatten):
                print(f"     {wavelength:7.1f} nm  {power:7.2f} dBm  OSNR {osnr:6.2f} dB")
        return
    results = results_by_label(graph)
    if key == "metro_ring":
        for k in (1, NODES):
            width, top = passband_ghz(results[f"osa_{k}"], results["osa_in"])
            print(f"   after {k} nodes: {width:6.2f} GHz wide, {top:6.2f} dB at the top")
        print(f"   theory after {NODES}: {cascade_width_ghz(NODES):.2f} GHz")
    elif key == "zr_dwdm":
        print(f"   OSNR {results['osnr']:.2f} dB, EVM {zr_evm(results):.2f} %")
    elif key == "rof":
        levels = rof_levels(results["esa"])
        for name, level in levels.items():
            print(f"   {name:9s} {level:8.2f} dBm")
        print(
            f"   C/I {levels['carrier'] - levels['imd_low']:.2f} dB, "
            f"theory {carrier_to_intermod_db(ROF_AMPLITUDE):.2f} dB"
        )
    elif key == "free_space":
        print(f"   received {results['pm_rx'].power_dbm:.2f} dBm, {results['ber']}")  # type: ignore[attr-defined]
    elif key == "lidar":
        beat = beat_hz(results["esa"])
        print(f"   beat {beat / 1e6:.1f} MHz -> {range_from_beat(beat):.2f} m")
    elif key == "qkd":
        print(f"   {results['bob'].summary}")  # type: ignore[attr-defined]


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
