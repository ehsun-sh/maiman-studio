"""Phase 1 of the example roadmap: first light, one idea per project.

Seven small graphs, each built to show one relation working before it is buried
in a link: what a dBm is, how fibre loss adds up, the Mach-Zehnder's cosine, a
pulse spreading, a chirped pulse compressing, a soliton holding its shape, and
power dividing through splitters and couplers. ``python first_light.py`` writes
all seven projects into ``examples/maiman/`` and prints what each one measures;
``python first_light.py loss_budget`` does one.

Each project carries its lesson from :mod:`lessons`, and
``tests/test_first_light.py`` holds every number a lesson quotes to what the
graph produces, so the prose cannot drift from the physics.

The three pulse examples share one time grid: 1024 samples over 400 ps, 0.39 ps
a sample. That is fine enough for a 10 ps pulse and long enough that one
broadened four-fold stays clear of the window's edges, which on a periodic grid
would wrap it round to the other side.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import lessons

from maiman import Graph, SimulationContext, sweep
from maiman.components import (
    Attenuator,
    Combiner,
    CWLaser,
    DCVoltage,
    DirectionalCoupler,
    Fiber,
    GaussianPulse,
    MachZehnderModulator,
    OpticalSpectrumAnalyzer,
    Oscilloscope,
    PowerMeter,
    SechPulse,
    Splitter,
)
from maiman.kernels import dispersion_to_beta2, soliton_peak_power, soliton_period
from maiman.project import save
from maiman.signals import OpticalSpectrum, PowerReading, ScopeTrace
from maiman.units import w_to_dbm

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"

Layout = dict[str, dict[str, float]]

#: A CW link has nothing to resolve in time: a short window runs instantly.
CW_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)
#: The pulse grid described above.
PULSE_CTX = SimulationContext(bit_rate=10e9, samples_per_symbol=256, sequence_length=4, seed=1)

#: Standard single-mode fibre at 1550 nm.
D_SMF = 17.0  # ps/(nm km)
GAMMA_SMF = 1.3  # 1/(W km)
BETA2 = dispersion_to_beta2(D_SMF * 1e-6, 1550e-9)  # s^2/m, negative
T0_PS = 10.0
#: The dispersion length T0^2/|beta2|, in km: 4.6 km for a 10 ps pulse.
LD_KM = (T0_PS * 1e-12) ** 2 / abs(BETA2) / 1e3


def col(i: int) -> float:
    """The x of the i-th column on the canvas, at the studio's usual pitch."""
    return 30.0 + 150.0 * i


# ---------------------------------------------------------------------------
# 1.1  A laser on a power meter and an optical spectrum analyser

LASER_METERS_LAYOUT: Layout = {
    "laser": {"x": col(0), "y": 150.0},
    "pm_tx": {"x": col(2), "y": 40.0},
    "osa": {"x": col(2), "y": 150.0},
    "att": {"x": col(1), "y": 270.0},
    "pm_rx": {"x": col(2), "y": 270.0},
}


def laser_meters() -> Graph:
    """One milliwatt, read three ways: in dBm, on a spectrum, and after 3 dB."""
    graph = Graph(CW_CTX)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, linewidth=100.0, label="laser"))
    graph.connect(laser, graph.add(PowerMeter(label="pm_tx"))["in"])
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            auto_span=False,
            center_wavelength=1550.0,
            span=100.0,
            points=1024.0,
            resolution_bandwidth=12.5,
            label="osa",
        )
    )
    graph.connect(laser, osa["in"])
    att = graph.add(Attenuator(attenuation=3.0, label="att"))
    graph.connect(laser, att["in"])
    graph.connect(att, graph.add(PowerMeter(label="pm_rx"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 1.2  A loss budget

LOSS_BUDGET_LAYOUT: Layout = {
    "laser": {"x": col(0), "y": 120.0},
    "patch_tx": {"x": col(1), "y": 120.0},
    "span": {"x": col(2), "y": 120.0},
    "patch_rx": {"x": col(3), "y": 120.0},
    "pm_rx": {"x": col(4), "y": 120.0},
}

#: Two connectors' worth at each end, as a field budget would book them.
CONNECTOR_DB = 0.5


def loss_budget() -> Graph:
    """0 dBm through 50 km at 0.2 dB/km and a connector loss at each end."""
    graph = Graph(CW_CTX)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    patch_tx = graph.add(Attenuator(attenuation=CONNECTOR_DB, label="patch_tx"))
    # No dispersion: a CW carrier has no width for it to act on.
    span = graph.add(Fiber(length=50.0, attenuation=0.2, dispersion=0.0, label="span"))
    patch_rx = graph.add(Attenuator(attenuation=CONNECTOR_DB, label="patch_rx"))
    meter = graph.add(PowerMeter(label="pm_rx"))
    graph.chain(laser, patch_tx, span)
    graph.connect(span["out"], patch_rx["in"])
    graph.chain(patch_rx, meter)
    return graph


# ---------------------------------------------------------------------------
# 1.3  The Mach-Zehnder's transfer curve

MZM_CURVE_LAYOUT: Layout = {
    "laser": {"x": col(0), "y": 60.0},
    "bias": {"x": col(0), "y": 190.0},
    "mzm": {"x": col(1), "y": 120.0},
    "pm_out": {"x": col(2), "y": 120.0},
}


def mzm_curve() -> Graph:
    """A DC voltage on a modulator's drive port: sweep it and the cosine appears."""
    graph = Graph(CW_CTX)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser"))
    bias = graph.add(DCVoltage(voltage=2.0, label="bias"))
    mzm = graph.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    graph.connect(laser, mzm["optical_in"])
    graph.connect(bias, mzm["electrical_in"])
    graph.connect(mzm, graph.add(PowerMeter(label="pm_out"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 1.4  A pulse spreading in fibre

PULSE_SPREADING_LAYOUT: Layout = {
    "pulse": {"x": col(0), "y": 60.0},
    "fibre_a": {"x": col(1), "y": 60.0},
    "fibre_b": {"x": col(2), "y": 60.0},
    "launch": {"x": col(0), "y": 220.0},
    "after_5km": {"x": col(1), "y": 220.0},
    "after_10km": {"x": col(2), "y": 220.0},
}

#: What the scopes show: 200 ps around the pulse, half the window. The figures
#: they report are measured inside it, and a pulse four times wider still fits.
SCOPE_SPAN_PS = 200.0


def pulse_spreading() -> Graph:
    """A 10 ps Gaussian, watched at the launch, after 5 km and after 10 km.

    Loss is off so that the only thing the pulse meets is dispersion; at a
    milliwatt the Kerr effect would take hundreds of kilometres to matter.
    """
    graph = Graph(PULSE_CTX)
    pulse = graph.add(GaussianPulse(peak_power=0.0, width=T0_PS, label="pulse"))
    fibre_a = graph.add(Fiber(length=5.0, attenuation=0.0, dispersion=D_SMF, label="fibre_a"))
    fibre_b = graph.add(Fiber(length=5.0, attenuation=0.0, dispersion=D_SMF, label="fibre_b"))
    graph.connect(pulse, fibre_a["in"])
    graph.connect(fibre_a["out"], fibre_b["in"])
    for source, label in ((pulse, "launch"), (fibre_a, "after_5km"), (fibre_b, "after_10km")):
        meter = graph.add(Oscilloscope(span=SCOPE_SPAN_PS, label=label))
        graph.connect(source["out"], meter["in"])
    return graph


# ---------------------------------------------------------------------------
# 1.5  Chirped pulse compression

CHIRP = 2.0
#: Where an up-chirped pulse is narrowest in anomalous fibre: z = L_D C / (1 + C^2).
COMPRESSION_KM = round(LD_KM * CHIRP / (1.0 + CHIRP**2), 2)

CHIRP_COMPRESSION_LAYOUT: Layout = {
    "up": {"x": col(0), "y": 60.0},
    "down": {"x": col(0), "y": 260.0},
    "fibre_up": {"x": col(1), "y": 60.0},
    "fibre_down": {"x": col(1), "y": 260.0},
    "up_out": {"x": col(2), "y": 60.0},
    "launch": {"x": col(2), "y": 160.0},
    "down_out": {"x": col(2), "y": 260.0},
}


def chirp_compression() -> Graph:
    """Two pulses with opposite chirp through the same fibre: one narrows, one spreads."""
    graph = Graph(PULSE_CTX)
    up = graph.add(GaussianPulse(peak_power=0.0, width=T0_PS, chirp=CHIRP, label="up"))
    down = graph.add(GaussianPulse(peak_power=0.0, width=T0_PS, chirp=-CHIRP, label="down"))
    for source, name in ((up, "up"), (down, "down")):
        fibre = graph.add(
            Fiber(length=COMPRESSION_KM, attenuation=0.0, dispersion=D_SMF, label=f"fibre_{name}")
        )
        graph.connect(source, fibre["in"])
        meter = graph.add(Oscilloscope(span=SCOPE_SPAN_PS, label=f"{name}_out"))
        graph.connect(fibre["out"], meter["in"])
    # Both launch with the same envelope; only their phase differs, so one
    # scope shows the input of either.
    graph.connect(up, graph.add(Oscilloscope(span=SCOPE_SPAN_PS, label="launch"))["in"])
    return graph


# ---------------------------------------------------------------------------
# 1.6  The fundamental soliton

#: N = 1: gamma P0 T0^2 / |beta2| = 1.
SOLITON_W = soliton_peak_power(BETA2, GAMMA_SMF * 1e-3, T0_PS * 1e-12)
#: Two soliton periods, z0 = (pi/2) L_D each.
SOLITON_KM = round(2.0 * soliton_period(BETA2, T0_PS * 1e-12) / 1e3, 2)

SOLITON_LAYOUT: Layout = {
    "soliton": {"x": col(0), "y": 160.0},
    "kerr_on": {"x": col(1), "y": 60.0},
    "kerr_off": {"x": col(1), "y": 260.0},
    "with_kerr": {"x": col(2), "y": 60.0},
    "launch": {"x": col(2), "y": 160.0},
    "dispersion_only": {"x": col(2), "y": 260.0},
}


def soliton() -> Graph:
    """One sech pulse at the N = 1 power, through the same fibre with and without Kerr."""
    graph = Graph(PULSE_CTX)
    source = graph.add(SechPulse(peak_power=w_to_dbm(SOLITON_W), width=T0_PS, label="soliton"))
    common = {"length": SOLITON_KM, "attenuation": 0.0, "dispersion": D_SMF}
    kerr_on = graph.add(Fiber(nonlinearity=GAMMA_SMF, label="kerr_on", **common))
    kerr_off = graph.add(Fiber(nonlinearity=0.0, label="kerr_off", **common))
    graph.connect(source, kerr_on["in"])
    graph.connect(source, kerr_off["in"])
    for end, label in ((source, "launch"), (kerr_on, "with_kerr"), (kerr_off, "dispersion_only")):
        meter = graph.add(Oscilloscope(span=SCOPE_SPAN_PS, label=label))
        graph.connect(end["out"], meter["in"])
    return graph


# ---------------------------------------------------------------------------
# 1.7  Splitters, couplers and a combiner

SPLITTERS_LAYOUT: Layout = {
    "laser_a": {"x": col(0), "y": 60.0},
    "laser_b": {"x": col(0), "y": 300.0},
    "split4": {"x": col(1), "y": 60.0},
    "tap": {"x": col(1), "y": 300.0},
    "combiner": {"x": col(2), "y": 180.0},
    "pm_quarter": {"x": col(3), "y": 60.0},
    "pm_sum": {"x": col(3), "y": 180.0},
    "pm_tap": {"x": col(3), "y": 300.0},
    "osa": {"x": col(3), "y": 420.0},
}


def splitters() -> Graph:
    """A 1:4 split, a 10 % tap, and two colours combined onto one fibre."""
    graph = Graph(CW_CTX)
    laser_a = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="laser_a"))
    laser_b = graph.add(CWLaser(power=0.0, wavelength=1551.0, label="laser_b"))
    split4 = graph.add(Splitter(4, label="split4"))
    tap = graph.add(DirectionalCoupler(coupling=0.1, label="tap"))
    combiner = graph.add(Combiner(2, label="combiner"))
    graph.connect(laser_a, split4["in"])
    graph.connect(split4["out0"], graph.add(PowerMeter(label="pm_quarter"))["in"])
    graph.connect(laser_b, tap["in1"])
    graph.connect(tap["out2"], graph.add(PowerMeter(label="pm_tap"))["in"])
    graph.connect(split4["out1"], combiner["in0"])
    graph.connect(tap["out1"], combiner["in1"])
    graph.connect(combiner, graph.add(PowerMeter(label="pm_sum"))["in"])
    # The two colours the combiner carries, each at its own wavelength: the
    # meter adds them, the spectrum keeps them apart.
    osa = graph.add(
        OpticalSpectrumAnalyzer(
            auto_span=False,
            center_wavelength=1550.5,
            span=300.0,
            points=1024.0,
            resolution_bandwidth=12.5,
            label="osa",
        )
    )
    graph.connect(combiner, osa["in"])
    return graph


# ---------------------------------------------------------------------------

#: key -> (builder, layout). The key names the project file and its lesson.
EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "laser_meters": (laser_meters, LASER_METERS_LAYOUT),
    "loss_budget": (loss_budget, LOSS_BUDGET_LAYOUT),
    "mzm_curve": (mzm_curve, MZM_CURVE_LAYOUT),
    "pulse_spreading": (pulse_spreading, PULSE_SPREADING_LAYOUT),
    "chirp_compression": (chirp_compression, CHIRP_COMPRESSION_LAYOUT),
    "soliton": (soliton, SOLITON_LAYOUT),
    "splitters": (splitters, SPLITTERS_LAYOUT),
}


def project_path(key: str) -> Path:
    return PROJECTS / f"{key}.maiman"


def write(key: str) -> Graph:
    """Build one example and save it, with its lesson and marks, where the studio reads it."""
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


def dbm(reading: object) -> float:
    assert isinstance(reading, PowerReading)
    return reading.power_dbm


def scope(reading: object) -> ScopeTrace:
    assert isinstance(reading, ScopeTrace)
    return reading


def report(key: str, graph: Graph) -> None:
    """Print what one example measures, in the terms its lesson uses."""
    if key == "loss_budget":
        span = next(c for c in graph.components if c.label == "span")
        meter = next(c for c in graph.components if c.label == "pm_rx")
        lengths = [0.0, 25.0, 50.0, 75.0, 100.0]
        print("   length    received")
        for length, point in zip(lengths, sweep(graph, {(span, "length"): lengths}), strict=True):
            print(f"   {length:5.0f} km  {dbm(point.runs[0][meter]):7.2f} dBm")
        return
    if key == "mzm_curve":
        bias = next(c for c in graph.components if c.label == "bias")
        meter = next(c for c in graph.components if c.label == "pm_out")
        volts_swept = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
        print("   drive     out")
        curve = sweep(graph, {(bias, "voltage"): volts_swept})
        for volts, point in zip(volts_swept, curve, strict=True):
            print(f"   {volts:4.1f} V  {dbm(point.runs[0][meter]):7.2f} dBm")
        return

    results = results_by_label(graph)
    for label, value in results.items():
        if isinstance(value, PowerReading):
            print(f"   {label:16s} {value.power_dbm:7.2f} dBm  ({value.power_w * 1e3:.3f} mW)")
        elif isinstance(value, ScopeTrace):
            print(
                f"   {label:16s} FWHM {value.fwhm * 1e12:6.2f} ps   "
                f"peak {value.peak_power_w * 1e3:8.3f} mW"
            )
        elif isinstance(value, OpticalSpectrum):
            shown = value.power_per_resolution()
            peak = int(shown.argmax())
            print(
                f"   {label:16s} peak {w_to_dbm(float(shown[peak])):6.2f} dBm "
                f"at {value.wavelengths_nm[peak]:.3f} nm"
            )


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)
    print(f"L_D = {LD_KM:.2f} km, soliton P0 = {SOLITON_W * 1e3:.1f} mW over {SOLITON_KM} km")
    print(f"compression length at C = {CHIRP:+.0f}: {COMPRESSION_KM} km")


if __name__ == "__main__":
    main(sys.argv[1:])
