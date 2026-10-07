"""Phase 6 of the example roadmap: sensing.

A grating's period is a length, so anything that stretches or warms the fibre
moves its reflection, and a cladding mode's index is set partly by what the
fibre is dipped in, so a liquid moves its notches. Each example lights its
gratings with an EDFA's ASE and reads them on an optical spectrum analyser, as
a laboratory interrogator does. The last one is a different instrument: a pulse
sent into a long fibre, and the faint light the glass scatters back.

``python sensing.py`` writes every project into ``examples/maiman/`` and prints
what each measures; ``python sensing.py otdr`` does one.
``tests/test_sensing.py`` holds the numbers the lessons quote.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import lessons
import numpy as np
from chip import results_by_label, transmission
from direct_detection import Layout, at

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    EDFA,
    BackscatterFiber,
    Circulator,
    Combiner,
    CWLaser,
    FiberBraggGrating,
    GaussianPulse,
    LongPeriodGrating,
    OpticalSpectrumAnalyzer,
    Oscilloscope,
    PowerMeter,
    Splitter,
    TiltedFiberBraggGrating,
)
from maiman.project import save
from maiman.signals import OpticalSpectrum, PowerReading, ScopeTrace
from maiman.units import C_LIGHT

PROJECTS = Path(__file__).resolve().parent.parent / "maiman"


def context() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=256, seed=1)


def spectrum(
    label: str, centre: float = 1550.0, span: float = 2000.0, resolution: float = 1.0
) -> OpticalSpectrumAnalyzer:
    """An OSA ``span`` GHz wide around ``centre`` nm."""
    return OpticalSpectrumAnalyzer(
        auto_span=False,
        center_wavelength=centre,
        span=span,
        points=4096.0,
        resolution_bandwidth=resolution,
        label=label,
    )


def white_light(graph: Graph, **window: float) -> Component:
    """An EDFA with almost nothing at its input, and ``osa_in`` reading what it sends."""
    seed = graph.add(CWLaser(power=-100.0, wavelength=1550.0, label="seed"))
    source = graph.add(EDFA(gain=30.0, noise_figure=5.0, label="ase"))
    graph.connect(seed, source["in"])
    graph.connect(source, graph.add(spectrum("osa_in", **window))["in"])
    return source


def centroid_nm(reading: object, reference: object) -> float:
    """Where a reflection peak sits [nm]: the centroid of its top half.

    That is what an interrogator reports.
    """
    frequencies, db = transmission(reading, reference)
    linear = 10.0 ** (db / 10.0)
    top = linear > 0.5 * linear.max()
    return float(C_LIGHT / ((frequencies[top] * linear[top]).sum() / linear[top].sum()) * 1e9)


def notches_nm(reading: object, reference: object, depth: float = 1.0) -> np.ndarray:
    """Wavelengths [nm] of every local minimum at least ``depth`` dB down."""
    frequencies, db = transmission(reading, reference)
    inner = np.arange(1, db.size - 1)
    found = inner[(db[1:-1] < db[:-2]) & (db[1:-1] <= db[2:]) & (db[1:-1] < -depth)]
    return np.sort(C_LIGHT / frequencies[found] * 1e9)


# ---------------------------------------------------------------------------
# 6.1  A Bragg grating as a strain gauge and a thermometer

STRAIN = 400.0  # microstrain on the working grating
WARMING = 15.0  # kelvin, on both
WORKING_NM, REFERENCE_NM = 1545.0, 1555.0

STRAIN_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "working": at(2, 1),
    "reference": at(3, 1),
    "osa_work": at(3, -0.4),
    "osa_ref": at(4, -0.4),
    "osa_thru": at(4, 1),
}


def sensor(bragg_nm: float, *, strain: float, label: str) -> FiberBraggGrating:
    """A short, strong grating: a narrow peak whose centre is easy to find."""
    return FiberBraggGrating(
        length=10.0,
        index_modulation=1.5e-4,
        bragg_wavelength=bragg_nm,
        strain=strain,
        temperature_change=WARMING,
        label=label,
    )


def fbg_strain() -> Graph:
    """A loaded grating and a loose one, in series on one fibre, both warmed."""
    graph = Graph(context())
    source = white_light(graph)
    working = graph.add(sensor(WORKING_NM, strain=STRAIN, label="working"))
    reference = graph.add(sensor(REFERENCE_NM, strain=0.0, label="reference"))
    graph.connect(source, working["in"])
    graph.connect(working["transmitted"], reference["in"])
    graph.connect(working["reflected"], graph.add(spectrum("osa_work"))["in"])
    graph.connect(reference["reflected"], graph.add(spectrum("osa_ref"))["in"])
    graph.connect(reference["transmitted"], graph.add(spectrum("osa_thru"))["in"])
    return graph


def recovered(results: dict[str, object]) -> tuple[float, float]:
    """Temperature [K] from the loose grating, then strain [microstrain] from the loaded one."""
    working = sensor(WORKING_NM, strain=0.0, label="w")
    reference = sensor(REFERENCE_NM, strain=0.0, label="r")
    reference_shift = (centroid_nm(results["osa_ref"], results["osa_in"]) - REFERENCE_NM) * 1e-9
    warming = reference_shift / reference.temperature_sensitivity()
    working_shift = (centroid_nm(results["osa_work"], results["osa_in"]) - WORKING_NM) * 1e-9
    strain = (working_shift - warming * working.temperature_sensitivity()) / (
        working.strain_sensitivity()
    )
    return warming, strain * 1e6


# ---------------------------------------------------------------------------
# 6.2  A grating and a circulator: dropping a channel

DROP_LAYOUT: Layout = {
    "ch1550": at(0, 0),
    "ch1552": at(0, 1.4),
    "mux": at(1, 0.7),
    "circ": at(2, 0.7),
    "fbg": at(3, 0.7),
    "pm_drop": at(3, 2.2),
    "osa_drop": at(4, 2.2),
    "pm_thru": at(4, -0.5),
    "osa_thru": at(5, 0.7),
}


def fbg_drop() -> Graph:
    graph = Graph(context())
    dropped = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="ch1550"))
    express = graph.add(CWLaser(power=0.0, wavelength=1552.0, label="ch1552"))
    mux = graph.add(Combiner(2, label="mux"))
    circ = graph.add(Circulator(insertion_loss=0.7, isolation=40.0, label="circ"))
    grating = graph.add(
        FiberBraggGrating(bragg_wavelength=1550.0, length=10.0, index_modulation=1e-4, label="fbg")
    )
    graph.connect(dropped, mux["in0"])
    graph.connect(express, mux["in1"])
    graph.connect(mux, circ["in1"])
    graph.connect(circ["out2"], grating["in"])
    graph.connect(grating["reflected"], circ["in2"])
    graph.connect(circ["out3"], graph.add(PowerMeter(label="pm_drop"))["in"])
    graph.connect(circ["out3"], graph.add(spectrum("osa_drop", span=1000.0))["in"])
    graph.connect(grating["transmitted"], graph.add(PowerMeter(label="pm_thru"))["in"])
    graph.connect(grating["transmitted"], graph.add(spectrum("osa_thru", span=1000.0))["in"])
    return graph


def band_dbm(reading: object, wavelength_nm: float) -> float:
    """The power one channel of a meter's reading carries [dBm]."""
    assert isinstance(reading, PowerReading)
    (band,) = [b for b in reading.bands if round(b.wavelength_nm, 2) == wavelength_nm]
    return float(band.power_dbm)


# ---------------------------------------------------------------------------
# 6.3  A tilted grating reads a liquid

TFBG_WINDOW = {"centre": 1544.0, "span": 2000.0, "resolution": 1.0}
WATER = 1.333

TILTED_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "sp": at(2, 1),
    "tfbg_air": at(3, 0),
    "tfbg_water": at(3, 2),
    "osa_air": at(4, 0),
    "osa_water": at(4, 2),
}


def tilted_grating() -> Graph:
    """The same tilted grating twice: dry, and dipped in water."""
    graph = Graph(context())
    source = white_light(graph, **TFBG_WINDOW)
    split = graph.add(Splitter(2, label="sp"))
    graph.connect(source, split["in"])
    for index, (name, outside) in enumerate((("air", 1.0), ("water", WATER))):
        device = graph.add(
            TiltedFiberBraggGrating(
                period=535.0,
                tilt=4.0,
                length=10.0,
                index_modulation=5e-4,
                surrounding_index=outside,
                azimuthal_orders=3.0,
                label=f"tfbg_{name}",
            )
        )
        graph.connect(split[f"out{index}"], device["in"])
        graph.connect(
            device["transmitted"], graph.add(spectrum(f"osa_{name}", **TFBG_WINDOW))["in"]
        )
    return graph


def comb_shift_pm(results: dict[str, object], lo_nm: float, hi_nm: float) -> float:
    """How far the water moves the comb between ``lo_nm`` and ``hi_nm`` [pm], longer positive.

    The two traces are slid over each other a bin at a time and the best match
    is refined between bins with a parabola through its neighbours, the way a
    peak is read off a correlation.
    """
    frequencies, air = transmission(results["osa_air"], results["osa_in"])
    _, water = transmission(results["osa_water"], results["osa_in"])
    wavelengths = C_LIGHT / frequencies * 1e9
    inside = (wavelengths > lo_nm) & (wavelengths < hi_nm)
    lags = np.arange(-60, 61)
    match = np.array([np.corrcoef(air[inside], np.roll(water, -lag)[inside])[0, 1] for lag in lags])
    best = int(np.argmax(match))
    a, b, c = match[best - 1], match[best], match[best + 1]
    lag = lags[best] + 0.5 * (a - c) / (a - 2.0 * b + c)
    step = float(np.mean(np.diff(wavelengths)))
    return float(lag * step * 1e3)


# ---------------------------------------------------------------------------
# 6.4  A long-period grating, and a pair of them

LPG_PERIOD = 485.0
LPG_FULL = 3.25e-4  # couples all of the core light into LP04 at the centre
LPG_SEPARATION = 200.0  # mm
LPG_WINDOW = {"centre": 1550.0, "span": 4000.0, "resolution": 1.0}

LPG_LAYOUT: Layout = {
    "seed": at(0, 1),
    "ase": at(1, 1),
    "osa_in": at(1, 2.5),
    "sp": at(2, 1),
    "lpg_one": at(3, 0),
    "lpg_pair": at(3, 2),
    "osa_one": at(4, 0),
    "osa_pair": at(4, 2),
}


def lpg_pair() -> Graph:
    """One grating strong enough to empty the core, and two of half its strength."""
    graph = Graph(context())
    source = white_light(graph, **LPG_WINDOW)
    split = graph.add(Splitter(2, label="sp"))
    graph.connect(source, split["in"])
    one = graph.add(
        LongPeriodGrating(period=LPG_PERIOD, index_modulation=LPG_FULL, label="lpg_one")
    )
    pair = graph.add(
        LongPeriodGrating(
            period=LPG_PERIOD,
            index_modulation=LPG_FULL / 2.0,
            pair=True,
            separation=LPG_SEPARATION,
            label="lpg_pair",
        )
    )
    for index, (name, device) in enumerate((("one", one), ("pair", pair))):
        graph.connect(split[f"out{index}"], device["in"])
        graph.connect(device["transmitted"], graph.add(spectrum(f"osa_{name}", **LPG_WINDOW))["in"])
    return graph


# ---------------------------------------------------------------------------
# 6.5  An OTDR finds a break

OTDR_KM = 25.0
SPLICE_KM, SPLICE_DB = 8.0, 0.5
BREAK_KM = 17.3
GROUP_INDEX = 1.468

OTDR_LAYOUT: Layout = {
    "pulse": at(0, 1),
    "fut": at(1, 1),
    "otdr": at(2, 0),
}


def otdr_context() -> SimulationContext:
    """327.68 us of window at 20 ns a sample: a 33 km round trip, read in 2 m steps."""
    return SimulationContext(bit_rate=12.5e6, samples_per_symbol=4, sequence_length=4096, seed=1)


def otdr() -> Graph:
    graph = Graph(otdr_context())
    pulse = graph.add(GaussianPulse(peak_power=20.0, width=60000.0, label="pulse"))
    fibre = graph.add(
        BackscatterFiber(
            length=OTDR_KM,
            group_index=GROUP_INDEX,
            splice_at=SPLICE_KM,
            splice_loss=SPLICE_DB,
            break_at=BREAK_KM,
            label="fut",
        )
    )
    scope = graph.add(Oscilloscope(decibels=True, from_start=True, points=4096.0, label="otdr"))
    graph.connect(pulse, fibre["in"])
    graph.connect(fibre["backscatter"], scope["in"])
    return graph


def trace_km(reading: object) -> tuple[np.ndarray, np.ndarray]:
    """Distance along the fibre [km] and the returned power [dBm] of an OTDR trace."""
    assert isinstance(reading, ScopeTrace)
    distance = np.asarray(reading.time) * C_LIGHT / GROUP_INDEX / 2.0 / 1e3
    with np.errstate(divide="ignore"):
        dbm = 10.0 * np.log10(np.asarray(reading.power_w) / 1e-3)
    return distance, dbm


def level_at(reading: object, km: float) -> float:
    distance, dbm = trace_km(reading)
    return float(dbm[int(np.argmin(np.abs(distance - km)))])


def last_echo_km(reading: object) -> float:
    """Where the trace's last spike is: the end of the glass [km]."""
    distance, dbm = trace_km(reading)
    beyond = distance > 1.0
    return float(distance[beyond][int(np.argmax(dbm[beyond]))])


# ---------------------------------------------------------------------------

EXAMPLES: dict[str, tuple[Callable[[], Graph], Layout]] = {
    "fbg_strain": (fbg_strain, STRAIN_LAYOUT),
    "fbg_drop": (fbg_drop, DROP_LAYOUT),
    "tilted_grating": (tilted_grating, TILTED_LAYOUT),
    "lpg_pair": (lpg_pair, LPG_LAYOUT),
    "otdr": (otdr, OTDR_LAYOUT),
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
    results = results_by_label(graph)
    if key == "fbg_strain":
        for label in ("osa_work", "osa_ref"):
            print(f"   {label:10s} peak at {centroid_nm(results[label], results['osa_in']):.5f} nm")
        warming, strain = recovered(results)
        print(f"   recovered {warming:.3f} K and {strain:.2f} microstrain")
    elif key == "fbg_drop":
        for label in ("pm_drop", "pm_thru"):
            print(
                f"   {label:8s} 1550 nm {band_dbm(results[label], 1550.0):8.3f} dBm   "
                f"1552 nm {band_dbm(results[label], 1552.0):8.3f} dBm"
            )
    elif key == "otdr":
        trace = results["otdr"]
        for km in (0.5, 1.0, 5.0, 7.9, 8.1, 15.0):
            print(f"   {km:5.1f} km  {level_at(trace, km):8.2f} dBm")
        print(f"   the last echo is at {last_echo_km(trace):.3f} km")
    else:
        for label, value in results.items():
            if isinstance(value, OpticalSpectrum) and label != "osa_in":
                _, db = transmission(value, results["osa_in"])
                found = notches_nm(value, results["osa_in"])
                print(f"   {label:10s} deepest {db.min():7.2f} dB, {found.size} notches")


def main(keys: list[str]) -> None:
    for key in keys or list(EXAMPLES):
        graph = write(key)
        print(f"{key}  ->  {project_path(key).name}")
        report(key, graph)


if __name__ == "__main__":
    main(sys.argv[1:])
