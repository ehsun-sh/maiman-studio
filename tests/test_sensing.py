"""Phase 6 of the examples: every number a sensing lesson quotes.

The graphs come from ``examples/python/sensing.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens, read back from its
project file, and holds it to the table in its lesson.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import lessons
import numpy as np
import pytest
import sensing
from chip import transmission

from maiman import Graph
from maiman.components import (
    BackscatterFiber,
    FiberBraggGrating,
    GaussianPulse,
    LongPeriodGrating,
    PowerMeter,
)
from maiman.components.tilted import TiltedFiberBraggGrating
from maiman.encoding import encode
from maiman.project import load
from maiman.signals import PowerReading, ScopeTrace
from maiman.units import C_LIGHT


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(sensing.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return sensing.results_by_label(opened(key))


def changed(key: str, label: str, param: str, value: float) -> dict[str, object]:
    """The project's results with one parameter changed, as a *Try this* asks."""
    run = sensing.EXAMPLES[key][0]().run(overrides={(label, param): value})
    return {name: reading for (name, _), reading in run.items()}


@pytest.mark.parametrize("key", sorted(sensing.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = sensing.EXAMPLES[key]
    on_disk = json.loads(sensing.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(sensing.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(sensing.EXAMPLES[key][1])


def test_6_1_a_reference_grating_separates_load_from_heat() -> None:
    working = sensing.sensor(sensing.WORKING_NM, strain=0.0, label="w")
    assert working.strain_sensitivity() * 1e-6 * 1e12 == pytest.approx(1.21, abs=0.005)
    assert working.temperature_sensitivity() * 1e12 == pytest.approx(11.2, abs=0.05)
    assert working.cross_sensitivity() * 1e6 == pytest.approx(9.26, abs=0.005)

    got = results("fbg_strain")
    work = sensing.centroid_nm(got["osa_work"], got["osa_in"])
    ref = sensing.centroid_nm(got["osa_ref"], got["osa_in"])
    assert work == pytest.approx(1545.6498, abs=0.00005)
    assert ref == pytest.approx(1555.1692, abs=0.00005)
    assert (work - sensing.WORKING_NM) * 1e3 == pytest.approx(649.8, abs=0.05)
    assert (ref - sensing.REFERENCE_NM) * 1e3 == pytest.approx(169.2, abs=0.05)

    warming, strain = sensing.recovered(got)
    assert warming == pytest.approx(15.07, abs=0.005)
    assert strain == pytest.approx(399.7, abs=0.05)
    # Read as strain alone, the heat shows up as load.
    uncompensated = (work - sensing.WORKING_NM) * 1e-9 / working.strain_sensitivity() * 1e6
    assert uncompensated == pytest.approx(539, abs=0.5)
    assert uncompensated - 400 == pytest.approx(15 * 9.26, abs=0.5)
    # Try this: unloading moves it back by 0.48 nm.
    assert sensing.STRAIN * working.strain_sensitivity() * 1e9 * 1e-6 == pytest.approx(
        0.48, abs=0.005
    )


def test_6_2_the_circulator_charges_twice_and_leaks() -> None:
    got = results("fbg_drop")
    assert sensing.band_dbm(got["pm_drop"], 1550.0) == pytest.approx(-1.70, abs=0.005)
    assert sensing.band_dbm(got["pm_drop"], 1552.0) == pytest.approx(-38.71, abs=0.005)
    assert sensing.band_dbm(got["pm_thru"], 1550.0) == pytest.approx(-12.43, abs=0.005)
    assert sensing.band_dbm(got["pm_thru"], 1552.0) == pytest.approx(-0.70, abs=0.005)
    grating = next(c for c in opened("fbg_drop").components if c.label == "fbg")
    assert isinstance(grating, FiberBraggGrating)
    reflectivity = grating.peak_reflectivity()
    assert reflectivity == pytest.approx(0.933, abs=0.0005)
    assert 10 * math.log10(reflectivity) == pytest.approx(-0.30, abs=0.005)
    assert 10 * math.log10(1 - reflectivity) == pytest.approx(-11.73, abs=0.005)
    assert grating.bandwidth() * 1e9 == pytest.approx(0.198, abs=0.0005)
    rejection = sensing.band_dbm(got["pm_drop"], 1550.0) - sensing.band_dbm(got["pm_drop"], 1552.0)
    assert rejection == pytest.approx(37.0, abs=0.05)

    leak = sensing.band_dbm(changed("fbg_drop", "circ", "isolation", 0.0)["pm_drop"], 1552.0)
    assert leak == pytest.approx(-47.36, abs=0.005)


def test_6_3_water_moves_the_comb_and_not_the_bragg_line() -> None:
    got = results("tilted_grating")
    for label in ("osa_air", "osa_water"):
        frequencies, db = transmission(got[label], got["osa_in"])
        assert C_LIGHT / frequencies[int(np.argmin(db))] * 1e9 == pytest.approx(1551.24, abs=0.005)
        assert db.min() == pytest.approx(-14.93, abs=0.005)
        wavelengths = C_LIGHT / frequencies * 1e9
        comb = (wavelengths > 1536.0) & (wavelengths < 1549.5)
        # The comb's notches, each about 10 dB deep.
        assert -10.5 < db[comb].min() < -9.5
    assert sensing.comb_shift_pm(got, 1536.5, 1538.5) == pytest.approx(88, abs=0.5)
    assert sensing.comb_shift_pm(got, 1546.5, 1548.5) == pytest.approx(17, abs=0.5)
    assert sensing.comb_shift_pm(got, 1550.5, 1552.0) == pytest.approx(0, abs=0.5)


def test_6_3_try_this() -> None:
    band = (1536e-9, 1552e-9)

    def deepest(index: float) -> float:
        grating = TiltedFiberBraggGrating(
            period=535.0,
            tilt=4.0,
            length=10.0,
            index_modulation=5e-4,
            surrounding_index=index,
            azimuthal_orders=3.0,
        )
        return grating.resonances(band)[0][2]

    assert deepest(1.40) - deepest(1.0) > deepest(1.333) - deepest(1.0) > 0.0

    got = changed("tilted_grating", "tfbg_air", "tilt", 8.0)
    frequencies, db = transmission(got["osa_air"], got["osa_in"])
    wavelengths = C_LIGHT / frequencies * 1e9
    tilted = TiltedFiberBraggGrating(period=535.0, tilt=8.0)
    assert tilted.bragg_wavelength() * 1e9 == pytest.approx(1562.6, abs=0.05)
    # Nothing in the window is as deep as the Bragg line was.
    assert db[wavelengths > 1550.5].min() > -5.0
    assert db.min() > -10.0


def test_6_4_two_gratings_make_an_interferometer() -> None:
    got = results("lpg_pair")
    frequencies, one = transmission(got["osa_one"], got["osa_in"])
    assert C_LIGHT / frequencies[int(np.argmin(one))] * 1e9 == pytest.approx(1550.14, abs=0.005)
    assert one.min() < -60.0
    fringes = sensing.notches_nm(got["osa_pair"], got["osa_in"], depth=3.0)
    near = fringes[np.abs(fringes - 1550.0) < 3.5]
    assert np.diff(near) == pytest.approx(2.31, abs=0.01)
    frequencies, pair = transmission(got["osa_pair"], got["osa_in"])
    assert C_LIGHT / frequencies[int(np.argmin(pair))] * 1e9 == pytest.approx(1549.93, abs=0.005)

    lam = np.linspace(1545e-9, 1555e-9, 20001)

    def spectrum(separation: float = sensing.LPG_SEPARATION, gap_loss: float = 0.0) -> np.ndarray:
        grating = LongPeriodGrating(
            period=sensing.LPG_PERIOD,
            index_modulation=sensing.LPG_FULL / 2.0,
            pair=True,
            separation=separation,
            gap_loss=gap_loss,
        )
        return 10 * np.log10(grating.scattering_matrix(C_LIGHT / lam).power("out", "in"))

    wide = spectrum(separation=400.0)
    inner = np.arange(1, lam.size - 1)
    dips = inner[(wide[1:-1] < wide[:-2]) & (wide[1:-1] <= wide[2:]) & (wide[1:-1] < -3.0)]
    spacing = np.diff(lam[dips]) * 1e9
    assert spacing[np.argmin(np.abs(lam[dips][1:] - 1550e-9))] == pytest.approx(1.24, abs=0.005)
    # A lossy gap washes the fringes out.
    assert spectrum(gap_loss=6.0).min() > -13.0 > spectrum().min() + 40.0


def test_6_5_the_trace_finds_the_splice_and_the_break() -> None:
    trace = results("otdr")["otdr"]
    assert sensing.level_at(trace, 1.0) == pytest.approx(-41.81, abs=0.005)
    assert sensing.level_at(trace, 5.0) == pytest.approx(-43.41, abs=0.005)
    slope = (sensing.level_at(trace, 1.0) - sensing.level_at(trace, 5.0)) / 4.0
    assert slope == pytest.approx(0.4, abs=0.005)
    step = sensing.level_at(trace, 7.9) - sensing.level_at(trace, 8.1) - 0.2 * 0.4
    assert step == pytest.approx(1.0, abs=0.005)
    assert sensing.last_echo_km(trace) == pytest.approx(17.30, abs=0.005)
    distance, dbm = sensing.trace_km(trace)
    assert dbm[np.abs(distance - 17.3) < 0.01].max() > sensing.level_at(trace, 17.0) + 20.0
    assert np.all(np.isneginf(dbm[distance > 17.5]) | (dbm[distance > 17.5] < -150.0))
    # A kilometre is 9.79 us of trace.
    assert pytest.approx(9.79, abs=0.005) == 2 * 1.468 * 1e3 / C_LIGHT * 1e6

    # Where it starts, from the pulse's energy.
    energy = 0.1 * math.sqrt(math.pi) * 60e-9
    scattering = 0.17 * math.log(10) / 10 / 1e3
    start = energy * 0.0017 * scattering * C_LIGHT / 1.468 / 2.0
    assert 10 * math.log10(start / 1e-3) == pytest.approx(-41.4, abs=0.05)
    assert 20.0 - 10 * math.log10(start / 1e-3) == pytest.approx(61, abs=0.5)


def test_6_5_try_this() -> None:
    full = changed("otdr", "fut", "break_at", 0.0)["otdr"]
    trace = results("otdr")["otdr"]
    assert sensing.last_echo_km(full) == pytest.approx(25.0, abs=0.005)
    distance, dbm = sensing.trace_km(full)
    _, broken = sensing.trace_km(trace)
    assert dbm[np.abs(distance - 25.0) < 0.01].max() > broken[np.abs(distance - 17.3) < 0.01].max()

    smeared = changed("otdr", "pulse", "width", 600000.0)["otdr"]
    assert sensing.level_at(smeared, 1.0) > sensing.level_at(trace, 1.0) + 9.0

    def sharpness(reading: object) -> float:
        distance, dbm = sensing.trace_km(reading)
        near = (distance > 7.5) & (distance < 8.5)
        return float(np.max(-np.diff(dbm[near])))

    assert sharpness(smeared) < sharpness(trace) / 3.0


def test_a_fibre_longer_than_the_window_is_drawn_to_its_edge() -> None:
    """40 km is a 392 us round trip in a 328 us window: the trace runs to the edge."""
    run = sensing.otdr().run(overrides={("fut", "break_at"): 0.0, ("fut", "length"): 40.0})
    trace = {name: value for (name, _), value in run.items()}["otdr"]
    distance, dbm = sensing.trace_km(trace)
    assert distance.max() < 40.0
    assert dbm[-2] == pytest.approx(
        sensing.level_at(trace, 30.0) - 0.4 * (distance[-2] - 30.0), abs=0.05
    )
    # Nothing wrapped round to the start.
    assert sensing.level_at(trace, 1.0) == pytest.approx(-41.81, abs=0.005)


def test_the_oscilloscope_counts_from_the_start_and_asks_for_decibels() -> None:
    trace = results("otdr")["otdr"]
    assert isinstance(trace, ScopeTrace)
    assert trace.time[0] >= 0.0 and trace.decibels
    assert encode(trace)["decibels"] is True


def test_a_broken_fibre_delivers_nothing_and_a_whole_one_its_loss() -> None:
    fibre = BackscatterFiber(length=10.0, splice_at=4.0, splice_loss=0.5)
    assert fibre.round_trip() == pytest.approx(2 * 10e3 * 1.468 / C_LIGHT)
    for broken in (0.0, 6.0):
        g = Graph(sensing.otdr_context())
        pulse = g.add(GaussianPulse(peak_power=20.0, width=60000.0))
        fut = g.add(BackscatterFiber(length=10.0, splice_at=4.0, splice_loss=0.5, break_at=broken))
        meter = g.add(PowerMeter(label="far"))
        g.connect(pulse, fut["in"])
        g.connect(fut["out"], meter["in"])
        launched = g.add(PowerMeter(label="near"))
        g.connect(pulse, launched["in"])
        got = sensing.results_by_label(g)
        far, near = got["far"], got["near"]
        assert isinstance(far, PowerReading) and isinstance(near, PowerReading)
        if broken:
            assert far.power_w == 0.0
        else:
            # 10 km at 0.2 dB/km, and the splice's 0.5 dB.
            assert near.power_dbm - far.power_dbm == pytest.approx(2.5, abs=1e-6)
