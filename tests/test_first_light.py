"""Phase 1 of the examples: every number a first-light lesson quotes.

The lessons in ``examples/python/lessons.py`` print tables of what each graph
measures. These tests run the same graphs from ``first_light.py`` and hold them
to those tables, and to the closed form each table illustrates, so neither the
prose nor the physics can move without the other.
"""

from __future__ import annotations

import json
import math
import re

import first_light as fl
import lessons
import pytest

from maiman import sweep
from maiman.component import Component
from maiman.graph import Graph
from maiman.signals import OpticalSpectrum, PowerReading, ScopeTrace
from maiman.units import w_to_dbm


def by_label(graph: Graph, label: str) -> Component:
    return next(c for c in graph.components if c.label == label)


def run(build: object) -> dict[str, object]:
    assert callable(build)
    return fl.results_by_label(build())


def dbm(value: object) -> float:
    assert isinstance(value, PowerReading)
    return value.power_dbm


def scope(value: object) -> ScopeTrace:
    assert isinstance(value, ScopeTrace)
    return value


@pytest.mark.parametrize("key", sorted(fl.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = fl.EXAMPLES[key]
    on_disk = json.loads(fl.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(fl.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(fl.EXAMPLES[key][1])


def test_1_1_one_milliwatt_read_three_ways() -> None:
    results = run(fl.laser_meters)
    assert dbm(results["pm_tx"]) == pytest.approx(0.0, abs=1e-4)
    assert dbm(results["pm_rx"]) == pytest.approx(-3.0, abs=1e-4)
    spectrum = results["osa"]
    assert isinstance(spectrum, OpticalSpectrum)
    shown = spectrum.power_per_resolution()
    peak = int(shown.argmax())
    # A 100 kHz line inside a 12.5 GHz filter reads the laser's whole power.
    assert w_to_dbm(float(shown[peak])) == pytest.approx(0.0, abs=0.05)
    assert spectrum.wavelengths_nm[peak] == pytest.approx(1550.0, abs=0.01)


def test_1_2_the_budget_table() -> None:
    graph = fl.loss_budget()
    span, meter = by_label(graph, "span"), by_label(graph, "pm_rx")
    lengths = [0.0, 25.0, 50.0, 75.0, 100.0]
    received = [dbm(p.runs[0][meter]) for p in sweep(graph, {(span, "length"): lengths})]
    assert received == pytest.approx([-1.0, -6.0, -11.0, -16.0, -21.0], abs=1e-4)


def test_1_3_the_cosine_and_its_floor() -> None:
    graph = fl.mzm_curve()
    bias, meter = by_label(graph, "bias"), by_label(graph, "pm_out")
    volts = [0.0, 1.0, 2.0, 4.0, 8.0]
    out = [dbm(p.runs[0][meter]) for p in sweep(graph, {(bias, "voltage"): volts})]
    assert out[0] == pytest.approx(0.0, abs=0.01)
    assert out[1] == pytest.approx(10 * math.log10(math.cos(math.pi / 8) ** 2), abs=0.01)
    assert out[2] == pytest.approx(-3.01, abs=0.01), "quadrature"
    assert out[3] == pytest.approx(-30.0, abs=0.05), "null, held up by the 30 dB extinction"
    assert out[4] == pytest.approx(0.0, abs=0.01), "periodic in 2 V_pi"


def test_1_4_the_spreading_table() -> None:
    results = run(fl.pulse_spreading)
    assert pytest.approx(4.61, abs=0.01) == fl.LD_KM
    launch = scope(results["launch"])
    assert launch.fwhm * 1e12 == pytest.approx(16.65, abs=0.01)
    for label, km, quoted in (("after_5km", 5.0, 24.6), ("after_10km", 10.0, 39.8)):
        factor = math.sqrt(1.0 + (km / fl.LD_KM) ** 2)
        width = scope(results[label]).fwhm
        assert width / launch.fwhm == pytest.approx(factor, rel=2e-3)
        assert width * 1e12 == pytest.approx(quoted, abs=0.05)


def test_1_5_the_sign_of_the_chirp_decides() -> None:
    results = run(fl.chirp_compression)
    assert pytest.approx(1.84, abs=1e-9) == fl.COMPRESSION_KM
    launch = scope(results["launch"]).fwhm
    up = scope(results["up_out"]).fwhm
    down = scope(results["down_out"]).fwhm
    assert up / launch == pytest.approx(0.447, abs=0.002)
    assert up * 1e12 == pytest.approx(7.45, abs=0.02)
    assert down / launch == pytest.approx(1.84, abs=0.005)
    assert down * 1e12 == pytest.approx(30.7, abs=0.05)


def test_1_6_the_soliton_holds_and_its_linear_twin_does_not() -> None:
    results = run(fl.soliton)
    assert pytest.approx(167.0, abs=0.5) == fl.SOLITON_W * 1e3
    assert pytest.approx(14.49, abs=1e-9) == fl.SOLITON_KM
    launch, kerr, linear = (scope(results[k]) for k in ("launch", "with_kerr", "dispersion_only"))
    assert launch.fwhm == pytest.approx(2.0 * math.acosh(math.sqrt(2.0)) * 10e-12, rel=2e-3)
    assert kerr.fwhm == pytest.approx(launch.fwhm, rel=0.01)
    assert kerr.peak_power_w == pytest.approx(launch.peak_power_w, rel=0.01)
    assert linear.fwhm * 1e12 == pytest.approx(49.5, abs=0.2)
    assert linear.peak_power_w * 1e3 == pytest.approx(65.0, abs=1.0)


def test_1_7_powers_divide_and_add_in_milliwatts() -> None:
    results = run(fl.splitters)
    assert dbm(results["pm_quarter"]) == pytest.approx(-6.02, abs=0.005)
    assert dbm(results["pm_tap"]) == pytest.approx(-10.0, abs=1e-4)
    total = results["pm_sum"]
    assert isinstance(total, PowerReading)
    assert total.power_w * 1e3 == pytest.approx(0.25 + 0.90, rel=1e-6)
    assert total.power_dbm == pytest.approx(0.61, abs=0.005)
