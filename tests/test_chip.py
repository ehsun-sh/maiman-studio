"""Phase 5 of the examples: every number a photonic-circuit lesson quotes.

The graphs come from ``examples/python/chip.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens, read back from its
project file, and holds it to the table in its lesson.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import chip
import lessons
import numpy as np
import pytest

from maiman import Graph
from maiman.photonics import (
    SILICON_STRIP_NEFF,
    coupled_waveguides,
    coupling_length,
    resonance_linewidth,
)
from maiman.project import load
from maiman.signals import PowerReading
from maiman.units import C_LIGHT

F0 = C_LIGHT / 1550e-9


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(chip.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return chip.results_by_label(opened(key))


def dbm(key: str, label: str) -> float:
    value = results(key)[label]
    assert isinstance(value, PowerReading)
    return value.power_dbm


def through(key: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    return chip.transmission(results(key)[label], results(key)["osa_in"])


@pytest.mark.parametrize("key", sorted(chip.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = chip.EXAMPLES[key]
    on_disk = json.loads(chip.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(chip.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(chip.EXAMPLES[key][1])


def test_5_1_the_power_walks_across_and_back() -> None:
    assert coupling_length(1550e-9, chip.DELTA_N) * 1e6 == pytest.approx(19.375)
    quoted = [0.000, 0.146, 0.500, 0.853, 1.000, 0.500, 0.000]
    for length, share in zip(chip.COUPLER_LENGTHS, quoted, strict=True):
        assert chip.cross_fraction(length) == pytest.approx(share, abs=0.0005)
    # The project opens as a 3 dB coupler.
    assert dbm("coupler_length", "pm_bar") == pytest.approx(
        dbm("coupler_length", "pm_cross"), abs=0.01
    )
    for wavelength, share in ((1534e-9, 0.508), (1566e-9, 0.492)):
        matrix = coupled_waveguides(
            np.array([C_LIGHT / wavelength]), length=9.69e-6, delta_n=chip.DELTA_N
        )
        assert matrix.power("out2", "in1")[0] == pytest.approx(share, abs=0.0005)
    _, bar = through("coupler_length", "osa_bar")
    _, cross = through("coupler_length", "osa_cross")
    for trace in (bar, cross):
        assert trace.max() == pytest.approx(-2.94, abs=0.005)
        assert trace.min() == pytest.approx(-3.08, abs=0.005)
    assert bar[0] > bar[-1] and cross[0] < cross[-1]


def test_5_1_the_coupler_conserves_power_at_every_length() -> None:
    grid = np.linspace(F0 - 2e12, F0 + 2e12, 7)
    for length in (0.0, 5e-6, 19.375e-6, 33e-6):
        s = coupled_waveguides(grid, length=length, delta_n=0.04).s
        for matrix in s:
            assert np.allclose(matrix.conj().T @ matrix, np.eye(4), atol=1e-12)


def test_5_2_the_interleaver_period_is_set_by_the_group_index() -> None:
    assert chip.fsr_ghz(4.20, chip.MZI_DELTA_L_UM) == pytest.approx(356.9, abs=0.05)
    assert chip.fsr_ghz(SILICON_STRIP_NEFF, chip.MZI_DELTA_L_UM) == pytest.approx(614, abs=0.5)
    graph = results("mzi")
    bar = chip.spacing_ghz(graph["osa_bar"], graph["osa_in"])
    cross = chip.spacing_ghz(graph["osa_cross"], graph["osa_in"])
    assert bar == pytest.approx(357.0, abs=0.05)
    assert cross == pytest.approx(356.9, abs=0.05)
    _, trace = through("mzi", "osa_bar")
    assert trace.min() == pytest.approx(-46.0, abs=0.05)
    # Complementary: the bar port's peaks are the cross port's nulls.
    frequencies, cross_db = through("mzi", "osa_cross")
    for peak in chip.peaks(graph["osa_bar"], graph["osa_in"]):
        assert cross_db[int(np.argmin(np.abs(frequencies - peak)))] < -30.0


def test_5_3_the_ring_drops_a_comb_and_notches_the_rest() -> None:
    assert chip.fsr_ghz(4.20, chip.RING_UM) == pytest.approx(713.8, abs=0.05)
    graph = results("ring")
    assert chip.spacing_ghz(graph["osa_drop"], graph["osa_in"]) == pytest.approx(714.2, abs=0.05)
    width = resonance_linewidth(
        chip.RING_UM * 1e-6, 4.20, coupling=0.05, drop_coupling=0.05, loss_db_per_m=200.0
    )
    assert width / 1e9 == pytest.approx(12.2, abs=0.05)
    assert through("ring", "osa_drop")[1].max() == pytest.approx(-0.40, abs=0.005)
    assert through("ring", "osa_through")[1].min() == pytest.approx(-21.7, abs=0.05)
    assert pytest.approx(0.0046, abs=5e-5) == 1.0 - 10 ** (-200.0 * chip.RING_UM * 1e-6 / 10.0)


def test_5_4_one_ring_two_combs() -> None:
    assert chip.fsr_ghz(4.20, chip.RING_UM) == pytest.approx(713.8, abs=0.05)
    assert chip.fsr_ghz(3.80, chip.RING_UM) == pytest.approx(788.9, abs=0.05)
    graph = results("birefringent_ring")
    plain = np.sort(chip.peaks(graph["osa_plain"], graph["osa_in"]) - F0) / 1e9
    biref = np.sort(chip.peaks(graph["osa_biref"], graph["osa_in"]) - F0) / 1e9
    assert plain == pytest.approx([-299, 415], abs=1.0)
    assert biref == pytest.approx([-661, -299, 128, 415, 916], abs=1.0)
    _, one = through("birefringent_ring", "osa_plain")
    _, two = through("birefringent_ring", "osa_biref")
    assert one.max() - two.max() == pytest.approx(3.0, abs=0.05)


def test_5_5_edge_couplers_lose_to_overlap_and_gratings_to_a_passband() -> None:
    eta = (2 * 10.4 * 3.0 / (10.4**2 + 3.0**2)) ** 2
    assert eta == pytest.approx(0.284, abs=0.0005)
    assert 10 * math.log10(eta) == pytest.approx(-5.47, abs=0.005)
    fresnel = sum(-10 * math.log10(1 - ((n - 1.0) / (n + 1.0)) ** 2) for n in (1.4682, 1.45))
    assert fresnel == pytest.approx(0.31, abs=0.005)
    assert -10 * math.log10(eta) + fresnel == pytest.approx(5.78, abs=0.005)
    assert dbm("chip_couplers", "pm_edge") == pytest.approx(-12.56, abs=0.005)
    frequencies, trace = through("chip_couplers", "osa_gc")
    assert trace.max() == pytest.approx(-10.0, abs=0.05)
    assert C_LIGHT / frequencies[int(np.argmax(trace))] * 1e9 == pytest.approx(1550, abs=1.0)
    passband = C_LIGHT / frequencies[trace > trace.max() - 1.0] * 1e9
    assert passband.max() - passband.min() == pytest.approx(24.7, abs=0.05)
    assert dbm("chip_couplers", "pm_ase") == pytest.approx(2.10, abs=0.005)
    assert dbm("chip_couplers", "pm_gc") == pytest.approx(-8.44, abs=0.005)


def test_5_6_the_kit_is_not_the_textbook() -> None:
    for k in range(1, 5):
        assert dbm("pdk_splitter", f"pm_ideal{k}") == pytest.approx(-6.03, abs=0.005)
    quoted = [-6.31, -6.42, -6.54, -6.66]
    arms = [dbm("pdk_splitter", f"pm_kit{k}") for k in range(1, 5)]
    assert arms == pytest.approx(quoted, abs=0.005)
    assert max(arms) - min(arms) == pytest.approx(0.35, abs=0.005)
    ideal = sum(10 ** (dbm("pdk_splitter", f"pm_ideal{k}") / 10) for k in range(1, 5))
    kit = sum(10 ** (a / 10) for a in arms)
    assert 10 * math.log10(ideal / kit) == pytest.approx(0.45, abs=0.005)
