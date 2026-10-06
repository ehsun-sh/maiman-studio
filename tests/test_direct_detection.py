"""Phase 2 of the examples: every number a direct-detection lesson quotes.

The graphs come from ``examples/python/direct_detection.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens -- read back from the
project file, so what is checked is what a user runs -- and holds it to the
table in its lesson.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import direct_detection as dd
import lessons
import numpy as np
import pytest

from maiman import Graph
from maiman.components import APDPhotodiode
from maiman.project import load
from maiman.signals import EyeMeasurement, PAMMeasurement, PowerReading, ScopeTrace


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(dd.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return dd.results_by_label(opened(key))


def q(key: str, label: str) -> float:
    value = results(key)[label]
    assert isinstance(value, EyeMeasurement)
    return value.q_factor


@pytest.mark.parametrize("key", sorted(dd.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = dd.EXAMPLES[key]
    on_disk = json.loads(dd.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(dd.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(dd.EXAMPLES[key][1])


def test_2_2_the_apd_buys_nine_decibels() -> None:
    table = dd.sensitivity_table(opened("receiver_sensitivity"))
    powers = [p for p, _, _ in table]
    pin = dd.sensitivity_at(6.0, powers, [x for _, x, _ in table])
    apd = dd.sensitivity_at(6.0, powers, [x for _, _, x in table])
    assert pin == pytest.approx(-19.2, abs=0.05)
    assert apd == pytest.approx(-28.1, abs=0.05)
    # Thermal-limited: Q in proportion to power, so 2 dB less is 0.63 of the Q.
    assert table[1][1] / table[0][1] == pytest.approx(10 ** (-0.2), rel=0.05)


def test_2_2_there_is_a_best_avalanche_gain() -> None:
    curve = dict(dd.apd_gain_table(opened("receiver_sensitivity")))
    assert curve[1.0] == pytest.approx(0.84, abs=0.01)
    assert curve[10.0] == pytest.approx(6.41, abs=0.01)
    assert curve[40.0] == pytest.approx(6.32, abs=0.01)
    best = max(curve, key=lambda gain: curve[gain])
    assert best == 20.0 and curve[best] == pytest.approx(7.32, abs=0.01)


def test_2_2_an_apd_at_unit_gain_is_the_pin() -> None:
    """The comparison is fair only if the two receivers differ by the avalanche alone."""
    graph = opened("receiver_sensitivity")
    apd = next(c for c in graph.components if c.label == "apd")
    assert isinstance(apd, APDPhotodiode) and apd.excess_noise_factor() > 1.0
    from maiman import sweep

    point = sweep(graph, {(apd, "gain"): [1.0]}).points[0].runs[0]
    by = {c.label: c for c in graph.components}
    pin_q = point[by["ber_pin"]].q_factor
    assert point[by["ber_apd"]].q_factor == pytest.approx(pin_q, rel=0.05)


def test_2_3_the_dml_chirps_and_loses_the_reach() -> None:
    assert dd.matched_external() == (14.85, 4.84)
    assert q("dml_reach", "ber_dml") == pytest.approx(2.7, abs=0.05)
    assert q("dml_reach", "ber_ext") == pytest.approx(39.8, abs=0.1)
    scope = results("dml_reach")["chirp"]
    assert isinstance(scope, ScopeTrace)
    assert np.nanmax(scope.chirp_hz) / 1e9 == pytest.approx(20.9, abs=0.2)
    # The lesson's delay: 17 ps/(nm km) x 20 km x the excursion as a wavelength.
    excursion_nm = 1550e-9**2 * np.nanmax(scope.chirp_hz) / 299792458.0 * 1e9
    assert 17.0 * 20.0 * excursion_nm == pytest.approx(58.0, abs=2.0)


def test_2_4_partition_is_what_separates_the_two_fabry_perot_rows() -> None:
    assert q("mode_partition", "ber_dfb") == pytest.approx(104.0, abs=1.0)
    assert q("mode_partition", "ber_fp_steady") == pytest.approx(28.6, abs=0.1)
    assert q("mode_partition", "ber_fp") == pytest.approx(13.0, abs=0.1)


def test_2_4_partition_noise_is_a_floor_power_does_not_lift() -> None:
    from maiman import sweep

    graph = opened("mode_partition")
    by = {c.label: c for c in graph.components}
    louder = sweep(graph, {(by["tx_fp"], "power"): [10.0]}).points[0].runs[0]
    assert louder[by["ber_fp"]].q_factor == pytest.approx(13.0, rel=0.05)
    assert louder[by["ber_dfb"]].q_factor == pytest.approx(104.0, rel=0.05)


def test_2_5_the_equaliser_takes_the_lane_from_failing_to_clean() -> None:
    off, on = (results("pam4_lane")[k] for k in ("eq_off", "eq"))
    assert isinstance(off, PAMMeasurement) and isinstance(on, PAMMeasurement)
    assert off.snr_db == pytest.approx(9.9, abs=0.05)
    assert off.symbol_errors == 609
    assert on.snr_db == pytest.approx(38.75, abs=0.05)
    assert on.symbol_errors == 0


def test_2_6_the_lanes_sit_on_the_coarse_grid_and_lose_by_their_dispersion() -> None:
    graph = opened("cwdm4")
    lasers = sorted(
        (c for c in graph.components if re.fullmatch(r"tx\d", c.label)), key=lambda c: c.label
    )
    assert [laser.si("wavelength") * 1e9 for laser in lasers] == pytest.approx(
        [1331, 1311, 1291, 1271]
    )
    quoted = {1331: 1.9, 1311: 0.1, 1291: -1.7, 1271: -3.6}
    for wavelength, d in quoted.items():
        assert dd.lane_dispersion(wavelength) == pytest.approx(d, abs=0.05)
    assert [q("cwdm4", f"ber{i}") for i in range(4)] == pytest.approx(
        [25.3, 26.0, 25.9, 20.3], abs=0.05
    )


def test_2_7_the_split_is_the_budget() -> None:
    reading = results("gpon")["pm_onu"]
    assert isinstance(reading, PowerReading)
    assert reading.power_dbm == pytest.approx(-21.56, abs=0.01)
    assert 10 * math.log10(32) == pytest.approx(15.05, abs=0.005)
    assert reading.power_dbm - (-28.0) == pytest.approx(6.45, abs=0.05)
    assert q("gpon", "ber") == pytest.approx(35.6, abs=0.05)
