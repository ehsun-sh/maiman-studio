"""Phase 4 of the examples: every number a coherent lesson quotes.

The graphs come from ``examples/python/coherent.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens, read back from its
project file, and holds it to the table in its lesson.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import acquisition_link
import coherent as coh
import dualpol_link
import lessons
import numpy as np
import pytest

from maiman import Graph
from maiman.analysis import required_osnr
from maiman.project import load
from maiman.signals import ConstellationMeasurement, ScopeTrace


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(coh.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return coh.results_by_label(opened(key))


def measured(key: str, label: str) -> ConstellationMeasurement:
    value = results(key)[label]
    assert isinstance(value, ConstellationMeasurement)
    return value


@pytest.mark.parametrize("key", sorted(coh.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = coh.EXAMPLES[key]
    on_disk = json.loads(coh.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(coh.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(coh.EXAMPLES[key][1])


def test_4_1_evm_follows_the_snr_and_the_snr_follows_the_osnr() -> None:
    quoted = [
        (26.0, 22.39, 18.21, 12.3),
        (29.0, 19.39, 16.47, 15.0),
        (32.0, 16.39, 14.28, 19.3),
        (35.0, 13.39, 11.75, 25.9),
        (38.0, 10.39, 9.00, 35.5),
    ]
    floor = coh.results_by_label(coh.qpsk_b2b(0.0))["vsa"]
    assert isinstance(floor, ConstellationMeasurement)
    assert floor.snr_db == pytest.approx(21.2, abs=0.05)
    floor_linear = 10 ** (floor.snr_db / 10)
    for (pad, osnr, snr, evm), row in zip(quoted, coh.qpsk_table(), strict=True):
        assert row[0] == pad
        assert row[1] == pytest.approx(osnr, abs=0.005)
        assert row[2] == pytest.approx(snr, abs=0.005)
        assert row[3] == pytest.approx(evm, abs=0.05)
        # EVM ~ 1/sqrt(SNR), and the SNR is the OSNR less 1.07 dB with the
        # transmitter's floor added as a noise power.
        assert row[3] / 100 == pytest.approx(10 ** (-row[2] / 20), rel=0.01)
        from_osnr = 10 ** ((row[1] + 10 * math.log10(2 * 12.5 / 32)) / 10)
        predicted = 10 * math.log10(1 / (1 / from_osnr + 1 / floor_linear))
        assert row[2] == pytest.approx(predicted, abs=0.15)
    assert 10 * math.log10(2 * 12.5 / 32) == pytest.approx(-1.07, abs=0.005)
    # The project opens at 35 dB, with no symbol errors.
    assert measured("qpsk_b2b", "vsa").snr_db == pytest.approx(11.75, abs=0.005)
    assert measured("qpsk_b2b", "vsa").symbol_errors == 0


def test_4_4_the_equaliser_separates_what_the_rotation_mixed() -> None:
    assert measured("dualpol", "vsa_x").evm * 100 == pytest.approx(2.51, abs=0.005)
    assert measured("dualpol", "vsa_y").evm * 100 == pytest.approx(2.52, abs=0.005)
    assert measured("dualpol", "vsa_x").symbol_errors == 0
    assert measured("dualpol", "vsa_y").symbol_errors == 0
    straight_x, straight_y, _ = dualpol_link.measure(0.0, equalize=False)
    assert straight_x.evm * 100 == pytest.approx(2.5, abs=0.05)
    assert straight_y.evm * 100 == pytest.approx(2.5, abs=0.05)
    mixed_x, mixed_y, _ = dualpol_link.measure(coh.ROTATION, equalize=False)
    assert mixed_x.evm * 100 == pytest.approx(218, abs=0.5)
    assert mixed_y.evm * 100 == pytest.approx(123, abs=0.5)
    assert mixed_x.symbol_errors + mixed_y.symbol_errors == 6217


def test_4_5_the_fine_stage_folds_and_the_coarse_stage_does_not() -> None:
    assert coh.FINE_LIMIT == 4e9
    graph, watched = acquisition_link.link(coh.ACQ_OFFSET, acquire=False)
    alone = graph.run(keep=list(watched.values()))
    assert alone.port(watched["fine"], "diagnostics").offset / 1e9 == pytest.approx(2.97, abs=0.005)
    assert alone[watched["vsa"]].evm * 100 == pytest.approx(8300, abs=50)
    assert alone[watched["vsa"]].symbol_errors == 1797
    graph, watched = acquisition_link.link(coh.ACQ_OFFSET, acquire=True)
    both = graph.run(keep=list(watched.values()))
    assert both.port(watched["coarse"], "diagnostics").offset / 1e9 == pytest.approx(
        20.07, abs=0.005
    )
    assert both.port(watched["fine"], "diagnostics").offset / 1e6 == pytest.approx(-71, abs=0.5)
    # The project is that second link, as the studio opens it.
    assert measured("acquisition", "vsa").evm * 100 == pytest.approx(6.7, abs=0.05)
    assert measured("acquisition", "vsa").symbol_errors == 0


def test_4_6_400zr_has_margin_and_800zr_sits_on_the_threshold() -> None:
    rate4, bits, _ = coh.ZR["zr400"]
    rate8, _, _ = coh.ZR["zr800"]
    assert rate4 * bits * 2 / 1e9 == pytest.approx(478.72)
    assert rate8 * bits * 2 / 1e9 == pytest.approx(957.44)
    assert 2 * rate4 * 4 / 6 / 1e9 == pytest.approx(79.8, abs=0.05)
    need4 = required_osnr(2e-2, bits, symbol_rate=rate4)
    need8 = required_osnr(2e-2, bits, symbol_rate=rate8)
    assert need4 == pytest.approx(19.47, abs=0.005)
    assert need8 == pytest.approx(22.48, abs=0.005)
    assert need8 - need4 == pytest.approx(3.01, abs=0.005)
    for key in ("zr400", "zr800"):
        osnr = results(key)["osnr"]
        assert isinstance(osnr, float) and osnr == pytest.approx(23.19, abs=0.005)
    quoted = {
        ("zr400", "vsa_x"): (16.1, 2.3e-3),
        ("zr400", "vsa_y"): (16.2, 1.8e-3),
        ("zr800", "vsa_x"): (23.3, 2.1e-2),
        ("zr800", "vsa_y"): (22.8, 2.0e-2),
    }
    for (key, label), (evm, ber) in quoted.items():
        assert measured(key, label).evm * 100 == pytest.approx(evm, abs=0.05)
        assert measured(key, label).ber_counted == pytest.approx(ber, rel=0.03)


def test_4_7_each_lap_arrives_800_ps_later_and_weaker() -> None:
    trace = results("loop")["scope"]
    assert isinstance(trace, ScopeTrace)
    time, power = np.asarray(trace.time), np.asarray(trace.power_w)
    loss = 10 ** (-coh.LOOP_LOSS_DB / 10)
    expected = [0.5, 0.25 * loss]
    while len(expected) < coh.LAPS:
        expected.append(expected[-1] * 0.5 * loss)
    quoted = [0.500, 0.199, 0.079, 0.031]
    for lap, (mw, want) in enumerate(zip(quoted, expected, strict=True)):
        near = np.abs(time - lap * coh.LOOP_PS * 1e-12) < 100e-12
        peak = float(power[near].max()) * 1e3
        assert peak == pytest.approx(mw, abs=0.0005)
        assert peak == pytest.approx(want, rel=1e-3)


def test_4_8_shaping_gains_until_the_entropy_caps_it() -> None:
    assert measured("pcs", "vsa_u").mutual_information == pytest.approx(2.96, abs=0.005)
    assert measured("pcs", "vsa_u").snr_db == pytest.approx(9.1, abs=0.05)
    assert measured("pcs", "vsa_s").mutual_information == pytest.approx(3.10, abs=0.005)
    assert measured("pcs", "vsa_s").entropy == pytest.approx(3.70, abs=0.005)
    quoted = {3.85: (1.25, 3.08), 3.7: (1.87, 3.10), 3.5: (2.60, 3.05), 3.28: (3.47, 2.94)}
    rows = {row[0]: row for row in coh.pcs_table()}
    for entropy, (backoff, mi) in quoted.items():
        assert rows[entropy][1] == pytest.approx(backoff, abs=0.005)
        assert rows[entropy][3] == pytest.approx(mi, abs=0.005)
        assert rows[entropy][3] <= entropy
