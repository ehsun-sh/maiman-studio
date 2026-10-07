"""Phase 7 of the examples: every number a real-world-systems lesson quotes.

The graphs come from ``examples/python/systems.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens, read back from its
project file, and holds it to the numbers in its lesson.
"""

from __future__ import annotations

import json
import re
from functools import cache

import lessons
import numpy as np
import pytest
import systems

from maiman import Graph
from maiman.components import (
    BB84Receiver,
    ElectricalSpectrumAnalyzer,
    FreeSpaceChannel,
    RFTone,
    SweptLaser,
)
from maiman.components.free_space import fade_db, geometric_loss_db, rytov_variance
from maiman.components.quantum import bb84
from maiman.components.rf import ElectricalSpectrum
from maiman.encoding import encode
from maiman.project import load
from maiman.signals import EyeMeasurement, PowerReading, Readout
from maiman.units import C_LIGHT


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(systems.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return systems.results_by_label(opened(key))


def changed(key: str, *edits: tuple[str, str, float]) -> dict[str, object]:
    """The project's results with parameters changed, as a *Try this* asks."""
    run = systems.EXAMPLES[key][0]().run(overrides={(label, p): v for label, p, v in edits})
    return {name: reading for (name, _), reading in run.items()}


@pytest.mark.parametrize("key", sorted(systems.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = systems.EXAMPLES[key]
    on_disk = json.loads(systems.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(systems.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(systems.EXAMPLES[key][1])


# ---------------------------------------------------------------------------
# 7.1


def test_7_1_eight_nodes_narrow_the_passband() -> None:
    got = results("metro_ring")
    one, loss_one = systems.passband_ghz(got["osa_1"], got["osa_in"])
    eight, loss_eight = systems.passband_ghz(got["osa_8"], got["osa_in"])
    assert one == pytest.approx(49.89, abs=0.005)
    assert eight == pytest.approx(35.24, abs=0.005)
    assert systems.cascade_width_ghz(8) == pytest.approx(35.36, abs=0.005)
    # Within one analyser bin of the formula.
    assert abs(eight - systems.cascade_width_ghz(8)) < 0.4
    assert loss_one == pytest.approx(-4.00, abs=0.005)
    assert loss_eight == pytest.approx(-32.00, abs=0.005)


def test_7_1_try_this() -> None:
    got = changed("metro_ring", *((f"node{k}", "order", 5.0) for k in range(1, 9)))
    width, _ = systems.passband_ghz(got["osa_8"], got["osa_in"])
    assert width == pytest.approx(40.51, abs=0.005)
    assert systems.cascade_width_ghz(8, order=5.0) == pytest.approx(40.61, abs=0.005)


def test_7_1_white_light_cannot_show_the_narrowing() -> None:
    """Why the lesson uses a pulse: noise through a passband is a flat rectangle."""
    from maiman.components.filters import apply_passband
    from maiman.signals import NoiseBin, OpticalSignal

    white = OpticalSignal(bands=(), noise=(NoiseBin(193.0e12, 194.0e12, 1e-15, 1e-15),))
    once = apply_passband(white, centre=193.4e12, bandwidth=50e9, order=3)
    twice = apply_passband(once, centre=193.4e12, bandwidth=50e9, order=3)
    assert [b.bandwidth for b in twice.noise] == [b.bandwidth for b in once.noise]


# ---------------------------------------------------------------------------
# 7.2


def test_7_2_the_rule_of_thumb() -> None:
    assert pytest.approx(23.9) == 58.0 + 0.0 - 4.5 - 9.6 - 20.0
    line = systems.cable()
    assert line.span_loss_db() == pytest.approx(9.6)
    assert line.length_km() == pytest.approx(6000.0)


def test_7_2_the_tilt_piles_up() -> None:
    rows = {round(nm): (power, osnr) for nm, power, osnr in systems.cable_channels(False)}
    assert rows[1532] == pytest.approx((-11.36, 17.00), abs=0.005)
    assert rows[1548] == pytest.approx((-1.25, 23.20), abs=0.005)
    assert rows[1560] == pytest.approx((6.20, 26.61), abs=0.005)
    assert rows[1560][0] - rows[1532][0] == pytest.approx(17.6, abs=0.05)
    # About a tenth of a decibel per repeater at the band's edge.
    line = systems.cable()
    assert line.tilt_db(C_LIGHT / 1532e-9) == pytest.approx(-0.11, abs=0.005)


def test_7_2_flattening_evens_the_channels() -> None:
    rows = systems.cable_channels(True)
    assert [power for _, power, _ in rows] == pytest.approx([0.0] * 8, abs=0.005)
    osnr = [value for _, _, value in rows]
    assert min(osnr) == pytest.approx(23.80, abs=0.005)
    assert max(osnr) == pytest.approx(23.88, abs=0.005)
    assert max(osnr) - 23.9 < 0.1


def test_7_2_the_line_reads_the_same_on_an_osnr_meter() -> None:
    """The per-channel table agrees with the block's own closed form."""
    line = systems.cable(True)
    f = C_LIGHT / 1548e-9
    rows = {round(nm): osnr for nm, _, osnr in systems.cable_channels(True)}
    assert line.osnr_db(f, 1e-3) == pytest.approx(rows[1548], abs=0.01)


# ---------------------------------------------------------------------------
# 7.3


def test_7_3_zr_on_a_dwdm_line() -> None:
    got = results("zr_dwdm")
    assert got["osnr"] == pytest.approx(32.95, abs=0.005)
    assert systems.zr_evm(got) == pytest.approx(6.19, abs=0.005)
    assert [got[f"vsa_{axis}"].symbol_errors for axis in "xy"] == [0, 0]  # type: ignore[attr-defined]


def test_7_3_the_pluggable_puts_out_what_the_lesson_says() -> None:
    import reference_rates

    from maiman.components import PowerMeter

    def tap(graph: Graph, tx: object) -> object:
        graph.connect(tx, graph.add(PowerMeter(label="pm"))["in"])  # type: ignore[arg-type]
        return tx

    rate, bits, _ = reference_rates.CONFIGURATIONS["400G DP-16QAM"]
    graph = reference_rates.build(rate, bits, line=tap)[0]  # type: ignore[arg-type]
    reading = systems.results_by_label(graph)["pm"]
    assert isinstance(reading, PowerReading)
    assert reading.power_dbm == pytest.approx(systems.ZR_NEIGHBOUR_DBM, abs=0.05)


def test_7_3_try_this() -> None:
    bare = systems.results_by_label(systems.zr_dwdm(demux=False))
    assert systems.zr_evm(bare) == pytest.approx(5.99, abs=0.005)
    assert [bare[f"vsa_{axis}"].symbol_errors for axis in "xy"] == [0, 0]  # type: ignore[attr-defined]

    low = changed("zr_dwdm", ("boost", "gain", 8.0))
    assert low["osnr"] == pytest.approx(26.43, abs=0.005)
    assert systems.zr_evm(low) == pytest.approx(11.59, abs=0.005)
    errors = [low[f"vsa_{axis}"].symbol_errors for axis in "xy"]  # type: ignore[attr-defined]
    assert 0 < sum(errors) < 10


# ---------------------------------------------------------------------------
# 7.4


def test_7_4_intermodulation_matches_bessel() -> None:
    levels = systems.rof_levels(results("rof")["esa"])
    assert levels["carrier"] == pytest.approx(-16.61, abs=0.005)
    assert levels["imd_low"] == pytest.approx(-50.71, abs=0.005)
    assert levels["carrier"] - levels["imd_low"] == pytest.approx(34.10, abs=0.005)
    assert levels["carrier"] - levels["imd_high"] == pytest.approx(34.10, abs=0.05)
    assert systems.carrier_to_intermod_db(0.5) == pytest.approx(34.07, abs=0.005)
    assert levels["harmonic"] == pytest.approx(-110.44, abs=0.005)


def test_7_4_quadrature_bias_leaves_no_second_harmonic() -> None:
    reading = results("rof")["esa"]
    assert isinstance(reading, ElectricalSpectrum)
    f = np.asarray(reading.frequencies)
    db = 10.0 * np.log10(np.asarray(reading.power_w) / 1e-3)
    beside = (f > 1.9e9) & (f < 2.1e9)
    assert reading.dbm_at(2e9) < float(np.median(db[beside])) + 3.0


def test_7_4_try_this() -> None:
    levels = systems.rof_levels(changed("rof", ("rf", "amplitude", 1.0))["esa"])
    base = systems.rof_levels(results("rof")["esa"])
    assert levels["carrier"] - base["carrier"] == pytest.approx(4.45, abs=0.005)
    assert levels["imd_low"] - base["imd_low"] == pytest.approx(17.23, abs=0.005)
    assert levels["carrier"] - levels["imd_low"] == pytest.approx(21.32, abs=0.005)
    assert systems.carrier_to_intermod_db(1.0) == pytest.approx(21.31, abs=0.005)


def test_the_bessel_integral_is_the_textbook_one() -> None:
    assert systems.bessel_j(0, 1.0) == pytest.approx(0.7651976866, abs=1e-9)
    assert systems.bessel_j(2, 1.0) == pytest.approx(0.1149034849, abs=1e-9)


def test_a_tone_lands_in_one_bin_of_the_analyser() -> None:
    from maiman import SimulationContext

    ctx = SimulationContext(bit_rate=1e9, samples_per_symbol=16, sequence_length=1000, seed=1)
    graph = Graph(ctx)
    tone = graph.add(RFTone(frequency=1.0003, amplitude=1.0, label="rf"))
    esa = graph.add(ElectricalSpectrumAnalyzer(label="esa"))
    graph.connect(tone, esa["in"])
    reading = graph.run()[esa]
    assert isinstance(reading, ElectricalSpectrum)
    # 1 V peak into 50 ohm is 10 mW: +10 dBm, moved onto the 1 MHz grid.
    assert reading.dbm_at(1.0e9) == pytest.approx(10.0, abs=1e-6)
    assert tone.tones(ctx) == [pytest.approx(1.0e9)]


def test_the_analyser_encodes_on_a_radio_axis() -> None:
    encoded = encode(results("rof")["esa"])
    assert encoded["kind"] == "spectrum"
    assert encoded["axis"] == "rf"
    assert encoded["frequencies_ghz"][0] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 7.5


def test_7_5_the_link_budget() -> None:
    air = FreeSpaceChannel()
    parts = air.budget()
    assert pytest.approx(1.025) == 0.025 + 1e-3 * 1000.0
    assert parts["geometric"] == pytest.approx(20.21, abs=0.005)
    assert parts["atmospheric"] == pytest.approx(0.5)
    assert air.loss_db() == pytest.approx(20.71, abs=0.005)
    assert parts["rytov"] == pytest.approx(0.199, abs=0.0005)
    assert fade_db(parts["rytov"], 1e-3) == pytest.approx(6.11, abs=0.005)
    strong = rytov_variance(1e-13, 1550e-9, 1000.0)
    assert fade_db(strong, 1e-3) == pytest.approx(16.43, abs=0.005)
    assert geometric_loss_db(1000.0, 1e-3, 0.025, 2.0) == 0.0

    got = results("free_space")
    assert isinstance(got["pm_rx"], PowerReading)
    assert got["pm_rx"].power_dbm == pytest.approx(-13.72, abs=0.005)
    assert isinstance(got["ber"], EyeMeasurement)
    assert got["ber"].q_factor == pytest.approx(20.86, abs=0.005)


@pytest.mark.parametrize(
    ("edits", "dbm", "errors"),
    [
        ((("air", "outage", 1e-3),), -19.83, 0),
        ((("air", "outage", 1e-3), ("air", "cn2", 1e-13)), -30.14, 630),
        ((("air", "attenuation", 10.0),), -23.22, 11),
    ],
)
def test_7_5_try_this(edits: tuple[tuple[str, str, float], ...], dbm: float, errors: int) -> None:
    got = changed("free_space", *edits)
    assert isinstance(got["pm_rx"], PowerReading)
    assert got["pm_rx"].power_dbm == pytest.approx(dbm, abs=0.005)
    assert isinstance(got["ber"], EyeMeasurement)
    assert got["ber"].errors == errors
    assert got["ber"].bits_evaluated == 2040
    if len(edits) == 1 and edits[0][1] == "outage":
        assert got["ber"].q_factor == pytest.approx(5.24, abs=0.005)


# ---------------------------------------------------------------------------
# 7.6


def test_7_6_the_beat_is_the_range() -> None:
    beat = systems.beat_hz(results("lidar")["esa"])
    assert beat == pytest.approx(200.0e6, abs=1.0)
    assert systems.range_from_beat(beat) == pytest.approx(29.98, abs=0.005)
    assert systems.round_trip_ps(30.0) == pytest.approx(200138.0, abs=1.0)
    # One analyser bin is c / 2B of range.
    assert pytest.approx(0.15, abs=0.001) == C_LIGHT / (2.0 * systems.LIDAR_SWEEP_GHZ * 1e9)
    assert SweptLaser(sweep=1.0).chirp_rate(systems.lidar_context()) == pytest.approx(1e15)


def test_7_6_try_this() -> None:
    assert round(systems.round_trip_ps(60.0)) == 400277
    got = changed("lidar", ("target", "delay", 400277.0))
    assert systems.beat_hz(got["esa"]) == pytest.approx(400.0e6, abs=1.0)


# ---------------------------------------------------------------------------
# 7.7


def test_7_7_bob_reads_the_lesson() -> None:
    assert systems.mu_attenuation_db() == pytest.approx(71.93, abs=0.005)
    reading = results("qkd")["bob"]
    assert isinstance(reading, Readout)
    assert reading["mean_photons"] == pytest.approx(0.5, abs=0.001)
    assert reading["channel_db"] == pytest.approx(10.0)
    assert reading["gain"] == pytest.approx(5.0e-3, abs=0.05e-3)
    assert reading["qber"] * 100 == pytest.approx(1.51, abs=0.005)
    assert reading["key_rate"] / 1e3 == pytest.approx(1019, abs=0.5)
    assert encode(reading)["kind"] == "readout"
    assert encode(reading)["qber"] == pytest.approx(reading["qber"])


@pytest.mark.parametrize(
    ("km", "qber", "key_kbps"),
    [
        (25.0, 1.50, 3233),
        (50.0, 1.51, 1019),
        (100.0, 1.60, 100.0),
        (150.0, 2.45, 8.26),
        (200.0, 9.58, 0.0),
    ],
)
def test_7_7_distance_table(km: float, qber: float, key_kbps: float) -> None:
    reading = changed("qkd", ("fib", "length", km))["bob"]
    assert isinstance(reading, Readout)
    assert reading["qber"] * 100 == pytest.approx(qber, abs=0.005)
    assert reading["key_rate"] / 1e3 == pytest.approx(key_kbps, rel=5e-4, abs=0.005)


def test_bb84_has_no_key_without_a_channel() -> None:
    assert bb84(0.5, 0.0, dark=1e-6, misalignment=0.015)["key_per_pulse"] == 0.0
    assert BB84Receiver().efficiency == pytest.approx(0.1)
