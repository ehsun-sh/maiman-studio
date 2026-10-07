"""Phase 3 of the examples: every number an amplified-link lesson quotes.

The graphs come from ``examples/python/amplified.py`` and the prose from
``lessons.py``. Each test runs the graph the studio opens, read back from its
project file, and holds it to the table in its lesson.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache

import amplified as amp
import lessons
import pytest

from maiman import Graph, sweep
from maiman.project import load
from maiman.signals import EyeMeasurement, OpticalSpectrum, PowerReading


@cache
def opened(key: str) -> Graph:
    """The project as the studio opens it, from its file."""
    return load(amp.project_path(key))


@cache
def results(key: str) -> dict[str, object]:
    return amp.results_by_label(opened(key))


def dbm(key: str, label: str) -> float:
    value = results(key)[label]
    assert isinstance(value, PowerReading)
    return value.power_dbm


def floor_at(key: str, label: str, nm: float) -> float:
    reading = results(key)[label]
    assert isinstance(reading, OpticalSpectrum)
    wavelengths = list(reading.wavelengths_nm)
    nearest = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - nm))
    return float(reading.power_dbm()[nearest])


@pytest.mark.parametrize("key", sorted(amp.EXAMPLES))
def test_every_example_is_the_file_its_script_writes(key: str) -> None:
    build, layout = amp.EXAMPLES[key]
    on_disk = json.loads(amp.project_path(key).read_text(encoding="utf-8"))
    assert {node["id"] for node in on_disk["nodes"]} == set(layout)
    assert on_disk["notes"] == lessons.NOTES[key]
    assert on_disk["annotations"] == lessons.marks(key, layout)
    assert len(on_disk["nodes"]) == len(build().components)


@pytest.mark.parametrize("key", sorted(amp.EXAMPLES))
def test_every_lesson_points_only_at_blocks_on_its_canvas(key: str) -> None:
    named = set(re.findall(r"\[\[([A-Za-z0-9_\-]+)\]\]", lessons.NOTES[key]["body"]))
    assert named, f"{key}'s lesson never points at a block"
    assert named <= set(amp.EXAMPLES[key][1])


def test_3_1_gain_and_the_noise_that_comes_with_it() -> None:
    assert dbm("edfa_basics", "pm_in") == pytest.approx(-20.00, abs=0.005)
    assert dbm("edfa_basics", "pm_out") == pytest.approx(0.65, abs=0.005)
    # The meter counts 0.16 mW of ASE on top of the 1 mW line.
    ase_mw = 10 ** (dbm("edfa_basics", "pm_out") / 10) - 1.0
    assert ase_mw == pytest.approx(0.16, abs=0.005)
    assert floor_at("edfa_basics", "osa_out", 1545.0) == pytest.approx(-32.95, abs=0.005)
    assert results("edfa_basics")["osnr"] == pytest.approx(32.95, abs=0.005)
    assert 58 + amp.EDFA_INPUT_DBM - amp.EDFA_NF_DB == 33


def test_3_2_each_doubling_of_the_spans_costs_three_decibels() -> None:
    quoted = {1: 36.95, 2: 33.94, 4: 30.93, 8: 27.92}
    for spans, osnr in quoted.items():
        assert results("amplified_chain")[f"osnr{spans}"] == pytest.approx(osnr, abs=0.005)
        assert osnr - quoted[1] == pytest.approx(-10 * math.log10(spans), abs=0.02)
    assert floor_at("amplified_chain", "osa1", 1545.0) == pytest.approx(-36.95, abs=0.05)
    assert floor_at("amplified_chain", "osa8", 1545.0) == pytest.approx(-27.92, abs=0.05)


def test_3_5_the_bathtub() -> None:
    quoted = [
        (0.0, -3.0, 20.2, 5.90),
        (3.0, 0.0, 23.1, 9.74),
        (6.0, 3.0, 26.1, 12.24),
        (9.0, 6.0, 29.1, 8.71),
        (12.0, 9.0, 32.1, 4.18),
        (15.0, 12.0, 35.0, 1.67),
    ]
    for (gain, launch, osnr, q), row in zip(quoted, amp.launch_table(), strict=True):
        assert row[0] == gain
        assert row[1] == pytest.approx(launch, abs=0.05)
        assert row[2] == pytest.approx(osnr, abs=0.05)
        assert row[3] == pytest.approx(q, abs=0.005)
    # The project opens at the bottom of the tub.
    ber = results("launch_power")["ber"]
    assert isinstance(ber, EyeMeasurement) and ber.q_factor == pytest.approx(12.24, abs=0.005)


def test_3_5_without_the_kerr_effect_power_only_helps() -> None:
    graph = amp.launch_power()
    by = {c.label: c for c in graph.components}
    for n in range(1, amp.LAUNCH_SPANS + 1):
        setattr(by[f"span{n}"], "nonlinearity", 0.0)  # noqa: B010 -- a Param, not typed
    result = sweep(graph, {(by["booster"], "gain"): amp.BOOSTER_SWEEP})
    qs = [point.runs[0][by["ber"]].q_factor for point in result.points]
    assert qs == sorted(qs)


def test_3_6_zero_dispersion_lets_the_products_grow() -> None:
    dsf = amp.slot_levels(results("fwm_dsf")["osa_dsf"])
    nz = amp.slot_levels(results("fwm_dsf")["osa_nzdsf"])
    assert dsf[-1] == pytest.approx(-24.1, abs=0.05)
    assert dsf[4] == pytest.approx(-23.4, abs=0.05)
    assert nz[-1] == pytest.approx(-63.2, abs=0.05)
    assert nz[4] == pytest.approx(-62.0, abs=0.05)
    assert [round(dsf[i], 1) for i in range(4)] == [-13.4, -13.6, -13.4, -13.7]
    assert all(nz[i] == pytest.approx(-13.0, abs=0.05) for i in range(4))
    assert nz[-1] - dsf[-1] == pytest.approx(-39.0, abs=0.2)


def test_3_9_the_leak_harms_only_the_channel_in_its_place() -> None:
    graph = opened("roadm")
    by = {c.label: c for c in graph.components}
    isolations = [80.0, 35.0, 25.0, 20.0]
    result = sweep(graph, {(by["wss"], "isolation"): isolations})
    added = [point.runs[0][by["ber_add"]].q_factor for point in result.points]
    dropped = [point.runs[0][by["ber_drop"]].q_factor for point in result.points]
    assert added == pytest.approx([41.2, 26.2, 10.8, 6.2], abs=0.05)
    assert dropped == pytest.approx([32.1] * 4, abs=0.05)


# ---------------------------------------------------------------------------
# 3.10


def test_3_10_the_pumped_span() -> None:
    from maiman.components import RamanAmplifiedSpan
    from maiman.units import C_LIGHT

    span = RamanAmplifiedSpan(length=100.0, pump_power=500.0)
    at = {nm: C_LIGHT / (nm * 1e-9) for nm in (1530, 1550, 1565)}
    assert (C_LIGHT / 1450e-9 - at[1550]) / 1e12 == pytest.approx(13.34, abs=0.005)
    assert span.on_off_gain_db(at[1550]) == pytest.approx(13.42, abs=0.005)
    assert span.on_off_gain_db(at[1530]) == pytest.approx(11.38, abs=0.005)
    assert span.on_off_gain_db(at[1565]) == pytest.approx(7.29, abs=0.005)
    assert span.noise_figure_db(at[1550]) == pytest.approx(-1.08, abs=0.005)
    assert amp.raman_makeup_db() == pytest.approx(6.58)
    # Unpumped, it is a span and nothing else.
    dark = RamanAmplifiedSpan(length=100.0, pump_power=0.0)
    assert dark.net_gain_db(at[1550]) == pytest.approx(-20.0)
    assert dark.ase_density(at[1550]) == 0.0


def test_3_10_raman_buys_osnr() -> None:
    got = results("raman")
    assert got["osnr_edfa"] == pytest.approx(32.95, abs=0.005)
    assert got["osnr_raman"] == pytest.approx(38.54, abs=0.005)
    assert got["osnr_raman"] - got["osnr_edfa"] == pytest.approx(5.59, abs=0.01)  # type: ignore[operator]


def test_3_10_try_this() -> None:
    from maiman.components import RamanAmplifiedSpan
    from maiman.units import C_LIGHT

    graph = amp.raman()
    half = {k[0]: v for k, v in graph.run(overrides={("span_raman", "pump_power"): 250.0}).items()}
    assert half["osnr_raman"] == pytest.approx(36.33, abs=0.005)
    weaker = RamanAmplifiedSpan(length=100.0, pump_power=250.0)
    assert weaker.on_off_gain_db(C_LIGHT / 1550e-9) == pytest.approx(6.71, abs=0.005)

    co = {
        k[0]: v for k, v in graph.run(overrides={("span_raman", "counter_pumped"): False}).items()
    }
    assert co["osnr_raman"] == pytest.approx(45.40, abs=0.005)
