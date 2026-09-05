"""The 400G and 800G reference configurations, and the relations behind them.

Two kinds of claim here. The relations in :mod:`maiman.analysis` that convert an
optical SNR into an electrical one and a target error rate into a required OSNR
— those are algebra and are checked as algebra, including against the
single-polarization relation the ASE tests already establish. And the reference
links themselves, which are checked by *measuring* what they need and comparing
it against those relations, which know nothing about the links.

The two shipped project files are checked the only way a project file can be:
loaded back and run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from maiman.analysis import (
    OSNR_REFERENCE_BANDWIDTH,
    osnr_from_snr,
    required_osnr,
    snr_for_ber,
    snr_from_osnr,
)
from maiman.modulation import ber_qam
from maiman.project import load

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

BAUD_400G = 59.84e9


# ---------------------------------------------------------------------------
# the relations


@pytest.mark.parametrize("polarizations", [1, 2])
@pytest.mark.parametrize("symbol_rate", [10e9, 32e9, BAUD_400G, 2 * BAUD_400G])
def test_the_osnr_conversion_inverts_itself(polarizations: int, symbol_rate: float) -> None:
    for osnr_db in (10.0, 20.0, 35.0):
        snr_db = snr_from_osnr(osnr_db, symbol_rate=symbol_rate, polarizations=polarizations)
        back = osnr_from_snr(snr_db, symbol_rate=symbol_rate, polarizations=polarizations)
        assert back == pytest.approx(osnr_db, abs=0.0, rel=1e-12)


def test_the_conversion_is_the_bandwidth_ratio_and_nothing_else() -> None:
    """``SNR = (2/pol) * OSNR * B_ref / R_s``, written out.

    A single-polarization receiver hears all of the signal and only the
    co-polarized half of the ASE, which is the factor of two the ASE tests
    already lean on. A dual-polarization one hears half of each, and those two
    halvings do not cancel, because the reference bandwidth has not halved.
    """
    for symbol_rate in (32e9, BAUD_400G):
        single = snr_from_osnr(20.0, symbol_rate=symbol_rate, polarizations=1)
        dual = snr_from_osnr(20.0, symbol_rate=symbol_rate, polarizations=2)
        assert single - dual == pytest.approx(10.0 * math.log10(2.0), abs=0.0, rel=1e-12)
        assert single == pytest.approx(
            20.0 + 10.0 * math.log10(2.0 * OSNR_REFERENCE_BANDWIDTH / symbol_rate),
            abs=0.0,
            rel=1e-12,
        )
    # Doubling the symbol rate costs 3 dB, whatever else is true.
    assert snr_from_osnr(20.0, symbol_rate=2 * BAUD_400G) - snr_from_osnr(
        20.0, symbol_rate=BAUD_400G
    ) == pytest.approx(-10.0 * math.log10(2.0), abs=0.0, rel=1e-12)


def test_the_conversion_refuses_what_it_cannot_mean() -> None:
    with pytest.raises(ValueError, match="symbol rate must be positive"):
        snr_from_osnr(20.0, symbol_rate=0.0)
    with pytest.raises(ValueError, match="polarizations must be 1 or 2"):
        snr_from_osnr(20.0, symbol_rate=32e9, polarizations=4)


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
@pytest.mark.parametrize("target", [1e-2, 2e-2, 1e-3, 1e-6])
def test_the_snr_inverse_lands_on_the_ber_it_was_given(bits: int, target: float) -> None:
    """Bisection over :func:`ber_qam`, so it has to reproduce it exactly.

    The inverse is bisected rather than derived precisely so that there is only
    one closed form in the project; this is the assertion that says the two are
    the same function seen from two sides.
    """
    snr_db = snr_for_ber(target, bits)
    assert ber_qam(10.0 ** (snr_db / 10.0), bits) == pytest.approx(target, rel=1e-6)


def test_a_denser_format_needs_more_signal_to_noise() -> None:
    """Monotone in the format order, which the bisection would not guarantee."""
    ladder = [snr_for_ber(2e-2, bits) for bits in (2, 4, 6, 8)]
    assert ladder == sorted(ladder)
    # 16-QAM to 64-QAM is the step the 800G choice turns on.
    assert ladder[2] - ladder[1] == pytest.approx(5.7, abs=0.2)


def test_the_ber_target_has_to_be_a_probability() -> None:
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="target BER"):
            snr_for_ber(bad, 4)


def test_required_osnr_is_the_two_relations_composed() -> None:
    """Nothing more, so it cannot disagree with either of them."""
    for bits, rate in ((4, BAUD_400G), (6, 2 * BAUD_400G * 4 / 6)):
        assert required_osnr(2e-2, bits, symbol_rate=rate) == pytest.approx(
            osnr_from_snr(snr_for_ber(2e-2, bits), symbol_rate=rate), abs=0.0, rel=1e-12
        )


def test_the_two_ways_to_reach_800g_cost_what_the_arithmetic_says() -> None:
    """The whole of the 400G-to-800G design choice, in three numbers.

    Doubling the symbol rate is 3 dB and nothing else. Going to 64-QAM instead
    costs 5.7 dB of symbol SNR and hands 1.8 dB back by needing a third less
    baud, for a net 4 dB paid to fit into two thirds of the spectrum.
    """
    four_hundred = required_osnr(2e-2, 4, symbol_rate=BAUD_400G)
    doubled = required_osnr(2e-2, 4, symbol_rate=2 * BAUD_400G)
    denser = required_osnr(2e-2, 6, symbol_rate=2 * BAUD_400G * 4 / 6)

    assert doubled - four_hundred == pytest.approx(10.0 * math.log10(2.0), abs=0.01)
    assert denser - doubled == pytest.approx(3.9, abs=0.2)
    assert denser - doubled == pytest.approx(
        snr_for_ber(2e-2, 6) - snr_for_ber(2e-2, 4) - 10.0 * math.log10(1.5), abs=0.01
    )


# ---------------------------------------------------------------------------
# the links


def test_the_configurations_carry_the_rates_they_claim() -> None:
    """Line rate is baud times bits times two polarizations, and that is all."""
    import reference_rates

    assert reference_rates.BAUD_400G == BAUD_400G
    CONFIGURATIONS = reference_rates.CONFIGURATIONS
    for name, (rate, bits, slot) in CONFIGURATIONS.items():
        line = rate * bits * 2
        payload = 400e9 if name.startswith("400G") else 800e9
        assert line == pytest.approx(payload / 0.8355, rel=0.01), name
        assert line < slot * 12.0, f"{name} would not fit its slot at any sane efficiency"

    # The two 800G rows carry the same payload by different means.
    assert CONFIGURATIONS["800G DP-16QAM"][0] == pytest.approx(2.0 * BAUD_400G)
    assert CONFIGURATIONS["800G DP-64QAM"][0] == pytest.approx(2.0 * BAUD_400G * 4 / 6)


@pytest.mark.parametrize(
    ("name", "bits"), [("400G DP-16QAM", 4), ("800G DP-16QAM", 4), ("800G DP-64QAM", 6)]
)
def test_each_reference_needs_the_osnr_the_relation_predicts(name: str, bits: int) -> None:
    """Measured against predicted, with the blind equaliser out of the way.

    The measurement bisects an attenuator until the *counted* bit error rate sits
    on the threshold — counted rather than estimated, because an estimate derived
    from the measured SNR through the same closed form would be comparing the
    relation with itself.

    What is left over is the transmitter's implementation penalty, and it grows
    with the format order because a denser constellation is less forgiving of the
    modulator's curvature. Half a decibel at 16-QAM, most of one at 64-QAM.
    """
    from reference_rates import CONFIGURATIONS, THRESHOLD, measured_required_osnr

    rate, _, _ = CONFIGURATIONS[name]
    measured = measured_required_osnr(rate, bits, equalize=False)
    predicted = required_osnr(THRESHOLD, bits, symbol_rate=rate)

    assert measured > predicted, "a real link cannot beat the optical limit"
    assert measured - predicted < 1.2, f"{name}: penalty {measured - predicted:.2f} dB"


def test_the_blind_equaliser_costs_nothing_at_16_qam_and_diverges_at_64() -> None:
    """A finding, pinned so that fixing it will show up as this test failing.

    Nothing rotates the polarization on this bench, so an ideal separator is the
    identity and the equaliser should be free. At 16-QAM it is, to within half a
    decibel. At 64-QAM it costs eleven, because nine constellation radii sit
    close enough together that a noisy sample snaps to the wrong one and the
    correction that follows is large and in the wrong direction.

    The structure is not what is wrong — a single tap recovers the whole of it,
    and so nearly does a step ten times smaller. A normalised update is the
    textbook fix and moves every other format's answer with it, which is why it
    is not folded in here.
    """
    from reference_rates import CONFIGURATIONS, measured_required_osnr

    for name, bits in (("400G DP-16QAM", 4), ("800G DP-16QAM", 4)):
        rate, _, _ = CONFIGURATIONS[name]
        clean = measured_required_osnr(rate, bits, equalize=False)
        blind = measured_required_osnr(rate, bits, equalize=True)
        assert blind - clean < 0.6, f"{name}: {blind - clean:.2f} dB"

    rate, _, _ = CONFIGURATIONS["800G DP-64QAM"]
    clean = measured_required_osnr(rate, 6, equalize=False)
    blind = measured_required_osnr(rate, 6, equalize=True)
    assert blind - clean > 5.0, f"64-QAM penalty has changed: {blind - clean:.2f} dB"


# ---------------------------------------------------------------------------
# the shipped projects


@pytest.mark.parametrize("filename", ["zr400.maiman", "zr800.maiman"])
def test_the_shipped_reference_projects_load_and_run(filename: str) -> None:
    """The only check a project file can be given that means anything."""
    graph = load(EXAMPLES / filename)
    assert len(graph.components) == 25

    results = graph.run()
    measurements = [
        results.port(component, "out")
        for component in graph.components
        if component.label.startswith("vsa")
    ]
    assert len(measurements) == 2
    for measurement in measurements:
        assert measurement.symbols_evaluated > 0
        assert 0.0 <= measurement.evm < 1.0


@pytest.mark.parametrize("filename", ["zr400.maiman", "zr800.maiman"])
def test_the_shipped_projects_carry_positions_for_every_block(filename: str) -> None:
    """A project the studio opens stacked on the origin is a project nobody reads."""
    document = json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))
    placed = {node["id"]: node.get("ui") for node in document["nodes"]}
    assert placed and all(position is not None for position in placed.values()), placed

    corners = {(position["x"], position["y"]) for position in placed.values()}
    assert len(corners) == len(placed), "two blocks share a position"


def test_the_two_shipped_projects_differ_only_in_their_symbol_rate() -> None:
    """800G here is 400G at twice the baud, and the files should say so."""
    four, eight = (
        json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
        for name in ("zr400.maiman", "zr800.maiman")
    )
    assert eight["context"]["bit_rate"] == pytest.approx(2.0 * four["context"]["bit_rate"])
    assert {node["id"] for node in eight["nodes"]} == {node["id"] for node in four["nodes"]}
    assert eight["edges"] == four["edges"]
