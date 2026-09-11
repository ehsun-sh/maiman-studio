"""Receiver DSP in the link: what carrier recovery is actually for.

The claim these tests exist to check is not "the block runs" but a physical one:
**with an uncompensated carrier, launching more power stops helping.** Laser
phase noise is a random walk, so it is not removable by subtracting a constant or
a line, and it puts a ceiling on the measured SNR that no power budget lifts.
Carrier recovery removes the ceiling. Both halves are asserted.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    FrequencyRecovery,
    IQDriver,
    IQModulator,
    IQSampler,
    PilotInserter,
    PRBSGenerator,
    QAMMapper,
    TimingRecovery,
    Waveguide,
)
from maiman.dsp import derotate, estimate_carrier_offset, estimate_timing, resample_to_instant
from maiman.modulation import qam_constellation, rotational_symmetry
from maiman.signals import ConstellationMeasurement, SymbolSignal
from maiman.units import C_LIGHT

#: Enough symbols for the walk to actually wander, and a window edge to discard.
SEQUENCE = 4096
EDGES = 64.0


def build(
    *,
    bits_per_symbol: int = 4,
    linewidth: float = 100.0,
    launch: float = -10.0,
    recovery: bool = True,
    seed: int = 31,
    **recovery_params: float,
) -> tuple[Graph, ConstellationAnalyzer]:
    ctx = SimulationContext(
        bit_rate=32e9,
        samples_per_symbol=8,
        sequence_length=SEQUENCE,
        seed=seed,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(bits_per_symbol), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(bits_per_symbol), label="map"))
    driver = graph.add(IQDriver(label="drv"))
    tx = graph.add(CWLaser(power=launch, linewidth=linewidth, label="tx"))
    modulator = graph.add(IQModulator(label="mod"))
    lo = graph.add(CWLaser(power=10.0, linewidth=linewidth, label="lo"))
    receiver = graph.add(CoherentReceiver(label="rx"))
    sampler = graph.add(IQSampler(label="smp"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=EDGES, label="vsa"))

    graph.chain(prbs, mapper, driver)
    graph.connect(tx, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])

    if recovery:
        stage = graph.add(CarrierRecovery(label="cr", **recovery_params))
        graph.connect(sampler["out"], stage["in"])
        graph.connect(stage["out"], analyzer["in"])
    else:
        graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer


def measure(**kwargs: Any) -> ConstellationMeasurement:
    graph, analyzer = build(**kwargs)
    return cast(ConstellationMeasurement, graph.run(keep=[])[analyzer])


# --------------------------------------------------------------------------
# The floor, and its removal
# --------------------------------------------------------------------------


def test_without_recovery_more_power_stops_buying_snr() -> None:
    """The ceiling. 12 dB of extra launch power buys barely one.

    This is the test that makes the next one mean something: if power still
    worked, carrier recovery would be solving a problem the model does not have.
    """
    quiet = measure(recovery=False, launch=-14.0)
    loud = measure(recovery=False, launch=-2.0)

    assert loud.snr_db - quiet.snr_db < 2.0
    assert loud.ber_estimated > 1e-4, "the link should still be failing at any power"


def test_with_recovery_power_works_again() -> None:
    """The same 12 dB now buys most of 12 dB, and the link closes."""
    quiet = measure(recovery=True, launch=-14.0)
    loud = measure(recovery=True, launch=-2.0)

    assert loud.snr_db - quiet.snr_db > 7.0
    assert loud.ber_estimated < 1e-12


@pytest.mark.parametrize("launch", [-14.0, -10.0, -6.0])
def test_recovery_is_worth_several_decibels_at_every_power(launch: float) -> None:
    without = measure(recovery=False, launch=launch)
    with_it = measure(recovery=True, launch=launch)
    assert with_it.snr_db > without.snr_db + 5.0


def test_a_wider_linewidth_costs_more_without_recovery_than_with() -> None:
    """The impairment being removed is specifically the laser's, not the shot noise."""
    narrow_off = measure(recovery=False, linewidth=1.0)
    wide_off = measure(recovery=False, linewidth=100.0)
    narrow_on = measure(recovery=True, linewidth=1.0)
    wide_on = measure(recovery=True, linewidth=100.0)

    assert narrow_off.snr_db - wide_off.snr_db > 5.0
    assert narrow_on.snr_db - wide_on.snr_db < 3.0


def test_recovery_does_not_damage_an_already_clean_signal() -> None:
    """A correction stage that hurt when there was nothing to correct would be worse
    than no stage at all, because it would be paid for on every link."""
    without = measure(recovery=False, linewidth=0.0, launch=-6.0)
    with_it = measure(recovery=True, linewidth=0.0, launch=-6.0)
    assert with_it.snr_db > without.snr_db - 0.5


# --------------------------------------------------------------------------
# Shape and wiring
# --------------------------------------------------------------------------


def test_recovery_preserves_the_sequence_and_the_alphabet() -> None:
    graph, _ = build()
    sampler = next(c for c in graph.components if isinstance(c, IQSampler))
    stage = next(c for c in graph.components if isinstance(c, CarrierRecovery))
    results = graph.run(keep=[sampler, stage])

    before = results.port(sampler, "out")
    after = results.port(stage, "out")

    assert after.num_symbols == before.num_symbols
    assert after.symbol_rate == before.symbol_rate
    assert np.array_equal(np.asarray(after.constellation), np.asarray(before.constellation))


def test_recovery_only_rotates_and_never_rescales() -> None:
    """Gain belongs to the AGC. A phase stage that also touched amplitude would be
    quietly doing two jobs, and only one of them would be tested."""
    graph, _ = build()
    sampler = next(c for c in graph.components if isinstance(c, IQSampler))
    stage = next(c for c in graph.components if isinstance(c, CarrierRecovery))
    results = graph.run(keep=[sampler, stage])

    before = np.abs(np.asarray(results.port(sampler, "out").symbols))
    after = np.abs(np.asarray(results.port(stage, "out").symbols))
    assert np.allclose(before, after, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_recovery_works_for_every_square_format(bits_per_symbol: int) -> None:
    without = measure(bits_per_symbol=bits_per_symbol, recovery=False, launch=-6.0)
    with_it = measure(bits_per_symbol=bits_per_symbol, recovery=True, launch=-6.0)
    assert with_it.snr_db > without.snr_db + 5.0


# ---------------------------------------------------------------------------
# timing recovery


def shaped_baseband(
    symbols: int = 512, sps: int = 16, roll_off: float = 0.3, seed: int = 5
) -> np.ndarray:
    """A pulse-shaped 16-QAM waveform, with the excess bandwidth the estimator needs."""
    rng = np.random.default_rng(seed)
    alphabet = (rng.integers(0, 4, symbols) * 2 - 3) + 1j * (rng.integers(0, 4, symbols) * 2 - 3)
    upsampled = np.zeros(symbols * sps, dtype=np.complex128)
    upsampled[::sps] = alphabet
    span = np.arange(-8 * sps, 8 * sps) / sps + 1e-9
    pulse = np.sinc(span) * np.cos(np.pi * roll_off * span) / (1 - (2 * roll_off * span) ** 2)
    return np.convolve(upsampled, pulse, mode="same")


@pytest.mark.parametrize("delay", [0.05, 0.1, 0.25, 0.4, -0.3, -0.45])
def test_the_timing_estimate_follows_a_known_delay(delay: float) -> None:
    """One for one, which is what makes it an estimator rather than an indicator.

    A detector that merely moved in the right direction would pass a sign test
    and be useless for correcting anything.
    """
    sample_rate, symbol_rate = 16.0, 1.0
    baseband = shaped_baseband()

    reference = estimate_timing(baseband, sample_rate, symbol_rate=symbol_rate)
    moved = resample_to_instant(baseband, sample_rate, delay=delay / symbol_rate)
    found = estimate_timing(moved, sample_rate, symbol_rate=symbol_rate)

    assert (found - reference) % 1.0 == pytest.approx(delay % 1.0, abs=2e-3)


def test_estimating_then_undoing_lands_on_zero() -> None:
    """The whole loop, in three lines, on a delay it was not told about."""
    sample_rate, symbol_rate = 16.0, 1.0
    skewed = resample_to_instant(shaped_baseband(), sample_rate, delay=0.37 / symbol_rate)

    found = estimate_timing(skewed, sample_rate, symbol_rate=symbol_rate)
    corrected = resample_to_instant(skewed, sample_rate, delay=-found / symbol_rate)

    residual = estimate_timing(corrected, sample_rate, symbol_rate=symbol_rate)
    assert min(residual, 1.0 - residual) == pytest.approx(0.0, abs=1e-6)


def test_the_fractional_delay_is_exact_and_reversible() -> None:
    """A phase ramp, not an interpolation: forward then back is the input again.

    Worth asserting because the alternative implementations — a polyphase filter,
    a cubic interpolator — are none of them exact, and swapping one in later
    would fail here rather than showing up as a slightly worse EVM nobody traces.
    """
    baseband = shaped_baseband(symbols=64)
    there = resample_to_instant(baseband, 16.0, delay=0.31)
    back = resample_to_instant(there, 16.0, delay=-0.31)
    assert np.allclose(back, baseband, atol=1e-12)


def test_a_whole_symbol_of_delay_is_invisible_to_it() -> None:
    """Not a limitation of this implementation — a property of ``|A|**2``.

    Delay by exactly one symbol period and the intensity waveform is identical,
    so the line it reads has the same phase. What that costs is not a worse
    decision but the sequence read off by one place, which is framing.
    """
    sample_rate, symbol_rate = 16.0, 1.0
    baseband = shaped_baseband()
    slipped = resample_to_instant(baseband, sample_rate, delay=1.0 / symbol_rate)

    assert estimate_timing(slipped, sample_rate, symbol_rate=symbol_rate) == pytest.approx(
        estimate_timing(baseband, sample_rate, symbol_rate=symbol_rate), abs=1e-9
    )


def timed_link(waveguide_um: float, *, recover: bool) -> tuple[Graph, Any, Any]:
    """A coherent link with an optional waveguide delay and an optional recovery stage."""
    ctx = SimulationContext(
        bit_rate=32e9, samples_per_symbol=16, sequence_length=2048, seed=5, precision="double"
    )
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, label="map"))
    driver = graph.add(
        IQDriver(v_pi=4.0, predistort=True, pulse_shaping=True, roll_off=0.2, label="drv")
    )
    laser = graph.add(CWLaser(power=2.0, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=4.0, label="mod"))
    lo = graph.add(CWLaser(power=10.0, wavelength=1550.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2, label="smp"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=EDGES, label="vsa"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])

    node: Any = modulator
    if waveguide_um:
        guide = graph.add(Waveguide(length=waveguide_um, propagation_loss=0.0, label="wg"))
        graph.connect(modulator, guide["in"])
        node = guide
    graph.connect(node, receiver["in"])
    graph.connect(lo, receiver["lo"])

    timing = None
    if recover:
        timing = graph.add(TimingRecovery(label="tr"))
        graph.connect(receiver["i"], timing["i"])
        graph.connect(receiver["q"], timing["q"])
        graph.connect(timing["i"], sampler["i"])
        graph.connect(timing["q"], sampler["q"])
    else:
        graph.connect(receiver["i"], sampler["i"])
        graph.connect(receiver["q"], sampler["q"])

    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, timing


@pytest.mark.parametrize(("waveguide_um", "delay_ps"), [(250.0, 3.50), (500.0, 7.00)])
def test_a_waveguide_breaks_the_link_and_timing_recovery_returns_it(
    waveguide_um: float, delay_ps: float
) -> None:
    """The impairment is not invented: it is one shipped block in the optical path.

    A silicon waveguide at a group index of 4.2 holds 14 ps per millimetre. At
    32 GBd a symbol is 31.2 ps, so a quarter of a millimetre is a ninth of a
    symbol — and the sampler, which takes a fixed column out of every symbol,
    has no way to know. Half a millimetre costs 675 symbol errors in 1920.

    With the stage in front of it the same link is back to its undelayed EVM and
    counts none.
    """
    baseline, analyzer, _ = timed_link(0.0, recover=False)
    clean = baseline.run()[analyzer]
    assert clean.symbol_errors == 0, "the undelayed link must be error free to compare against"

    broken, analyzer, _ = timed_link(waveguide_um, recover=False)
    without = broken.run()[analyzer]
    assert without.symbol_errors > 0, f"{delay_ps} ps of delay should have cost something"
    assert without.evm > 2.5 * clean.evm

    fixed, analyzer, timing = timed_link(waveguide_um, recover=True)
    results = fixed.run(keep=[timing])
    with_recovery = results[analyzer]
    estimate = results.port(timing, "diagnostics")

    assert with_recovery.symbol_errors == 0
    assert with_recovery.evm == pytest.approx(clean.evm, rel=0.05)
    # And it moved by what the waveguide actually holds, not by whatever
    # happened to help — the delay is n_g L / c and nothing here was told it.
    assert -estimate.applied * 1e12 == pytest.approx(delay_ps, abs=0.1)


def test_timing_recovery_does_nothing_when_there_is_nothing_to_do() -> None:
    """A block that quietly moved a correctly timed signal would be worse than absent."""
    graph, analyzer, timing = timed_link(0.0, recover=True)
    results = graph.run(keep=[timing])
    estimate = results.port(timing, "diagnostics")

    assert abs(estimate.applied) * 1e12 < 0.05, "it moved a signal that was already aligned"
    assert results[analyzer].symbol_errors == 0


def test_past_half_a_symbol_the_frame_slips_and_that_is_a_different_problem() -> None:
    """Half a symbol is the cliff, and the failure past it is framing, not timing.

    At 1115 um the waveguide holds 15.6 ps against a 31.2 ps symbol — exactly
    half. Both directions are the same distance, so whichever it takes lands the
    instant correctly and the *sequence* one place out. The constellation is
    fine and the errors are total, which is the signature of a slip rather than
    of bad timing, and no estimator reading ``|A|**2`` can do better: it cannot
    see a whole symbol at all.
    """
    graph, analyzer, timing = timed_link(1115.0, recover=True)
    results = graph.run(keep=[timing])
    measurement = results[analyzer]
    estimate = results.port(timing, "diagnostics")

    assert abs(estimate.applied) * 1e12 == pytest.approx(15.6, abs=0.2), (
        "it should still find half a symbol; what it cannot do is choose a side"
    )
    assert measurement.symbol_errors > measurement.symbols_evaluated // 2, (
        "a slipped frame is nearly all wrong, which is what distinguishes it from a timing error"
    )


def test_the_estimate_carries_the_strength_of_the_line_it_came_from() -> None:
    """Because a confident number from a line that is not there is the trap.

    A signal shaped to exactly the Nyquist bandwidth carries no symbol-rate
    line, and a heavily dispersed one nearly none. The phase of nothing is a
    number like any other; the magnitude beside it is what says whether to
    believe it.
    """
    graph, _analyzer, timing = timed_link(500.0, recover=True)
    estimate = graph.run(keep=[timing]).port(timing, "diagnostics")
    assert estimate.strength > 0.0
    assert 0.0 <= estimate.fraction < 1.0


# ---------------------------------------------------------------------------
# carrier frequency offset


def analyzer_of(graph: Graph) -> Any:
    return next(c for c in graph.components if isinstance(c, ConstellationAnalyzer))


def detuned_link(
    offset: float, *, bits_per_symbol: int = 4, recover: bool, launch: float = -18.0
) -> tuple[Graph, Any, Any]:
    """A coherent link whose LO is detuned from the transmitter by ``offset`` Hz.

    Not an injected impairment: the LO is an ordinary :class:`CWLaser` tuned to a
    different wavelength, and the beat comes out of the receiver's own mixing.
    Which is the point — if the offset had to be synthesised, the block removing
    it would be answering a question nothing asks.
    """
    ctx = SimulationContext(
        bit_rate=32e9, samples_per_symbol=4, sequence_length=1920, seed=2026, precision="double"
    )
    graph = Graph(ctx)
    bps = float(bits_per_symbol)
    prbs = graph.add(PRBSGenerator(order=23.0, bits_per_symbol=bps, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=bps, label="map"))
    driver = graph.add(IQDriver(v_pi=4.0, predistort=True, label="drv"))
    laser = graph.add(CWLaser(power=launch, wavelength=1550.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=4.0, label="mod"))
    # Tuned below the signal by the offset, so the beat comes out positive.
    detuned_nm = C_LIGHT / (C_LIGHT / 1550e-9 - offset) * 1e9
    lo = graph.add(CWLaser(power=10.0, wavelength=detuned_nm, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    sampler = graph.add(IQSampler(label="smp"))
    # Blind here, or the analyser's own data-aided removal would hide the very
    # thing under test — it holds the transmitted sequence and this block does not.
    analyzer = graph.add(ConstellationAnalyzer(remove_frequency_offset=False, label="vsa"))

    graph.chain(prbs, mapper, driver)
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], sampler["i"])
    graph.connect(receiver["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])

    recovery = None
    if recover:
        recovery = graph.add(FrequencyRecovery(label="fo"))
        graph.connect(sampler["out"], recovery["in"])
        graph.connect(recovery["out"], analyzer["in"])
    else:
        graph.connect(sampler["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, recovery


def modulated(offset: float, symbol_rate: float, count: int = 2048, bits: int = 4) -> np.ndarray:
    """A clean QAM sequence spun by a known offset, with no channel in the way."""
    rng = np.random.default_rng(7)
    points = qam_constellation(bits)
    symbols = points[rng.integers(0, points.size, count)]
    index = np.arange(count, dtype=np.float64)
    return symbols * np.exp(2j * np.pi * offset * index / symbol_rate)


@pytest.mark.parametrize("offset", [0.0, 137e6, -137e6, 1e9, -1e9, 3.9e9, -3.9e9])
def test_the_offset_estimate_lands_on_the_offset(offset: float) -> None:
    """Across the whole unambiguous range and both signs, to well under a MHz.

    A sign error here is not a small error: it doubles the residual rather than
    removing it, so both directions are asserted rather than one and a shrug.
    """
    symbol_rate = 32e9
    found, confidence = estimate_carrier_offset(
        modulated(offset, symbol_rate), symbol_rate, symmetry=4
    )
    assert found == pytest.approx(offset, abs=0.5e6)
    assert confidence > 10.0


def test_past_half_the_stripped_bandwidth_it_aliases_rather_than_degrades() -> None:
    """The one failure mode worth knowing, because it does not look like failure.

    The tone sits at ``M`` times the offset, so ``symbol_rate / (2 * M)`` is where
    it wraps. Past that the estimate is not noisy — it is confidently wrong by
    exactly ``symbol_rate / M``, with the same confidence as a correct one. That
    is why a real receiver sweeps its LO to acquire, and why the docstring says so.
    """
    symbol_rate = 32e9
    beyond = symbol_rate / 8.0 + 100e6  # M = 4 for square QAM

    found, confidence = estimate_carrier_offset(
        modulated(beyond, symbol_rate), symbol_rate, symmetry=4
    )
    assert found == pytest.approx(beyond - symbol_rate / 4.0, abs=0.5e6)
    assert confidence > 10.0, "and it is confident about it, which is the trap"


def test_an_alphabet_with_no_symmetry_is_refused() -> None:
    """There is no power that strips it, so there is no estimate to return.

    Returning an argmax over noise with a low confidence beside it would be the
    other option; refusing is better, because a caller that ignores confidence is
    the normal case and a caller that ignores an exception is not.
    """
    with pytest.raises(ValueError, match="half-turn"):
        estimate_carrier_offset(modulated(0.0, 32e9), 32e9, symmetry=1)


def test_derotating_by_what_was_found_is_reversible() -> None:
    """Forward then back is the input again — a phasor, not a filter."""
    symbols = modulated(0.0, 32e9, count=256)
    there = derotate(symbols, 32e9, offset=411e6)
    back = derotate(there, 32e9, offset=-411e6)
    assert np.allclose(back, symbols, atol=1e-12)


def test_the_symmetry_used_is_the_one_the_geometry_reports() -> None:
    """Not four by assumption: a rectangular alphabet only has a half turn.

    8-QAM here is 4x2, which a quarter turn maps onto a 2x4 grid — a different
    alphabet. Raising it to the fourth power would leave the data in.
    """
    assert rotational_symmetry(qam_constellation(3)) == 2
    assert rotational_symmetry(qam_constellation(4)) == 4

    symbol_rate = 32e9
    found, _ = estimate_carrier_offset(
        modulated(500e6, symbol_rate, bits=3), symbol_rate, symmetry=2
    )
    assert found == pytest.approx(500e6, abs=0.5e6)


def test_confidence_collapses_when_there_is_no_line() -> None:
    """Circular noise has no tone at any power of itself, and says so.

    The number this guards is the one a user would otherwise read as an answer:
    the argmax exists either way, and only the ratio beside it distinguishes a
    measurement from an accident.
    """
    rng = np.random.default_rng(11)
    noise = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    _, confidence = estimate_carrier_offset(noise, 32e9, symmetry=4)

    _, real_line = estimate_carrier_offset(modulated(200e6, 32e9), 32e9, symmetry=4)
    assert confidence < real_line / 5.0


@pytest.mark.parametrize("offset", [10e6, 100e6, 1e9])
def test_the_block_takes_a_detuned_link_back_to_zero_errors(offset: float) -> None:
    """The physical claim, end to end, on a link nothing else in the chain fixes.

    10 MHz is three hundredths of one per cent of the symbol rate and is enough
    to destroy the link, which is the whole reason this block exists: the offset
    a real LO carries is orders of magnitude larger than the offset the link
    survives.
    """
    broken = detuned_link(offset, recover=False)[0]
    fixed_graph, analyzer, recovery = detuned_link(offset, recover=True)

    without = broken.run()[analyzer_of(broken)]
    results = fixed_graph.run(keep=[recovery])
    with_block = results[analyzer]
    estimate = results.port(recovery, "diagnostics")

    assert without.symbol_errors > without.symbols_evaluated // 2
    assert with_block.symbol_errors == 0
    assert estimate.offset == pytest.approx(offset, abs=0.5e6)
    assert estimate.symmetry == 4


def test_it_does_nothing_when_there_is_nothing_to_do() -> None:
    """A block that quietly costs something when idle is a block nobody leaves in."""
    aligned, analyzer, recovery = detuned_link(0.0, recover=True)
    baseline, plain_analyzer, _ = detuned_link(0.0, recover=False)

    results = aligned.run(keep=[recovery])
    estimate = results.port(recovery, "diagnostics")

    assert estimate.offset == pytest.approx(0.0, abs=0.5e6)
    assert results[analyzer].evm == pytest.approx(baseline.run()[plain_analyzer].evm, rel=1e-9)


def test_a_rectangular_format_is_recovered_too() -> None:
    """8-QAM has half the symmetry, so half the range and the same outcome."""
    graph, analyzer, recovery = detuned_link(1e9, bits_per_symbol=3, recover=True)
    results = graph.run(keep=[recovery])
    estimate = results.port(recovery, "diagnostics")

    assert estimate.symmetry == 2
    assert estimate.offset == pytest.approx(1e9, abs=0.5e6)
    assert results[analyzer].symbol_errors == 0


def test_an_empty_sequence_is_answered_rather_than_divided_by() -> None:
    empty = np.array([], dtype=np.complex128)
    assert estimate_carrier_offset(empty, 32e9, symmetry=4) == (0.0, 0.0)
    assert derotate(empty, 32e9, offset=1e9).size == 0


def shipped_variant(*, timing: bool, frequency: bool, detuned: bool = True) -> Any:
    """The shipped coherent project, with either front-end stage optionally taken out.

    Edited as a project document rather than rebuilt, so what is measured is the
    graph the studio actually opens and not a copy of it that could drift.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "examples"))
    from export_ui_data import build

    from maiman.project import graph_from_dict, graph_to_dict

    document = graph_to_dict(build(sequence_length=4096))
    if not detuned:
        for node in document["nodes"]:
            if node["id"] == "lo":
                node["params"]["wavelength"] = 1550.0

    def bypass(label: str, pairs: list[tuple[str, str]]) -> None:
        document["nodes"] = [n for n in document["nodes"] if n["id"] != label]
        upstream: dict[str, Any] = {}
        downstream: dict[str, Any] = {}
        kept = []
        for edge in document["edges"]:
            if edge["to"][0] == label:
                upstream[edge["to"][1]] = edge["from"]
            elif edge["from"][0] == label:
                downstream[edge["from"][1]] = edge["to"]
            else:
                kept.append(edge)
        for source, sink in pairs:
            kept.append({"from": upstream[source], "to": downstream[sink]})
        document["edges"] = kept

    if not timing:
        bypass("tr", [("i", "i"), ("q", "q")])
    if not frequency:
        bypass("fo", [("in", "out")])

    graph = graph_from_dict(document)
    results = graph.run()
    # One analyser reports both numbers since the quadrant stopped being
    # resolved by a decision device.
    soft = next(c for c in graph.components if c.label == "vsa")
    counted = soft
    return results[soft], results[counted]


def test_the_reference_design_needs_both_front_end_stages() -> None:
    """The shipped link's LO is 200 MHz off, and both stages earn that back.

    Guards the table in the README, and guards it on the *shipped* graph rather
    than a test fixture — the claim is about what a user opens, so measuring
    anything else would be measuring the wrong thing.

    Also pins the order of magnitude of what each is worth, which is the part a
    refactor can quietly lose: frequency recovery is the difference between a
    link and no link, and timing recovery is worth several points of EVM but
    only once the offset is gone.
    """
    _, broken_errors = shipped_variant(timing=False, frequency=False)
    assert broken_errors.symbol_errors > broken_errors.symbols_evaluated // 2, (
        "200 MHz on a 32 GBd link should destroy it; if this passes the LO is no longer detuned"
    )

    _, timing_errors = shipped_variant(timing=True, frequency=False)
    assert timing_errors.symbol_errors > timing_errors.symbols_evaluated // 2, (
        "a spinning constellation is not a timing problem, and timing alone must not fix it"
    )

    frequency_only, frequency_errors = shipped_variant(timing=False, frequency=True)
    # Essentially clean rather than exactly clean. The pilots are the alphabet's
    # *outermost* points, which is what makes them good for a phase estimate and
    # also what makes them the first to fall over when the sampling instant is
    # wrong -- so a mistimed link now leaves a couple of errors here where the
    # differentially-coded one left none. A tenth of a symbol error per thousand
    # does not affect the claim, and pinning it at exactly zero was brittle.
    assert frequency_errors.symbol_errors < frequency_errors.symbols_evaluated // 1000
    assert 0.11 < frequency_only.evm < 0.15, "13 % EVM: the offset is gone, the instant is not"

    both, both_errors = shipped_variant(timing=True, frequency=True)
    assert both_errors.symbol_errors == 0
    assert both.evm < frequency_only.evm / 1.5, "the timing stage is worth most of that 13 %"

    ideal, _ = shipped_variant(timing=True, frequency=True, detuned=False)
    assert both.evm == pytest.approx(ideal.evm, abs=0.002), (
        "the two stages together should return the detuned link to the co-tuned one"
    )


# ---------------------------------------------------------------------------
# pilot symbols


def piloted(spacing: float = 64.0, turns: int = 0, seed: int = 12) -> tuple[Any, Any, Any]:
    """A mapper, a pilot inserter and a recovery stage, with a rotation injected.

    No channel: the point under test is whether a known sequence resolves a
    constant rotation, and a fiber in the way would only make that harder to see.
    """
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=4, sequence_length=4096, seed=seed)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, differential=False, label="map"))
    pilots = graph.add(PilotInserter(spacing=spacing, label="pil"))
    graph.chain(prbs, mapper, pilots)
    return graph, mapper, pilots


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_pilots_resolve_every_quarter_turn_identically(turns: int) -> None:
    """The whole job. A blind stage lands on one of four answers; this picks it.

    Identically is the assertion that matters: if one rotation came back better
    than another, the block would be resolving *some* of the ambiguity and the
    link would work three times in four.
    """
    from maiman.components.mapping import PilotPhaseRecovery as _Recovery

    graph, _mapper, pilots = piloted()
    transmitted = graph.run(keep=[pilots])[pilots]
    symbols = np.asarray(transmitted.symbols)
    constellation = np.asarray(transmitted.constellation)

    rotated = symbols * np.exp(1j * np.pi / 2.0 * turns)
    reference = SymbolSignal(
        symbols=symbols, symbol_rate=transmitted.symbol_rate, constellation=constellation
    )
    received = SymbolSignal(
        symbols=rotated, symbol_rate=transmitted.symbol_rate, constellation=constellation
    )

    block = _Recovery(spacing=64.0, label="pqr")
    out = block.run(graph.ctx, {"in": received, "reference": reference})
    recovered = np.asarray(out["out"].symbols)
    estimate = out["diagnostics"]

    assert estimate.quarter_turns == turns
    assert estimate.residual < 1e-6, "with no noise there is nothing but the ambiguity"
    assert np.allclose(recovered, symbols, atol=1e-9)


def test_the_pilots_are_constellation_points_and_are_not_all_the_same() -> None:
    """Two failure modes that both look fine until they are measured.

    An off-grid pilot — unit-power QPSK inside a 16-QAM link, say — sits exactly
    between four legal points, so every pilot is counted as a symbol error and
    the error rate measures the pilots rather than the channel. And a *constant*
    pilot puts a line in the spectrum at the pilot rate, which is a real
    impairment on a real link and an invisible one here.
    """
    graph, mapper, pilots = piloted(spacing=32.0)
    results = graph.run(keep=[mapper, pilots])
    plain = np.asarray(results[mapper].symbols)
    with_pilots = np.asarray(results[pilots].symbols)
    constellation = np.asarray(results[pilots].constellation)

    positions = np.arange(0, with_pilots.size, 32)
    inserted = with_pilots[positions]

    distance = np.abs(inserted[:, None] - constellation[None, :]).min(axis=1)
    assert np.allclose(distance, 0.0, atol=1e-12), "a pilot must be a legal symbol"

    assert len(set(np.round(inserted, 9))) > 1, "a constant pilot is a spectral line"
    # Drawn from the outermost ring, which is what a phase estimate is made of.
    assert np.allclose(np.abs(inserted), np.abs(constellation).max(), atol=1e-12)

    # Everything else is untouched.
    mask = np.ones(with_pilots.size, dtype=bool)
    mask[positions] = False
    assert np.array_equal(with_pilots[mask], plain[mask])


def test_the_recovery_reads_the_reference_only_where_the_pilots_are() -> None:
    """Otherwise it is a data-aided estimator wearing a pilot block's name.

    It is handed the whole transmitted sequence, the way ``IQSampler`` is, and
    nothing in the type system stops it using all of it — which would make it far
    better than any receiver could be and would look entirely normal. So the
    reference is corrupted everywhere *except* the pilot positions and the answer
    has to be unchanged.
    """
    from maiman.components.mapping import PilotPhaseRecovery as _Recovery

    graph, _mapper, pilots = piloted()
    transmitted = graph.run(keep=[pilots])[pilots]
    symbols = np.asarray(transmitted.symbols)
    constellation = np.asarray(transmitted.constellation)
    rate = transmitted.symbol_rate

    rng = np.random.default_rng(2)
    received = SymbolSignal(
        symbols=symbols * np.exp(1j * np.pi / 2.0),
        symbol_rate=rate,
        constellation=constellation,
    )

    honest = SymbolSignal(symbols=symbols, symbol_rate=rate, constellation=constellation)
    corrupted = symbols.copy()
    mask = np.ones(symbols.size, dtype=bool)
    mask[np.arange(0, symbols.size, 64)] = False
    corrupted[mask] = constellation[rng.integers(0, constellation.size, int(mask.sum()))]
    lying = SymbolSignal(symbols=corrupted, symbol_rate=rate, constellation=constellation)

    block = _Recovery(spacing=64.0, label="pqr")
    clean = block.run(graph.ctx, {"in": received, "reference": honest})["diagnostics"]
    messy = block.run(graph.ctx, {"in": received, "reference": lying})["diagnostics"]

    assert messy.rotation == pytest.approx(clean.rotation, abs=1e-12)
    assert messy.quarter_turns == clean.quarter_turns


def test_soft_information_survives_the_flagship_now() -> None:
    """The reason the flagship stopped using differential coding.

    ``DifferentialDecoder`` resolves the same ambiguity by slicing, and a slice
    puts every symbol exactly on a constellation point — measured on the shipped
    graph before this change, the distance to the nearest point was exactly zero
    and a demapper returned log-likelihood ratios of order 1e29. Pilot recovery
    moves every symbol by one constant angle and decides nothing, so what leaves
    it is still a measurement.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "examples"))
    from export_ui_data import build

    from maiman.modulation import soft_demap

    graph = build(sequence_length=4096)
    quadrant = next(c for c in graph.components if c.label == "pqr")
    output = graph.run(keep=[quadrant]).port(quadrant, "out")

    symbols = np.asarray(output.symbols)
    constellation = np.asarray(output.constellation)
    distance = np.abs(symbols[:, None] - constellation[None, :]).min(axis=1)
    assert distance.mean() > 1e-3, "the output is decisions, not measurements"

    llr = soft_demap(symbols, constellation, 4)
    assert np.all(np.isfinite(llr))
    assert float(np.std(np.abs(llr))) > 0.0, "every bit equally certain is a hard decision"


def test_a_spacing_with_too_few_pilots_is_refused() -> None:
    """One pilot in a window estimates nothing, and saying so beats a NaN."""
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=4, sequence_length=64, seed=1)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=4.0, label="prbs"))
    mapper = graph.add(QAMMapper(bits_per_symbol=4.0, differential=False, label="map"))
    pilots = graph.add(PilotInserter(spacing=4096.0, label="pil"))
    graph.chain(prbs, mapper, pilots)
    with pytest.raises(ValueError, match="not enough"):
        graph.run()
