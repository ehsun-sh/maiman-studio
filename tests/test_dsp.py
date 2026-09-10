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
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
    TimingRecovery,
    Waveguide,
)
from maiman.dsp import estimate_timing, resample_to_instant
from maiman.signals import ConstellationMeasurement

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
