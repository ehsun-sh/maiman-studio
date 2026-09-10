"""Validation of RS(255, 239) — the code, the decoder, and the link it buys.

The headline is the one a link engineer cares about: a line running at a bit
error rate that would be a broken link, delivering none.

Nothing in the implementation is written in those terms. It builds a field,
divides by a generator polynomial, solves a shift-register synthesis and
evaluates two more polynomials. That the result corrects exactly eight symbol
errors and not nine, and that a counted link agrees with the bounded-distance
expression, is the result.

References: ITU-T G.709/Y.1331 Annex A; Berlekamp 1968; Massey 1969; Forney 1965.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import Graph, SimulationContext, fec
from maiman.components import (
    CWLaser,
    ElectricalFilter,
    FECDecoder,
    FECEncoder,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    Slicer,
    SoftDemapper,
)
from maiman.modulation import (
    estimate_noise_variance,
    indices_to_bits,
    nearest_indices,
    qam_constellation,
    soft_demap,
)

# --------------------------------------------------------------------------
# The field and the code
# --------------------------------------------------------------------------


def test_the_field_is_a_field() -> None:
    """Every non-zero element has an inverse, and the tables agree with each other.

    Cheap, and it is the assumption every other line in the module rests on: a
    log table built against the wrong primitive polynomial still looks like a
    table and fails silently everywhere downstream.
    """
    for element in range(1, 256):
        assert int(fec.gf_multiply(element, fec.gf_inverse(element))) == 1
    # 2 is primitive: its powers must visit all 255 non-zero elements.
    powers = {fec.gf_power(2, exponent) for exponent in range(255)}
    assert powers == set(range(1, 256))


def test_the_generator_polynomial_has_the_roots_it_is_built_from() -> None:
    """``g(alpha^i) = 0`` for each of the sixteen consecutive roots G.709 names.

    This is the property, not the coefficients. Quoting a coefficient table
    would be transcription; deriving it and checking the roots is the same claim
    made in a way that catches a wrong primitive polynomial or an off-by-one in
    the first root exponent.
    """
    assert fec._GENERATOR.size - 1 == fec.PARITY_SYMBOLS
    for index in range(fec.PARITY_SYMBOLS):
        root = int(fec._ANTILOG[(fec.FIRST_ROOT_EXPONENT + index) % 255])
        assert fec._evaluate(fec._GENERATOR[::-1], root) == 0


def test_the_overhead_is_the_six_point_seven_percent_an_otn_frame_spends() -> None:
    assert fec.overhead() == pytest.approx(255.0 / 239.0 - 1.0)
    assert fec.overhead() == pytest.approx(0.0669, abs=1e-4)


def test_encoding_is_systematic_and_produces_codewords() -> None:
    """The message comes back unchanged, and the result has zero syndromes."""
    rng = np.random.default_rng(0)
    messages = rng.integers(0, 256, (8, fec.MESSAGE_SYMBOLS)).astype(np.int32)
    codewords = fec.encode_blocks(messages)

    assert np.array_equal(codewords[:, : fec.MESSAGE_SYMBOLS], messages)
    assert not np.any(fec._syndromes(codewords))


# --------------------------------------------------------------------------
# The decoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("errors", [0, 1, 2, 5, 7, 8])
def test_it_corrects_up_to_eight_symbol_errors(errors: int) -> None:
    """Anywhere in the codeword, any magnitude, exactly recovered.

    ``t = 8`` is not a tuning parameter — a Reed-Solomon code meets the
    Singleton bound with equality, so it corrects exactly half its parity. The
    boundary is asserted from both sides: this passes at eight and the next test
    fails at nine.
    """
    rng = np.random.default_rng(errors + 1)
    messages = rng.integers(0, 256, (4, fec.MESSAGE_SYMBOLS)).astype(np.int32)
    received = fec.encode_blocks(messages).copy()

    for block in range(received.shape[0]):
        positions = rng.choice(fec.CODEWORD_SYMBOLS, errors, replace=False)
        received[block, positions] ^= rng.integers(1, 256, errors).astype(np.int32)

    result = fec.decode_blocks(received)
    assert np.array_equal(result.messages, messages)
    assert result.failed == 0
    assert result.corrected == errors * received.shape[0]


def test_nine_errors_is_past_the_code_and_it_says_so() -> None:
    """It does not silently return something wrong-looking; it reports failure.

    The distinction matters because the *other* thing a Reed-Solomon decoder can
    do past its limit is find a wrong error pattern and hand back a codeword that
    is confidently incorrect. That is undetectable from inside and is why a coded
    link is quoted with an output error rate rather than with the word
    "corrected".
    """
    rng = np.random.default_rng(99)
    messages = rng.integers(0, 256, (4, fec.MESSAGE_SYMBOLS)).astype(np.int32)
    received = fec.encode_blocks(messages).copy()
    for block in range(received.shape[0]):
        positions = rng.choice(fec.CODEWORD_SYMBOLS, 9, replace=False)
        received[block, positions] ^= rng.integers(1, 256, 9).astype(np.int32)

    result = fec.decode_blocks(received)
    assert result.failed == received.shape[0]
    assert not np.array_equal(result.messages, messages)


def test_a_burst_inside_one_byte_costs_the_same_as_a_single_bit() -> None:
    """The property the code was chosen for, asserted rather than described.

    Eight bad bits in one byte are one symbol error. Eight bad bits spread over
    eight bytes are eight. A bit-oriented code would see those as the same event;
    this one does not, and that is the whole argument for a symbol code on a
    channel whose errors arrive in bursts.
    """
    rng = np.random.default_rng(5)
    messages = rng.integers(0, 256, (1, fec.MESSAGE_SYMBOLS)).astype(np.int32)
    codeword = fec.encode_blocks(messages)

    burst = codeword.copy()
    burst[0, 0:8] ^= 0xFF  # 64 bad bits, in eight bytes — at the limit
    assert fec.decode_blocks(burst).failed == 0

    spread = codeword.copy()
    spread[0, 0:9] ^= 0x01  # 9 bad bits, in nine bytes — past it
    assert fec.decode_blocks(spread).failed == 1


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------


def test_the_closed_form_agrees_with_a_counted_decode() -> None:
    """The expression and the implementation are independent, and must agree.

    They disagreed by a factor of four the first time, and the reason was in the
    expression rather than the decoder: a wrong byte is wrong in *one* bit at
    these error rates, not in four. That is what conditioning on "at least one of
    eight flipped" does at small p, and assuming the half it would tend to at
    p = 0.5 makes the whole curve pessimistic.
    """
    rng = np.random.default_rng(11)
    blocks = 2000
    for rate in (3e-3, 2e-3):
        messages = rng.integers(0, 256, (blocks, fec.MESSAGE_SYMBOLS)).astype(np.int32)
        codewords = fec.encode_blocks(messages)
        bits = np.unpackbits(
            codewords.astype(np.uint8).reshape(blocks, fec.CODEWORD_SYMBOLS, 1), axis=2
        )
        received = np.packbits(bits ^ (rng.random(bits.shape) < rate), axis=2)
        result = fec.decode_blocks(received.reshape(blocks, fec.CODEWORD_SYMBOLS).astype(np.int32))

        wrong = np.unpackbits(np.bitwise_xor(result.messages, messages).astype(np.uint8)).sum()
        counted = wrong / (blocks * fec.MESSAGE_SYMBOLS * 8)
        assert counted == pytest.approx(fec.output_bit_error_rate(rate), rel=0.25)


def test_the_g709_threshold_is_a_ten_thousandth_and_not_a_thousandth() -> None:
    """What this code can actually carry, which is not what "1e-3" suggests.

    The sensitivity tables in this project are quoted at a pre-FEC BER of 1e-3.
    RS(255, 239) takes 1e-3 to about 1e-6 — nowhere near the 1e-15 a transport
    system is specified at. To reach that it needs roughly 1e-4 in. So 1e-3 is a
    *soft-decision* threshold and this is a hard-decision code, and the README
    now says which is which instead of leaving the word "FEC" to cover both.
    """
    assert fec.output_bit_error_rate(1e-3) == pytest.approx(1.1e-6, rel=0.2)
    assert fec.output_bit_error_rate(1e-4) < 1e-14
    assert fec.output_bit_error_rate(2e-4) > 1e-13


# --------------------------------------------------------------------------
# The link
# --------------------------------------------------------------------------


def coded_link(power_dbm: float, sequence_length: int = 20400) -> tuple[Graph, FECDecoder]:
    """A 10 Gb/s OOK link with the code in the path and a real slicer deciding."""
    ctx = SimulationContext(
        bit_rate=10e9, samples_per_symbol=16, sequence_length=sequence_length, seed=4
    )
    graph = Graph(ctx)
    encoder = graph.add(FECEncoder(order=23.0, bits_per_symbol=1.0, label="fec"))
    driver = graph.add(NRZDriver(v_low=4.0, v_high=0.0, label="drv"))
    laser = graph.add(CWLaser(power=power_dbm, label="tx"))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0, label="mzm"))
    detector = graph.add(PINPhotodiode(responsivity=0.8, label="pin"))
    lowpass = graph.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    slicer = graph.add(Slicer(label="sl"))
    decoder = graph.add(FECDecoder(label="dec"))

    graph.connect(encoder["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver, modulator["electrical_in"])
    graph.chain(modulator, detector, lowpass, slicer)
    graph.connect(slicer["out"], decoder["in"])
    graph.connect(encoder["payload"], decoder["payload"])
    return graph, decoder


def test_the_code_turns_a_broken_line_into_a_clean_one() -> None:
    """The whole point, counted end to end rather than predicted.

    At -19 dBm the line is running at a bit error rate a few times 1e-4 — an
    unusable link by any uncoded standard — and the payload comes out with no
    errors at all. Nothing here is a formula: the bits went through a modulator,
    a photodiode with real shot and thermal noise, a filter and a blind slicer,
    and the decoder repaired what it found.
    """
    graph, decoder = coded_link(-19.0)
    report = graph.run(keep=[decoder]).port(decoder, "diagnostics")

    assert report.pre_fec_ber > 1e-4, (
        "the line has to be genuinely broken for this to mean anything"
    )
    assert report.post_fec_ber == 0.0
    assert report.corrected_symbols > 0
    assert report.failed_blocks == 0


def test_past_the_cliff_the_decoder_makes_things_slightly_worse() -> None:
    """Miscorrection, which is the honest half of a FEC story.

    Below the threshold a bounded-distance decoder does not degrade gracefully —
    it adds errors, by confidently applying a wrong error pattern to a codeword
    it cannot decode. A model that clipped the output rate at the input rate
    would look tidier and would be wrong in the direction that flatters the code.
    """
    graph, decoder = coded_link(-21.0, sequence_length=8160)
    report = graph.run(keep=[decoder]).port(decoder, "diagnostics")

    assert report.failed_blocks == report.blocks
    assert report.post_fec_ber > report.pre_fec_ber


def test_the_window_must_hold_whole_codewords() -> None:
    """A codeword split across two runs cannot be decoded in either.

    Refused at run time with a sequence length that would work, rather than
    padded — padding would report a corrected error rate for data nobody sent.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=4096, seed=1)
    graph = Graph(ctx)
    graph.add(FECEncoder(bits_per_symbol=1.0, label="fec"))
    with pytest.raises(ValueError, match="whole number of"):
        graph.run()


def test_the_payload_leaves_at_the_rate_it_is_actually_carried_at() -> None:
    """Coding does not make the line faster; it makes the payload smaller.

    Worth an assertion because the opposite convention — quoting the payload at
    the line rate — makes a coded link look like free capacity, which is the one
    thing a FEC block must never imply.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=8160, seed=1)
    graph = Graph(ctx)
    encoder = graph.add(FECEncoder(bits_per_symbol=1.0, label="fec"))
    results = graph.run(keep=[encoder])

    line = results.port(encoder, "out")
    payload = results.port(encoder, "payload")
    assert payload.symbol_rate == pytest.approx(line.symbol_rate / (1.0 + fec.overhead()))
    assert payload.num_bits * fec.CODEWORD_SYMBOLS == line.num_bits * fec.MESSAGE_SYMBOLS


# --------------------------------------------------------------------------
# The slicer the code needed
# --------------------------------------------------------------------------


def test_the_slicer_decides_without_being_told_the_answer() -> None:
    """Blind in both choices a decision circuit makes, and it has to be.

    Every bit decided in this library used to be decided inside the BER analyser,
    which is handed the transmitted sequence. That is the right way to measure a
    link and no way to build one — a decoder needs the bits a receiver actually
    produced, mistakes included.
    """
    from maiman.analysis import blind_threshold, slice_waveform

    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 4000).astype(np.uint8)
    samples_per_symbol = 8
    clean = np.repeat(bits.astype(np.float64) * 2e-3, samples_per_symbol)
    noisy = clean + rng.normal(0.0, 1e-4, clean.size)

    decided, offset, level = slice_waveform(noisy, samples_per_symbol)
    assert np.count_nonzero(decided != bits) == 0
    assert 0 <= offset < samples_per_symbol
    # Lloyd's converges on the midpoint of the rails, which is what the docstring
    # says it does and where its penalty against the optimum comes from.
    assert level == pytest.approx(1e-3, rel=0.05)
    assert blind_threshold(np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Soft information
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits_per_symbol", [1, 2, 3, 4, 6, 8])
def test_positive_means_zero_in_every_format(bits_per_symbol: int) -> None:
    """The sign convention, asserted rather than assumed.

    Both orders are in common use and neither is more correct, so the only thing
    that makes one of them right here is that everything agrees on it. A decoder
    written against the other convention does not fail loudly — it decodes the
    complement, converges happily, and reports a bit error rate near one half,
    which looks like a broken channel rather than a broken convention.
    """
    constellation = qam_constellation(bits_per_symbol)
    indices = np.arange(constellation.size)
    labels = indices_to_bits(indices, bits_per_symbol)

    llr = soft_demap(constellation[indices], constellation, bits_per_symbol, noise_variance=0.1)
    assert np.array_equal((llr < 0.0).astype(np.uint8), labels)


@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
def test_the_soft_sign_is_the_hard_decision(bits_per_symbol: int) -> None:
    """Max-log is exact in the sign, so it can never change a decision.

    That is the property that makes the approximation safe: it compresses
    confidence, always towards less, and never moves a bit. A soft demapper that
    disagreed with the slicer would be a second, quieter decision rule in the
    receiver — which is exactly the kind of thing that produces two BER numbers
    from one link.
    """
    rng = np.random.default_rng(4)
    constellation = qam_constellation(bits_per_symbol)
    indices = rng.integers(0, constellation.size, 4000)
    noisy = constellation[indices] + (
        rng.normal(0.0, 0.15, indices.size) + 1j * rng.normal(0.0, 0.15, indices.size)
    )

    soft_bits = (soft_demap(noisy, constellation, bits_per_symbol) < 0.0).astype(np.uint8)
    hard_bits = indices_to_bits(nearest_indices(noisy, constellation), bits_per_symbol)
    assert np.array_equal(soft_bits, hard_bits)


def test_confidence_falls_as_a_symbol_approaches_a_boundary() -> None:
    """The magnitude has to mean something, or this is a hard decision in a float.

    A symbol sitting exactly on a decision boundary carries no information about
    that bit, and its LLR must be zero. One placed well inside a region must
    carry a large one. Without this a demapper that returned ``+-1`` would pass
    every sign test in this file.
    """
    constellation = qam_constellation(2)  # QPSK: boundaries on the axes
    on_boundary = np.array([0.0 + 0.7j], dtype=np.complex128)
    well_inside = np.array([0.7 + 0.7j], dtype=np.complex128)

    boundary_llr = soft_demap(on_boundary, constellation, 2, noise_variance=0.1)
    inside_llr = soft_demap(well_inside, constellation, 2, noise_variance=0.1)

    # The in-phase bit is the ambiguous one at x = 0.
    assert abs(boundary_llr[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(inside_llr[0]) > 10.0


def test_the_noise_estimate_finds_the_noise_it_was_given() -> None:
    """Blind, and it has to be: a receiver does not hold the transmitted sequence.

    Checked against the variance actually injected. The estimator is biased low
    at high error rates — a symbol that crossed a boundary is measured against
    the wrong point and looks closer than it is — so this runs where that bias is
    small and the docstring says where it is not.
    """
    rng = np.random.default_rng(7)
    constellation = qam_constellation(4)
    symbols = constellation[rng.integers(0, constellation.size, 20000)]

    for sigma in (0.05, 0.1):
        noisy = symbols + (
            rng.normal(0.0, sigma, symbols.size) + 1j * rng.normal(0.0, sigma, symbols.size)
        )
        assert estimate_noise_variance(noisy, constellation) == pytest.approx(
            2.0 * sigma**2, rel=0.1
        )


def test_scaling_the_variance_scales_every_llr_together() -> None:
    """So it cannot change a hard decision — only how much a decoder believes it.

    Worth pinning because it is the whole of the variance's influence. A
    demapper that let the estimate move a decision would make the noise estimate
    a second decision rule, and a wrong estimate would then cost errors rather
    than only costing confidence.
    """
    rng = np.random.default_rng(9)
    constellation = qam_constellation(4)
    noisy = constellation[rng.integers(0, constellation.size, 500)] + rng.normal(0.0, 0.1, 500)

    tight = soft_demap(noisy, constellation, 4, noise_variance=0.01)
    loose = soft_demap(noisy, constellation, 4, noise_variance=0.04)
    assert np.allclose(tight, 4.0 * loose)


def test_a_soft_output_cannot_be_wired_into_a_block_that_wants_bits() -> None:
    """Refused at edit time, which is the reason the port type exists at all.

    Wiring soft data into a hard input is not a crash — the array has the right
    length and the wrong meaning, and the link would run and quietly give back
    the decibels the soft path exists to recover. The type system is what makes
    that unrepresentable instead of merely unlikely.
    """
    from maiman.component import PortType
    from maiman.graph import GraphError

    assert SoftDemapper.outputs["out"] is PortType.SOFT

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=4, sequence_length=64, seed=1)
    graph = Graph(ctx)
    demapper = graph.add(SoftDemapper(label="soft"))
    decoder = graph.add(FECDecoder(label="dec"))
    with pytest.raises((GraphError, TypeError, ValueError)):
        graph.connect(demapper["out"], decoder["in"])


def test_soft_and_hard_disagree_about_nothing_but_confidence() -> None:
    """End to end on a real link: the soft path is the same decisions, plus more.

    The point of the whole soft path is that it adds information without
    changing any conclusion. If a real received constellation gave different bits
    through the two routes, one of them would be wrong and there would be no way
    to tell which from inside.
    """
    from maiman.signals import SoftSignal

    rng = np.random.default_rng(21)
    constellation = qam_constellation(4)
    indices = rng.integers(0, constellation.size, 2000)
    noisy = constellation[indices] + (
        rng.normal(0.0, 0.12, indices.size) + 1j * rng.normal(0.0, 0.12, indices.size)
    )

    signal = SoftSignal(
        llr=soft_demap(noisy, constellation, 4), symbol_rate=32e9, bits_per_symbol=4
    )
    assert signal.num_bits == indices.size * 4
    assert np.array_equal(signal.hard(), indices_to_bits(nearest_indices(noisy, constellation), 4))
    # And it really is soft: the confidences are not all the same number.
    assert float(np.std(np.abs(signal.llr))) > 0.0
