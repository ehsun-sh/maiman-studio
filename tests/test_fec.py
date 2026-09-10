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
