"""The W-Port framing around OFEC, against the specification's own test vectors.

:mod:`maiman.ofec` reproduces TP2 to TP6. This is everything on either side of
it: TP0 to TP2 through the FlexO adaptation and the scrambler, and TP6 to TP7
through the symbol mapper and the DSP framer. The specification says compliance
needs only TP7 from TP0; with the vectors present, every point between is
checked too.

**Without the vectors**, what holds it is the framing's own arithmetic. The
frame's symbol budget has to add up exactly -- 2,736 pilots, 240 training
symbols that are not pilots, 22 of alignment and 74 reserved leave precisely the
172,032 the symbol mapper makes -- every stage has to be the inverse of its
partner, the CRC32 has to catch a flipped bit, and the pilot sequence has to be
the PRBS10 the clause names, DC balanced, with its first symbol shared with the
training sequence. Two fingerprints, taken while the code matched the vectors,
fail on any bit that moves.
"""

from __future__ import annotations

import hashlib
import math
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from maiman.wport import (
    DSP_FRAME_SYMBOLS,
    ERROR_CONTROL_BLOCK,
    ERROR_MARKING_BLOCK,
    FAW_SYMBOLS,
    FLEXO_ROW_BITS,
    PAYLOAD_SYMBOLS,
    PILOT_SPACING,
    PILOTS_PER_SUBFRAME,
    RESERVED_SYMBOLS,
    SUBFRAME_SYMBOLS,
    SUBFRAMES,
    TRAINING_SYMBOLS,
    _pam_constellation,  # the constellation table the brute-force LLR check reads directly
    codec_bits,
    crc32,
    dsp_deframe,
    dsp_frame,
    faw_symbols,
    flexo_adapt,
    flexo_deadapt,
    flexo_rows,
    information_bits,
    jones,
    line_modulation,
    pilot_symbols,
    scramble,
    symbol_bits,
    symbol_levels,
    symbol_llr,
    training_symbols,
    wport_receive,
    wport_transmit,
)

VECTORS = os.environ.get("MAIMAN_OFEC_VECTORS")
MODULATIONS = ["qpsk", "16qam"]

#: SHA-256 of the DSP frame made from a seeded information frame, taken while the
#: chain reproduced the specification's TP7 symbol for symbol.
FINGERPRINTS = {
    "qpsk": "0d228997110b2dbf43140c50b519cd626ccc577bf95cab9b806381a2a2a50285",
    "16qam": "50032843e6efd878f1e857f267ef91b7208f6297fbb05ad149d5aaf24931296e",
}

#: The check value every CRC-32/BZIP2 implementation publishes for "123456789",
#: which is this clause's CRC32 read MSB-first over a byte string's bits.
CRC_CHECK = 0xFC891918


def information(modulation: str, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, information_bits(modulation), dtype=np.uint8)


def vector(modulation: str, point: str) -> np.ndarray:
    folder = Path(str(VECTORS)) / modulation
    dtype = np.int8 if point == "TP7" else np.uint8
    return np.loadtxt(folder / f"{point}_{modulation}.txt", dtype=dtype)


# ---------------------------------------------------------------------------
# Bit-exact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not VECTORS, reason="set MAIMAN_OFEC_VECTORS to the specification's vectors")
@pytest.mark.parametrize("modulation", MODULATIONS)
def test_the_specification_vectors_end_to_end(modulation: str) -> None:
    """TP0 to TP7, which is the only point the specification requires."""
    tp0 = vector(modulation, "TP0")
    tp7 = vector(modulation, "TP7")
    frames = tp0.size // information_bits(modulation)
    assert frames == 4
    reserved = tp7[:DSP_FRAME_SYMBOLS].reshape(SUBFRAMES, SUBFRAME_SYMBOLS, 4)
    carried = np.ones(SUBFRAME_SYMBOLS, dtype=bool)
    carried[::PILOT_SPACING] = False
    head = TRAINING_SYMBOLS - 1 + FAW_SYMBOLS
    reserved = reserved[0, carried][head : head + RESERVED_SYMBOLS]

    # All four frames at once: OFEC carries state from one into the next.
    made = wport_transmit(tp0, modulation=modulation, reserved=reserved)
    assert np.array_equal(made, tp7)


@pytest.mark.skipif(not VECTORS, reason="set MAIMAN_OFEC_VECTORS to the specification's vectors")
@pytest.mark.parametrize("modulation", MODULATIONS)
def test_every_test_point_on_the_way(modulation: str) -> None:
    """TP0 -> TP1 -> TP2, and TP6 -> TP7: the optional points, all of them."""
    tp0, tp1, tp2 = (vector(modulation, point) for point in ("TP0", "TP1", "TP2"))
    tp6, tp7 = vector(modulation, "TP6"), vector(modulation, "TP7")
    info_bits = information_bits(modulation)
    codec = codec_bits(modulation)
    line = tp6.size // 4

    for index in range(4):
        info = tp0[index * info_bits : (index + 1) * info_bits]
        adapted = flexo_adapt(info, modulation=modulation)
        assert np.array_equal(adapted, tp1[index * codec : (index + 1) * codec]), f"TP1 {index}"
        assert np.array_equal(scramble(adapted), tp2[index * codec : (index + 1) * codec])

        mapped = symbol_levels(tp6[index * line : (index + 1) * line], modulation=modulation)
        want = tp7[index * DSP_FRAME_SYMBOLS : (index + 1) * DSP_FRAME_SYMBOLS]
        taken = dsp_deframe(want, modulation=modulation)
        assert np.array_equal(mapped, taken.payload), f"TP7 payload {index}"
        assert np.array_equal(taken.faw, faw_symbols(modulation)), f"TP7 FAW {index}"
        assert np.array_equal(taken.pilots[0], pilot_symbols(modulation)), f"TP7 pilots {index}"
        assert np.array_equal(taken.training[0], training_symbols(modulation))


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_the_output_has_not_moved_since_it_matched_the_vectors(modulation: str) -> None:
    """What the vector tests check when the vectors are not here: one digest, no bit spare."""
    frame = wport_transmit(information(modulation), modulation=modulation)
    digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    assert digest == FINGERPRINTS[modulation]


# ---------------------------------------------------------------------------
# The adaptation
# ---------------------------------------------------------------------------


def test_the_rows_and_the_pad_add_up_to_the_codec_s_own_width() -> None:
    """Every mode, DO and DPO: rows, CRC32s and pad are the codec's input width."""
    for modulation, rows, crcs, pad in (
        ("qpsk", 58, 15, 16),
        ("16qam", 116, 29, 64),
        ("b72", 433, 22, 1464),
        ("b106", 522, 27, 1104),
        ("b116", 548, 28, 1376),
    ):
        assert flexo_rows(modulation) == rows
        carried = rows * FLEXO_ROW_BITS[modulation]
        assert carried + crcs * 32 + pad == codec_bits(modulation)
        adapted = flexo_adapt(information(modulation), modulation=modulation)
        assert adapted.size == codec_bits(modulation)
        assert np.array_equal(adapted[-pad:], np.zeros(pad, dtype=np.uint8)), "the pad is zeros"
        back = flexo_deadapt(adapted, modulation=modulation)
        assert np.array_equal(back.information, information(modulation)) and back.clean


def test_a_crc32_covers_the_same_bits_however_the_rows_are_cut() -> None:
    """Four rows of 10,280 and twenty of 2,056 are both 41,120 bits."""
    assert 4 * FLEXO_ROW_BITS["qpsk"] == 20 * FLEXO_ROW_BITS["b116"] == 41120
    assert line_modulation("b116") == "16qam", "shaped or not, the line is DP-16QAM"
    assert line_modulation("qpsk") == "qpsk"


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_the_adaptation_is_undone_exactly(modulation: str) -> None:
    info = information(modulation)
    back = flexo_deadapt(flexo_adapt(info, modulation=modulation), modulation=modulation)
    assert np.array_equal(back.information, info)
    assert back.clean


def test_a_flipped_bit_is_marked_by_the_crc_that_covers_it() -> None:
    info = information("qpsk")
    adapted = flexo_adapt(info, modulation="qpsk")
    spoiled = adapted.copy()
    spoiled[5 * FLEXO_ROW_BITS["qpsk"]] ^= 1  # the second group of four rows
    back = flexo_deadapt(spoiled, modulation="qpsk")
    assert not back.clean
    assert back.crc_ok.sum() == back.crc_ok.size - 1
    assert not back.crc_ok[1], "and it is the group the bit was in"


#: Figure 9's bottom row, "257-bit error marking block encoding (binary value)",
#: in transmission order: copied from the figure, not built from the fields.
FIGURE_9 = "0" + "01110000" + "0111100" * 8 + ("01111000" + "0111100" * 8) * 3


def test_the_error_marking_block_is_figure_9() -> None:
    """Built from clause 8.2's fields, it is the bit string the figure prints.

    The figure's top row gives the fields -- header 0, flags 0000, 0xE, eight
    /E/, then 0x1E and eight /E/ three times -- and its bottom row the bits on
    the wire. The two agree only if every field goes out least significant bit
    first and the flags ride in the high nibble of the first block's type byte.
    """
    assert "".join(map(str, ERROR_MARKING_BLOCK)) == FIGURE_9
    assert ERROR_MARKING_BLOCK.size == len(FIGURE_9) == 257

    # "sync=10, control block type=0x1E, and eight 7-bit /E/ control characters".
    assert "".join(map(str, ERROR_CONTROL_BLOCK)) == "10" + "01111000" + "0111100" * 8
    # The last three blocks of the 257 are three error control blocks less their sync.
    assert np.array_equal(ERROR_MARKING_BLOCK[-192:], np.tile(ERROR_CONTROL_BLOCK[2:], 3))
    # The first differs only where the flags displaced the top of its type byte.
    first = ERROR_MARKING_BLOCK[1:65]
    assert np.array_equal(first[:4], ERROR_CONTROL_BLOCK[2:6])
    assert np.array_equal(first[8:], ERROR_CONTROL_BLOCK[10:])


def test_every_crc_covers_whole_marking_blocks() -> None:
    """A wide row is forty blocks of 257 and a narrow one eight, so no span is cut."""
    assert FLEXO_ROW_BITS["qpsk"] == 40 * 257 and FLEXO_ROW_BITS["b72"] == 8 * 257
    for modulation in FLEXO_ROW_BITS:
        total = information_bits(modulation)
        spans = np.diff([*range(0, total, 41120), total])
        assert all(span % 257 == 0 for span in spans), modulation


@pytest.mark.parametrize("modulation", ["qpsk", "b72"])
def test_a_failed_crc_s_rows_come_back_marked(modulation: str) -> None:
    """Clause 8.2 with ``error_marking`` set: the failed span, and only it, is overwritten.

    Two hits: one inside a full span and one in the short span at the end, which
    ends on a row boundary rather than a four-row one and still tiles exactly.
    """
    info = information(modulation)
    adapted = flexo_adapt(info, modulation=modulation)
    spoiled = adapted.copy()
    spoiled[41120 + 32 + 100] ^= 1  # the second span, past the first span and its CRC32
    spoiled[info.size + 32 * (info.size // 41120) - 1] ^= 1  # the last information bit
    plain = flexo_deadapt(spoiled, modulation=modulation)
    marked = flexo_deadapt(spoiled, modulation=modulation, error_marking=True)
    assert np.array_equal(marked.crc_ok, plain.crc_ok), "marking does not change the verdict"
    assert list(np.flatnonzero(~marked.crc_ok)) == [1, marked.crc_ok.size - 1]

    starts = [*range(0, info.size, 41120), info.size]
    for index, (start, stop) in enumerate(pairwise(starts)):
        got = marked.information[start:stop]
        if marked.crc_ok[index]:
            assert np.array_equal(got, info[start:stop])
        else:
            assert np.array_equal(got, np.tile(ERROR_MARKING_BLOCK, (stop - start) // 257))
    assert not np.array_equal(plain.information, info), "off, the damage comes through as is"


def test_the_crc_is_the_one_802_3_specifies() -> None:
    """Against the check value published for it, which this code has never seen.

    The clause's procedure -- complement the leading 32 bits, divide, complement
    the remainder, transmit ``x^31`` first -- is CRC-32/BZIP2 when the bits come
    from bytes most significant first. Its catalogued check value for the nine
    characters ``123456789`` is ``0xFC891918``, and that is an independent
    answer: nothing here was fitted to it.
    """
    bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    value = int("".join(str(bit) for bit in crc32(bits)), 2)
    assert value == CRC_CHECK

    # Complementing the leading bits is part of the definition, so leading zeros
    # are not invisible the way a bare remainder would leave them.
    assert not np.array_equal(
        crc32(np.zeros(64, dtype=np.uint8)), crc32(np.ones(64, dtype=np.uint8))
    )
    padded = np.concatenate([np.zeros(32, dtype=np.uint8), np.ones(32, dtype=np.uint8)])
    assert not np.array_equal(crc32(padded[32:]), crc32(padded))


def test_what_is_not_a_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="information bits"):
        flexo_adapt(np.zeros(10, dtype=np.uint8))
    with pytest.raises(ValueError, match="modulation must be"):
        flexo_rows("bpsk")
    with pytest.raises(ValueError, match="codec frame"):
        flexo_deadapt(np.zeros(10, dtype=np.uint8))


# ---------------------------------------------------------------------------
# The scrambler
# ---------------------------------------------------------------------------


def test_the_scrambler_is_its_own_inverse_and_starts_from_all_ones() -> None:
    bits = np.random.default_rng(2).integers(0, 2, 5000, dtype=np.uint8)
    assert np.array_equal(scramble(scramble(bits)), bits)
    zeros = scramble(np.zeros(40, dtype=np.uint8))
    assert np.array_equal(zeros[:16], np.ones(16, dtype=np.uint8)), "the 0xFFFF seed, put out first"


def test_the_sequence_is_the_polynomial_the_clause_names() -> None:
    keystream = scramble(np.zeros(70000, dtype=np.uint8))
    tail = keystream[16:]
    assert np.array_equal(
        tail, keystream[15:-1] ^ keystream[13:-3] ^ keystream[4:-12] ^ keystream[:-16]
    ), "x^16 + x^12 + x^3 + x + 1"
    assert np.array_equal(keystream[:4000], keystream[65535 : 65535 + 4000]), "period 65535"
    assert keystream.mean() == pytest.approx(0.5, abs=0.01), "and balanced"


# ---------------------------------------------------------------------------
# The symbol mapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("modulation", "per"), [("qpsk", 4), ("16qam", 8)])
def test_the_mapper_is_undone_by_the_slicer(modulation: str, per: int) -> None:
    bits = np.random.default_rng(3).integers(0, 2, per * 1000, dtype=np.uint8)
    levels = symbol_levels(bits, modulation=modulation)
    assert levels.shape == (1000, 4)
    assert np.array_equal(symbol_bits(levels, modulation=modulation), bits)


def test_the_polarizations_are_interleaved_bit_by_bit() -> None:
    """Clause 11.1.2: the even bits are X's, the odd ones Y's."""
    bits = np.array([1, 0, 0, 1], dtype=np.uint8)  # c0..c3
    levels = symbol_levels(bits, modulation="qpsk")
    assert levels.tolist() == [[1, -1, -1, 1]], "XI from c0, XQ from c2, YI from c1, YQ from c3"


def test_the_16qam_labels_are_the_table_s() -> None:
    """(0,0) -> -3, (0,1) -> -1, (1,1) -> +1, (1,0) -> +3, per signalling dimension."""
    seen = {}
    for first in (0, 1):
        for second in (0, 1):
            bits = np.zeros(8, dtype=np.uint8)
            bits[0], bits[2] = first, second
            seen[(first, second)] = int(symbol_levels(bits, modulation="16qam")[0, 0])
    assert seen == {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}


def test_a_slicer_takes_the_nearest_point() -> None:
    noisy = np.array([[2.7, -0.8, -3.4, 1.1]])
    assert np.array_equal(
        symbol_bits(noisy, modulation="16qam"),
        symbol_bits(np.array([[3, -1, -3, 1]]), modulation="16qam"),
    )


def test_a_jones_vector_is_the_two_polarizations() -> None:
    x, y = jones(np.array([[1, -1, -1, 1]]))
    assert x[0] == 1 - 1j
    assert y[0] == -1 + 1j


# ---------------------------------------------------------------------------
# The DSP frame
# ---------------------------------------------------------------------------


def test_the_frame_s_symbol_budget_adds_up() -> None:
    pilots = SUBFRAMES * PILOTS_PER_SUBFRAME
    shared = SUBFRAMES  # the first training symbol of each subframe is its first pilot
    training = SUBFRAMES * TRAINING_SYMBOLS - shared
    assert pilots == 2736
    assert training == 240
    assert pilots + training + FAW_SYMBOLS + RESERVED_SYMBOLS + PAYLOAD_SYMBOLS == DSP_FRAME_SYMBOLS
    assert SUBFRAMES * SUBFRAME_SYMBOLS == DSP_FRAME_SYMBOLS


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_the_frame_is_taken_apart_into_what_went_into_it(modulation: str) -> None:
    payload = np.random.default_rng(4).integers(0, 2, (PAYLOAD_SYMBOLS, 4), dtype=np.int8) * 2 - 1
    frame = dsp_frame(payload.astype(np.int8), modulation=modulation)
    assert frame.shape == (DSP_FRAME_SYMBOLS, 4)
    taken = dsp_deframe(frame, modulation=modulation)
    assert np.array_equal(taken.payload, payload)
    assert np.array_equal(taken.faw, faw_symbols(modulation))
    assert np.array_equal(taken.training[7], training_symbols(modulation))
    assert np.array_equal(taken.pilots[13], pilot_symbols(modulation))
    assert taken.reserved.shape == (RESERVED_SYMBOLS, 4)


def test_the_pilots_sit_on_the_grid_and_nothing_displaces_them() -> None:
    payload = np.ones((PAYLOAD_SYMBOLS, 4), dtype=np.int8)
    frame = dsp_frame(payload, modulation="qpsk").reshape(SUBFRAMES, SUBFRAME_SYMBOLS, 4)
    pilots = pilot_symbols("qpsk")
    for index in range(SUBFRAMES):
        assert np.array_equal(frame[index, ::PILOT_SPACING], pilots), index
    # The reserved symbols straddle the pilot at 64 rather than displacing it:
    # they run 33 to 63 and 65 to 107 of the first subframe.
    assert np.array_equal(frame[0, PILOT_SPACING], pilots[1]), "still a pilot"
    assert np.array_equal(frame[0, 33], frame[0, 63]), "reserved either side of it"
    assert np.array_equal(frame[0, 65], frame[0, 107])


def test_the_training_sequence_shares_its_first_symbol_with_a_pilot() -> None:
    for modulation in MODULATIONS:
        assert np.array_equal(training_symbols(modulation)[0], pilot_symbols(modulation)[0])


def test_the_pilots_are_dc_balanced_as_the_seeds_were_chosen_to_be() -> None:
    pilots = pilot_symbols("qpsk").astype(np.float64)
    assert np.abs(pilots.mean(axis=0)).max() < 0.1, pilots.mean(axis=0)


def test_the_overhead_sits_on_the_outer_points() -> None:
    for modulation, amplitude in (("qpsk", 1), ("16qam", 3)):
        for block in (
            faw_symbols(modulation),
            training_symbols(modulation),
            pilot_symbols(modulation),
        ):
            assert set(np.unique(block).tolist()) == {-amplitude, amplitude}


def test_the_two_polarizations_carry_different_words() -> None:
    """Which is what lets a receiver tell X from Y, and I from Q, at all."""
    faw = faw_symbols("qpsk")
    assert not np.array_equal(faw[:, :2], faw[:, 2:])
    pilots = pilot_symbols("qpsk")
    assert not np.array_equal(pilots[:, :2], pilots[:, 2:])


def test_a_frame_that_is_not_one_is_refused() -> None:
    with pytest.raises(ValueError, match="payload symbols"):
        dsp_frame(np.zeros((10, 4), dtype=np.int8))
    with pytest.raises(ValueError, match="reserved symbols"):
        dsp_frame(
            np.zeros((PAYLOAD_SYMBOLS, 4), dtype=np.int8), reserved=np.zeros((2, 4), dtype=np.int8)
        )
    with pytest.raises(ValueError, match="DSP frame is"):
        dsp_deframe(np.zeros((10, 4), dtype=np.int8))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_what_goes_in_comes_back_out(modulation: str) -> None:
    info = information(modulation)
    frame = wport_transmit(info, modulation=modulation)
    assert frame.shape == (DSP_FRAME_SYMBOLS, 4)
    received = wport_receive(frame, modulation=modulation, iterations=1)
    assert np.array_equal(received.information, info)
    assert received.clean
    assert received.corrections == 0, "nothing was wrong with it"


def test_the_decoder_repairs_what_the_line_did_and_the_crc_agrees() -> None:
    info = information("qpsk")
    frame = wport_transmit(info, modulation="qpsk")
    rng = np.random.default_rng(6)
    spoiled = frame.copy()
    # Flip the sign of a few hundred payload symbols, well inside what the code holds.
    hit = rng.choice(np.arange(200, DSP_FRAME_SYMBOLS), size=300, replace=False)
    spoiled[hit, 0] *= -1
    received = wport_receive(spoiled, modulation="qpsk", error_marking=True)
    assert received.corrections > 0
    assert np.array_equal(received.information, info)
    assert received.clean


# ---------------------------------------------------------------------------
# Soft-decision demapping
# ---------------------------------------------------------------------------


def test_qpsk_llr_is_the_closed_form_awgn_ratio() -> None:
    """Each axis of DP-QPSK is BPSK on amplitudes +/-1: ``LLR = -2r/sigma**2``.

    Derivation, for bit=1 -> +1 and bit=0 -> -1 (the labelling
    :func:`symbol_levels` builds): with a Gaussian channel of variance
    ``sigma**2`` per real dimension,

        LLR = ln P(bit=0|r) / P(bit=1|r)
            = [-(r+1)**2 + (r-1)**2] / (2*sigma**2)
            = -4r / (2*sigma**2) = -2r/sigma**2,

    which is what :func:`ofec_decode` expects: positive means 0, negative
    means 1. This is the amplitude scaling clause 11.1 actually uses (+/-1,
    not the unit-energy +/-1/sqrt(2) a normalised BPSK reference would use), so
    there is no extra factor of sqrt(2) here.
    """
    rng = np.random.default_rng(11)
    levels = symbol_levels(rng.integers(0, 2, 4 * 64, dtype=np.uint8), modulation="qpsk")
    noisy = levels.astype(np.float64) + rng.normal(0.0, 0.6, levels.shape)
    sigma2 = 0.37
    llr = symbol_llr(noisy, sigma2, modulation="qpsk")
    axes = -2.0 * noisy / sigma2
    expected = np.stack([axes[:, 0], axes[:, 2], axes[:, 1], axes[:, 3]], axis=1).reshape(-1)
    np.testing.assert_allclose(llr, expected)


def test_16qam_llr_matches_a_brute_force_log_sum_exp() -> None:
    """The exact LLR against a plain-Python log-sum-exp over all four points.

    :func:`symbol_llr` vectorises the same sum :func:`_pam_llr` documents; this
    recomputes it with nothing but ``math.exp`` and ``math.log``, reading the
    points and bit labels straight off :func:`_pam_constellation` so a labelling
    mistake in the library can't also be baked into the reference.
    """
    points, labels = _pam_constellation("16qam")
    rng = np.random.default_rng(12)
    axis_values = rng.uniform(-4.0, 4.0, size=20)
    sigma2 = 0.6

    def brute(r: float, column: int) -> float:
        def log_sum(mask: np.ndarray) -> float:
            terms = [-((r - float(p)) ** 2) / (2.0 * sigma2) for p in points[mask]]
            peak = max(terms)
            return peak + math.log(sum(math.exp(term - peak) for term in terms))

        zero = labels[:, column] == 0
        return log_sum(zero) - log_sum(~zero)

    # A minimal levels array that puts every sample on the XI axis (column 0),
    # so this exercises exactly the bits the brute-force reference computes.
    levels = np.zeros((axis_values.size, 4), dtype=np.float64)
    levels[:, 0] = axis_values
    llr = symbol_llr(levels, sigma2, modulation="16qam")
    llr = llr.reshape(axis_values.size, 8)

    expected_bit0 = np.array([brute(r, 0) for r in axis_values])
    expected_bit1 = np.array([brute(r, 1) for r in axis_values])
    np.testing.assert_allclose(llr[:, 0], expected_bit0, rtol=1e-10)
    np.testing.assert_allclose(llr[:, 2], expected_bit1, rtol=1e-10)


def test_max_log_agrees_with_exact_away_from_the_boundary() -> None:
    """Tosato & Bisaglia's approximation is exact in the sign and close in value.

    Far from a decision boundary one term dominates every log-sum-exp, so
    max-log and the exact LLR should coincide to a small tolerance; and,
    because both read off the same nearest/farthest ordering of points, neither
    can flip a hard decision relative to the other -- tested here at the level
    of the sign alone, since the value tolerance already implies it.
    """
    rng = np.random.default_rng(13)
    bits = rng.integers(0, 2, 8 * 256, dtype=np.uint8)
    levels = symbol_levels(bits, modulation="16qam").astype(np.float64)
    # Well inside a decision region: sigma is a fraction of the 2-unit spacing.
    sigma2 = 0.05
    exact = symbol_llr(levels, sigma2, modulation="16qam", exact=True)
    approx = symbol_llr(levels, sigma2, modulation="16qam", exact=False)
    np.testing.assert_array_equal(np.sign(exact), np.sign(approx))
    np.testing.assert_allclose(exact, approx, atol=1e-6)


def test_symbol_llr_refuses_bad_shapes_and_variance() -> None:
    with pytest.raises(ValueError, match="XI, XQ, YI, YQ"):
        symbol_llr(np.zeros((4, 3)), 1.0)
    with pytest.raises(ValueError, match="noise_variance"):
        symbol_llr(np.zeros((4, 4)), 0.0)
    with pytest.raises(ValueError, match="noise_variance"):
        symbol_llr(np.zeros((4, 4)), -1.0)


@pytest.mark.parametrize("modulation", MODULATIONS)
def test_the_llr_s_sign_is_the_hard_decision_on_every_axis(modulation: str) -> None:
    """Every bit of every axis, in TP6 order: a slightly noisy symbol's LLR says what slicing says.

    The brute-force test above holds the value of one axis's two bits; this holds
    where each of all four axes' bits lands in the stream, against
    :func:`symbol_bits`, which the vector tests hold to the specification. A 1 is
    a negative ratio. (Unchanged defaults need no test of their own: every
    vector and fingerprint test above runs the receiver with none given.)
    """
    rng = np.random.default_rng(14)
    width = 4 if modulation == "qpsk" else 8
    levels = symbol_levels(rng.integers(0, 2, width * 512, dtype=np.uint8), modulation=modulation)
    noisy = levels.astype(np.float64) + rng.normal(0.0, 0.2, levels.shape)
    llr = symbol_llr(noisy, 0.04, modulation=modulation)
    sliced = symbol_bits(levels, modulation=modulation)
    assert np.array_equal((llr < 0).astype(np.uint8), sliced)


def test_soft_decoding_cleans_up_where_hard_decoding_cannot() -> None:
    """One frame, one seed, at an SNR chosen so the hard path fails and the soft path does not.

    DP-QPSK, information seed 3 (:func:`information`'s default), AWGN of
    variance 0.15 per real dimension added with ``numpy.random.default_rng(42)``.
    That measures a pre-FEC BER of about 0.50 % on the sliced bits -- inside
    what :mod:`maiman.ofec`'s own docstring table shows three passes clearing at
    1.0 % and missing at 1.5 % raw, so a miss here is not a cherry-picked
    surprise. The hard path hands the decoder a confidence of 8.0 regardless of
    how close the sample sat to the boundary; the soft path hands it the real
    ratio, and clears frames the hard path leaves dirty.
    """
    info = information("qpsk")
    frame = wport_transmit(info, modulation="qpsk")
    sigma2 = 0.15
    noise = np.random.default_rng(42).normal(0.0, sigma2**0.5, frame.shape)
    noisy = frame.astype(np.float64) + noise

    taken = dsp_deframe(noisy, modulation="qpsk")
    clean = dsp_deframe(frame.astype(np.float64), modulation="qpsk")
    pre_fec_ber = float(
        np.mean(
            symbol_bits(taken.payload, modulation="qpsk")
            != symbol_bits(clean.payload, modulation="qpsk")
        )
    )
    assert 0.001 < pre_fec_ber < 0.02, f"pick a noisier or quieter seed, got {pre_fec_ber}"

    hard = wport_receive(noisy, modulation="qpsk", noise_variance=None)
    soft = wport_receive(noisy, modulation="qpsk", noise_variance=sigma2)

    assert not hard.clean, "the hard-decision LLR was expected to leave this frame dirty"
    assert soft.clean, "the soft LLR was expected to decode this frame cleanly"
    assert np.array_equal(soft.information, info)


@pytest.mark.parametrize("sigma2", [0.10, 0.15, 0.25])
def test_pre_fec_ber_of_the_soft_demapper_matches_the_q_function(sigma2: float) -> None:
    """DP-QPSK in AWGN: the sign of each LLR errs at Q(a / sigma), a = the level, sigma^2 per axis.

    Each bit rides one real axis at amplitude ``a`` with noise of variance
    ``sigma2``, so the raw bit error rate is ``Q(a / sigma) = erfc(a / sqrt(2 sigma2)) / 2``
    (Proakis, *Digital Communications*, eq. 4.2-20 for antipodal signalling).
    The measured rate over 4e6 bits sits within four binomial standard errors.
    """
    points, _ = _pam_constellation("qpsk")
    amplitude = float(np.max(np.abs(points)))
    rng = np.random.default_rng(21)
    symbols = 1_000_000
    sent = rng.choice([-amplitude, amplitude], size=(symbols, 4))
    received = sent + rng.normal(0.0, sigma2**0.5, sent.shape)

    llr = symbol_llr(received, sigma2, modulation="qpsk")
    hard = symbol_llr(sent, sigma2, modulation="qpsk")  # the transmitted bits, same bit order
    measured = float(np.mean((llr > 0) != (hard > 0)))

    theory = 0.5 * math.erfc(amplitude / math.sqrt(2.0 * sigma2))
    tolerance = 4.0 * math.sqrt(theory * (1.0 - theory) / (4 * symbols))
    assert abs(measured - theory) < tolerance, (measured, theory)


def brute_force_llr(levels: np.ndarray, sigma2: float, prior: np.ndarray, bit: int) -> np.ndarray:
    """``ln P(b=0 | y) / P(b=1 | y)`` for one axis's bit, summed over the four 16QAM points."""
    points, labels = _pam_constellation("16qam")
    likelihood = np.exp(-((levels[:, None] - points[None, :]) ** 2) / (2.0 * sigma2)) * prior
    zero = likelihood[:, labels[:, bit] == 0].sum(axis=1)
    one = likelihood[:, labels[:, bit] == 1].sum(axis=1)
    return np.log(zero / one)


@pytest.mark.parametrize("inner", [0.623, 0.681, 0.8265])
def test_the_shaped_prior_is_the_brute_force_map_ratio(inner: float) -> None:
    """The ratio with ``inner_probability`` is Bayes' rule over the four points, written out.

    Prior ``(1 - p) / 2, p / 2, p / 2, (1 - p) / 2`` on ``-3, -1, 1, 3``: sign
    uniform, magnitude ``p`` on the inner pair. Checked on the first axis's two
    bits, which the symbol order puts at positions 0 and 2 of each row of eight.
    """
    rng = np.random.default_rng(31)
    levels = rng.normal(0.0, 2.0, size=(500, 4))
    sigma2 = 0.4
    prior = np.array([(1 - inner) / 2, inner / 2, inner / 2, (1 - inner) / 2])
    got = symbol_llr(levels, sigma2, modulation="b72", inner_probability=inner).reshape(-1, 8)
    assert np.allclose(got[:, 0], brute_force_llr(levels[:, 0], sigma2, prior, 0))
    assert np.allclose(got[:, 2], brute_force_llr(levels[:, 0], sigma2, prior, 1))


def test_an_even_prior_is_the_uniform_one() -> None:
    levels = np.random.default_rng(32).normal(0.0, 2.0, size=(200, 4))
    even = symbol_llr(levels, 0.3, modulation="b72", inner_probability=0.5)
    assert np.allclose(even, symbol_llr(levels, 0.3, modulation="b72"), rtol=0, atol=1e-12)


@pytest.mark.parametrize("inner", [0.623, 0.8265])
def test_with_no_information_in_the_sample_the_ratio_is_the_prior(inner: float) -> None:
    """Noise so large the amplitude carries nothing: the magnitude ratio is ``ln (1 - p) / p``.

    The sign bit's stays at 0, the shaper leaving it even -- and the ratio is
    positive where the bit is likelier a 0, the outer amplitude, which is the
    shaping's rarer one.
    """
    levels = np.zeros((10, 4))
    got = symbol_llr(levels, 1e8, modulation="b72", inner_probability=inner).reshape(-1, 8)
    assert np.allclose(got[:, 2], math.log((1 - inner) / inner), atol=1e-6)
    assert np.allclose(got[:, 0], 0.0, atol=1e-6)


def test_max_log_carries_the_prior_too() -> None:
    """Away from a decision boundary max-log and exact agree with a prior as without one."""
    levels = np.array([[2.9, -1.05, 0.98, -3.1]])
    exact = symbol_llr(levels, 0.05, modulation="b72", inner_probability=0.8)
    approx = symbol_llr(levels, 0.05, modulation="b72", inner_probability=0.8, exact=False)
    assert np.allclose(exact, approx, rtol=1e-3)


def test_a_prior_needs_an_amplitude_bit_and_a_probability() -> None:
    with pytest.raises(ValueError, match="no amplitude bit"):
        symbol_llr(np.zeros((1, 4)), 0.1, modulation="qpsk", inner_probability=0.6)
    with pytest.raises(ValueError, match="probability"):
        symbol_llr(np.zeros((1, 4)), 0.1, modulation="b72", inner_probability=1.0)
