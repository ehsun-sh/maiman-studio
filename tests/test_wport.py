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
import os
from pathlib import Path

import numpy as np
import pytest

from maiman.ofec import ofec_frame_bits
from maiman.wport import (
    DSP_FRAME_SYMBOLS,
    FAW_SYMBOLS,
    FLEXO_ROW_BITS,
    PAYLOAD_SYMBOLS,
    PILOT_SPACING,
    PILOTS_PER_SUBFRAME,
    RESERVED_SYMBOLS,
    SUBFRAME_SYMBOLS,
    SUBFRAMES,
    TRAINING_SYMBOLS,
    crc32,
    dsp_deframe,
    dsp_frame,
    faw_symbols,
    flexo_adapt,
    flexo_deadapt,
    flexo_rows,
    information_bits,
    jones,
    pilot_symbols,
    scramble,
    symbol_bits,
    symbol_levels,
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
    codec = ofec_frame_bits(modulation)
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
    for modulation, rows, crcs, pad in (("qpsk", 58, 15, 16), ("16qam", 116, 29, 64)):
        assert flexo_rows(modulation) == rows
        carried = rows * FLEXO_ROW_BITS
        assert carried + crcs * 32 + pad == ofec_frame_bits(modulation)
        adapted = flexo_adapt(information(modulation), modulation=modulation)
        assert adapted.size == ofec_frame_bits(modulation)
        assert np.array_equal(adapted[-pad:], np.zeros(pad, dtype=np.uint8)), "the pad is zeros"


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
    spoiled[5 * FLEXO_ROW_BITS] ^= 1  # the second group of four rows
    back = flexo_deadapt(spoiled, modulation="qpsk")
    assert not back.clean
    assert back.crc_ok.sum() == back.crc_ok.size - 1
    assert not back.crc_ok[1], "and it is the group the bit was in"


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
    received = wport_receive(spoiled, modulation="qpsk")
    assert received.corrections > 0
    assert np.array_equal(received.information, info)
    assert received.clean
