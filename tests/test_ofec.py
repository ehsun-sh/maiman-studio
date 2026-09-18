"""OFEC, bit-exact to the Open ROADM MSA 6.0 W-Port Digital Specification.

**Against the specification's own test vectors.** The document publishes them
(``ps_ofec_test_vectors_v3.zip`` on the Open ROADM MSA wiki). With them, the
encoder reproduces test point TP5 from TP2 and the interleavers TP6 from TP5 with
no bit different -- 2,752,512 bits for DP-QPSK and 5,505,024 for DP-16QAM. The
vectors carry no licence to redistribute them, so they are not in this
repository: set ``MAIMAN_OFEC_VECTORS`` to the directory they unzip to (holding
``qpsk/`` and ``16qam/``) and :func:`test_the_specification_vectors` runs them.

**Without them**, three things hold the code in place. The encoder's output for a
seeded payload is fingerprinted: the digests below were taken while the encoder
matched the vectors, so any change that moves one bit fails here. Every
component codeword is checked against the component code the specification
defines, and every bit found in exactly two of them. And the interleaver is
checked to be a permutation that the de-interleaver undoes exactly.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from maiman.ofec import (
    _BACK,
    _FRONT_OFFSET,
    _INTER,
    _INTRA_SOURCE_COLUMN,
    _INTRA_SOURCE_ROW,
    INFORMATION_BITS_PER_BLOCK,
    REACH,
    ROW_BITS,
    _demultiplex,
    _front_rows,
    _output_order,
    ofec_decode,
    ofec_decode_stream,
    ofec_deinterleave,
    ofec_encode,
    ofec_encode_stream,
    ofec_frame_bits,
    ofec_interleave,
)
from maiman.softfec import bch_code

#: SHA-256 of the packed line bits for a seeded payload, taken while the encoder
#: and interleavers matched the specification's TP5 and TP6 bit for bit.
FINGERPRINTS = {
    ("qpsk", 7): "400dc59ea165156355a3ec9bdff7795fb028799a8ab99e7fc832f9a8a29afd37",
    ("16qam", 8): "3235f3bc01f3a951a5c161245917281f505d9b253fc8a601bce629a0c829d6c8",
}

GENERATOR = sum(1 << p for p in (16, 14, 13, 11, 10, 9, 8, 6, 5, 1, 0))


def payload(modulation: str, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, ofec_frame_bits(modulation), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Bit-exact
# ---------------------------------------------------------------------------

VECTORS = os.environ.get("MAIMAN_OFEC_VECTORS")


@pytest.mark.skipif(
    not VECTORS, reason="set MAIMAN_OFEC_VECTORS to the specification's test vectors"
)
@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_the_specification_vectors(modulation: str) -> None:
    """TP2 -> TP5 through the encoders, TP5 -> TP6 through the interleavers: no bit different."""
    folder = Path(str(VECTORS)) / modulation
    tp2 = np.loadtxt(folder / f"TP2_{modulation}.txt", dtype=np.uint8)
    tp5 = np.loadtxt(folder / f"TP5_{modulation}.txt", dtype=np.uint8)
    tp6 = np.loadtxt(folder / f"TP6_{modulation}.txt", dtype=np.uint8)
    lanes = [ofec_encode_stream(lane) for lane in _demultiplex(tp2)]
    for lane, encoded in enumerate(lanes):
        assert np.array_equal(encoded, tp5[:, lane]), f"encoder {lane}"
    assert np.array_equal(ofec_interleave(lanes, modulation), tp6)
    assert np.array_equal(ofec_encode(tp2, modulation=modulation), tp6)


@pytest.mark.parametrize(("modulation", "seed"), sorted(FINGERPRINTS))
def test_the_output_has_not_moved_since_it_matched_the_vectors(modulation: str, seed: int) -> None:
    line = ofec_encode(payload(modulation, seed), modulation=modulation)
    digest = hashlib.sha256(np.packbits(line).tobytes()).hexdigest()
    assert digest == FINGERPRINTS[(modulation, seed)]


# ---------------------------------------------------------------------------
# The code the specification defines
# ---------------------------------------------------------------------------


def component_words(information: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The encoder's matrix, and each row's (16, 256) component codewords as bit positions."""
    y = ofec_encode_stream(information)
    rows = y.size // ROW_BITS
    matrix = np.empty(rows * ROW_BITS, dtype=np.uint8)
    matrix[_output_order(rows)] = y
    words = []
    for row in range(REACH, rows):
        front = _front_rows(row) * ROW_BITS + _FRONT_OFFSET
        words.append(np.concatenate([front, row * ROW_BITS + _BACK], axis=1))
    return matrix, words


def test_every_codeword_is_an_extended_bch_codeword_of_the_specified_generator() -> None:
    """Divisible by ``t^16 + t^14 + ... + t + 1`` with bit 0 as ``t^254``, and of even weight."""
    assert bch_code(8, 2).generator == GENERATOR
    information = np.random.default_rng(3).integers(
        0, 2, 30 * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8
    )
    matrix, words = component_words(information)
    checked = 0
    for positions in words:
        for word in matrix[positions]:
            value = int("".join(str(bit) for bit in word[:255]), 2)
            for power in range(254, 15, -1):
                if value >> power & 1:
                    value ^= GENERATOR << (power - 16)
            assert value == 0
            assert int(word.sum()) % 2 == 0
            checked += 1
    assert checked == (60 - REACH) * 16


def test_every_bit_is_the_front_of_one_codeword_and_the_back_of_another() -> None:
    information = np.zeros(40 * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8)
    matrix, words = component_words(information)
    fronts = np.zeros(matrix.size, dtype=np.int64)
    backs = np.zeros(matrix.size, dtype=np.int64)
    for positions in words:
        np.add.at(fronts, positions[:, :128].reshape(-1), 1)
        np.add.at(backs, positions[:, 128:].reshape(-1), 1)
    rows = matrix.size // ROW_BITS
    settled = slice(REACH * ROW_BITS, (rows - REACH - 2) * ROW_BITS)
    assert np.all(backs[settled] == 1)
    assert np.all(fronts[settled] == 1), "twenty rows back, and exactly once"


def test_the_rate_is_one_hundred_and_eleven_in_one_hundred_and_twenty_eight() -> None:
    frame = ofec_frame_bits("qpsk")
    line = ofec_encode(np.zeros(frame, dtype=np.uint8), modulation="qpsk")
    assert line.size / frame == pytest.approx(128 / 111, rel=1e-15)
    assert frame == 42 * 14208 and line.size == 4 * 172032


def test_the_interleaver_is_a_permutation_the_deinterleaver_undoes() -> None:
    assert np.array_equal(np.sort(_INTER), np.arange(_INTER.size))
    cells = _INTRA_SOURCE_ROW * 16 + _INTRA_SOURCE_COLUMN
    assert np.array_equal(np.sort(cells.ravel()), np.arange(256)), "Table 8, a permutation"
    for modulation in ("qpsk", "16qam"):
        lanes = _demultiplex(payload(modulation, 11))
        encoded = [ofec_encode_stream(lane) for lane in lanes]
        back = ofec_deinterleave(ofec_interleave(encoded, modulation), modulation)
        assert all(np.array_equal(a, b) for a, b in zip(back, encoded, strict=True))


def test_what_is_not_an_ofec_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="specified for"):
        ofec_frame_bits("64qam")
    with pytest.raises(ValueError, match="whole frames"):
        ofec_encode(np.zeros(100, dtype=np.uint8))
    with pytest.raises(ValueError, match="3552-bit blocks"):
        ofec_encode_stream(np.zeros(100, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def channel(bits: np.ndarray, raw_error_rate: float, seed: int) -> np.ndarray:
    """BPSK in white Gaussian noise at a given raw error rate, as log-likelihood ratios."""
    low, high = 0.01, 10.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if 0.5 * math.erfc(middle / math.sqrt(2.0)) > raw_error_rate:
            low = middle
        else:
            high = middle
    sigma = 1.0 / low
    received = (1.0 - 2.0 * bits) + np.random.default_rng(seed).normal(0.0, sigma, bits.size)
    return np.asarray(2.0 * received / sigma**2)


def test_a_clean_frame_decodes_to_itself_through_the_whole_chain() -> None:
    bits = payload("qpsk", 5)
    line = ofec_encode(bits, modulation="qpsk")
    result = ofec_decode(np.where(line > 0, -8.0, 8.0), modulation="qpsk", iterations=1)
    assert np.array_equal(result.payload, bits)
    assert result.corrections == 0


BLOCKS = 40
SETTLED = (BLOCKS - 12) * INFORMATION_BITS_PER_BLOCK


@pytest.fixture(scope="module")
def stream() -> tuple[np.ndarray, np.ndarray]:
    information = np.random.default_rng(1).integers(
        0, 2, BLOCKS * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8
    )
    return information, ofec_encode_stream(information)


def test_one_percent_raw_errors_come_out_clean(stream: tuple[np.ndarray, np.ndarray]) -> None:
    information, encoded = stream
    llr = channel(encoded, 0.01, seed=2)
    assert float(np.mean((llr < 0) != encoded)) == pytest.approx(0.01, rel=0.05)
    decoded, corrections, _ = ofec_decode_stream(llr)
    assert np.array_equal(decoded[:SETTLED], information[:SETTLED])
    assert corrections > 1000


def test_at_the_specified_threshold_three_passes_of_chase_four_are_not_enough(
    stream: tuple[np.ndarray, np.ndarray],
) -> None:
    """The specification quotes 2.0e-2 for its decoders; this one needs more to get there.

    With the three iterations the specification quotes and four Chase test bits,
    a raw error rate of 2 % leaves about one bit in eighty wrong. Six test bits
    over six passes take it to seventeen in a hundred thousand: the code can reach
    the specification's threshold, and the gap is the decoder's.
    """
    information, encoded = stream
    llr = channel(encoded, 0.02, seed=4)
    quick, _, _ = ofec_decode_stream(llr)
    thorough, _, _ = ofec_decode_stream(llr, iterations=6, test_bits=6)
    quick_errors = int(np.count_nonzero(quick[:SETTLED] != information[:SETTLED]))
    thorough_errors = int(np.count_nonzero(thorough[:SETTLED] != information[:SETTLED]))
    assert quick_errors > 500
    assert thorough_errors < 60
