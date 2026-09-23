"""OFEC, bit-exact: the Open ROADM MSA's open forward error correction, encoder to line.

:mod:`maiman.softfec` builds the *family* OFEC belongs to -- braided BCH
component codes, Chase-decoded -- and was careful to say it was not OFEC,
because being OFEC means matching a normative document bit for bit, and that
cannot be done from memory. This module is written with the document open:

    Open ROADM MSA 6.0 W-Port Digital Specification (400G-800G), Rev 1.0.1,
    clause 10 "Open Forward Error Correction (OFEC)" -- the formal encoder
    definition of 10.1.2 and the post-encoder interleavers of 10.3.

and it is checked against that document's own test vectors, which is what
bit-exact means: the encoder reproduces test point TP5 from TP2, and the
interleavers TP6 from TP5, with no bit different, for DP-QPSK and DP-16QAM
alike. The same code is specified identically in ITU-T G.709.6, OpenZR+ and the
OIF 800ZR Implementation Agreement. (It is *not* in OIF's 400ZR agreement,
which uses a different, concatenated code; an earlier note in this project said
otherwise and was wrong.)

**The code.** Each of four parallel encoders keeps a semi-infinite matrix of
bits, 128 columns wide, in 16 x 16 square blocks. Every bit belongs to two
component codewords of an extended BCH(256, 239) code -- as the *front* half of
one and the *back* half of another -- and the fronts are read from rows twenty
square blocks back, which is what makes it block-convolutional rather than a
product code. Within a square block the bits are taken with an ``^ r`` twist
the specification credits with raising the minimum distance from 36 to at least
42. The component code's generator is ``t^16 + t^14 + t^13 + t^11 + t^10 + t^9
+ t^8 + t^6 + t^5 + t + 1`` with an overall parity bit; it is the same polynomial
:func:`maiman.softfec.bch_code` builds for ``m = 8, t = 2``, which is why the
hard component decoder here is that module's. Rate 111/128, 15.3 % overhead.

**What this module covers and what it does not.** Covered: the payload bits
entering the four encoders (test point TP2, after the specification's scrambler)
to the interleaved, block-merged bits leaving for the symbol mapper (TP6), and
the reverse. Not covered here, because it is the W-Port framing rather than the
code: the FlexO adaptation, CRC and scrambler before TP2, and the symbol
mapping, pilots and framing after TP6. Those are :mod:`maiman.wport`, which is
held to the same vectors and joins on either side of this.

The decoder is this project's, not the specification's: the specification says
only that "any of the iterative algorithms designed for turbo decoding of
Product Codes" can be adapted, and this is one -- Pyndiah's soft-in soft-out
Chase decoding, each bit's two component codewords exchanging extrinsic
information.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .softfec import BCHCode, bch_code, bch_decode_hard

__all__ = [
    "INFORMATION_BITS_PER_BLOCK",
    "OFECDecodeResult",
    "ofec_decode",
    "ofec_decode_stream",
    "ofec_deinterleave",
    "ofec_encode",
    "ofec_encode_stream",
    "ofec_frame_bits",
    "ofec_interleave",
]

#: Columns of the semi-infinite matrix, ``N``.
COLUMNS = 128
#: Square block size, ``B``.
BLOCK = 16
#: Component code: extended BCH(256, 239).
COMPONENT_LENGTH = 2 * COLUMNS
COMPONENT_MESSAGE = 239
#: Fresh information bits in each component codeword's back half, ``k - N``.
INFORMATION = COMPONENT_MESSAGE - COLUMNS
#: Encoder input and output blocks: two square-block rows of each.
INFORMATION_BITS_PER_BLOCK = 2 * BLOCK * INFORMATION  # 3552
OUTPUT_BITS_PER_BLOCK = 2 * BLOCK * COLUMNS  # 4096
#: Square-block rows between a codeword's back and the first of its front: 2G + 2N/B.
REACH = 20
#: Bits per square-block row of one encoder's matrix.
ROW_BITS = (COLUMNS // BLOCK) * BLOCK * BLOCK  # 2048
#: Bits in one OFEC codec block (OFCB), across the four encoders.
OFCB_BITS = 4 * INFORMATION_BITS_PER_BLOCK  # 14208
#: Codec blocks per frame and bits per symbol, by modulation (clause 10, 11.1).
_FORMATS = {"qpsk": (42, 4), "16qam": (84, 8)}

#: The generator the specification gives, as the bit mask ``bch_code`` builds.
_GENERATOR = sum(1 << p for p in (16, 14, 13, 11, 10, 9, 8, 6, 5, 1, 0))


def _component() -> BCHCode:
    code = bch_code(8, 2)
    if code.generator != _GENERATOR:  # pragma: no cover - a construction check
        raise AssertionError("softfec's BCH(255, 239) is not OFEC's")
    return code


_CODE = _component()


def _parity_matrix() -> np.ndarray:
    """``(239, 17)``: the parity each message bit contributes, bit 0 being ``t^254``.

    Sixteen remainder bits of ``m(t) t^16 mod g(t)``, then the overall parity
    that makes the whole 256-bit word even -- the specification's "textbook
    encoding".
    """
    rows = np.zeros((COMPONENT_MESSAGE, 17), dtype=np.uint8)
    for index in range(COMPONENT_MESSAGE):
        register = 1 << (254 - index)
        for power in range(254, 15, -1):
            if register >> power & 1:
                register ^= _GENERATOR << (power - 16)
        remainder = [(register >> (15 - j)) & 1 for j in range(16)]
        rows[index, :16] = remainder
        rows[index, 16] = (1 + sum(remainder)) & 1
    return rows


_PARITY = _parity_matrix().astype(np.int64)

# -- index arithmetic, straight from clause 10.1.2 ---------------------------
#
# The matrix V(R, C, r, c) is held flat, one square-block row after another:
# position R * 2048 + C * 256 + r * 16 + c.

_R = np.arange(BLOCK)[:, None]  # the codeword's r, down the rows of a (16, 128) table
_K = np.arange(COLUMNS)[None, :]  # the codeword's k, within a half


def _back_offsets() -> np.ndarray:
    """``V(R, [(k - 128)/16], r, (k % 16) ^ r)`` relative to row R: the back halves."""
    return (_K // BLOCK) * 256 + _R * BLOCK + ((_K % BLOCK) ^ _R)


def _front_offsets() -> tuple[np.ndarray, np.ndarray]:
    """``V((R ^ 1) - 20 + 2 [k/16], [k/16], (k % 16) ^ r, r)``: row step, then offset in row."""
    step = 2 * (_K // BLOCK) - REACH + 0 * _R
    offset = (_K // BLOCK) * 256 + ((_K % BLOCK) ^ _R) * BLOCK + _R
    return step, offset


_BACK = _back_offsets()
_FRONT_STEP, _FRONT_OFFSET = _front_offsets()


def _input_offsets(parity_of_row: int) -> np.ndarray:
    """Where the 111 fresh bits of each of a row's 16 codewords sit in its input block."""
    k = np.arange(INFORMATION)[None, :]
    row = parity_of_row * BLOCK + np.arange(BLOCK)[:, None]
    return row * (BLOCK - k // 96) + (k // BLOCK) * 512 + k % BLOCK


_INPUT = (_input_offsets(0), _input_offsets(1))


def _output_order(rows: int) -> np.ndarray:
    """Flat V index of each output bit ``y``, for ``rows`` square-block rows."""
    y = np.arange(rows * ROW_BITS)
    block, within = np.divmod(y, OUTPUT_BITS_PER_BLOCK)
    column, rest = np.divmod(within, 512)
    odd, rest = np.divmod(rest, 256)
    r, c = np.divmod(rest, BLOCK)
    row = 2 * block + odd
    return np.asarray(row * ROW_BITS + column * 256 + r * BLOCK + c)


def _front_rows(row: int) -> np.ndarray:
    return (row ^ 1) + _FRONT_STEP


# ---------------------------------------------------------------------------
# One encoder
# ---------------------------------------------------------------------------


def ofec_encode_stream(information: np.ndarray) -> np.ndarray:
    """One OFEC encoder, clause 10.1.2: input bits ``u`` to output bits ``y``.

    ``information`` is a whole number of 3552-bit input blocks; each makes one
    4096-bit output block. The stream starts where the specification's test
    vectors do, at square-block row 0, where the fronts of the first twenty rows
    are undefined and the codewords are completed with the specification's
    ``H'`` -- parity over the back half alone -- so that the output is fully
    determined.
    """
    u = np.asarray(information, dtype=np.uint8).reshape(-1)
    if u.size % INFORMATION_BITS_PER_BLOCK:
        raise ValueError(
            f"an OFEC encoder takes whole {INFORMATION_BITS_PER_BLOCK}-bit blocks, "
            f"got {u.size} bits"
        )
    blocks = u.size // INFORMATION_BITS_PER_BLOCK
    rows = 2 * blocks
    matrix = np.zeros(rows * ROW_BITS, dtype=np.uint8)
    for row in range(rows):
        fresh = u[(row // 2) * INFORMATION_BITS_PER_BLOCK + _INPUT[row % 2]]
        if row >= REACH:
            front = matrix[_front_rows(row) * ROW_BITS + _FRONT_OFFSET]
        else:
            front = np.zeros((BLOCK, COLUMNS), dtype=np.uint8)
        parity = (np.concatenate([front, fresh], axis=1).astype(np.int64) @ _PARITY) & 1
        back = np.concatenate([fresh, parity.astype(np.uint8)], axis=1)
        matrix[row * ROW_BITS + _BACK] = back
    return matrix[_output_order(rows)]


# ---------------------------------------------------------------------------
# The codec block: four encoders, intra- and inter-block interleaving, merge
# ---------------------------------------------------------------------------


def ofec_frame_bits(modulation: str) -> int:
    """Payload bits one OFEC frame carries: 42 codec blocks for DP-QPSK, 84 for DP-16QAM."""
    blocks, _ = _format(modulation)
    return blocks * OFCB_BITS


def _format(modulation: str) -> tuple[int, int]:
    try:
        return _FORMATS[modulation.lower()]
    except KeyError:
        raise ValueError(f"OFEC is specified for {sorted(_FORMATS)}; got {modulation!r}") from None


def _demultiplex(payload: np.ndarray) -> list[np.ndarray]:
    """Clause 10: bit ``j`` of each 14208-bit codec block goes to encoder ``j%2 + 2 ((j/2)%2)``.

    The specification's words are "even bits to encoders 0 and 2, odd bits to 1
    and 3"; which of each pair was settled against its test vectors, where only
    this reading reproduces them.
    """
    blocks = payload.reshape(-1, OFCB_BITS)
    return [blocks[:, lane::4].reshape(-1) for lane in (0, 1, 2, 3)]


def _multiplex(lanes: list[np.ndarray]) -> np.ndarray:
    blocks = lanes[0].size // INFORMATION_BITS_PER_BLOCK
    out = np.empty((blocks, OFCB_BITS), dtype=lanes[0].dtype)
    for lane, stream in enumerate(lanes):
        out[:, lane::4] = stream.reshape(blocks, INFORMATION_BITS_PER_BLOCK)
    return out.reshape(-1)


#: Table 8 of the specification, as printed: for each destination bit (row, column)
#: of a square block, the ``row,column`` of the encoder-output bit it takes.
_TABLE_8 = """
0,0 1,1 2,2 3,3 4,4 5,5 6,6 7,7 8,8 9,9 10,10 11,11 12,12 13,13 14,14 15,15
14,15 15,0 0,1 1,2 2,3 3,4 4,5 5,6 6,7 7,8 8,9 9,10 10,11 11,12 12,13 13,14
12,14 13,15 14,0 15,1 0,2 1,3 2,4 3,5 4,6 5,7 6,8 7,9 8,10 9,11 10,12 11,13
10,13 11,14 12,15 13,0 14,1 15,2 0,3 1,4 2,5 3,6 4,7 5,8 6,9 7,10 8,11 9,12
8,12 9,13 10,14 11,15 12,0 13,1 14,2 15,3 0,4 1,5 2,6 3,7 4,8 5,9 6,10 7,11
6,11 7,12 8,13 9,14 10,15 11,0 12,1 13,2 14,3 15,4 0,5 1,6 2,7 3,8 4,9 5,10
4,10 5,11 6,12 7,13 8,14 9,15 10,0 11,1 12,2 13,3 14,4 15,5 0,6 1,7 2,8 3,9
2,9 3,10 4,11 5,12 6,13 7,14 8,15 9,0 10,1 11,2 12,3 13,4 14,5 15,6 0,7 1,8
15,7 0,8 1,9 2,10 3,11 4,12 5,13 6,14 7,15 8,0 9,1 10,2 11,3 12,4 13,5 14,6
13,6 14,7 15,8 0,9 1,10 2,11 3,12 4,13 5,14 6,15 7,0 8,1 9,2 10,3 11,4 12,5
11,5 12,6 13,7 14,8 15,9 0,10 1,11 2,12 3,13 4,14 5,15 6,0 7,1 8,2 9,3 10,4
9,4 10,5 11,6 12,7 13,8 14,9 15,10 0,11 1,12 2,13 3,14 4,15 5,0 6,1 7,2 8,3
7,3 8,4 9,5 10,6 11,7 12,8 13,9 14,10 15,11 0,12 1,13 2,14 3,15 4,0 5,1 6,2
5,2 6,3 7,4 8,5 9,6 10,7 11,8 12,9 13,10 14,11 15,12 0,13 1,14 2,15 3,0 4,1
3,1 4,2 5,3 6,4 7,5 8,6 9,7 10,8 11,9 12,10 13,11 14,12 15,13 0,14 1,15 2,0
1,0 2,1 3,2 4,3 5,4 6,5 7,6 8,7 9,8 10,9 11,10 12,11 13,12 14,13 15,14 0,15
"""


def _parse_table_8() -> tuple[np.ndarray, np.ndarray]:
    cells = [
        [tuple(int(value) for value in cell.split(",")) for cell in line.split()]
        for line in _TABLE_8.strip().splitlines()
    ]
    rows = np.array([[cell[0] for cell in line] for line in cells])
    columns = np.array([[cell[1] for cell in line] for line in cells])
    if rows.shape != (BLOCK, BLOCK) or len(set((rows * BLOCK + columns).ravel())) != BLOCK * BLOCK:
        raise AssertionError("Table 8 is not a permutation of a square block")
    return rows, columns


_INTRA_SOURCE_ROW, _INTRA_SOURCE_COLUMN = _parse_table_8()


def _intra_block(matrix: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Clause 10.3.1 on every square block of ``(..., 16, 16)``."""
    if not inverse:
        return np.asarray(matrix[..., _INTRA_SOURCE_ROW, _INTRA_SOURCE_COLUMN])
    out = np.empty_like(matrix)
    out[..., _INTRA_SOURCE_ROW, _INTRA_SOURCE_COLUMN] = matrix
    return out


def _inter_order() -> np.ndarray:
    """Clause 10.3.2: the buffer bit (row, column) read out at each position of a block.

    84 square-block rows by 8, even rows from one encoder and odd from its pair,
    split into four subsets of 21 block rows; out go eight bits at a time from
    each subset in turn, down a bit column, before the next column.
    """
    subsets = [
        np.concatenate([np.arange(b * BLOCK, b * BLOCK + BLOCK) for b in rows])
        for rows in (range(0, 42, 2), range(1, 42, 2), range(42, 84, 2), range(43, 84, 2))
    ]
    order = []
    for column in range(COLUMNS):
        for cycle in range(42):
            for subset in subsets:
                rows = subset[cycle * 8 : cycle * 8 + 8]
                order.append(rows * COLUMNS + column)
    return np.concatenate(order)


_INTER = _inter_order()
_INTER_BITS = _INTER.size  # 172032


def _as_blocks(stream: np.ndarray) -> np.ndarray:
    """An encoder's output stream as square blocks ``(rows, 8, 16, 16)``."""
    rows = stream.size // ROW_BITS
    flat = np.empty(rows * ROW_BITS, dtype=stream.dtype)
    flat[_output_order(rows)] = stream
    return flat.reshape(rows, COLUMNS // BLOCK, BLOCK, BLOCK)


def _as_stream(blocks: np.ndarray) -> np.ndarray:
    return np.asarray(blocks.reshape(-1)[_output_order(blocks.shape[0])])


def _buffer_bits(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    """Two encoders' 42 block rows each into one 84-row buffer, as a (1344, 128) bit array."""
    buffer = np.empty((84, COLUMNS // BLOCK, BLOCK, BLOCK), dtype=even.dtype)
    buffer[0::2], buffer[1::2] = even, odd
    return buffer.transpose(0, 2, 1, 3).reshape(84 * BLOCK, COLUMNS)


def _unbuffer(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    buffer = bits.reshape(84, BLOCK, COLUMNS // BLOCK, BLOCK).transpose(0, 2, 1, 3)
    return buffer[0::2], buffer[1::2]


def ofec_interleave(lanes: list[np.ndarray], modulation: str) -> np.ndarray:
    """Four encoder outputs to the line bits of clause 10.3 and 11.1 (test point TP6).

    Each square block is permuted by Table 8; each half frame of encoders 0 and
    1 fills one inter-block buffer and of encoders 2 and 3 another; and the two
    buffers' outputs are merged ``S`` bits at a time -- 4 for DP-QPSK, 8 for
    DP-16QAM -- as the symbol mapper takes them.
    """
    _, per_symbol = _format(modulation)
    blocks = [_intra_block(_as_blocks(np.asarray(lane))) for lane in lanes]
    rows = blocks[0].shape[0]
    if rows % 42:
        raise ValueError("the inter-block interleaver takes whole half frames of 42 block rows")
    merged = []
    for start in range(0, rows, 42):
        part = slice(start, start + 42)
        first = _buffer_bits(blocks[0][part], blocks[1][part]).reshape(-1)[_INTER]
        second = _buffer_bits(blocks[2][part], blocks[3][part]).reshape(-1)[_INTER]
        merged.append(
            np.stack(
                [first.reshape(-1, per_symbol), second.reshape(-1, per_symbol)], axis=1
            ).reshape(-1)
        )
    return np.concatenate(merged)


def ofec_deinterleave(line: np.ndarray, modulation: str) -> list[np.ndarray]:
    """The inverse of :func:`ofec_interleave`, for bits or for log-likelihood ratios."""
    _, per_symbol = _format(modulation)
    values = np.asarray(line)
    if values.size % (2 * _INTER_BITS):
        raise ValueError(f"the line holds whole pairs of {_INTER_BITS}-bit interleaver blocks")
    lanes: list[list[np.ndarray]] = [[], [], [], []]
    for start in range(0, values.size, 2 * _INTER_BITS):
        pair = values[start : start + 2 * _INTER_BITS].reshape(-1, 2, per_symbol)
        for which, (a, b) in enumerate(((0, 1), (2, 3))):
            bits = np.empty(_INTER_BITS, dtype=values.dtype)
            bits[_INTER] = pair[:, which, :].reshape(-1)
            even, odd = _unbuffer(bits.reshape(84 * BLOCK, COLUMNS))
            lanes[a].append(_intra_block(even, inverse=True))
            lanes[b].append(_intra_block(odd, inverse=True))
    return [_as_stream(np.concatenate(parts)) for parts in lanes]


def ofec_encode(payload: np.ndarray, *, modulation: str = "qpsk") -> np.ndarray:
    """Payload bits (TP2) to interleaved line bits (TP6), for whole frames."""
    bits = np.asarray(payload, dtype=np.uint8).reshape(-1)
    frame = ofec_frame_bits(modulation)
    if bits.size % frame:
        raise ValueError(f"OFEC {modulation} carries whole frames of {frame} bits, got {bits.size}")
    return ofec_interleave([ofec_encode_stream(lane) for lane in _demultiplex(bits)], modulation)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OFECDecodeResult:
    """Decoded payload, and what the iterations did."""

    payload: np.ndarray
    """Decoded information bits, in the order :func:`ofec_encode` takes them."""
    corrections: int
    """Bits whose hard decision the decoder changed."""
    iterations: int
    """Passes run; fewer than asked when a pass changed nothing."""


def _extended_chase(
    llr: np.ndarray, test_bits: int, no_competitor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-in soft-out Chase-II on the extended BCH(256, 239).

    Candidates come from flipping the least reliable of the first 255 bits and
    hard-decoding BCH(255, 239); the parity bit then follows from the other 255,
    and a candidate is scored over all 256. A hard decision that is already a
    codeword -- almost every one, at an operating point -- is returned at once.
    """
    hard = (llr < 0.0).astype(np.uint8)
    reliability = np.abs(llr)
    if not int(hard.sum()) & 1:
        body, ok = bch_decode_hard(_CODE, hard[:255])
        if ok and np.array_equal(body, hard[:255]):
            magnitude = np.maximum(reliability, no_competitor)
            return hard, np.where(hard > 0, -magnitude, magnitude)

    weakest = np.argsort(reliability[:255])[:test_bits]
    candidates: list[np.ndarray] = []
    metrics: list[float] = []
    seen: set[bytes] = set()
    for flips in range(test_bits + 1):
        for combination in itertools.combinations(weakest, flips):
            trial = hard[:255].copy()
            if combination:
                trial[list(combination)] ^= 1
            body, ok = bch_decode_hard(_CODE, trial)
            if not ok:
                continue
            word = np.append(body, np.uint8(int(body.sum()) & 1))
            key = word.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(word)
            metrics.append(float(np.sum(reliability[word != hard])))
    if not candidates:
        return hard, llr.copy()
    best_index = int(np.argmin(metrics))
    best, best_metric = candidates[best_index], metrics[best_index]
    competitor = np.full(COMPONENT_LENGTH, np.inf)
    for word, metric in zip(candidates, metrics, strict=True):
        differing = word != best
        competitor[differing] = np.minimum(competitor[differing], metric)
    magnitude = np.where(
        np.isinf(competitor), np.maximum(reliability, no_competitor), competitor - best_metric
    )
    return best, np.where(best > 0, -magnitude, magnitude)


def ofec_decode_stream(
    llr: np.ndarray,
    *,
    iterations: int = 3,
    test_bits: int = 4,
    confidence: float = 0.5,
    clip: float = 3.0,
) -> tuple[np.ndarray, int, int]:
    """Iteratively decode one encoder's output. Returns ``(information, corrections, passes)``.

    ``llr`` is in the order :func:`ofec_encode_stream` emits, positive meaning
    zero. Every bit is the back of one component codeword and the front of
    another, so two extrinsic terms are kept per bit and each codeword is fed the
    channel plus only the *other* one -- the same discipline
    :func:`maiman.softfec.staircase_decode` explains. A pass decodes every
    codeword in row order; the specification's three iterations are the
    default. The first twenty rows' codewords check their back halves alone, as
    the encoder built them.

    **The last twenty rows are provisional.** Their bits are fronts of codewords
    that have not arrived, so they carry one check where every other bit has
    two; a stream that ends is weaker at its end, which is why a real decoder
    runs a window and never releases its newest rows.
    """
    channel = np.asarray(llr, dtype=np.float64).reshape(-1)
    if channel.size % OUTPUT_BITS_PER_BLOCK:
        raise ValueError(f"expected whole {OUTPUT_BITS_PER_BLOCK}-bit output blocks")
    rows = channel.size // ROW_BITS
    order = _output_order(rows)
    received = np.empty(rows * ROW_BITS)
    received[order] = channel
    scale = float(np.mean(np.abs(received)))
    bound = clip * scale if scale > 0.0 else clip
    back_extrinsic = np.zeros_like(received)
    front_extrinsic = np.zeros_like(received)
    known_zero = np.full((BLOCK, COLUMNS), bound)
    before = (received < 0.0).astype(np.uint8)

    passes = 0
    for _ in range(iterations):
        passes += 1
        moved = 0
        for row in range(rows):
            back_index = row * ROW_BITS + _BACK
            back = np.clip(
                received[back_index] + confidence * front_extrinsic[back_index], -bound, bound
            )
            if row >= REACH:
                front_index = _front_rows(row) * ROW_BITS + _FRONT_OFFSET
                front = np.clip(
                    received[front_index] + confidence * back_extrinsic[front_index], -bound, bound
                )
            else:
                front = known_zero
            words = np.concatenate([front, back], axis=1)
            for r in range(BLOCK):
                decided, soft = _extended_chase(words[r], test_bits, bound)
                extrinsic = soft - words[r]
                moved += int(np.count_nonzero(decided != (words[r] < 0.0)))
                back_extrinsic[back_index[r]] = extrinsic[COLUMNS:]
                if row >= REACH:
                    front_extrinsic[front_index[r]] = extrinsic[:COLUMNS]
        if moved == 0:
            break

    total = received + confidence * (back_extrinsic + front_extrinsic)
    decided = (total < 0.0).astype(np.uint8)
    corrections = int(np.count_nonzero(decided != before))
    information = np.empty(rows // 2 * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8)
    for row in range(rows):
        fresh = decided[row * ROW_BITS + _BACK][:, :INFORMATION]
        information[(row // 2) * INFORMATION_BITS_PER_BLOCK + _INPUT[row % 2]] = fresh
    return information, corrections, passes


def ofec_decode(
    llr: np.ndarray, *, modulation: str = "qpsk", iterations: int = 3, test_bits: int = 4
) -> OFECDecodeResult:
    """Line log-likelihood ratios (TP6 order) back to payload bits (TP2 order)."""
    lanes = ofec_deinterleave(np.asarray(llr, dtype=np.float64), modulation)
    decoded = [
        ofec_decode_stream(lane, iterations=iterations, test_bits=test_bits) for lane in lanes
    ]
    return OFECDecodeResult(
        payload=_multiplex([bits for bits, _, _ in decoded]),
        corrections=sum(count for _, count, _ in decoded),
        iterations=max(passes for _, _, passes in decoded),
    )
