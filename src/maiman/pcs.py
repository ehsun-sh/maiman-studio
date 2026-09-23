"""Probabilistic constellation shaping: the W-Port's DPO path, clause 9.

:mod:`maiman.wport` is the framing either side of OFEC and :mod:`maiman.ofec` is
the code. Between them, on the DPO modes, sits a shaper. It is what makes
FlexO-6(e), FlexO-8e and FlexO-8 *DPO* different from the DO modes: the bits do
not go into the encoder as they arrive, they are transcoded first so that the
constellation's inner points come up more often than its outer ones.

    Open ROADM MSA 6.0 W-Port Digital Specification, Rev 1.0.1, clause 8.5
    (the adaptation, which :mod:`maiman.wport` does) and clause 9.

**What the shaping buys.** A 16QAM symbol's amplitude on each axis is one bit of
the pair. Shaping makes that bit a 1 -- the inner amplitude -- more often than a
0: 0.623 of the time for FlexO-8, 0.681 for FlexO-8e, 0.8265 for FlexO-6(e),
which is Table 4's quoted figure and what the tests measure coming out of the
tables. Lower mean power for the same minimum distance, at the cost of spectral
efficiency: 3.281 bits a symbol instead of four.

**How.** The scrambled stream is split a bit at a time between the four
encoders, each encoder's share is cut into 84 groups of one coder block each,
and each group alternates chunks of *shaping* bits with runs of *sign* bits::

    b=116:  [464][512][464][512][464][480][464]   = 3360 bits
    b=106:  [424][512][424][512][424][480][424]   = 3200
    b=72:   [288][512][288][512][288][480][288]   = 2656

Each 4b-bit chunk feeds four lookup arrays round robin -- array ``j`` takes every
fourth bit -- and each array turns its ``b`` bits into 128. Those ``b`` bits are
cut into twelve fields, four for the 10-bit table and eight for the 11-bit one,
each read **least significant bit first** as an index; the table's row is the
field's output. A table is a bijection on its words ordered by falling weight,
so an input shorter than the index -- 9 bits into a 10-bit table -- reaches only
its heavy half, and that is the whole trick: fewer indices, all of them
low-energy words.

Each 128-bit output is then rewired, a different permutation per polarization
and quadrature and per encoder, to keep the bit classes' power balance; the four
are bit-interleaved into a 512-bit column. Four columns of shaping and 1,504
sign bits are the 3,552 an encoder takes.

**The tables are not here.** ``SCS_LUT10``, ``SCS_LUT11`` and the rewiring table
are normative data published inside the specification document rather than
derived from anything -- their within-weight ordering is not any enumeration
this project could reproduce, and it was checked against the obvious ones. They
carry no licence to redistribute, so, exactly as the OFEC test vectors are
handled, they are not in this repository: point ``MAIMAN_WPORT_TABLES`` at a
directory holding ``SCS_LUT10.txt``, ``SCS_LUT11.txt`` and ``rewire.json`` and
this module works. Without them it raises and says so.

**Where this stops.** At the encoder's input, test point TP4, which is where the
specification's own vectors are checked against. What the DPO path then does to
the *encoder* -- its output ordering differs from the DO path's, and clause
9.2.4 permutes the last 35 bits of each codeword back through it -- is not here,
so a DPO transmitter cannot yet be run end to end. Said plainly rather than
approximated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ofec import (
    ofec_decode_stream,
    ofec_deinterleave,
    ofec_encode_stream,
    ofec_interleave,
)
from .wport import (
    DSP_FRAME_SYMBOLS,
    PAYLOAD_SYMBOLS,
    SHAPED,
    codec_bits,
    codec_payload,
    dsp_deframe,
    dsp_frame,
    flexo_deadapt,
    scramble,
    symbol_bits,
    symbol_levels,
)

__all__ = [
    "AMPLITUDE_BITS",
    "ENCODER_BITS",
    "LUT_WIDTHS",
    "SIGN_BITS",
    "TABLES_ENV",
    "TAIL_PERMUTATION",
    "DPOReception",
    "ShapingTables",
    "amplitude_and_sign",
    "dpo_receive",
    "dpo_transmit",
    "group_layout",
    "load_tables",
    "shaped_encode",
    "shaped_lanes",
    "unshape_lanes",
]

#: Where to find the specification's own tables. See the module docstring.
TABLES_ENV = "MAIMAN_WPORT_TABLES"

#: The four axes a rewiring function is defined for, in the order the four
#: lookup arrays of a chunk take them.
DIMENSIONS = ("XI", "YI", "XQ", "YQ")

#: How the ``b`` shaping bits of one lookup are cut into fields, per mode.
#: Four for the 10-bit table then eight for the 11-bit one; the widths differ by
#: mode because the shaping rate does.
LUT_WIDTHS = {
    "b72": [6] * 12,
    "b106": [8, 8] + [9] * 10,
    "b116": [9] * 4 + [10] * 8,
}

#: Bits one encoder takes for one coder block, and what they are made of.
ENCODER_BITS = 3552
AMPLITUDE_BITS = 2048
SIGN_BITS = 1504
#: Coder blocks in a group, and lookups in a chunk.
BLOCKS = 84
ARRAYS = 4
#: Sign runs in a group, in order.
SIGN_RUNS = (512, 512, 480)


def lut_bits(modulation: str) -> int:
    """``b``: shaping bits one lookup takes, which is what the mode is named for."""
    if modulation not in LUT_WIDTHS:
        raise ValueError(f"modulation must be one of {sorted(LUT_WIDTHS)}, got {modulation!r}")
    return sum(LUT_WIDTHS[modulation])


def group_layout(
    modulation: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """``(shaping, sign)`` spans of one coder block's share, as ``(offset, length)``.

    Clause 9.2's figure, which alternates a chunk of shaping bits with a run of
    sign bits and ends on shaping: four chunks of ``4b`` and three runs.
    """
    chunk = ARRAYS * lut_bits(modulation)
    shaping: list[tuple[int, int]] = []
    sign: list[tuple[int, int]] = []
    cursor = 0
    for index, length in enumerate(
        (chunk, SIGN_RUNS[0], chunk, SIGN_RUNS[1], chunk, SIGN_RUNS[2], chunk)
    ):
        (shaping if index % 2 == 0 else sign).append((cursor, length))
        cursor += length
    return tuple(shaping), tuple(sign)


def group_bits(modulation: str) -> int:
    """Bits of one coder block's share of one encoder's stream."""
    return ARRAYS * ARRAYS * lut_bits(modulation) + SIGN_BITS


@dataclass(frozen=True)
class ShapingTables:
    """The specification's own lookup and rewiring tables, and the block order."""

    lut10: np.ndarray
    """``(1024, 10)``: a bijection on 10-bit words, ordered by falling weight."""

    lut11: np.ndarray
    """``(2048, 11)``: the same for 11-bit words."""

    rewire: np.ndarray
    """``(4, 4, 128)``: source index per output index, by encoder then dimension."""

    block_order: np.ndarray
    """``(3552,)``: where each bit of an encoder's block comes from.

    Indices below 2,048 are that block's shaping bits in order; the rest, minus
    2,048, are its sign bits. The sign bits keep their order; the shaping bits do
    not, because the encoder's matrix has its own bit classes and the shaping and
    sign columns interleave through them. Shipped with the package because it is
    a fact about the block's layout rather than a table quoted from the document,
    and checked against the specification's vectors for every mode and encoder.
    """

    def tables(self) -> list[np.ndarray]:
        """One table per field, in the order a lookup's fields are cut."""
        return [self.lut10] * 4 + [self.lut11] * 8


@lru_cache(maxsize=4)
def _packed_order() -> np.ndarray:
    return np.load(Path(__file__).parent / "data" / "pcs_block_order.npy")


@lru_cache(maxsize=4)
def load_tables(directory: str | None = None) -> ShapingTables:
    """Read the shaping tables, from ``directory`` or from ``MAIMAN_WPORT_TABLES``.

    Cached, because they never change and a frame wants them a thousand times.
    """
    where = directory or os.environ.get(TABLES_ENV)
    if not where:
        raise RuntimeError(
            "probabilistic shaping needs the specification's own LUT and rewiring tables, "
            f"which are not in this repository. Set {TABLES_ENV} to a directory holding "
            "SCS_LUT10.txt, SCS_LUT11.txt and rewire.json -- see maiman.pcs's docstring."
        )
    root = Path(where)
    lut10 = np.loadtxt(root / "SCS_LUT10.txt", dtype=np.uint8)
    lut11 = np.loadtxt(root / "SCS_LUT11.txt", dtype=np.uint8)
    if lut10.shape != (1024, 10) or lut11.shape != (2048, 11):
        raise ValueError(
            f"expected (1024, 10) and (2048, 11) lookup tables, got {lut10.shape} and {lut11.shape}"
        )
    raw = json.loads((root / "rewire.json").read_text(encoding="utf-8"))
    rewire = np.empty((4, 4, 128), dtype=np.int64)
    for encoder in range(4):
        for index, dimension in enumerate(DIMENSIONS):
            row = raw[f"encoder{encoder}:{dimension}"]
            if sorted(row) != list(range(128)):
                raise ValueError(f"encoder{encoder}:{dimension} is not a permutation of 128")
            rewire[encoder, index] = row
    return ShapingTables(lut10, lut11, rewire, _packed_order())


def _lookup(bits: np.ndarray, widths: list[int], tables: list[np.ndarray]) -> np.ndarray:
    """One lookup: ``b`` shaping bits to the 128 the encoder sees, before rewiring."""
    pieces = []
    cursor = 0
    for width, table in zip(widths, tables, strict=True):
        field = bits[cursor : cursor + width]
        cursor += width
        # Least significant bit first, which is the direction the figure's
        # register runs and the only one that reproduces the vectors.
        index = int("".join(str(bit) for bit in field[::-1]), 2)
        pieces.append(table[index])
    return np.concatenate(pieces)


#: Bits each field of a lookup's output takes: the 10-bit table's four, then the
#: 11-bit table's eight. 4 x 10 + 8 x 11 = 128.
OUTPUT_WIDTHS = [10] * 4 + [11] * 8


def _unlookup(word: np.ndarray, widths: list[int], inverses: list[dict[int, int]]) -> np.ndarray:
    """The inverse of :func:`_lookup`: the 128 bits back to ``b``."""
    out: list[int] = []
    cursor = 0
    for width, span, inverse in zip(widths, OUTPUT_WIDTHS, inverses, strict=True):
        value = int("".join(str(bit) for bit in word[cursor : cursor + span]), 2)
        cursor += span
        index = inverse[value]
        out.extend((index >> position) & 1 for position in range(width))
    return np.array(out, dtype=np.uint8)


def amplitude_and_sign(
    payload: np.ndarray, *, modulation: str, directory: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Scrambled codec payload (TP2) to the shaped bits (TP3), per encoder.

    Returns ``(amplitude, sign)`` of shapes ``(frames * 172032, 4)`` and
    ``(frames * 126336, 4)``, the columns being the four parallel encoders --
    the shape the specification's own vectors are published in.
    """
    if modulation not in SHAPED:
        raise ValueError(f"shaping is for the DPO modes {sorted(SHAPED)}, got {modulation!r}")
    bits = np.asarray(payload, dtype=np.uint8).reshape(-1)
    span = codec_bits(modulation)
    if bits.size % span:
        raise ValueError(f"a {modulation} codec frame is {span} bits, got {bits.size}")
    kit = load_tables(directory)
    widths, tables = LUT_WIDTHS[modulation], kit.tables()
    shaping_spans, sign_spans = group_layout(modulation)
    group = group_bits(modulation)

    frames = bits.size // span
    amplitude = np.empty((frames * BLOCKS * AMPLITUDE_BITS, 4), dtype=np.uint8)
    sign = np.empty((frames * BLOCKS * SIGN_BITS, 4), dtype=np.uint8)
    for frame in range(frames):
        window = bits[frame * span : (frame + 1) * span]
        for encoder in range(4):
            stream = window[encoder::4]
            amp_at = frame * BLOCKS * AMPLITUDE_BITS
            sgn_at = frame * BLOCKS * SIGN_BITS
            for block in range(BLOCKS):
                share = stream[block * group : (block + 1) * group]
                for offset, length in shaping_spans:
                    chunk = share[offset : offset + length]
                    column = np.empty(512, dtype=np.uint8)
                    for array in range(ARRAYS):
                        word = _lookup(chunk[array::ARRAYS], widths, tables)
                        column[array::ARRAYS] = word[kit.rewire[encoder, array]]
                    amplitude[amp_at : amp_at + 512, encoder] = column
                    amp_at += 512
                for offset, length in sign_spans:
                    sign[sgn_at : sgn_at + length, encoder] = share[offset : offset + length]
                    sgn_at += length
    return amplitude, sign


def shaped_lanes(
    payload: np.ndarray, *, modulation: str, directory: str | None = None
) -> np.ndarray:
    """Scrambled codec payload (TP2) to what the four encoders take (TP4).

    ``(frames * 84 * 3552, 4)``. Each column is one OFEC encoder's stream, with
    the shaping and sign bits laid into the block's own bit classes.
    """
    amplitude, sign = amplitude_and_sign(payload, modulation=modulation, directory=directory)
    order = load_tables(directory).block_order
    blocks = amplitude.shape[0] // AMPLITUDE_BITS
    lanes = np.empty((blocks * ENCODER_BITS, 4), dtype=np.uint8)
    for block in range(blocks):
        source = np.concatenate(
            [
                amplitude[block * AMPLITUDE_BITS : (block + 1) * AMPLITUDE_BITS],
                sign[block * SIGN_BITS : (block + 1) * SIGN_BITS],
            ]
        )
        lanes[block * ENCODER_BITS : (block + 1) * ENCODER_BITS] = source[order]
    return lanes


def unshape_lanes(
    lanes: np.ndarray, *, modulation: str, directory: str | None = None
) -> np.ndarray:
    """The inverse of :func:`shaped_lanes`: the encoders' bits back to the payload."""
    if modulation not in SHAPED:
        raise ValueError(f"shaping is for the DPO modes {sorted(SHAPED)}, got {modulation!r}")
    taken = np.asarray(lanes, dtype=np.uint8)
    if taken.ndim != 2 or taken.shape[1] != 4 or taken.shape[0] % ENCODER_BITS:
        raise ValueError(
            f"expected whole blocks of {ENCODER_BITS} bits by four encoders, got {taken.shape}"
        )
    kit = load_tables(directory)
    widths = LUT_WIDTHS[modulation]
    inverses = [
        {int("".join(map(str, row)), 2): index for index, row in enumerate(table)}
        for table in kit.tables()
    ]
    shaping_spans, sign_spans = group_layout(modulation)
    group = group_bits(modulation)
    blocks = taken.shape[0] // ENCODER_BITS
    frames = blocks // BLOCKS
    span = codec_bits(modulation)
    out = np.empty(frames * span, dtype=np.uint8)
    inverse_order = np.empty_like(kit.block_order)
    inverse_order[kit.block_order] = np.arange(kit.block_order.size)
    for frame in range(frames):
        window = np.empty(span, dtype=np.uint8)
        for encoder in range(4):
            stream = np.empty(span // 4, dtype=np.uint8)
            for block in range(BLOCKS):
                source = taken[
                    (frame * BLOCKS + block) * ENCODER_BITS : (frame * BLOCKS + block + 1)
                    * ENCODER_BITS,
                    encoder,
                ][inverse_order]
                amplitude = source[:AMPLITUDE_BITS]
                sign = source[AMPLITUDE_BITS:]
                share = np.empty(group, dtype=np.uint8)
                for index, (offset, length) in enumerate(shaping_spans):
                    column = amplitude[index * 512 : (index + 1) * 512]
                    chunk = np.empty(length, dtype=np.uint8)
                    for array in range(ARRAYS):
                        word = np.empty(128, dtype=np.uint8)
                        word[kit.rewire[encoder, array]] = column[array::ARRAYS]
                        chunk[array::ARRAYS] = _unlookup(word, widths, inverses)
                    share[offset : offset + length] = chunk
                cursor = 0
                for offset, length in sign_spans:
                    share[offset : offset + length] = sign[cursor : cursor + length]
                    cursor += length
                stream[block * group : (block + 1) * group] = share
            window[encoder::4] = stream
        out[frame * span : (frame + 1) * span] = window
    return out


# ---------------------------------------------------------------------------
# Clause 9.2.4: the post-encode 35-bit permute
# ---------------------------------------------------------------------------

#: Clause 9.2.4's permutation of a codeword's last 35 bits -- eighteen
#: information bits and all seventeen parity ones -- which the shaped modes
#: apply after the parity is computed, "to preserve the bit classes through the
#: OFCBG". Entry ``j`` says where the output's ``j``-th tail bit comes from.
#:
#: There are four orderings and a codeword takes the one its index selects,
#: ``r % 4``, which is the same period the block's own ``^ r`` twist runs on.
#: Read off the specification's TP4 and TP5 for every mode and every encoder,
#: because Figure 27's own table did not survive the document's conversion to
#: a page of run-together digits.
TAIL_PERMUTATION = np.array(
    [
        # r % 4 == 0
        [
            18,
            1,
            2,
            19,
            20,
            0,
            3,
            21,
            22,
            4,
            5,
            23,
            24,
            6,
            7,
            25,
            26,
            8,
            9,
            27,
            28,
            10,
            11,
            29,
            30,
            12,
            13,
            31,
            32,
            14,
            15,
            33,
            34,
            16,
            17,
        ],
        # r % 4 == 1
        [
            0,
            1,
            18,
            19,
            5,
            2,
            20,
            21,
            3,
            4,
            22,
            23,
            9,
            6,
            24,
            25,
            7,
            8,
            26,
            27,
            13,
            10,
            28,
            29,
            11,
            12,
            30,
            31,
            17,
            14,
            32,
            33,
            15,
            16,
            34,
        ],
        # r % 4 == 2
        [
            18,
            1,
            2,
            19,
            20,
            4,
            5,
            21,
            22,
            0,
            3,
            23,
            24,
            8,
            9,
            25,
            26,
            6,
            7,
            27,
            28,
            12,
            13,
            29,
            30,
            10,
            11,
            31,
            32,
            16,
            17,
            33,
            34,
            14,
            15,
        ],
        # r % 4 == 3
        [
            0,
            1,
            18,
            19,
            5,
            4,
            20,
            21,
            3,
            2,
            22,
            23,
            9,
            8,
            24,
            25,
            7,
            6,
            26,
            27,
            13,
            12,
            28,
            29,
            11,
            10,
            30,
            31,
            17,
            16,
            32,
            33,
            15,
            14,
            34,
        ],
    ],
    dtype=np.int64,
)


def shaped_encode(lanes: np.ndarray) -> np.ndarray:
    """The four encoders' input (TP4) to their output (TP5), with the tail permute.

    The permute is applied inside the encoder rather than to its output, because
    the rows twenty behind read these bits as their front halves: reordering the
    output afterwards would leave every parity bit that followed wrong. See
    :func:`maiman.ofec.ofec_encode_stream`.
    """
    taken = np.asarray(lanes, dtype=np.uint8)
    if taken.ndim != 2 or taken.shape[1] != 4:
        raise ValueError(f"expected four encoder lanes, got {taken.shape}")
    return np.stack(
        [ofec_encode_stream(taken[:, lane], tail=TAIL_PERMUTATION) for lane in range(4)],
        axis=1,
    )


def dpo_transmit(
    information: np.ndarray,
    *,
    modulation: str,
    directory: str | None = None,
    reserved: np.ndarray | None = None,
) -> np.ndarray:
    """A shaped mode end to end: FlexO information (TP0) to the symbols (TP7).

    The DO path's :func:`maiman.wport.wport_transmit` with the shaper and the
    tail permute in the middle. Everything after the encoder -- the interleavers,
    the symbol mapper and the DSP frame -- is the same code the DO modes use,
    because the shaping changed which constellation points are likely and
    nothing else.
    """
    payload = codec_payload(information, modulation=modulation)
    lanes = shaped_lanes(payload, modulation=modulation, directory=directory)
    line = ofec_interleave([shaped_encode(lanes)[:, lane] for lane in range(4)], "16qam")
    levels = symbol_levels(line, modulation=modulation)
    return np.concatenate(
        [
            dsp_frame(
                levels[start : start + PAYLOAD_SYMBOLS], modulation=modulation, reserved=reserved
            )
            for start in range(0, levels.shape[0], PAYLOAD_SYMBOLS)
        ]
    )


@dataclass(frozen=True)
class DPOReception:
    """What came back out of a shaped frame."""

    information: np.ndarray
    """The FlexO-x(e) information bits."""

    crc_ok: np.ndarray
    """One flag per CRC32: clause 8.2's error marking, as far as this goes."""

    corrections: int
    """Bits the decoder changed."""

    @property
    def clean(self) -> bool:
        return bool(self.crc_ok.all())


def dpo_receive(
    frame: np.ndarray,
    *,
    modulation: str,
    directory: str | None = None,
    confidence: float = 8.0,
    iterations: int = 3,
) -> DPOReception:
    """Received symbols (TP7) back to FlexO information: the whole shaped path, backwards.

    The counterpart of :func:`dpo_transmit`. The decoding is
    :func:`maiman.ofec.ofec_decode_stream` with ``tail`` set, which is where the
    permute is undone -- one view of the matrix for the front halves and another
    for the backs. What comes out of it is each encoder's input, which goes back
    through the shaper rather than through the DO path's multiplexer.

    As on the DO path, the ratios here are a hard decision dressed as a soft
    one; a receiver with a real channel behind it should form its own from its
    own noise and call the decoder itself.
    """
    symbols = np.asarray(frame)
    if symbols.ndim != 2 or symbols.shape[0] % DSP_FRAME_SYMBOLS:
        raise ValueError(f"whole DSP frames of {DSP_FRAME_SYMBOLS} symbols, got {symbols.shape}")
    payloads = [
        dsp_deframe(symbols[start : start + DSP_FRAME_SYMBOLS], modulation=modulation).payload
        for start in range(0, symbols.shape[0], DSP_FRAME_SYMBOLS)
    ]
    bits = symbol_bits(np.concatenate(payloads), modulation=modulation)
    llr = np.where(bits == 1, -confidence, confidence)
    decoded = [
        ofec_decode_stream(lane, iterations=iterations, tail=TAIL_PERMUTATION)
        for lane in ofec_deinterleave(llr, "16qam")
    ]
    lanes = np.stack([bits for bits, _, _ in decoded], axis=1)
    payload = unshape_lanes(lanes, modulation=modulation, directory=directory)
    span = codec_bits(modulation)
    recovered = [
        flexo_deadapt(scramble(payload[start : start + span]), modulation=modulation)
        for start in range(0, payload.size, span)
    ]
    return DPOReception(
        information=np.concatenate([block.information for block in recovered]),
        crc_ok=np.concatenate([block.crc_ok for block in recovered]),
        corrections=sum(count for _, count, _ in decoded),
    )
