"""The W-Port framing around OFEC: a FlexO information structure in, a DSP frame out.

:mod:`maiman.ofec` is the code -- test point TP2 to TP6, the four encoders and the
interleavers -- and it said in its own docstring what it did not cover: *"the
FlexO adaptation, CRC and scrambler before TP2, and the symbol mapping, pilots
and framing after TP6 -- those are the W-Port framing, not the code."* This
module is that framing, from the same document:

    Open ROADM MSA 6.0 W-Port Digital Specification (400G-800G), Rev 1.0.1,
    clause 8 (FEC adaptation), clause 11 (symbol mapping and polarization
    distribution) and clause 12 (DSP framing).

and it is held to the same standard: the specification's own test vectors, TP0
through TP7, with no bit and no symbol different. The specification says
compliance *only* needs the final output TP7 from the input TP0 -- the points
between are implementation choices -- and this reproduces every one of them.

**The chain, for FlexO-x(e)-DO.** Each stage is a function here, and the arrows
are the test points::

    TP0  58 x 10,280 bits of FlexO-4(e)      (116 rows for FlexO-8(e))
         flexo_adapt:  a CRC32 every four rows, then a CRC32 and zero pad
    TP1  596,736 bits                        (1,193,472)
         scramble:     x^16 + x^12 + x^3 + x + 1, reset to 0xFFFF per block
    TP2  the OFEC codec payload -- maiman.ofec takes it from here
    TP6  688,128 line bits                   (1,376,256)
         symbol_levels: four bits to one DP-QPSK symbol, eight to a DP-16QAM one
         dsp_frame:     FAW, training, reserved and pilots inserted
    TP7  175,104 symbols in each polarization

**What the frame is made of.** 24 subframes of 7,296 symbols. A pilot every 64
symbols, from a PRBS10 reset at the head of each subframe; 11 training symbols
at each subframe's head, the first of which *is* that subframe's first pilot;
and in the first subframe only, a 22-symbol frame alignment word and 74 reserved
symbols. That leaves 172,032 payload symbols, which is exactly what the symbol
mapper made: 2,736 pilots, 240 training symbols that are not pilots, 22 + 74 of
alignment and reserve, and the sum is the 175,104 a DSP frame holds.

**What is not here.** The DPO path: FlexO-6(e)/8(e) with probabilistic
constellation shaping (clause 9), whose adaptation runs on 2,056-bit rows
through a shaping LUT and a 35-bit permute. The published vectors this was
checked against are the DO ones, and a shaping LUT written from memory would be
the thing :mod:`maiman.ofec` was careful not to do. Error marking (clause 8.2)
is reported here as a per-CRC32 flag rather than expanded into 802.3 error
blocks, which is a client-layer action this project has no client layer for.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .ofec import ofec_decode, ofec_encode

__all__ = [
    "DSP_FRAME_SYMBOLS",
    "FLEXO_ROW_BITS",
    "PAYLOAD_SYMBOLS",
    "PILOT_SPACING",
    "SHAPED",
    "DSPFrame",
    "WPortReception",
    "codec_bits",
    "codec_payload",
    "crc32",
    "dsp_deframe",
    "dsp_frame",
    "faw_symbols",
    "flexo_adapt",
    "flexo_deadapt",
    "flexo_rows",
    "information_bits",
    "jones",
    "line_modulation",
    "pilot_symbols",
    "scramble",
    "symbol_bits",
    "symbol_levels",
    "training_symbols",
    "wport_receive",
    "wport_transmit",
]

# ---------------------------------------------------------------------------
# Clause 8: FlexO adaptation
# ---------------------------------------------------------------------------

#: One bit-column row of the FlexO-x(e) information structure: 10,280 bits on the
#: DO path, 2,056 on the DPO one, which is the same structure read in narrower
#: columns. Twenty of the narrow rows are four of the wide ones, which is why the
#: CRC32 below covers the same 41,120 bits either way.
FLEXO_ROW_BITS = {"qpsk": 10280, "16qam": 10280, "b72": 2056, "b106": 2056, "b116": 2056}
#: Bits a CRC32 covers, clause 8.1: four wide rows, or twenty narrow ones.
CRC_SPAN = 41120
CRC_BITS = 32
#: Rows that go into one coder block group, clause 8.4 and clause 8.5.
FLEXO_ROWS = {"qpsk": 58, "16qam": 116, "b72": 433, "b106": 522, "b116": 548}
#: Bits the codec takes, which is those rows plus the CRC32s and the pad. The DO
#: modes' number is :func:`maiman.ofec.ofec_frame_bits`; the DPO modes shape
#: theirs up to 1,193,472 before the code sees it, so it is smaller here.
CODEC_BITS = {
    "qpsk": 596736,
    "16qam": 1193472,
    "b72": 892416,
    "b106": 1075200,
    "b116": 1128960,
}
#: The DPO modes, by the LUT input width that names them: FlexO-6(e), FlexO-8e
#: and FlexO-8 with probabilistic constellation shaping. See :mod:`maiman.pcs`.
SHAPED = ("b72", "b106", "b116")


def _known(modulation: str) -> str:
    """Refuse a mode this framing is not defined for, and say which are."""
    if modulation not in FLEXO_ROWS:
        raise ValueError(f"modulation must be one of {sorted(FLEXO_ROWS)}, got {modulation!r}")
    return modulation


def flexo_rows(modulation: str) -> int:
    """Bit-column rows of FlexO-x(e) information one codec block group carries."""
    return FLEXO_ROWS[_known(modulation)]


def codec_bits(modulation: str) -> int:
    """Bits the adaptation hands on: rows, CRC32s and pad (test point TP1)."""
    return CODEC_BITS[_known(modulation)]


def information_bits(modulation: str) -> int:
    """Bits of FlexO information one frame carries (test point TP0)."""
    return flexo_rows(modulation) * FLEXO_ROW_BITS[modulation]


def line_modulation(modulation: str) -> str:
    """What the symbols on the line are: the shaped modes are DP-16QAM too.

    Clause 9's shaping changes which points of the 16QAM constellation are
    likely, not which constellation it is -- so everything from the symbol
    mapper onward treats a shaped mode as ``"16qam"``.
    """
    return "16qam" if _known(modulation) in SHAPED else modulation


def crc32(bits: np.ndarray) -> np.ndarray:
    """The IEEE 802.3 CRC32 of a bit sequence, as 32 bits, ``x^31`` first.

    Clause 8.1.1, and the procedure is the one 802.3 states rather than a
    byte-wise table: the first 32 bits of the block are complemented, the block
    is taken as the coefficients of a polynomial, multiplied by ``x^32``, divided
    by ``G(x)``, and the remainder complemented.

    Bit-serial because the blocks are 41,120 bits long and not a whole number of
    bytes in any useful alignment -- a byte-wise table would have to be fed a
    repacking of exactly this, and this is what the clause says.
    """
    data = np.asarray(bits, dtype=np.uint8).reshape(-1).copy()
    if data.size < CRC_BITS:
        raise ValueError(f"a CRC32 covers at least 32 bits, got {data.size}")
    data[:CRC_BITS] ^= 1
    register, polynomial = 0, 0x04C11DB7
    for bit in data.tolist():
        top = (register >> 31) & 1
        register = ((register << 1) & 0xFFFFFFFF) ^ (polynomial if top ^ bit else 0)
    register ^= 0xFFFFFFFF
    return np.array([(register >> (31 - i)) & 1 for i in range(CRC_BITS)], dtype=np.uint8)


def flexo_adapt(information: np.ndarray, *, modulation: str = "qpsk") -> np.ndarray:
    """One frame of FlexO information (TP0) to the codec's input width (TP1).

    Clause 8.4. A CRC32 after every four bit-column rows, a CRC32 over whatever
    rows are left, and all-zero pad up to the codec block group: 15 CRC32s and
    16 bits of pad for FlexO-4(e)'s 58 rows, 29 and 64 for FlexO-8(e)'s 116. The
    pad exists because the FlexO structure and the codec block group are not
    ratio-locked without it, and the CRC32s are what a receiver may use to mark
    errors the decoder could not repair.
    """
    bits = np.asarray(information, dtype=np.uint8).reshape(-1)
    want = information_bits(modulation)
    if bits.size != want:
        raise ValueError(f"a {modulation} frame carries {want} information bits, got {bits.size}")
    pieces: list[np.ndarray] = []
    for start in range(0, bits.size, CRC_SPAN):
        covered = bits[start : start + CRC_SPAN]
        pieces.extend((covered, crc32(covered)))
    built = np.concatenate(pieces)
    total = codec_bits(modulation)
    return np.concatenate([built, np.zeros(total - built.size, dtype=np.uint8)])


@dataclass(frozen=True)
class FlexOFrame:
    """A de-adapted frame: the information, and what its CRC32s said about it."""

    information: np.ndarray
    """The FlexO-x(e) bit-column rows, without CRC32s or pad."""

    crc_ok: np.ndarray
    """One flag per CRC32, in order; a false one marks its four rows as errored."""

    @property
    def clean(self) -> bool:
        return bool(self.crc_ok.all())


def flexo_deadapt(frame: np.ndarray, *, modulation: str = "qpsk") -> FlexOFrame:
    """The inverse of :func:`flexo_adapt`, checking each CRC32 as it goes.

    The flags are clause 8.2's error marking, stopping where this project stops:
    a false flag says the four rows it covers did not survive, and what an
    Ethernet client does about that is the client's.
    """
    bits = np.asarray(frame, dtype=np.uint8).reshape(-1)
    total = codec_bits(modulation)
    if bits.size != total:
        raise ValueError(f"a {modulation} codec frame is {total} bits, got {bits.size}")
    group = CRC_SPAN
    rows, cursor = information_bits(modulation), 0
    recovered: list[np.ndarray] = []
    checks: list[bool] = []
    taken = 0
    while taken < rows:
        span = min(group, rows - taken)
        covered = bits[cursor : cursor + span]
        carried = bits[cursor + span : cursor + span + CRC_BITS]
        recovered.append(covered)
        checks.append(bool(np.array_equal(crc32(covered), carried)))
        cursor += span + CRC_BITS
        taken += span
    return FlexOFrame(np.concatenate(recovered), np.array(checks, dtype=bool))


# ---------------------------------------------------------------------------
# Clause 8.3: the frame-synchronous scrambler
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _keystream(length: int) -> np.ndarray:
    """``x^16 + x^12 + x^3 + x + 1`` from a register of ones, as a bit sequence.

    The shift register seeded with 0xFFFF puts those sixteen ones out first, and
    every bit after them is the recurrence the polynomial states. Cached because
    a frame's worth is the same sequence every time: the scrambler is *frame
    synchronous*, reset at the head of each block rather than running on.
    """
    bits = np.ones(length, dtype=np.uint8)
    for index in range(16, length):
        bits[index] = bits[index - 1] ^ bits[index - 3] ^ bits[index - 12] ^ bits[index - 16]
    return bits


def scramble(bits: np.ndarray) -> np.ndarray:
    """Scramble -- or descramble -- one codec block, clause 8.3.

    Additive, so it is its own inverse; the register is reset to 0xFFFF before
    the first bit of every block, which is what makes a receiver able to
    synchronise on the block alone rather than having to track the transmitter's
    register. The sequence's period is 65,535 and a block is longer than that,
    so it wraps within one frame and is still a permutation of nothing -- it
    adds, it does not move anything.
    """
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return np.asarray(data ^ _keystream(data.size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Clause 11: symbol mapping and polarization distribution
# ---------------------------------------------------------------------------

#: Bits one dual-polarization symbol carries.
BITS_PER_SYMBOL = {"qpsk": 4, "16qam": 8}
#: The outer constellation point's amplitude, which every overhead symbol sits on.
OUTER = {"qpsk": 1, "16qam": 3}
#: Payload symbols in one DSP frame, per polarization.
PAYLOAD_SYMBOLS = 172032

#: Clause 11.1.1: two bits to one 16QAM amplitude, in the order the table gives.
_AMPLITUDE = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}


def symbol_levels(line: np.ndarray, *, modulation: str = "qpsk") -> np.ndarray:
    """Line bits (TP6) to ``(symbols, 4)`` of ``[XI, XQ, YI, YQ]`` amplitudes.

    Clause 11.1. The two polarizations are interleaved bit by bit rather than
    block by block -- ``c_{4i}`` and ``c_{4i+2}`` are X's in-phase and quadrature
    and ``c_{4i+1}`` and ``c_{4i+3}`` are Y's -- so a burst on one polarization
    lands on both halves of the code rather than all of it on one.

    Amplitudes are the specification's relative ones: ±1 for DP-QPSK, ±1 and ±3
    for DP-16QAM, with the Gray-like labelling of Table 10 in which the *first*
    bit of a pair chooses the sign and the second the magnitude.
    """
    bits = np.asarray(line, dtype=np.uint8).reshape(-1)
    per = BITS_PER_SYMBOL[line_modulation(modulation)]
    if bits.size % per:
        raise ValueError(f"{modulation} takes {per} bits a symbol, got {bits.size}")
    grouped = bits.reshape(-1, per)
    if modulation == "qpsk":
        levels = np.where(grouped == 1, 1, -1).astype(np.int8)
        return np.stack([levels[:, 0], levels[:, 2], levels[:, 1], levels[:, 3]], axis=1)
    table = _amplitude_table()
    return np.stack(
        [
            table[grouped[:, 0], grouped[:, 2]],
            table[grouped[:, 4], grouped[:, 6]],
            table[grouped[:, 1], grouped[:, 3]],
            table[grouped[:, 5], grouped[:, 7]],
        ],
        axis=1,
    )


@lru_cache(maxsize=1)
def _amplitude_table() -> np.ndarray:
    table = np.zeros((2, 2), dtype=np.int8)
    for (first, second), value in _AMPLITUDE.items():
        table[first, second] = value
    return table


def symbol_bits(levels: np.ndarray, *, modulation: str = "qpsk") -> np.ndarray:
    """The inverse of :func:`symbol_levels`, by nearest amplitude.

    Nearest rather than exact, so a receiver's slicer is this same function: an
    amplitude of 2.7 is a 3 and there is no other reading of it.
    """
    values = np.asarray(levels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"levels are (symbols, 4) of [XI, XQ, YI, YQ], got {values.shape}")
    if line_modulation(modulation) == "qpsk":
        bits = (values > 0.0).astype(np.uint8)
        return np.stack([bits[:, 0], bits[:, 2], bits[:, 1], bits[:, 3]], axis=1).reshape(-1)
    pairs = sorted((value, label) for label, value in _AMPLITUDE.items())
    points = np.array([value for value, _ in pairs], dtype=np.float64)
    labels = np.array([label for _, label in pairs], dtype=np.uint8)
    nearest = np.abs(values[..., None] - points[None, None, :]).argmin(axis=-1)
    chosen = labels[nearest]  # (symbols, 4, 2)
    out = np.empty((values.shape[0], 8), dtype=np.uint8)
    out[:, 0], out[:, 2] = chosen[:, 0, 0], chosen[:, 0, 1]
    out[:, 4], out[:, 6] = chosen[:, 1, 0], chosen[:, 1, 1]
    out[:, 1], out[:, 3] = chosen[:, 2, 0], chosen[:, 2, 1]
    out[:, 5], out[:, 7] = chosen[:, 3, 0], chosen[:, 3, 1]
    return out.reshape(-1)


def jones(levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` complex symbols from ``[XI, XQ, YI, YQ]`` amplitudes."""
    values = np.asarray(levels, dtype=np.float64)
    return values[:, 0] + 1j * values[:, 1], values[:, 2] + 1j * values[:, 3]


# ---------------------------------------------------------------------------
# Clause 12: the DSP frame
# ---------------------------------------------------------------------------

DSP_FRAME_SYMBOLS = 175104
SUBFRAMES = 24
SUBFRAME_SYMBOLS = DSP_FRAME_SYMBOLS // SUBFRAMES  # 7296
PILOT_SPACING = 64
PILOTS_PER_SUBFRAME = SUBFRAME_SYMBOLS // PILOT_SPACING  # 114
TRAINING_SYMBOLS = 11
FAW_SYMBOLS = 22
RESERVED_SYMBOLS = 74

#: Clause 12.3, Table 12. Each entry is the sign of ``I`` then of ``Q``.
_FAW_X = "+- ++ ++ ++ +- +- -- ++ -- -+ -+ +- -- -- -+ ++ -- +- -+ ++ -- -+"
_FAW_Y = "++ -+ -- -+ +- ++ +- +- -- +- ++ -+ -+ ++ -- ++ -- -+ +- -- +- -+"
#: Clause 12.4, Table 13. The first symbol is also that subframe's first pilot.
_TRAINING_X = "-+ ++ -+ ++ -- ++ -- -- ++ +- +-"
_TRAINING_Y = "-- -- +- -+ -+ ++ -- -+ +- ++ +-"

#: Clause 12.5, Table 14: PRBS10 and the two seeds, one per polarization.
PILOT_SEEDS = {"x": 0x34E, "y": 0x084}


def _signs(pattern: str) -> np.ndarray:
    pairs = pattern.split()
    return np.array([[1 if c == "+" else -1 for c in pair] for pair in pairs], dtype=np.int8)


def faw_symbols(modulation: str = "qpsk") -> np.ndarray:
    """The 22-symbol frame alignment word, ``(22, 4)`` of ``[XI, XQ, YI, YQ]``.

    On the constellation's outer points, and different between the
    polarizations -- which is what lets a receiver decide which of its two
    recovered polarizations is X, and which quadrature is which, from the word
    alone. That is why clause 12.6 can allow every polarization and quadrature
    mapping: the FAW resolves them all.
    """
    amplitude = OUTER[line_modulation(modulation)]
    return np.concatenate([_signs(_FAW_X), _signs(_FAW_Y)], axis=1).astype(np.int8) * amplitude


def training_symbols(modulation: str = "qpsk") -> np.ndarray:
    """The 11-symbol training sequence at the head of every subframe, ``(11, 4)``."""
    amplitude = OUTER[line_modulation(modulation)]
    return (
        np.concatenate([_signs(_TRAINING_X), _signs(_TRAINING_Y)], axis=1).astype(np.int8)
        * amplitude
    )


def _prbs10(seed: int, length: int) -> np.ndarray:
    """``x^10 + x^7 + x^3 + x + 1``, the register's ten bits out first.

    Read in the direction the specification's own figure runs: the seed appears
    at the output before anything is computed, least significant bit first, and
    the recurrence that follows is that polynomial's reciprocal -- which is the
    same register looked at from the other end, and is what its Table 15 holds.
    """
    bits = np.empty(length, dtype=np.uint8)
    for index in range(10):
        bits[index] = (seed >> index) & 1
    for index in range(10, length):
        bits[index] = bits[index - 3] ^ bits[index - 7] ^ bits[index - 9] ^ bits[index - 10]
    return bits


@lru_cache(maxsize=2)
def pilot_symbols(modulation: str = "qpsk") -> np.ndarray:
    """The 114-symbol pilot sequence of one subframe, ``(114, 4)``.

    Clause 12.5: a PRBS10 per polarization with its own seed, two bits a symbol
    -- in-phase then quadrature -- onto the outer constellation points, reset at
    the head of every subframe. The seeds were chosen so that the pilots are DC
    balanced and so that the first pilot of a subframe is also the first symbol
    of its training sequence, which is why the two tables agree on their first
    entry and why 114 pilots and 11 training symbols cost 124 slots and not 125.
    """
    amplitude = OUTER[line_modulation(modulation)]
    out = np.empty((PILOTS_PER_SUBFRAME, 4), dtype=np.int8)
    for axis, key in ((0, "x"), (2, "y")):
        bits = _prbs10(PILOT_SEEDS[key], 2 * PILOTS_PER_SUBFRAME).reshape(-1, 2)
        out[:, axis] = np.where(bits[:, 0] == 1, 1, -1)
        out[:, axis + 1] = np.where(bits[:, 1] == 1, 1, -1)
    return np.asarray(out * amplitude, dtype=np.int8)


def _pilot_slots() -> np.ndarray:
    """Positions within a subframe that carry a pilot."""
    return np.arange(0, SUBFRAME_SYMBOLS, PILOT_SPACING)


#: Overhead symbols at the head of a subframe, once the pilots are taken out:
#: ten training symbols (the eleventh is the pilot at position zero), and in the
#: first subframe the alignment word and the reserved symbols after them.
_HEAD = TRAINING_SYMBOLS - 1
_FIRST_HEAD = _HEAD + FAW_SYMBOLS + RESERVED_SYMBOLS


@dataclass(frozen=True)
class DSPFrame:
    """What one DSP frame was carrying, taken apart again."""

    payload: np.ndarray
    """``(172032, 4)`` of ``[XI, XQ, YI, YQ]``: the symbol mapper's own output."""

    faw: np.ndarray
    """``(22, 4)``: the alignment word as received."""

    training: np.ndarray
    """``(24, 11, 4)``: each subframe's training sequence, its first symbol the pilot."""

    pilots: np.ndarray
    """``(24, 114, 4)``: every pilot, subframe by subframe."""

    reserved: np.ndarray
    """``(74, 4)``: the symbols reserved for future use, which mean nothing here."""


def dsp_frame(
    payload: np.ndarray, *, modulation: str = "qpsk", reserved: np.ndarray | None = None
) -> np.ndarray:
    """Payload symbols to a whole DSP frame, clause 12: ``(175104, 4)``.

    Two insertions, in the order the specification's figure has them. First the
    alignment word, the reserved symbols and the training sequences go in, which
    takes 172,032 symbols to 172,368; then a pilot every 64 symbols, which takes
    it to 175,104. Doing it in that order is what puts the reserved symbols
    *around* a pilot rather than over it -- they run 33 to 63 and 65 to 107 of
    the first subframe, and the slot between them belongs to the pilot grid,
    which nothing is allowed to displace.

    ``reserved`` is the 74 symbols the specification sets aside and asks a
    transmitter to randomise, so that they do not stand as a tone. Left unset
    they are the constellation's ``(-, -)`` corner, which is what the
    specification's own test vectors carry; a receiver ignores them either way.
    """
    symbols = np.asarray(payload, dtype=np.int8)
    if symbols.shape != (PAYLOAD_SYMBOLS, 4):
        raise ValueError(
            f"a DSP frame carries {PAYLOAD_SYMBOLS} payload symbols, got {symbols.shape}"
        )
    amplitude = OUTER[line_modulation(modulation)]
    if reserved is None:
        reserved = np.full((RESERVED_SYMBOLS, 4), -amplitude, dtype=np.int8)
    reserved = np.asarray(reserved, dtype=np.int8)
    if reserved.shape != (RESERVED_SYMBOLS, 4):
        raise ValueError(f"{RESERVED_SYMBOLS} reserved symbols, got {reserved.shape}")

    training = training_symbols(modulation)
    faw = faw_symbols(modulation)
    pilots = pilot_symbols(modulation)

    width = SUBFRAME_SYMBOLS - PILOTS_PER_SUBFRAME
    stage = np.empty((SUBFRAMES, width, 4), dtype=np.int8)
    cursor = 0
    for index in range(SUBFRAMES):
        head = _FIRST_HEAD if index == 0 else _HEAD
        stage[index, :_HEAD] = training[1:]
        if index == 0:
            stage[index, _HEAD : _HEAD + FAW_SYMBOLS] = faw
            stage[index, _HEAD + FAW_SYMBOLS : head] = reserved
        stage[index, head:] = symbols[cursor : cursor + width - head]
        cursor += width - head
    assert cursor == PAYLOAD_SYMBOLS

    frame = np.empty((SUBFRAMES, SUBFRAME_SYMBOLS, 4), dtype=np.int8)
    slots = _pilot_slots()
    carried = np.ones(SUBFRAME_SYMBOLS, dtype=bool)
    carried[slots] = False
    frame[:, slots, :] = pilots
    frame[:, carried, :] = stage
    return frame.reshape(DSP_FRAME_SYMBOLS, 4)


def dsp_deframe(frame: np.ndarray, *, modulation: str = "qpsk") -> DSPFrame:
    """The inverse of :func:`dsp_frame`: take one DSP frame apart.

    The overhead is handed back rather than discarded, because a receiver's
    whole business with it is measurement -- the pilots are its phase reference,
    the training sequence its equaliser's, and the alignment word says which
    polarization and which quadrature it is looking at.
    """
    symbols = np.asarray(frame)
    if symbols.shape != (DSP_FRAME_SYMBOLS, 4):
        raise ValueError(f"a DSP frame is {DSP_FRAME_SYMBOLS} symbols of 4, got {symbols.shape}")
    blocks = symbols.reshape(SUBFRAMES, SUBFRAME_SYMBOLS, 4)
    slots = _pilot_slots()
    carried = np.ones(SUBFRAME_SYMBOLS, dtype=bool)
    carried[slots] = False
    pilots = blocks[:, slots, :]
    stage = blocks[:, carried, :]

    training = np.concatenate([pilots[:, :1, :], stage[:, :_HEAD, :]], axis=1)
    faw = stage[0, _HEAD : _HEAD + FAW_SYMBOLS]
    reserved = stage[0, _HEAD + FAW_SYMBOLS : _FIRST_HEAD]
    payload = np.concatenate(
        [stage[0, _FIRST_HEAD:]] + [stage[index, _HEAD:] for index in range(1, SUBFRAMES)]
    )
    del modulation  # the layout is modulation independent; only the amplitudes are not
    return DSPFrame(payload=payload, faw=faw, training=training, pilots=pilots, reserved=reserved)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def codec_payload(information: np.ndarray, *, modulation: str = "qpsk") -> np.ndarray:
    """FlexO information (TP0) to the OFEC codec's payload (TP2), frame by frame.

    Both stages before the code are *per frame*: the CRC32s and pad align one
    block group, and the scrambler is reset to 0xFFFF at the head of each. So
    this cuts the input into frames and treats each on its own, which is what
    lets a receiver synchronise on a block rather than on the stream.
    """
    bits = np.asarray(information, dtype=np.uint8).reshape(-1)
    span = information_bits(modulation)
    if bits.size % span:
        raise ValueError(f"a {modulation} frame carries {span} information bits, got {bits.size}")
    return np.concatenate(
        [
            scramble(flexo_adapt(bits[start : start + span], modulation=modulation))
            for start in range(0, bits.size, span)
        ]
    )


def wport_transmit(
    information: np.ndarray, *, modulation: str = "qpsk", reserved: np.ndarray | None = None
) -> np.ndarray:
    """FlexO-x(e) information (TP0) to the symbols on the line (TP7).

    Takes whole frames, and **more than one frame is not the same as one frame
    twice**: OFEC is block-convolutional, and each block's parity is read from
    rows twenty square blocks behind it, so the second frame's line bits depend
    on the first frame's payload. The adaptation and the scrambler reset per
    frame; the code does not, and neither does this.
    """
    line = ofec_encode(codec_payload(information, modulation=modulation), modulation=modulation)
    levels = symbol_levels(line, modulation=modulation)
    return np.concatenate(
        [
            dsp_frame(
                levels[start : start + PAYLOAD_SYMBOLS],
                modulation=modulation,
                reserved=reserved,
            )
            for start in range(0, levels.shape[0], PAYLOAD_SYMBOLS)
        ]
    )


@dataclass(frozen=True)
class WPortReception:
    """What came back out of a received frame."""

    information: np.ndarray
    """The FlexO-x(e) information bits."""

    crc_ok: np.ndarray
    """One flag per CRC32: clause 8.2's error marking, as far as this goes."""

    corrections: int
    """Bits the OFEC decoder changed."""

    frames: tuple[DSPFrame, ...]
    """Each DSP frame taken apart, overhead included."""

    @property
    def clean(self) -> bool:
        return bool(self.crc_ok.all())


def wport_receive(
    frame: np.ndarray, *, modulation: str = "qpsk", confidence: float = 8.0, iterations: int = 3
) -> WPortReception:
    """Received symbols (TP7) back to FlexO information, decoding on the way.

    ``confidence`` is the magnitude of the log-likelihood ratios the sliced bits
    are handed to the decoder with. It is a hard decision dressed as a soft one,
    which is what a frame of clean symbols deserves; a receiver with a real
    channel behind it should form its own ratios from its own noise and call
    :func:`maiman.ofec.ofec_decode` itself.
    """
    symbols = np.asarray(frame)
    if symbols.ndim != 2 or symbols.shape[0] % DSP_FRAME_SYMBOLS:
        raise ValueError(f"whole DSP frames of {DSP_FRAME_SYMBOLS} symbols, got {symbols.shape}")
    taken = [
        dsp_deframe(symbols[start : start + DSP_FRAME_SYMBOLS], modulation=modulation)
        for start in range(0, symbols.shape[0], DSP_FRAME_SYMBOLS)
    ]
    bits = symbol_bits(np.concatenate([frame.payload for frame in taken]), modulation=modulation)
    llr = np.where(bits == 1, -confidence, confidence)
    decoded = ofec_decode(llr, modulation=modulation, iterations=iterations)
    span = codec_bits(modulation)
    recovered = [
        flexo_deadapt(scramble(decoded.payload[start : start + span]), modulation=modulation)
        for start in range(0, decoded.payload.size, span)
    ]
    return WPortReception(
        information=np.concatenate([block.information for block in recovered]),
        crc_ok=np.concatenate([block.crc_ok for block in recovered]),
        corrections=decoded.corrections,
        frames=tuple(taken),
    )
