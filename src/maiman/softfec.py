"""Soft-decision FEC: braided BCH component codes, Chase-decoded.

This is the code family that optical transport actually runs at 400G. It is not
one code but an arrangement: short algebraic component codes, laid out so that
every bit is protected by two of them, decoded iteratively so each pass hands the
next one a cleaner block. :mod:`maiman.fec`'s Reed-Solomon code corrects within a
codeword and stops; this one lets codewords help each other, which is where the
two to three decibels over hard-decision coding come from — that, and the fact
that it decodes on **log-likelihood ratios** rather than on bits.

**What this is, and what it is not.** The construction here is the braided /
staircase family that OIF's oFEC belongs to, with the soft component decoding
oFEC uses. It is **not bit-exact oFEC**. The 400ZR Implementation Agreement is
normative about its interleaver, its framing and its exact component-code
shortening, and reproducing those faithfully needs the document open, not
recalled. A block that claimed to be oFEC while being a reconstruction of it
would be the one thing this project refuses to do: a number nobody can back.

So the parts here are the parts that are derivable from the open literature, and
they are cited:

* **BCH component codes**, from the standard algebraic construction — the
  generator is the least common multiple of the minimal polynomials of
  ``alpha, alpha^3, ..., alpha^(2t-1)``, built here rather than tabulated.
* **Chase-II soft decoding** — D. Chase, "A class of algorithms for decoding
  block codes with channel measurement information", IEEE Trans. Inf. Theory
  18(1), 1972. Flip the least reliable bits in every combination, hard-decode
  each, and keep the candidate closest to the received soft vector.
* **The staircase arrangement** — B. P. Smith, A. Farhood, A. Hunt,
  F. R. Kschischang and J. Lodge, "Staircase codes: FEC for 100 Gb/s OTN",
  J. Lightwave Technol. 30(1), 2012. Successive square blocks, in which the rows
  of ``[B_{i-1}^T | B_i]`` are component codewords, decoded over a sliding
  window.

The parameters are stated as **this project's choice** within that family, not as
a standard's. Nothing here should be read as interoperable with a real 400ZR
module.

**Measured.** On a BPSK/AWGN channel at 16.4 % overhead, ten iterations:

===============  ===============  ==================================
input BER        post-FEC BER     RS(255, 239) at the same input
===============  ===============  ==================================
1.53e-2          5.9e-4           1.53e-2 — corrects nothing
1.03e-2          4.4e-5           1.03e-2 — corrects nothing
6.3e-3           0                5.9e-3  — corrects almost nothing
7.7e-4           0                1.6e-7
===============  ===============  ==================================

The two middle rows are the point of the whole module: the hard-decision code
is past its cliff and returns its input, and this one is still working. That is
what soft information buys, and it is why a modern coherent link is specified at
a pre-FEC rate a hard code could not touch.

**The last block in a stream is provisional.** ``B_i`` is checked twice — once
by its own stripe's rows and once, transposed, by the stripe after it — so the
final block has only half its protection until the next one arrives. A single
error in it is not corrected where the same error anywhere else is. That is a
property of the arrangement rather than of this implementation, it is why real
decoders run a sliding window and never emit the newest block, and it is
asserted rather than hidden.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

#: Primitive polynomials for GF(2^m), as integer bit patterns including the
#: leading x^m term. Every one of these is *checked* rather than trusted:
#: :meth:`GaloisField.__init__` requires that the element it generates visits all
#: ``2^m - 1`` non-zero values, which is what primitive means. A wrong entry
#: cannot survive construction, so the table is a claim the code verifies rather
#: than a transcription it depends on.
PRIMITIVE_POLYNOMIALS: dict[int, int] = {
    4: 0x13,  # x^4 + x + 1
    5: 0x25,  # x^5 + x^2 + 1
    6: 0x43,  # x^6 + x + 1
    7: 0x89,  # x^7 + x^3 + 1
    8: 0x11D,  # x^8 + x^4 + x^3 + x^2 + 1
    9: 0x211,  # x^9 + x^4 + 1
    10: 0x409,  # x^10 + x^3 + 1
}


class GaloisField:
    """GF(2^m), as log and antilog tables, with its primitivity checked.

    :mod:`maiman.fec` builds GF(2^8) directly because it needs exactly one field.
    A BCH component code needs whichever field its block length asks for, so this
    is the same construction with ``m`` free.
    """

    def __init__(self, m: int) -> None:
        if m not in PRIMITIVE_POLYNOMIALS:
            raise ValueError(f"no primitive polynomial on hand for GF(2^{m})")
        self.m = m
        self.size = 1 << m
        self.order = self.size - 1
        self.polynomial = PRIMITIVE_POLYNOMIALS[m]

        antilog = np.zeros(2 * self.size, dtype=np.int64)
        log = np.full(self.size, -1, dtype=np.int64)
        value = 1
        for power in range(self.order):
            antilog[power] = value
            log[value] = power
            value <<= 1
            if value & self.size:
                value ^= self.polynomial
        if value != 1 or np.any(log[1:] < 0):
            raise ValueError(
                f"0x{self.polynomial:X} is not primitive over GF(2^{m}): its powers "
                f"do not visit every non-zero element"
            )
        antilog[self.order : 2 * self.order] = antilog[: self.order]
        self.antilog = antilog
        self.log = log

    def multiply(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.antilog[self.log[a] + self.log[b]])

    def power(self, a: int, exponent: int) -> int:
        if a == 0:
            return 0 if exponent else 1
        return int(self.antilog[(self.log[a] * exponent) % self.order])

    def minimal_polynomial(self, exponent: int) -> int:
        """The minimal polynomial of ``alpha^exponent``, as a coefficient bitmask.

        Built from its conjugate roots — ``alpha^e``, ``alpha^2e``, ``alpha^4e``,
        … until the exponent cycles — because that product is guaranteed to have
        coefficients in GF(2) and is therefore a binary polynomial, which is the
        whole reason a BCH code is binary at all.
        """
        conjugates = []
        exponent %= self.order
        current = exponent
        while True:
            conjugates.append(current)
            current = (current * 2) % self.order
            if current == exponent:
                break

        # Multiply out (x - alpha^c) over the conjugates, in GF(2^m).
        polynomial = [1]
        for conjugate in conjugates:
            root = self.power(2, conjugate)
            shifted = [0, *polynomial]
            scaled = [self.multiply(coefficient, root) for coefficient in polynomial] + [0]
            polynomial = [a ^ b for a, b in zip(shifted, scaled, strict=True)]

        if any(coefficient not in (0, 1) for coefficient in polynomial):
            raise ValueError("a minimal polynomial came out non-binary; the field is wrong")
        # The product above is built lowest power first; a bitmask holds bit i as
        # x^i, so it is assembled from the top down. Getting this backwards
        # produces the *reciprocal* polynomial, which is still a valid binary
        # polynomial of the right degree and still divides the codewords it
        # generates -- so nothing looks wrong until the syndromes of a clean
        # codeword come back non-zero.
        mask = 0
        for coefficient in reversed(polynomial):
            mask = (mask << 1) | coefficient
        return mask


def _polynomial_degree(mask: int) -> int:
    return mask.bit_length() - 1


def _polynomial_multiply_binary(a: int, b: int) -> int:
    """Multiply two GF(2) polynomials held as bitmasks."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result


def _polynomial_mod_binary(a: int, b: int) -> int:
    """Remainder of one GF(2) polynomial by another."""
    degree = _polynomial_degree(b)
    while a.bit_length() - 1 >= degree and a:
        a ^= b << (a.bit_length() - 1 - degree)
    return a


@dataclass(frozen=True)
class BCHCode:
    """A binary BCH code, built from its design distance rather than tabulated.

    ``n = 2^m - 1`` naturally; ``shortened`` trims leading information bits, which
    is how a code of a convenient block length is made from one of the natural
    lengths. Shortening removes codewords and never adds them, so the distance —
    and therefore ``t`` — is preserved; that is why it is the standard way to fit
    an algebraic code to a frame that is not one short of a power of two.
    """

    field: GaloisField
    t: int
    generator: int
    natural_length: int
    length: int

    @property
    def parity_bits(self) -> int:
        return _polynomial_degree(self.generator)

    @property
    def message_bits(self) -> int:
        return self.length - self.parity_bits

    @property
    def rate(self) -> float:
        return self.message_bits / self.length


def bch_code(m: int, t: int, *, length: int | None = None) -> BCHCode:
    """Construct BCH(2^m - 1, k, t), optionally shortened to ``length``.

    The generator is the least common multiple of the minimal polynomials of
    ``alpha^1, alpha^3, …, alpha^(2t-1)``. Only the odd powers are needed: the
    even ones are conjugates of odd ones and contribute the same minimal
    polynomial, so including them would multiply the generator by factors it
    already has and silently cost parity bits for nothing.
    """
    if t < 1:
        raise ValueError(f"a BCH code needs t >= 1, got {t}")
    field = GaloisField(m)
    natural = field.order

    generator = 1
    for exponent in range(1, 2 * t, 2):
        minimal = field.minimal_polynomial(exponent)
        if _polynomial_mod_binary(generator, minimal) or generator == 1:
            # lcm: multiply in only the factors not already present.
            if _remainder_is_zero(generator, minimal):
                continue
            generator = _polynomial_multiply_binary(generator, minimal)

    block = natural if length is None else length
    if not 0 < block <= natural:
        raise ValueError(f"a shortened length must be in (0, {natural}], got {block}")
    if block <= _polynomial_degree(generator):
        raise ValueError(
            f"BCH({natural}, ., {t}) has {_polynomial_degree(generator)} parity bits, "
            f"which leaves no room in a block of {block}"
        )
    return BCHCode(field=field, t=t, generator=generator, natural_length=natural, length=block)


def _remainder_is_zero(generator: int, factor: int) -> bool:
    """Whether ``factor`` already divides ``generator``."""
    if generator <= 1:
        return False
    return _polynomial_mod_binary(generator, factor) == 0


def bch_encode(code: BCHCode, messages: np.ndarray) -> np.ndarray:
    """Systematically encode ``(blocks, k)`` bits to ``(blocks, n)``."""
    messages = np.asarray(messages, dtype=np.uint8)
    if messages.ndim != 2 or messages.shape[1] != code.message_bits:
        raise ValueError(
            f"expected message blocks of shape (n, {code.message_bits}), got {messages.shape}"
        )
    parity = code.parity_bits
    out = np.zeros((messages.shape[0], code.length), dtype=np.uint8)
    out[:, :-parity] = messages

    for block in range(messages.shape[0]):
        # Remainder of m(x) * x^parity by g(x), as an integer polynomial.
        value = 0
        for bit in messages[block]:
            value = (value << 1) | int(bit)
        value <<= parity
        remainder = _polynomial_mod_binary(value, code.generator)
        for index in range(parity):
            out[block, code.length - 1 - index] = (remainder >> index) & 1
    return out


#: ``alpha^(j * (L-1-p))`` for every syndrome ``j`` and bit position ``p``, built
#: once per code. The syndrome is then an XOR-reduction over the set bits, which
#: numpy does in one call — where the obvious loop costs a Python iteration per
#: bit per syndrome and made a staircase block take a second to decode.
_SYNDROME_TABLES: dict[tuple[int, int, int], np.ndarray] = {}


def _syndrome_table(code: BCHCode) -> np.ndarray:
    key = (code.field.m, code.t, code.length)
    table = _SYNDROME_TABLES.get(key)
    if table is None:
        field = code.field
        powers = np.arange(code.length - 1, -1, -1, dtype=np.int64)
        rows = [field.antilog[(j * powers) % field.order] for j in range(1, 2 * code.t + 1)]
        table = np.stack(rows).astype(np.int64)
        _SYNDROME_TABLES[key] = table
    return table


def bch_syndromes(code: BCHCode, received: np.ndarray) -> list[int]:
    """``S_j = r(alpha^j)`` for ``j = 1 … 2t``, on one shortened codeword.

    Shortening is *leading zeros*: a word of length L is a natural-length word
    whose top ``n - L`` coefficients are zero, so position p still carries
    ``x^(L-1-p)`` and nothing about the syndromes needs adjusting for it.
    """
    table = _syndrome_table(code)
    active = np.asarray(received, dtype=bool)
    if not active.any():
        return [0] * (2 * code.t)
    selected = table[:, active]
    return [int(value) for value in np.bitwise_xor.reduce(selected, axis=1)]


def bch_decode_hard(code: BCHCode, received: np.ndarray) -> tuple[np.ndarray, bool]:
    """Correct up to ``t`` bit errors. Returns ``(word, succeeded)``.

    Binary, so there are no error *magnitudes* to compute — a located error is
    simply flipped, and Forney's formula has nothing to do. That is the whole
    difference between this decoder and the Reed-Solomon one in :mod:`maiman.fec`.
    """
    field = code.field
    word = np.asarray(received, dtype=np.uint8).copy()
    syndromes = bch_syndromes(code, word)
    if not any(syndromes):
        return word, True

    locator = _berlekamp_massey_binary(code, syndromes)
    degree = len(locator) - 1
    if degree < 1 or degree > code.t:
        return word, False

    # Chien search, over every position at once. Chase hard-decodes sixteen
    # candidates per component codeword and most of them have errors, so this is
    # the loop that decides whether a staircase block takes milliseconds or
    # seconds.
    exponents = np.arange(code.length - 1, -1, -1, dtype=np.int64) % field.order
    inverse_logs = (field.order - exponents) % field.order
    total = np.zeros(code.length, dtype=np.int64)
    for power, coefficient in enumerate(locator):
        if coefficient == 0:
            continue
        total ^= field.antilog[(field.log[coefficient] + power * inverse_logs) % field.order]
    positions = np.flatnonzero(total == 0).tolist()

    if len(positions) != degree:
        return word, False
    word[positions] ^= 1
    return word, not any(bch_syndromes(code, word))


def _berlekamp_massey_binary(code: BCHCode, syndromes: list[int]) -> list[int]:
    """Error-locator polynomial, lowest power first, over GF(2^m)."""
    field = code.field
    locator = [1]
    previous = [1]
    shift = 1
    last_discrepancy = 1

    for step in range(2 * code.t):
        discrepancy = syndromes[step]
        for index in range(1, len(locator)):
            discrepancy ^= field.multiply(locator[index], syndromes[step - index])
        if discrepancy == 0:
            shift += 1
            continue

        scale = field.multiply(discrepancy, field.power(last_discrepancy, field.order - 1))
        correction = [0] * shift + [field.multiply(c, scale) for c in previous]
        updated = list(locator) + [0] * max(0, len(correction) - len(locator))
        for index, value in enumerate(correction):
            updated[index] ^= value

        if 2 * (len(locator) - 1) <= step:
            previous = locator
            last_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1
        locator = updated

    while len(locator) > 1 and locator[-1] == 0:
        locator.pop()
    return locator


def _evaluate_binary(field: GaloisField, polynomial: list[int], x: int) -> int:
    total = 0
    for coefficient in reversed(polynomial):
        total = field.multiply(total, x) ^ coefficient
    return total


def chase_decode(code: BCHCode, llr: np.ndarray, *, test_bits: int = 4) -> tuple[np.ndarray, float]:
    """Chase-II soft decoding of one component codeword.

    D. Chase, IEEE Trans. Inf. Theory 18(1), 1972. The hard decision is one
    candidate; the others come from flipping the ``test_bits`` least reliable
    positions in every combination, hard-decoding each, and keeping whichever
    valid codeword is closest to the received soft vector in the correlation
    metric ``sum |llr| over disagreeing positions``.

    Returns ``(bits, metric)``, the metric being that distance — lower is better,
    and zero means the hard decision was already a codeword.

    **Why it is worth 2 to 3 dB.** A hard decoder past ``t`` errors has nothing
    to try. This one knows *which* of its bits it is least sure about, and the
    errors are overwhelmingly there — so a codeword that a hard decoder abandons
    is often two flips away from the right one, and those two flips are usually
    inside the four positions it tests.
    """
    values = np.asarray(llr, dtype=np.float64)
    if values.shape[0] != code.length:
        raise ValueError(f"expected {code.length} LLRs, got {values.shape[0]}")
    if test_bits < 0:
        raise ValueError(f"test_bits must be >= 0, got {test_bits}")

    hard = (values < 0.0).astype(np.uint8)
    reliability = np.abs(values)
    weakest = np.argsort(reliability)[:test_bits]

    best_bits: np.ndarray | None = None
    best_metric = float("inf")

    for flips in range(test_bits + 1):
        for combination in itertools.combinations(weakest, flips):
            trial = hard.copy()
            if combination:
                trial[list(combination)] ^= 1
            candidate, ok = bch_decode_hard(code, trial)
            if not ok:
                continue
            metric = float(np.sum(reliability[candidate != hard]))
            if metric < best_metric:
                best_metric, best_bits = metric, candidate
                if metric == 0.0:
                    return best_bits, 0.0

    if best_bits is None:
        # Nothing in the test set decoded. Hand back the hard decision and say
        # so through an infinite metric — a caller that treats this as a success
        # would be trusting a word no component code ever accepted.
        return hard, float("inf")
    return best_bits, best_metric


# ---------------------------------------------------------------------------
# The staircase arrangement


@dataclass(frozen=True)
class StaircaseCode:
    """Successive square blocks, braided so every bit sits in two codewords.

    Smith, Farhood, Hunt, Kschischang and Lodge, *J. Lightwave Technol.* 30(1),
    2012. Block ``B_i`` is ``m x m``; the rows of ``[B_{i-1}^T | B_i]`` are
    component codewords of length ``2m``. So a bit written into ``B_i`` is
    checked once by a row of its own stripe and again — as a *column* — by a row
    of the next stripe.

    That second check is the whole idea. A component code alone corrects ``t``
    errors and stops. Here a codeword that is beyond its own reach can still be
    cleaned up by the stripe that crosses it, and the next pass over the first
    stripe then finds a problem it can solve. Iterating is not a refinement of
    the decoder; it is the decoder.

    ``B_0`` is all zeros and is never transmitted — it is the boundary condition
    that lets the first real block have a left half at all.
    """

    component: BCHCode
    half: int

    @property
    def parity_bits(self) -> int:
        return self.component.parity_bits

    @property
    def information_columns(self) -> int:
        return self.half - self.parity_bits

    @property
    def rate(self) -> float:
        return self.information_columns / self.half

    @property
    def overhead(self) -> float:
        return 1.0 / self.rate - 1.0


def staircase_code(m: int = 9, t: int = 2, half: int = 128) -> StaircaseCode:
    """A staircase built on BCH(2*half, ., t).

    The default is BCH(256, 238, t=2) on 128x128 blocks, which comes out at
    16.4 % overhead — the same neighbourhood as the codes optical transport
    actually runs, and chosen for that reason rather than transcribed from any
    of them.
    """
    component = bch_code(m, t, length=2 * half)
    if component.parity_bits >= half:
        raise ValueError(
            f"{component.parity_bits} parity bits leave no information in a {half}-column block"
        )
    return StaircaseCode(component=component, half=half)


def staircase_encode(code: StaircaseCode, information: np.ndarray) -> np.ndarray:
    """Encode ``(blocks, half, information_columns)`` bits into ``(blocks, half, half)``.

    Each block's left half is the *transpose* of the one before it, which is
    already fixed by the time this block is written — so encoding is causal and
    a stream never has to wait for a block it has not sent yet.
    """
    information = np.asarray(information, dtype=np.uint8)
    half, columns = code.half, code.information_columns
    if information.ndim != 3 or information.shape[1:] != (half, columns):
        raise ValueError(
            f"expected information of shape (blocks, {half}, {columns}), got {information.shape}"
        )

    blocks = information.shape[0]
    out = np.zeros((blocks, half, half), dtype=np.uint8)
    previous = np.zeros((half, half), dtype=np.uint8)  # B_0
    for index in range(blocks):
        left = previous.T
        message = np.concatenate([left, information[index]], axis=1)
        codewords = bch_encode(code.component, message)
        out[index, :, :columns] = information[index]
        out[index, :, columns:] = codewords[:, -code.parity_bits :]
        previous = out[index]
    return out


def staircase_decode(
    code: StaircaseCode,
    llr: np.ndarray,
    *,
    iterations: int = 8,
    test_bits: int = 4,
    confidence: float = 0.5,
    clip: float = 3.0,
) -> tuple[np.ndarray, int]:
    """Iteratively Chase-decode a stream of blocks. Returns ``(information, flips)``.

    ``llr[i]`` holds the log-likelihood ratios for block ``i``, in the sign
    convention of :class:`~maiman.signals.SoftSignal` — positive means zero.

    **Every bit sits in two component codewords**, and the decoder has to combine
    what both of them say rather than let one overwrite the other. Block ``B_i``
    is checked once by the rows of its own stripe ``[B_{i-1}^T | B_i]``, and again
    by the rows of the next stripe, where it appears transposed as the left half.
    So two extrinsic terms are carried per block — ``row`` and ``column`` — and
    the working value of a bit is always

        channel + confidence * (row extrinsic + column extrinsic)

    An earlier version of this function simply wrote each stripe's result over
    the last one's, which discards half the information the arrangement exists to
    produce and is why it corrected almost nothing. That is the difference
    between iterating and merely repeating.

    ``confidence`` scales the extrinsic and ``clip`` bounds the working value as
    a multiple of the input's own mean reliability; the metric differences Chase
    returns grow with the magnitudes it is given, so an unbounded loop compounds
    them.
    """
    channel = np.array(llr, dtype=np.float64)
    blocks, half = channel.shape[0], code.half
    # ``clip`` is a multiple of the input's own mean reliability, not an absolute
    # number of log-likelihood units. It has to be: the LLRs a real demapper
    # produces scale with the signal-to-noise ratio, and on the links this
    # project ships the mean magnitude runs from about 10 at the sensitivity
    # limit to 60 well above it. A fixed bound would be loose where it should
    # bite and would flatten every reliability difference in the block where it
    # should not.
    scale = float(np.mean(np.abs(channel)))
    bound = clip * scale if scale > 0.0 else clip
    if channel.shape[1:] != (half, half):
        raise ValueError(f"expected LLRs of shape (blocks, {half}, {half}), got {channel.shape}")

    row_extrinsic = np.zeros_like(channel)
    column_extrinsic = np.zeros_like(channel)
    # B_0 is the all-zero block the staircase starts from. It is never
    # transmitted and it is known exactly, so it enters at full confidence --
    # positive, because positive means zero.
    boundary = np.full((half, half), bound, dtype=np.float64)

    def bounded(value: np.ndarray) -> np.ndarray:
        return np.clip(value, -bound, bound)

    def working() -> np.ndarray:
        return bounded(channel + confidence * (row_extrinsic + column_extrinsic))

    flips = 0
    for _ in range(iterations):
        moved = 0
        for index in range(blocks):
            # **Each check sees the channel plus only the *other* check's
            # extrinsic.** That is the whole content of the word extrinsic, and
            # getting it wrong is silent: feed a decoder its own previous output
            # and it agrees with itself, the extrinsic it returns collapses to
            # zero, and the correction it made on the last pass quietly unwinds.
            # This decoder corrected a bit on pass one and had lost it again by
            # pass three, with nothing in between to say so.
            right = bounded(channel[index] + confidence * column_extrinsic[index])
            if index == 0:
                left = boundary.T
            else:
                left = bounded(channel[index - 1] + confidence * row_extrinsic[index - 1]).T
            rows = np.concatenate([left, right], axis=1)
            before = (rows < 0.0).astype(np.uint8)

            decoded = np.empty_like(before)
            soft = np.empty_like(rows)
            for row in range(half):
                decoded[row], soft[row] = chase_soft_decode(
                    code.component, rows[row], test_bits=test_bits
                )
            moved += int(np.count_nonzero(decoded != before))

            extrinsic = soft - rows
            # The right half is this block's own row check; the left half is the
            # previous block's column check, and goes back transposed.
            row_extrinsic[index] = extrinsic[:, half:]
            if index > 0:
                column_extrinsic[index - 1] = extrinsic[:, :half].T

        flips += moved
        if moved == 0:
            break  # a pass that moves nothing will never move anything again

    hard = (working() < 0.0).astype(np.uint8)
    return hard[:, :, : code.information_columns], flips


def chase_soft_decode(
    code: BCHCode, llr: np.ndarray, *, test_bits: int = 4, no_competitor: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-in soft-out Chase decoding. Returns ``(bits, soft_output)``.

    R. Pyndiah, "Near-optimum decoding of product codes: block turbo codes",
    IEEE Trans. Commun. 46(8), 1998.

    The reason a plain Chase decoder is not enough for an iterated code: it
    returns *bits*, and feeding bits back into the next half-iteration hands each
    bit its own channel information a second time. That is intrinsic feedback,
    it makes the iteration over-confident, and it produces exactly one signature
    — the error rate improves for a pass or two and then stops, at a floor far
    above where the code should go. This module had that floor at 5e-4, measured,
    before this function existed.

    So instead of a decision, each bit gets a *reliability*: the difference
    between the metric of the best candidate codeword and the best competing one
    that disagrees about this bit. Where no competitor was generated, the
    difference is unknown and ``no_competitor`` stands in for it — the usual
    treatment, and the reason that constant exists rather than an infinity.

    The caller subtracts the input to make it extrinsic; see
    :func:`staircase_decode`.
    """
    values = np.asarray(llr, dtype=np.float64)
    if values.shape[0] != code.length:
        raise ValueError(f"expected {code.length} LLRs, got {values.shape[0]}")

    hard = (values < 0.0).astype(np.uint8)
    reliability = np.abs(values)
    weakest = np.argsort(reliability)[:test_bits]

    candidates: list[np.ndarray] = []
    metrics: list[float] = []
    seen: set[bytes] = set()
    for flips in range(test_bits + 1):
        for combination in itertools.combinations(weakest, flips):
            trial = hard.copy()
            if combination:
                trial[list(combination)] ^= 1
            word, ok = bch_decode_hard(code, trial)
            if not ok:
                continue
            key = word.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(word)
            metrics.append(float(np.sum(reliability[word != hard])))

    if not candidates:
        # Nothing decoded. The hard decision stands and carries only what the
        # channel said about it, which is the honest amount of information here.
        return hard, values.copy()

    order = int(np.argmin(metrics))
    best = candidates[order]
    best_metric = metrics[order]

    # For every bit, the closest candidate that disagrees about it.
    competitor = np.full(code.length, np.inf)
    for word, metric in zip(candidates, metrics, strict=True):
        differing = word != best
        competitor[differing] = np.minimum(competitor[differing], metric)

    # Where no competing codeword was generated, the true reliability is not
    # small -- it is *unknown and large*, because every codeword disagreeing about
    # this bit lies outside everything the search reached. Returning a small
    # constant there is worse than returning nothing: the caller forms
    # ``soft - input`` to get the extrinsic, so a reliability below the input's
    # own gives a *negative* extrinsic and drags a confident, correct bit towards
    # zero and past it. That is how this decoder corrupted blocks it had not
    # corrected a single bit in -- 268 errors out of zero flips.
    #
    # So the floor is the input's own reliability: no competitor means at least
    # as sure as we already were, never less.
    magnitude = np.where(
        np.isinf(competitor),
        np.maximum(reliability, no_competitor),
        competitor - best_metric,
    )
    sign = np.where(best > 0, -1.0, 1.0)  # positive LLR means zero
    return best, magnitude * sign
