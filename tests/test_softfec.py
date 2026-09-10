"""Validation of the soft-decision FEC pieces — and of what does not work yet.

The component-level machinery is verified here: the fields, the BCH codes built
from their design distance, hard decoding to exactly ``t``, Chase-II recovering
past it, and a staircase encoding whose every stripe row is a codeword.

The iterative decoder is **not** verified, because it does not work, and two
tests pin exactly why: Chase-II with four test bits produces no competing
codeword, so Pyndiah's reliability collapses to a constant, so the extrinsic
carries nothing and the iteration cannot repair even a single bit. Both are
written as assertions of the current failure rather than as ``xfail``, so that
fixing the decoder makes them fail and demand to be rewritten instead of quietly
passing beside a stale warning.

References: Chase 1972; Pyndiah 1998; Smith, Farhood, Hunt, Kschischang and
Lodge, J. Lightwave Technol. 30(1), 2012.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import softfec as sf

# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", sorted(sf.PRIMITIVE_POLYNOMIALS))
def test_every_tabulated_polynomial_is_actually_primitive(m: int) -> None:
    """The table is a claim the constructor checks, not one the code trusts.

    A non-primitive polynomial still builds tables and still multiplies; what it
    does not do is generate the whole multiplicative group, and everything
    algebraic downstream then fails in ways that look like arithmetic bugs.
    """
    field = sf.GaloisField(m)
    assert field.order == (1 << m) - 1
    assert {field.power(2, e) for e in range(field.order)} == set(range(1, 1 << m))


def test_a_non_primitive_polynomial_is_refused() -> None:
    """Constructed by hand, because the shipped table has no bad entry to use."""
    original = sf.PRIMITIVE_POLYNOMIALS[4]
    sf.PRIMITIVE_POLYNOMIALS[4] = 0x1F  # x^4+x^3+x^2+x+1: order 5, not 15
    try:
        with pytest.raises(ValueError, match="not primitive"):
            sf.GaloisField(4)
    finally:
        sf.PRIMITIVE_POLYNOMIALS[4] = original


@pytest.mark.parametrize(("m", "exponent"), [(4, 1), (8, 1), (8, 3), (9, 3), (10, 5)])
def test_a_minimal_polynomial_vanishes_at_its_own_root(m: int, exponent: int) -> None:
    """And is binary, which is what makes a BCH code a binary code at all.

    This is the check that caught the real bug in this module: the coefficient
    mask was assembled in the wrong bit order, which produces the *reciprocal*
    polynomial. That is still binary, still the right degree, and still divides
    the codewords built from it — so nothing looked wrong until a clean
    codeword's syndromes came back non-zero.
    """
    field = sf.GaloisField(m)
    mask = field.minimal_polynomial(exponent)
    coefficients = [(mask >> i) & 1 for i in range(mask.bit_length())]
    assert sf._evaluate_binary(field, coefficients, field.power(2, exponent)) == 0


# --------------------------------------------------------------------------
# BCH component codes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "t"), [(8, 2), (9, 2), (9, 3), (10, 3)])
def test_bch_spends_exactly_m_times_t_parity_bits(m: int, t: int) -> None:
    """The standard result for a narrow-sense BCH code, and a real guard.

    Only the *odd* powers contribute minimal polynomials — the even ones are
    conjugates and bring nothing new. A construction that multiplied them in
    anyway would still make a valid code and would silently pay twice the parity
    for the same correcting power.
    """
    code = sf.bch_code(m, t)
    assert code.parity_bits == m * t
    assert code.length == (1 << m) - 1


@pytest.mark.parametrize(("m", "t"), [(9, 2), (10, 3)])
def test_every_generator_root_annihilates_the_generator(m: int, t: int) -> None:
    """``g(alpha^j) = 0`` for j = 1 … 2t, which is the definition of the code."""
    code = sf.bch_code(m, t)
    field = code.field
    coefficients = [(code.generator >> i) & 1 for i in range(code.generator.bit_length())]
    for j in range(1, 2 * t + 1):
        assert sf._evaluate_binary(field, coefficients, field.power(2, j)) == 0


def test_encoding_is_systematic_and_shortening_preserves_it() -> None:
    """A shortened word is a natural one with leading zeros, and decodes as one."""
    code = sf.bch_code(9, 2, length=256)
    rng = np.random.default_rng(0)
    messages = rng.integers(0, 2, (8, code.message_bits)).astype(np.uint8)
    codewords = sf.bch_encode(code, messages)

    assert np.array_equal(codewords[:, : code.message_bits], messages)
    for word in codewords:
        assert not any(sf.bch_syndromes(code, word))


@pytest.mark.parametrize("errors", [0, 1, 2])
def test_bch_corrects_up_to_t_bit_errors(errors: int) -> None:
    """Binary, so a located error is simply flipped — no magnitudes to compute."""
    code = sf.bch_code(9, 2, length=256)
    rng = np.random.default_rng(errors + 5)
    messages = rng.integers(0, 2, (12, code.message_bits)).astype(np.uint8)
    codewords = sf.bch_encode(code, messages)

    for word in codewords:
        received = word.copy()
        if errors:
            received[rng.choice(code.length, errors, replace=False)] ^= 1
        decoded, ok = sf.bch_decode_hard(code, received)
        assert ok and np.array_equal(decoded, word)


def test_past_t_the_hard_decoder_gives_up_rather_than_guessing() -> None:
    code = sf.bch_code(9, 2, length=256)
    rng = np.random.default_rng(17)
    word = sf.bch_encode(code, rng.integers(0, 2, (1, code.message_bits)).astype(np.uint8))[0]

    failures = 0
    for _ in range(20):
        received = word.copy()
        received[rng.choice(code.length, 3, replace=False)] ^= 1
        _, ok = sf.bch_decode_hard(code, received)
        failures += not ok
    assert failures >= 15, "three errors is past t=2 and should usually be reported as such"


# --------------------------------------------------------------------------
# Chase
# --------------------------------------------------------------------------


def test_chase_recovers_twice_the_hard_limit_when_the_errors_are_the_weak_bits() -> None:
    """The whole soft-decision argument, in one assertion.

    A hard decoder past ``t`` has nothing left to try. This one knows which of
    its bits it is least sure about, and channel errors are overwhelmingly there
    — so four errors on a ``t = 2`` code come back exactly, which no amount of
    hard decoding could do.
    """
    code = sf.bch_code(9, 2, length=256)
    rng = np.random.default_rng(2)
    word = sf.bch_encode(code, rng.integers(0, 2, (1, code.message_bits)).astype(np.uint8))[0]

    llr = np.where(word > 0, -6.0, 6.0)
    weak = [3, 77, 120, 200]
    llr[weak] = -llr[weak] * 0.15  # wrong sign, and barely believed

    decoded, metric = sf.chase_decode(code, llr, test_bits=4)
    assert np.array_equal(decoded, word)
    assert metric > 0.0, "it had to move bits, so the metric cannot be zero"

    hard, ok = sf.bch_decode_hard(code, (llr < 0.0).astype(np.uint8))
    assert not (ok and np.array_equal(hard, word)), (
        "if hard decoding managed it, this proves nothing"
    )


def test_the_soft_output_degenerates_to_a_constant_and_that_is_the_bug() -> None:
    """Why the iteration cannot work, located precisely.

    Chase-II with four test bits on a 256-bit codeword generates sixteen trials,
    and on anything but a badly corrupted input **all of them decode to the same
    codeword**. Pyndiah's reliability is the metric gap to the best *competing*
    codeword, so with no competitor every bit falls back to the same constant —
    and an extrinsic that is constant carries no information from one component
    decode to the next, which is precisely what an iterative decoder runs on.

    So the fault is not in the wiring and not in the tuning: it is that the
    candidate set is too small to produce soft output at all. Fixing it means
    more test bits (exponentially more trials) or a different way of generating
    competitors. Recorded here because a constant is easy to mistake for a
    working reliability — the signs are all correct, and only the magnitudes
    give it away.
    """
    code = sf.bch_code(9, 2, length=256)
    rng = np.random.default_rng(6)
    word = sf.bch_encode(code, rng.integers(0, 2, (1, code.message_bits)).astype(np.uint8))[0]

    llr = np.where(word > 0, -6.0, 6.0)
    llr[[10, 50]] *= 0.05

    decoded, soft = sf.chase_soft_decode(code, llr, test_bits=4)
    assert np.array_equal(decoded, word)
    assert np.array_equal((soft < 0.0).astype(np.uint8), word), "the sign must be the decision"
    assert float(np.std(np.abs(soft))) == 0.0, (
        "the magnitudes now vary, so competitors are being generated. That is the "
        "fix the module warning waits for: rewrite this test and the decoder one."
    )


# --------------------------------------------------------------------------
# The staircase
# --------------------------------------------------------------------------


def test_the_staircase_overhead_lands_where_optical_transport_lives() -> None:
    assert sf.staircase_code().overhead == pytest.approx(0.1636, abs=1e-3)
    assert sf.staircase_code(m=10, t=3, half=256).overhead == pytest.approx(0.1327, abs=1e-3)


def test_every_stripe_row_is_a_component_codeword() -> None:
    """The braiding, asserted structurally rather than inferred from a curve.

    ``[B_{i-1}^T | B_i]`` must be codewords row by row — that is what makes each
    bit of ``B_{i-1}`` checked a second time, as a column, by the stripe after
    it. Encoding is causal: a block's left half is already fixed when it is
    written, so a stream never waits for a block it has not sent.
    """
    code = sf.staircase_code()
    rng = np.random.default_rng(3)
    information = rng.integers(0, 2, (3, code.half, code.information_columns)).astype(np.uint8)
    blocks = sf.staircase_encode(code, information)

    previous = np.zeros((code.half, code.half), dtype=np.uint8)
    for index in range(blocks.shape[0]):
        rows = np.concatenate([previous.T, blocks[index]], axis=1)
        for row in rows:
            assert not any(sf.bch_syndromes(code.component, row))
        previous = blocks[index]


def test_the_information_survives_encoding_unchanged() -> None:
    """Systematic all the way up: the payload is visible in the transmitted block."""
    code = sf.staircase_code()
    rng = np.random.default_rng(4)
    information = rng.integers(0, 2, (3, code.half, code.information_columns)).astype(np.uint8)
    blocks = sf.staircase_encode(code, information)
    assert np.array_equal(blocks[:, :, : code.information_columns], information)


def test_the_iterative_decoder_is_broken_and_this_records_how() -> None:
    """A single injected bit error is not repaired. That is the current state.

    Written as an assertion of the failure rather than an ``xfail`` so that
    fixing the iteration makes *this* test fail and demand to be rewritten,
    instead of quietly passing and leaving the module's warning stale.

    The component pieces above are all verified, so the fault is in the
    iteration's wiring rather than in the codes: encoding produces valid stripe
    rows, and hard decoding repairs two errors in any one of them.
    """
    code = sf.staircase_code()
    rng = np.random.default_rng(3)
    information = rng.integers(0, 2, (3, code.half, code.information_columns)).astype(np.uint8)
    blocks = sf.staircase_encode(code, information)

    llr = np.where(blocks > 0, -8.0, 8.0).astype(np.float64)
    llr[1, 40, 60] *= -1  # exactly one bit, deep inside a block

    decoded, _ = sf.staircase_decode(code, llr, iterations=4, confidence=0.2)
    assert not np.array_equal(decoded, information), (
        "the iterative decoder now repairs a single bit error. That is the fix "
        "this module's warning is waiting for — delete the warning, replace this "
        "test with a waterfall, and only then claim a coding gain."
    )
