"""Probabilistic constellation shaping, against the specification's own vectors.

The DPO modes put a shaper between the scrambler and the code, and the
specification publishes test points either side of it: TP2 going in, TP3 (split
into its amplitude and sign halves) and TP4 coming out. With the vectors and the
document's own lookup tables present, this reproduces all of them for
FlexO-6(e), FlexO-8e and FlexO-8, bit for bit.

**Without them** what holds the module in place is arithmetic that has to close:
each mode's group is four chunks of ``4b`` and three runs of sign bits and must
come to the block's own width, the field widths must come to ``b``, the block
order shipped with the package must be a permutation whose sign half is in
order, and a mode that is not shaped must be refused. The shaping itself cannot
run without the tables, and says so rather than approximating them.
"""

from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from maiman.ofec import (
    INFORMATION_BITS_PER_BLOCK,
    TAIL_START,
    ofec_encode_stream,
    untangle_tail,
)
from maiman.pcs import (
    AMPLITUDE_BITS,
    ARRAYS,
    BLOCKS,
    ENCODER_BITS,
    LUT_WIDTHS,
    SIGN_BITS,
    TABLES_ENV,
    TAIL_PERMUTATION,
    amplitude_and_sign,
    dpo_decode_note,
    dpo_transmit,
    group_bits,
    group_layout,
    load_tables,
    lut_bits,
    shaped_encode,
    shaped_lanes,
    unshape_lanes,
)
from maiman.wport import SHAPED, codec_bits, flexo_adapt, information_bits, scramble

VECTORS = os.environ.get("MAIMAN_OFEC_VECTORS")
TABLES = os.environ.get(TABLES_ENV)
BOTH = pytest.mark.skipif(
    not (VECTORS and TABLES),
    reason=f"set MAIMAN_OFEC_VECTORS and {TABLES_ENV} to the specification's vectors and tables",
)
#: Symbols one DSP frame carries per polarization, which is what a frame's worth
#: of shaped bits has to come to.
FRAME_AMPLITUDE = BLOCKS * AMPLITUDE_BITS
FRAME_SIGN = BLOCKS * SIGN_BITS


def vector(mode: str, point: str) -> np.ndarray:
    return np.loadtxt(Path(str(VECTORS)) / mode / f"{point}_{mode}.txt", dtype=np.uint8)


# ---------------------------------------------------------------------------
# Bit-exact
# ---------------------------------------------------------------------------


@BOTH
@pytest.mark.parametrize("mode", SHAPED)
def test_the_specification_vectors_through_the_shaper(mode: str) -> None:
    """TP2 -> TP3 and TP4, for one frame of each mode."""
    payload = vector(mode, "TP2")[: codec_bits(mode)]
    amplitude, sign = amplitude_and_sign(payload, modulation=mode)
    assert np.array_equal(amplitude, vector(mode, "TP3_amp")[:FRAME_AMPLITUDE]), "TP3 amplitude"
    assert np.array_equal(sign, vector(mode, "TP3_sgn")[:FRAME_SIGN]), "TP3 sign"
    assert np.array_equal(
        shaped_lanes(payload, modulation=mode), vector(mode, "TP4")[: BLOCKS * ENCODER_BITS]
    ), "TP4"


@BOTH
@pytest.mark.parametrize("mode", SHAPED)
def test_the_adaptation_and_scrambler_reach_the_shaper_s_input(mode: str) -> None:
    """TP0 -> TP1 -> TP2 on the DPO modes, which is the same code as the DO ones."""
    info = vector(mode, "TP0")[: information_bits(mode)]
    adapted = flexo_adapt(info, modulation=mode)
    assert np.array_equal(adapted, vector(mode, "TP1")[: codec_bits(mode)])
    assert np.array_equal(scramble(adapted), vector(mode, "TP2")[: codec_bits(mode)])


@BOTH
@pytest.mark.parametrize("mode", SHAPED)
def test_the_shaping_is_undone_exactly(mode: str) -> None:
    payload = vector(mode, "TP2")[: codec_bits(mode)]
    lanes = shaped_lanes(payload, modulation=mode)
    assert np.array_equal(unshape_lanes(lanes, modulation=mode), payload)


@BOTH
@pytest.mark.parametrize(
    ("mode", "probability"), [("b72", 0.8265), ("b106", 0.681), ("b116", 0.623)]
)
def test_the_inner_amplitude_comes_up_as_often_as_the_table_says(
    mode: str, probability: float
) -> None:
    """Table 4's quoted amplitude probabilities, measured coming out of the shaper.

    This is the whole point of the exercise: a 1 is the inner amplitude, and
    shaping makes it more likely than a 0. Nothing in the module is fitted to
    these numbers -- they fall out of how far into a weight-ordered table an
    input of ``b`` bits can reach.
    """
    payload = vector(mode, "TP2")[: codec_bits(mode)]
    amplitude, sign = amplitude_and_sign(payload, modulation=mode)
    assert float(amplitude.mean()) == pytest.approx(probability, abs=0.005)
    assert float(sign.mean()) == pytest.approx(0.5, abs=0.005), "and the sign bits are not shaped"


@BOTH
def test_the_tables_are_the_shape_the_clause_describes() -> None:
    kit = load_tables()
    assert kit.lut10.shape == (1024, 10)
    assert kit.lut11.shape == (2048, 11)
    for table in (kit.lut10, kit.lut11):
        words = {int("".join(map(str, row)), 2) for row in table}
        assert len(words) == table.shape[0], "a bijection on its words"
        weights = table.sum(axis=1).astype(np.int64)
        assert np.all(np.diff(weights) <= 0), "ordered by falling weight"
    assert kit.rewire.shape == (4, 4, 128)


# ---------------------------------------------------------------------------
# What holds without the specification's data
# ---------------------------------------------------------------------------


def test_each_mode_s_group_adds_up_to_its_block() -> None:
    for mode, b in (("b72", 72), ("b106", 106), ("b116", 116)):
        assert lut_bits(mode) == b, "the mode is named for its lookup width"
        assert sum(LUT_WIDTHS[mode]) == b
        assert len(LUT_WIDTHS[mode]) == 12, "four fields of ten bits, eight of eleven"
        shaping, sign = group_layout(mode)
        assert [length for _, length in shaping] == [ARRAYS * b] * 4
        assert [length for _, length in sign] == [512, 512, 480]
        assert sum(length for _, length in sign) == SIGN_BITS
        assert group_bits(mode) == 16 * b + SIGN_BITS
        spans = sorted(shaping + sign)
        assert spans[0][0] == 0
        for (offset, length), (following, _) in pairwise(spans):
            assert offset + length == following, "the spans tile the group"


def test_the_shaping_takes_more_bits_out_than_it_puts_in() -> None:
    """Sixteen lookups of ``b`` bits become 2,048, which is where the rate goes.

    Table 4 states the shaping rate as ``222 / (94 + b) - 1``: 222 bit-columns
    of which 94 carry sign bits, against the ``b`` a lookup is fed.
    """
    for mode, b in (("b72", 72), ("b106", 106), ("b116", 116)):
        assert 16 * b < AMPLITUDE_BITS, "more amplitude bits come out than went in"
        assert 222 / (94 + b) - 1.0 == pytest.approx(
            {"b72": 0.337, "b106": 0.110, "b116": 0.0571}[mode], abs=0.001
        ), "Table 4's shaping rate"
        assert SIGN_BITS == 94 * 16 and AMPLITUDE_BITS == 128 * 16, "94 + 128 = 222 columns"


def test_the_block_order_is_a_permutation_with_its_sign_half_in_order() -> None:
    order = np.load(Path(__file__).resolve().parent.parent / "src/maiman/data/pcs_block_order.npy")
    assert order.shape == (ENCODER_BITS,)
    assert sorted(order.tolist()) == list(range(ENCODER_BITS))
    classes = order >= AMPLITUDE_BITS
    assert int(classes.sum()) == SIGN_BITS
    assert int((~classes).sum()) == AMPLITUDE_BITS
    sign = order[classes] - AMPLITUDE_BITS
    assert np.array_equal(sign, np.arange(SIGN_BITS)), "the sign bits keep their order"


def test_a_mode_that_is_not_shaped_is_refused() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        lut_bits("qpsk")
    with pytest.raises(ValueError, match="DPO modes"):
        amplitude_and_sign(np.zeros(8, dtype=np.uint8), modulation="16qam")
    with pytest.raises(ValueError, match="DPO modes"):
        unshape_lanes(np.zeros((ENCODER_BITS, 4), dtype=np.uint8), modulation="qpsk")


def test_a_payload_that_is_not_whole_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="codec frame"):
        amplitude_and_sign(np.zeros(1000, dtype=np.uint8), modulation="b116")
    with pytest.raises(ValueError, match="whole blocks"):
        unshape_lanes(np.zeros((10, 4), dtype=np.uint8), modulation="b116")


@pytest.mark.skipif(bool(TABLES), reason="the tables are present, so nothing is missing")
def test_without_the_tables_it_says_what_is_missing() -> None:
    load_tables.cache_clear()
    with pytest.raises(RuntimeError, match=TABLES_ENV):
        load_tables()


# ---------------------------------------------------------------------------
# Clause 9.2.4: the tail permute, and the whole shaped path
# ---------------------------------------------------------------------------


@BOTH
@pytest.mark.parametrize("mode", SHAPED)
def test_the_encoder_reproduces_tp5_with_the_tail_permute(mode: str) -> None:
    assert np.array_equal(shaped_encode(vector(mode, "TP4")), vector(mode, "TP5"))


@BOTH
@pytest.mark.parametrize("mode", SHAPED)
def test_the_whole_shaped_path_reproduces_tp7(mode: str) -> None:
    """TP0 to TP7, which is the only point the specification requires."""
    made = dpo_transmit(vector(mode, "TP0"), modulation=mode)
    assert np.array_equal(
        made, np.loadtxt(Path(str(VECTORS)) / mode / f"TP7_{mode}.txt", dtype=np.int8)
    )


def test_the_tail_permute_is_four_orderings_of_thirty_five() -> None:
    assert TAIL_PERMUTATION.shape == (4, 35)
    for ordering in TAIL_PERMUTATION:
        assert sorted(ordering.tolist()) == list(range(35))
    assert len({tuple(row) for row in TAIL_PERMUTATION.tolist()}) == 4, "and they differ"


def test_the_permute_moves_the_codeword_s_last_bits_and_nothing_else() -> None:
    """Eighteen information bits and all seventeen parity ones, per codeword."""
    rng = np.random.default_rng(2)
    u = rng.integers(0, 2, 4 * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8)
    plain = ofec_encode_stream(u)
    tailed = ofec_encode_stream(u, tail=TAIL_PERMUTATION)
    assert not np.array_equal(plain, tailed)
    # Inside the first twenty square-block rows no front half is read, so the
    # permute is only a reordering there and untangling it restores the stream.
    assert np.array_equal(untangle_tail(tailed, TAIL_PERMUTATION), plain)
    assert TAIL_START == 93, "the last 35 of a back half's 128"


def test_past_twenty_rows_the_permute_is_in_the_parity_and_cannot_be_undone() -> None:
    """The reason it is applied to the matrix and not to the output.

    A row twenty back is read as this one's front half, and by then it has been
    permuted -- so the parity that follows is parity over permuted bits, and no
    reordering of the output can put that right.
    """
    rng = np.random.default_rng(3)
    u = rng.integers(0, 2, 16 * INFORMATION_BITS_PER_BLOCK, dtype=np.uint8)
    plain = ofec_encode_stream(u)
    tailed = ofec_encode_stream(u, tail=TAIL_PERMUTATION)
    restored = untangle_tail(tailed, TAIL_PERMUTATION)
    assert np.array_equal(restored[: 10 * 4096], plain[: 10 * 4096]), "before the reach"
    assert not np.array_equal(restored, plain), "and after it, no longer"


def test_what_is_not_a_permutation_is_refused() -> None:
    with pytest.raises(ValueError, match="four orderings"):
        ofec_encode_stream(
            np.zeros(INFORMATION_BITS_PER_BLOCK, dtype=np.uint8),
            tail=np.zeros((2, 35), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="output blocks"):
        untangle_tail(np.zeros(100, dtype=np.uint8), TAIL_PERMUTATION)


def test_the_missing_decoder_says_why_it_is_missing() -> None:
    assert "front" in dpo_decode_note() and "unshape_lanes" in dpo_decode_note()
