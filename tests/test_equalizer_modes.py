"""A fractionally spaced equaliser, and one that trains on its own decisions.

Two things a receiver in the field needs and a reference receiver does not. It
cannot choose where in the symbol its clock lands, so it wants an equaliser that
does not care: sampled half a symbol late, a symbol-spaced FFE is left with an
MSE of 0.32 and six percent of its symbols wrong, and a T/2-spaced one with the
same span barely notices. And it has no reference to train on, so it trains on
its own decisions -- which gets it exactly where the reference would, as long as
enough of those decisions are right to begin with, and nowhere at all when they
are not. The tests find that edge rather than hide it.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components import FFEDFEEqualizer
from maiman.dsp import ffe_dfe_equalize
from maiman.signals import BinarySignal, ElectricalSignal

ALPHABET = np.array([-3.0, -1.0, 1.0, 3.0])
COUNT = 4096
OVERSAMPLING = 16


def symbols(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(ALPHABET, size=COUNT)


def band_limited(sent: np.ndarray, *, seed: int) -> np.ndarray:
    """Impulses through a Gaussian low-pass of 0.3 baud and an echo, sixteen samples a symbol."""
    impulses = np.zeros(COUNT * OVERSAMPLING)
    impulses[::OVERSAMPLING] = sent
    frequency = np.fft.fftfreq(impulses.size, 1.0 / OVERSAMPLING)
    shaped = np.real(
        np.fft.ifft(np.fft.fft(impulses) * np.exp(-np.log(2) / 2 * (frequency / 0.3) ** 2))
    )
    shaped = shaped + 0.3 * np.roll(shaped, int(1.5 * OVERSAMPLING))
    shaped = shaped / np.std(shaped)
    return shaped + np.random.default_rng(seed).normal(0.0, 0.05, shaped.size)


def sampled(waveform: np.ndarray, phase: float, per_symbol: int) -> np.ndarray:
    """``per_symbol`` samples a symbol, the first ``phase`` of a symbol late."""
    index = np.arange(COUNT * per_symbol) * (OVERSAMPLING // per_symbol)
    return waveform[(index + round(phase * OVERSAMPLING)) % waveform.size]


def symbol_error_rate(sent: np.ndarray, decisions: np.ndarray) -> float:
    return float(np.mean(ALPHABET[decisions] != sent))


# ---------------------------------------------------------------------------
# Fractional spacing
# ---------------------------------------------------------------------------


def test_half_a_symbol_late_closes_a_symbol_spaced_eye_and_not_a_fractional_one() -> None:
    sent = symbols(1)
    waveform = band_limited(sent, seed=2)

    def spaced(phase: float) -> tuple[float, float]:
        result = ffe_dfe_equalize(
            sampled(waveform, phase, 1), sent, levels=ALPHABET, ffe_taps=7, step=0.02, passes=8
        )
        return result.mse, symbol_error_rate(sent, result.decisions)

    def fractional(phase: float) -> tuple[float, float]:
        result = ffe_dfe_equalize(
            sampled(waveform, phase, 2),
            sent,
            levels=ALPHABET,
            ffe_taps=13,
            step=0.02,
            passes=8,
            samples_per_symbol=2,
        )
        return result.mse, symbol_error_rate(sent, result.decisions)

    on_time, late = spaced(0.0), spaced(0.5)
    assert on_time[1] == 0.0
    assert late[0] > 10.0 * on_time[0]
    assert late[1] > 0.03, "six percent of the symbols wrong, from where the clock landed"

    for phase in (0.0, 0.25, 0.5):
        mse, errors = fractional(phase)
        assert errors == 0.0
        assert mse == pytest.approx(on_time[0], rel=0.3), phase


def test_one_sample_a_symbol_is_the_symbol_spaced_equaliser_exactly() -> None:
    sent = symbols(3)
    received = sent + 0.3 * np.roll(sent, 1)
    one = ffe_dfe_equalize(received, sent, levels=ALPHABET, ffe_taps=5, passes=3)
    named = ffe_dfe_equalize(
        received, sent, levels=ALPHABET, ffe_taps=5, passes=3, samples_per_symbol=1
    )
    assert np.array_equal(one.output, named.output)


def test_the_samples_have_to_match_the_symbols() -> None:
    sent = symbols(4)
    with pytest.raises(ValueError, match="2 a symbol"):
        ffe_dfe_equalize(sent, sent, levels=ALPHABET, ffe_taps=3, samples_per_symbol=2)
    with pytest.raises(ValueError, match="at least 1"):
        ffe_dfe_equalize(sent, sent, levels=ALPHABET, ffe_taps=3, samples_per_symbol=0)


# ---------------------------------------------------------------------------
# Blind
# ---------------------------------------------------------------------------


def postcursor(amount: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sent = symbols(seed)
    noise = np.random.default_rng(seed + 100).normal(0.0, 0.05, COUNT)
    return sent, sent + amount * np.roll(sent, 1) + noise


@pytest.mark.parametrize("amount", [0.3, 0.4])
def test_blind_gets_where_the_reference_does(amount: float) -> None:
    """Even from a start where a third of the raw decisions are wrong."""
    sent, received = postcursor(amount, seed=2)
    raw = np.clip(
        np.searchsorted([-2.0, 0.0, 2.0], received / np.std(received) * np.std(ALPHABET)), 0, 3
    )
    assert np.mean(ALPHABET[raw] != sent) > 0.05, "the eye starts partly closed"

    settings = {"levels": ALPHABET, "ffe_taps": 15, "step": 0.02, "passes": 8}
    reference = ffe_dfe_equalize(received, sent, **settings)  # type: ignore[arg-type]
    blind = ffe_dfe_equalize(received, sent, blind=True, **settings)  # type: ignore[arg-type]
    assert blind.mse == pytest.approx(reference.mse, rel=0.01)
    assert symbol_error_rate(sent, blind.decisions) == 0.0


def test_blind_fails_where_the_eye_is_too_closed_to_start() -> None:
    """Half a symbol of postcursor: the reference equalises it, its own decisions cannot."""
    sent, received = postcursor(0.5, seed=2)
    settings = {"levels": ALPHABET, "ffe_taps": 15, "step": 0.02, "passes": 8}
    reference = ffe_dfe_equalize(received, sent, **settings)  # type: ignore[arg-type]
    blind = ffe_dfe_equalize(received, sent, blind=True, **settings)  # type: ignore[arg-type]
    assert symbol_error_rate(sent, reference.decisions) == 0.0
    assert symbol_error_rate(sent, blind.decisions) > 0.2


def test_a_blind_dfe_feeds_back_its_own_decisions_and_still_converges() -> None:
    sent, received = postcursor(0.4, seed=5)
    result = ffe_dfe_equalize(
        received, sent, levels=ALPHABET, ffe_taps=3, dfe_taps=1, step=0.02, passes=8, blind=True
    )
    assert symbol_error_rate(sent, result.decisions) == 0.0
    assert result.dfe[0] == pytest.approx(0.4, abs=0.02)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def test_the_block_refuses_a_half_symbol_it_cannot_take() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=5, sequence_length=64)
    waveform = ElectricalSignal(samples=np.zeros(ctx.num_samples), fs=ctx.sample_rate, unit="A")
    reference = BinarySignal(bits=np.zeros(128, dtype=np.uint8), symbol_rate=ctx.bit_rate)
    block = FFEDFEEqualizer(fractional=True, sample_offset=0.0, label="eq")
    with pytest.raises(ValueError, match="even number of samples"):
        block.run(ctx, {"in": waveform, "reference": reference})
