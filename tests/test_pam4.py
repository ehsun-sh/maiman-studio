"""PAM4, and the feed-forward and decision-feedback equalisers a short-reach lane needs.

The error-rate formula is checked two ways: against the QAM formula it is one
axis of, exactly, and against errors counted in Gaussian noise. The equaliser is
checked against the minimum-mean-square-error solution for the very samples it
was trained on -- the least-squares fit, which is what adaptation converges to --
and a decision-feedback tap against the postcursor it exists to cancel. The
driver is checked for putting a modulator's powers where it says.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import (
    CWLaser,
    ElectricalFilter,
    FFEDFEEqualizer,
    MachZehnderModulator,
    PAM4Driver,
    PINPhotodiode,
    PRBSGenerator,
)
from maiman.dsp import ffe_dfe_equalize
from maiman.modulation import (
    bits_to_indices,
    gray_pam_levels,
    ser_pam,
    ser_qam,
)
from maiman.signals import BinarySignal, PAMMeasurement

ALPHABET = np.array([-3.0, -1.0, 1.0, 3.0])


def symbols(count: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).choice(ALPHABET, size=count)


def circular_channel(sent: np.ndarray, taps: list[float], cursor: int) -> np.ndarray:
    """``y_k = sum_j h_j a_{k - j + cursor}``, circular, as every window here is.

    Tap ``cursor`` is the main one; taps after it are postcursors -- earlier
    symbols leaking into this one -- and taps before it precursors.
    """
    received = np.zeros_like(sent)
    for j, h in enumerate(taps):
        received += h * np.roll(sent, j - cursor)
    return received


# ---------------------------------------------------------------------------
# The error rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [0.0, 6.0, 12.0, 18.0, 24.0])
def test_square_qam_is_two_pam_axes_exactly(snr_db: float) -> None:
    snr = 10 ** (snr_db / 10)
    for bits in (2, 4, 6, 8):
        per_axis = 1 << (bits // 2)
        assert ser_qam(snr, bits) == pytest.approx(
            1.0 - (1.0 - ser_pam(snr, per_axis)) ** 2, rel=1e-12, abs=0.0
        )


@pytest.mark.parametrize("snr_db", [10.0, 13.0, 16.0])
def test_pam4_error_rate_matches_errors_counted_in_gaussian_noise(snr_db: float) -> None:
    snr = 10 ** (snr_db / 10)
    sent = symbols(400_000, seed=int(snr_db))
    sigma = math.sqrt(5.0 / snr)  # mean power of +-1, +-3 is 5
    received = sent + np.random.default_rng(7).normal(0.0, sigma, sent.size)
    decided = ALPHABET[np.searchsorted([-2.0, 0.0, 2.0], received)]
    counted = float(np.mean(decided != sent))
    assert counted == pytest.approx(ser_pam(snr, 4), rel=0.05)


def test_gray_coding_costs_one_bit_between_neighbours() -> None:
    gray = gray_pam_levels(2)
    by_amplitude = sorted(range(4), key=lambda pattern: gray[pattern])
    for index in range(len(by_amplitude) - 1):
        assert bin(by_amplitude[index] ^ by_amplitude[index + 1]).count("1") == 1


# ---------------------------------------------------------------------------
# The equaliser
# ---------------------------------------------------------------------------


def test_the_ffe_converges_to_the_minimum_mean_square_error_solution() -> None:
    """NLMS on a known channel in noise lands on the least-squares fit to the same samples.

    The least-squares FFE-plus-bias over the training window is the MMSE solution
    for those samples, computed in one linear solve with nothing adaptive in it.
    Adaptation that has converged sits within its own misadjustment of it.
    """
    channel, cursor, sigma, taps = [0.15, 1.0, 0.35], 1, 0.15, 7
    sent = symbols(8192, seed=3)
    received = circular_channel(sent, channel, cursor) + np.random.default_rng(4).normal(
        0.0, sigma, sent.size
    )
    result = ffe_dfe_equalize(received, sent, levels=ALPHABET, ffe_taps=taps, step=0.01, passes=12)

    centre = taps // 2
    n = sent.size
    windows = received[(np.arange(n)[:, None] + np.arange(taps)[None, :] - centre) % n]
    design = np.hstack([windows, np.ones((n, 1))])
    solution, *_ = np.linalg.lstsq(design, sent, rcond=None)
    minimum = float(np.mean((design @ solution - sent) ** 2))

    assert result.ffe == pytest.approx(solution[:taps], abs=0.01)
    assert result.bias == pytest.approx(solution[taps], abs=0.01)
    assert result.mse == pytest.approx(minimum, rel=0.02)
    assert result.mse >= minimum * (1.0 - 1e-9), "nothing linear beats least squares on its data"


def test_a_dfe_tap_is_the_postcursor_it_cancels() -> None:
    sent = symbols(4096, seed=5)
    received = circular_channel(sent, [1.0, 0.6], 0)
    result = ffe_dfe_equalize(
        received, sent, levels=ALPHABET, ffe_taps=1, dfe_taps=1, step=0.05, passes=6
    )
    assert result.dfe[0] == pytest.approx(0.6, abs=1e-3)
    assert result.ffe[0] == pytest.approx(1.0, abs=1e-3)
    assert np.array_equal(ALPHABET[result.decisions], sent)


def test_decision_feedback_beats_feed_forward_on_a_heavy_postcursor() -> None:
    """A linear equaliser inverting a deep postcursor amplifies noise; a DFE does not."""
    sent = symbols(8192, seed=6)
    received = circular_channel(sent, [1.0, 0.8], 0) + np.random.default_rng(8).normal(
        0.0, 0.2, sent.size
    )
    linear = ffe_dfe_equalize(received, sent, levels=ALPHABET, ffe_taps=15, step=0.02, passes=8)
    feedback = ffe_dfe_equalize(
        received, sent, levels=ALPHABET, ffe_taps=3, dfe_taps=1, step=0.02, passes=8
    )
    assert feedback.mse < 0.6 * linear.mse


def test_a_scaled_and_offset_detector_is_absorbed_by_gain_and_bias() -> None:
    """A photocurrent is ten thousand times smaller than a level; that must not matter."""
    sent = symbols(2048, seed=9)
    received = 3.7e-4 * sent + 1.2e-3
    result = ffe_dfe_equalize(received, sent, levels=ALPHABET, ffe_taps=3, passes=2)
    assert result.mse < 1e-12
    assert np.array_equal(ALPHABET[result.decisions], sent)
    assert result.ffe[1] == pytest.approx(1.0 / 3.7e-4, rel=1e-6)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"ffe_taps": 4}, "odd"),
        ({"ffe_taps": 3, "dfe_taps": -1}, "negative"),
        ({"ffe_taps": 3, "step": 2.5}, "step"),
    ],
)
def test_an_equaliser_that_cannot_be_is_refused(settings: dict[str, float], message: str) -> None:
    sent = symbols(64)
    with pytest.raises(ValueError, match=message):
        ffe_dfe_equalize(sent, sent, levels=ALPHABET, **settings)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def test_predistortion_spaces_the_modulator_powers_evenly() -> None:
    modulator = MachZehnderModulator(v_pi=4.0, extinction_ratio=30.0)
    straight = PAM4Driver(v_low=3.2, v_high=0.8)
    corrected = PAM4Driver(v_low=3.2, v_high=0.8, predistort=True)

    def spacing(driver: PAM4Driver) -> np.ndarray:
        return np.diff(modulator.power_transmission(driver.level_voltages()))

    assert np.allclose(spacing(corrected), spacing(corrected)[0], rtol=1e-12, atol=0.0)
    uneven = spacing(straight)
    assert uneven.max() / uneven.min() > 1.2, "evenly spaced volts are not evenly spaced powers"


def test_the_driver_needs_two_bits_a_symbol() -> None:
    ctx = SimulationContext(bit_rate=26.5625e9, samples_per_symbol=4, sequence_length=16)
    bits = BinarySignal(bits=np.zeros(16, dtype=np.uint8), symbol_rate=ctx.bit_rate)
    with pytest.raises(ValueError, match="bits_per_symbol to 2"):
        PAM4Driver().run(ctx, {"in": bits})


def test_the_driver_maps_gray_pairs_to_levels() -> None:
    ctx = SimulationContext(bit_rate=26.5625e9, samples_per_symbol=2, sequence_length=4)
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    driver = PAM4Driver(v_low=0.0, v_high=3.0)
    out = driver.run(ctx, {"in": BinarySignal(bits=bits, symbol_rate=2 * ctx.bit_rate)})["out"]
    held = np.asarray(out.samples)[::2]
    expected = gray_pam_levels(2)[bits_to_indices(bits, 2)] / 2.0 + 1.5
    assert held == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# A lane, end to end
# ---------------------------------------------------------------------------


def lane(**equaliser: float) -> PAMMeasurement:
    ctx = SimulationContext(bit_rate=26.5625e9, samples_per_symbol=8, sequence_length=4096)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=2.0, label="prbs"))
    driver = graph.add(PAM4Driver(v_low=3.2, v_high=0.8, predistort=True, label="pam4"))
    laser = graph.add(CWLaser(power=0.0, wavelength=1310.0, label="tx"))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0, label="mzm"))
    pin = graph.add(PINPhotodiode(label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=7.0, label="lpf"))
    eq = graph.add(FFEDFEEqualizer(label="eq", **equaliser))
    graph.connect(prbs, driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver, modulator["electrical_in"])
    graph.connect(modulator, pin["in"])
    graph.connect(pin, lpf["in"])
    graph.connect(lpf, eq["in"])
    graph.connect(prbs, eq["reference"])
    result = graph.run()[eq]
    assert isinstance(result, PAMMeasurement)
    return result


def test_a_narrow_receiver_closes_the_eye_and_the_ffe_opens_it() -> None:
    """7 GHz at 26.5625 GBd: 642 errors in 4096 symbols unequalised, none with nine taps."""
    bare = lane(ffe_taps=1.0, dfe_taps=0.0)
    equalised = lane(ffe_taps=9.0, dfe_taps=0.0)
    assert bare.symbol_errors > 100
    assert bare.snr_db == pytest.approx(9.70, abs=0.1)
    assert equalised.symbol_errors == 0
    assert equalised.snr_db > bare.snr_db + 20.0
