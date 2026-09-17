"""The coherent polarization term, and PMD applied along the span rather than after it.

The coherent term ``(1/3) A_y**2 A_x*`` is checked where it has closed forms: a
circular state picks up two thirds of the nonlinear phase a linear one does, and
an elliptical state's axes rotate at ``(2/3) gamma S3`` while its power and its
ellipticity stay put. The step it is taken with is checked against a direct
Runge-Kutta integration of the x/y equations it is supposed to solve.

PMD along the span is checked where it must agree with the old arrangement --
with no Kerr effect the two orders are the same operator -- and where it must
not: the Manakov average. That average, 8/9, holds with the coherent term or
without it, and the test says so rather than pretending it separates them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components.fiber import Fiber
from maiman.kernels import (
    PMDSection,
    apply_pmd,
    propagate_coupled_ssfm,
    random_pmd_sections,
    random_unitary_2x2,
)
from maiman.signals import Band, OpticalSignal

SAMPLE_RATE = 160e9
SAMPLES = 64
GAMMA = 1.3e-3
LENGTH = 20e3
POWER = 0.01
ONES = np.ones(SAMPLES, dtype=np.complex128)


def propagate(
    ex: np.ndarray,
    ey: np.ndarray,
    *,
    coherent: bool,
    pmd: tuple[PMDSection, ...] | None = None,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    out, _ = propagate_coupled_ssfm(
        [ex, ey],
        SAMPLE_RATE,
        beta2=[0.0, 0.0],
        walkoff=[0.0, 0.0],
        gamma=gamma,
        polarization=[0, 1],
        pairs=[(0, 1)],
        coherent_polarization=coherent,
        pmd=pmd,
        alpha=0.0,
        distance=LENGTH,
        max_nonlinear_phase=1e-3,
    )
    return out[0], out[1]


def stokes(ex: np.ndarray, ey: np.ndarray) -> tuple[float, float, float, float]:
    cross = ex * np.conj(ey)
    return (
        float(np.mean(np.abs(ex) ** 2 + np.abs(ey) ** 2)),
        float(np.mean(np.abs(ex) ** 2 - np.abs(ey) ** 2)),
        float(np.mean(2.0 * np.real(cross))),
        float(np.mean(-2.0 * np.imag(cross))),
    )


# ---------------------------------------------------------------------------
# The coherent term
# ---------------------------------------------------------------------------


def test_the_circular_step_is_the_x_y_equations_solved() -> None:
    """Against a fine RK4 of the coupled x/y equations, coherent term included."""
    ax0, ay0 = 0.02 * np.exp(0.3j), 0.013 * np.exp(-1.1j)

    def rhs(state: np.ndarray) -> np.ndarray:
        ax, ay = state
        return np.array(
            [
                -1j
                * GAMMA
                * ((abs(ax) ** 2 + (2 / 3) * abs(ay) ** 2) * ax + (1 / 3) * ay**2 * np.conj(ax)),
                -1j
                * GAMMA
                * ((abs(ay) ** 2 + (2 / 3) * abs(ax) ** 2) * ay + (1 / 3) * ax**2 * np.conj(ay)),
            ]
        )

    state = np.array([ax0, ay0])
    count = 20_000
    h = 5000.0 / count
    for _ in range(count):
        k1 = rhs(state)
        k2 = rhs(state + h / 2 * k1)
        k3 = rhs(state + h / 2 * k2)
        k4 = rhs(state + h * k3)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    out, _ = propagate_coupled_ssfm(
        [ax0 * ONES, ay0 * ONES],
        SAMPLE_RATE,
        beta2=[0.0, 0.0],
        walkoff=[0.0, 0.0],
        gamma=GAMMA,
        polarization=[0, 1],
        pairs=[(0, 1)],
        coherent_polarization=True,
        alpha=0.0,
        distance=5000.0,
    )
    assert out[0][0] == pytest.approx(state[0], abs=1e-12)
    assert out[1][0] == pytest.approx(state[1], abs=1e-12)


@pytest.mark.parametrize(("coherent", "share"), [(True, 2.0 / 3.0), (False, 5.0 / 6.0)])
def test_circular_light_takes_two_thirds_of_the_phase(coherent: bool, share: float) -> None:
    """Two thirds with the coherent term; five sixths without, which is the difference it makes."""
    ex, ey = math.sqrt(POWER / 2) * ONES, 1j * math.sqrt(POWER / 2) * ONES
    ox, _ = propagate(ex, ey, coherent=coherent)
    phase = -float(np.angle(np.mean(ox / ex)))
    assert phase == pytest.approx(share * GAMMA * POWER * LENGTH, rel=1e-9)


def test_linear_light_is_the_scalar_model_exactly() -> None:
    ex = math.sqrt(POWER) * ONES
    ox, oy = propagate(ex, 0 * ONES, coherent=True)
    assert -float(np.angle(np.mean(ox / ex))) == pytest.approx(GAMMA * POWER * LENGTH, rel=1e-9)
    assert float(np.max(np.abs(oy))) < 1e-15, "no power leaks onto an empty axis"


def test_an_ellipse_turns_at_two_thirds_gamma_s3() -> None:
    """Its power and its ellipticity stay put; its axes rotate, in proportion to S3."""
    chi = 0.3
    ex = math.sqrt(POWER) * math.cos(chi) * ONES
    ey = 1j * math.sqrt(POWER) * math.sin(chi) * ONES
    before = stokes(ex, ey)
    after = stokes(*propagate(ex, ey, coherent=True))

    assert after[0] == pytest.approx(before[0], rel=1e-12)
    assert after[3] == pytest.approx(before[3], rel=1e-9)
    turned = math.atan2(after[2], after[1]) - math.atan2(before[2], before[1])
    assert abs(turned) == pytest.approx((2.0 / 3.0) * GAMMA * before[3] * LENGTH, rel=1e-6)


# ---------------------------------------------------------------------------
# PMD along the span
# ---------------------------------------------------------------------------


def pulse() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(SAMPLES) / SAMPLE_RATE
    ex = 0.03 * np.exp(-(((t - t.mean()) / 20e-12) ** 2)).astype(np.complex128)
    return ex, 0.5 * ex * np.exp(0.4j)


def test_without_the_kerr_effect_along_the_span_is_the_same_as_after() -> None:
    """Linear operators commute, so where the waveplates sit cannot matter."""
    sections = random_pmd_sections(5e-12, 30, np.random.default_rng(7))
    ex, ey = pulse()
    along_x, along_y = propagate(ex, ey, coherent=True, pmd=sections, gamma=0.0)
    after_x, after_y = apply_pmd(ex, ey, SAMPLE_RATE, sections)
    assert np.allclose(along_x, after_x, rtol=0.0, atol=1e-14)
    assert np.allclose(along_y, after_y, rtol=0.0, atol=1e-14)


def test_the_kerr_effect_and_the_waveplates_conserve_energy_together() -> None:
    sections = random_pmd_sections(5e-12, 30, np.random.default_rng(7))
    ex, ey = pulse()
    ox, oy = propagate(ex, ey, coherent=True, pmd=sections)
    before = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    after = float(np.sum(np.abs(ox) ** 2 + np.abs(oy) ** 2))
    assert after == pytest.approx(before, rel=1e-12)


@pytest.mark.parametrize("coherent", [True, False])
def test_fast_scrambling_averages_the_nonlinearity_to_eight_ninths(coherent: bool) -> None:
    """The Manakov average, with or without the coherent term.

    Averaged over every state of polarization the two nonlinear forms have the
    same invariant part, ``(4/9) S0**2``, so scrambling reduces both to a common
    phase of 8/9 of ``gamma P L``. The coherent term changes how the state
    evolves on the way, not this average -- which is why this test runs both and
    expects one number.
    """
    rng = np.random.default_rng(1)
    sections = tuple(PMDSection(unitary=random_unitary_2x2(rng), dgd=0.0) for _ in range(800))
    ex, ey = math.sqrt(POWER) * ONES, 0.0 * ONES
    kerr_x, kerr_y = propagate(ex, ey, coherent=coherent, pmd=sections)
    linear_x, linear_y = propagate(ex, ey, coherent=coherent, pmd=sections, gamma=0.0)
    common = -float(np.angle(np.mean(kerr_x * np.conj(linear_x) + kerr_y * np.conj(linear_y))))
    assert common / (GAMMA * POWER * LENGTH) == pytest.approx(8.0 / 9.0, rel=0.01)


def test_pairs_must_name_an_x_and_a_y() -> None:
    with pytest.raises(ValueError, match="name them with pairs"):
        propagate_coupled_ssfm(
            [ONES, ONES],
            SAMPLE_RATE,
            beta2=[0.0, 0.0],
            walkoff=[0.0, 0.0],
            gamma=GAMMA,
            polarization=[0, 1],
            coherent_polarization=True,
            alpha=0.0,
            distance=10.0,
        )
    with pytest.raises(ValueError, match="label the first 0"):
        propagate_coupled_ssfm(
            [ONES, ONES],
            SAMPLE_RATE,
            beta2=[0.0, 0.0],
            walkoff=[0.0, 0.0],
            gamma=GAMMA,
            polarization=[1, 0],
            pairs=[(0, 1)],
            alpha=0.0,
            distance=10.0,
        )


# ---------------------------------------------------------------------------
# The fibre block
# ---------------------------------------------------------------------------

CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=8, seed=3, precision="double"
)


def carrier(chi: float = 0.3) -> OpticalSignal:
    n = CTX.num_samples
    band = Band(
        Ex=np.full(n, math.sqrt(POWER) * math.cos(chi), dtype=np.complex128),
        Ey=np.full(n, 1j * math.sqrt(POWER) * math.sin(chi), dtype=np.complex128),
        f0=193.4e12,
        fs=CTX.sample_rate,
    )
    return OpticalSignal(bands=(band,))


def test_the_block_needs_cross_polarization_for_either() -> None:
    for settings in ({"coherent_polarization": True}, {"interleave_pmd": True}):
        span = Fiber(
            length=20.0,
            attenuation=0.0,
            dispersion=0.0,
            nonlinearity=1.3,
            pmd_coefficient=0.1,
            label="span",
            **settings,
        )
        with pytest.raises(ValueError, match="cross_polarization"):
            span.run(CTX, {"in": carrier()})


def test_the_block_turns_an_ellipse_when_asked_and_not_otherwise() -> None:
    def turn(coherent: bool) -> float:
        span = Fiber(
            length=20.0,
            attenuation=0.0,
            dispersion=0.0,
            nonlinearity=1.3,
            cross_polarization=True,
            coherent_polarization=coherent,
            max_nonlinear_phase=1e-3,
            label="span",
        )
        (band,) = span.run(CTX, {"in": carrier()})["out"].bands
        before = stokes(carrier().bands[0].Ex, carrier().bands[0].Ey)
        after = stokes(band.Ex, band.Ey)
        return math.atan2(after[2], after[1]) - math.atan2(before[2], before[1])

    assert abs(turn(True)) == pytest.approx(
        (2.0 / 3.0) * GAMMA * POWER * math.sin(0.6) * LENGTH, rel=1e-6
    )
    assert abs(turn(False)) > 1e-3, "the phase-only form turns it too, at its own rate"
    assert turn(True) != pytest.approx(turn(False), rel=1e-3)
