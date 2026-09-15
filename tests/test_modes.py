"""The step-index mode solver, against numbers that owe it nothing.

Bessel functions against tabulated values; the core's cutoffs against the zeros
of ``J_0`` and ``J_1``; the core mode against Gloge's closed approximation; and
the cladding modes against a finite-difference solution of the same radial
equation, built here from nothing but a tridiagonal matrix. The fields are held
to the one property a correct set of modes cannot fake: they are orthogonal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.modes import (
    StepIndexFibre,
    bessel_j,
    bessel_k_scaled,
    bessel_y,
    cladding_modes,
    core_modes,
    core_overlap,
    lpg_coupling,
)

FIBRE = StepIndexFibre()
WAVELENGTH = 1.55e-6
K = 2.0 * math.pi / WAVELENGTH


# ---------------------------------------------------------------------------
# Bessel functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "order", "x", "expected"),
    [
        (bessel_j, 0, 1.0, 0.7651976865579666),
        (bessel_j, 1, 1.0, 0.44005058574493355),
        (bessel_j, 0, 100.0, 0.019985850304223122),
        (bessel_y, 0, 1.0, 0.08825696421567696),
        (bessel_y, 1, 1.0, -0.7812128213002887),
        (bessel_y, 1, 0.1, -6.458951094702027),
        (bessel_y, 0, 100.0, -0.07724431336886477),
    ],
)
def test_bessel_functions_match_their_tables(
    function: object, order: int, x: float, expected: float
) -> None:
    assert callable(function)
    assert function(order, x)[0] == pytest.approx(expected, rel=0.0, abs=1e-11)


@pytest.mark.parametrize(
    ("order", "x", "expected"),
    [(0, 1.0, 0.42102443824070834), (1, 1.0, 0.6019072301972346), (0, 0.01, 4.721244730161)],
)
def test_the_scaled_k_is_k_times_e_to_the_x(order: int, x: float, expected: float) -> None:
    assert bessel_k_scaled(order, x)[0] * math.exp(-x) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_the_zeros_are_where_the_tables_put_them() -> None:
    assert abs(bessel_j(0, 2.404825557695773)[0]) < 1e-14
    assert abs(bessel_j(1, 3.8317059702075125)[0]) < 1e-14
    assert abs(bessel_y(0, 0.8935769662791675)[0]) < 1e-13


def test_negative_orders_follow_the_reflection_formula() -> None:
    x = np.array([0.3, 2.0, 17.0])
    assert np.allclose(bessel_j(-3, x), -bessel_j(3, x), rtol=0.0, atol=1e-15)
    assert np.allclose(bessel_y(-1, x), -bessel_y(1, x), rtol=0.0, atol=1e-13)
    assert np.allclose(bessel_k_scaled(-2, x), bessel_k_scaled(2, x), rtol=1e-14, atol=0.0)


def test_non_positive_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        bessel_y(0, 0.0)


# ---------------------------------------------------------------------------
# Core modes
# ---------------------------------------------------------------------------


def wavelength_for(v: float) -> float:
    numerical_aperture = math.sqrt(FIBRE.core_index**2 - FIBRE.cladding_index**2)
    return 2.0 * math.pi * FIBRE.core_radius * numerical_aperture / v


def test_standard_fibre_is_single_mode_at_1550() -> None:
    modes = core_modes(FIBRE, WAVELENGTH)
    assert len(modes) == 1
    assert FIBRE.cladding_index < modes[0].effective_index < FIBRE.core_index
    assert core_modes(FIBRE, WAVELENGTH, order=1) == []


@pytest.mark.parametrize(("order", "cutoff"), [(1, 2.404825557695773), (0, 3.8317059702075125)])
def test_a_higher_mode_appears_exactly_at_its_bessel_zero(order: int, cutoff: float) -> None:
    """LP11 at the first zero of J_0, LP02 at the first nonzero zero of J_1.

    Checked a tenth of a percent below each cutoff and two percent above. The
    margins differ because the modes do: an LP0m mode leaves cutoff with ``w``
    rising only logarithmically -- ``w K_1(w) / K_0(w)`` goes to zero like
    ``1 / ln(1/w)`` -- so LP02 is not bound until V is between half a percent and
    one percent past its zero, where LP11 is bound a tenth of a percent past its.
    """
    expected = 1 if order == 1 else 2
    margin = 1e-3 if order == 1 else 2e-2
    below = core_modes(FIBRE, wavelength_for(cutoff * (1 - 1e-3)), order=order)
    above = core_modes(FIBRE, wavelength_for(cutoff * (1 + margin)), order=order)
    assert len(below) == expected - 1
    assert len(above) == expected


def test_the_core_mode_agrees_with_gloge() -> None:
    """``w = 1.1428 V - 0.9960``, which Gloge gives to about a tenth of a percent."""
    v = FIBRE.v_number(WAVELENGTH)
    (mode,) = core_modes(FIBRE, WAVELENGTH)
    w = FIBRE.core_radius * K * math.sqrt(mode.effective_index**2 - FIBRE.cladding_index**2)
    assert w == pytest.approx(1.1428 * v - 0.9960, rel=1e-3)


# ---------------------------------------------------------------------------
# Cladding modes, against finite differences
# ---------------------------------------------------------------------------


def finite_difference_indices(step: float) -> np.ndarray:
    """Effective indices of every l = 0 mode, from a cell-centred radial grid.

    The interfaces sit on cell faces, which keeps the discretisation second order
    across the index steps; the operator is symmetrised by ``sqrt(r)``.
    """
    outer = FIBRE.cladding_radius + 3e-6
    cells = round(outer / step)
    r = (np.arange(cells) + 0.5) * step
    n = np.where(
        r < FIBRE.core_radius,
        FIBRE.core_index,
        np.where(r < FIBRE.cladding_radius, FIBRE.cladding_index, FIBRE.surrounding_index),
    )
    diagonal = -2.0 / step**2 + K**2 * n**2
    off = (r[:-1] + 0.5 * step) / step**2 / np.sqrt(r[:-1] * r[1:])
    matrix = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    eigenvalues = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    guided = eigenvalues[eigenvalues > (K * FIBRE.surrounding_index) ** 2]
    return np.sqrt(guided) / K


def test_core_and_cladding_modes_match_an_independent_finite_difference_solve() -> None:
    """Richardson-extrapolated to 1e-9, where the solver and the grid meet."""
    coarse = finite_difference_indices(0.05e-6)
    fine = finite_difference_indices(0.025e-6)
    reference = (4.0 * fine[:9] - coarse[:9]) / 3.0

    (core,) = core_modes(FIBRE, WAVELENGTH)
    cladding = cladding_modes(FIBRE, WAVELENGTH, count=8)
    assert core.effective_index == pytest.approx(reference[0], rel=0.0, abs=1e-9)
    for mode, expected in zip(cladding, reference[1:], strict=True):
        assert mode.effective_index == pytest.approx(expected, rel=0.0, abs=1e-9), mode.name


def test_cladding_modes_are_orthogonal() -> None:
    """Distinct solutions of one self-adjoint problem, so their fields must be."""
    modes = cladding_modes(FIBRE, WAVELENGTH, count=5)
    radii = np.linspace(1e-9, FIBRE.cladding_radius + 2e-6, 400_001)
    fields = [mode.field(radii) for mode in modes]
    weight = radii * (radii[1] - radii[0])
    norms = [math.sqrt(float(np.sum(f * f * weight))) for f in fields]
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            inner = float(np.sum(fields[i] * fields[j] * weight)) / (norms[i] * norms[j])
            assert abs(inner) < 1e-4, (modes[i].name, modes[j].name, inner)


def test_the_surroundings_move_the_cladding_and_not_the_core() -> None:
    """Which is the whole reason a long-period grating is a refractometer."""
    wet = StepIndexFibre(surrounding_index=1.333)
    assert core_modes(wet, WAVELENGTH)[0].effective_index == pytest.approx(
        core_modes(FIBRE, WAVELENGTH)[0].effective_index, rel=0.0, abs=1e-15
    )
    dry_mode = cladding_modes(FIBRE, WAVELENGTH, count=5)[4]
    wet_mode = cladding_modes(wet, WAVELENGTH, count=5)[4]
    assert wet_mode.effective_index - dry_mode.effective_index > 1e-6


# ---------------------------------------------------------------------------
# Overlaps and coupling
# ---------------------------------------------------------------------------


def test_a_mode_overlaps_itself_by_its_confinement() -> None:
    (core,) = core_modes(FIBRE, WAVELENGTH)
    confinement = core_overlap(core, core)
    assert 0.7 < confinement < 0.8, (
        "LP01 at V = 2.04 carries three quarters of its power in the core"
    )


def test_modes_of_different_order_do_not_couple() -> None:
    (core,) = core_modes(FIBRE, WAVELENGTH)
    (other,) = cladding_modes(FIBRE, WAVELENGTH, order=1, count=1)
    assert core_overlap(core, other) == 0.0
    assert lpg_coupling(core, other, 3e-4) == 0.0


def test_self_coupling_is_the_bragg_coefficient_scaled_by_confinement() -> None:
    """``pi dn / lambda`` times the overlap and ``n1 / n_eff``: the Bragg grating's own kappa."""
    (core,) = core_modes(FIBRE, WAVELENGTH)
    modulation = 2e-4
    expected = (
        math.pi
        * modulation
        / WAVELENGTH
        * core_overlap(core, core)
        * FIBRE.core_index
        / core.effective_index
    )
    assert lpg_coupling(core, core, modulation) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_an_impossible_fibre_is_refused() -> None:
    with pytest.raises(ValueError, match="core radius"):
        StepIndexFibre(core_radius=70e-6)
    with pytest.raises(ValueError, match="core index"):
        StepIndexFibre(surrounding_index=1.45)
