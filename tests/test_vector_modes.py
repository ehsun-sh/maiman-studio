"""The vector modes of a fibre, against the closed equations and against the scalar solver.

The scalar solver is checked against a finite-difference solution of the scalar
equation, which pins its numerics and says nothing about the approximation. This
is the other half: two layers against the characteristic equation of Snyder and
Love, which is exact and owes this code nothing; three layers against two where
the core step vanishes; the families against the LP modes they are made of, by
counting them in a fibre that guides several; and the weak-guidance limit against
:mod:`maiman.modes`, where the two must agree and do, to a part in a thousand in
coupling and 2.5e-7 in effective index.

Then what the split is worth: at the glass-air boundary the HE1m sit up to 4.4e-6
below the LP0m they stand in for, which moves a long-period notch by a nanometre,
and the EH1m -- modes the scalar model has no counterpart for at this order --
couple to the core at a percent of the HE1m.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from maiman.modes import (
    StepIndexFibre,
    bessel_j,
    bessel_k_scaled,
    cladding_modes,
    core_modes,
    lpg_coupling,
)
from maiman.photonics import long_period_grating, long_period_resonances
from maiman.units import C_LIGHT
from maiman.vector_modes import (
    vector_cladding_modes,
    vector_core_modes,
    vector_coupling,
    vector_modes,
)

WAVELENGTH = 1.55e-6
K = 2.0 * math.pi / WAVELENGTH
FIBRE = StepIndexFibre()
#: A fibre with V = 6, which guides enough modes for the families to be counted.
WIDE = StepIndexFibre(core_radius=12e-6)


def snyder_love(order: int, n1: float, n2: float, radius: float, effective_index: float) -> float:
    """The exact two-layer characteristic function, Snyder and Love, *Optical Waveguide Theory*.

        ``(J'/uJ + K'/wK) (J'/uJ + (n2/n1)^2 K'/wK) = nu^2 (n_eff/n1)^2 (1/u^2 + 1/w^2)^2``

    written as a residual. Nothing of the solver is in it but the Bessel
    functions, which have their own tests.
    """
    u = radius * K * math.sqrt(n1**2 - effective_index**2)
    w = radius * K * math.sqrt(effective_index**2 - n2**2)
    j, j_prime = bessel_j(order, u), bessel_j(order - 1, u) - (order / u) * bessel_j(order, u)
    k = bessel_k_scaled(order, w)
    k_prime = -bessel_k_scaled(order - 1, w) - (order / w) * k
    a = float(j_prime[0] / (u * j[0]))
    b = float(k_prime[0] / (w * k[0]))
    return (a + b) * (a + (n2 / n1) ** 2 * b) - order**2 * (effective_index / n1) ** 2 * (
        1.0 / u**2 + 1.0 / w**2
    ) ** 2


# ---------------------------------------------------------------------------
# Against the closed equation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_a_glass_rod_in_air_solves_the_exact_equation(order: int) -> None:
    """The boundary the LP approximation has no business at: a third of an index step."""
    modes = vector_modes([62.5e-6], [1.444, 1.0], WAVELENGTH, order=order, window=(1.435, 1.4439))
    assert len(modes) >= 4
    for mode in modes:
        assert snyder_love(order, 1.444, 1.0, 62.5e-6, mode.effective_index) == pytest.approx(
            0.0, abs=1e-10
        ), mode.name


def test_the_core_mode_solves_it_too() -> None:
    (mode,) = vector_core_modes(FIBRE, WAVELENGTH)
    assert mode.name == "HE11 (core)"
    assert snyder_love(
        1, FIBRE.core_index, FIBRE.cladding_index, FIBRE.core_radius, mode.effective_index
    ) == pytest.approx(0.0, abs=1e-12)
    (scalar,) = core_modes(FIBRE, WAVELENGTH)
    assert mode.effective_index - scalar.effective_index == pytest.approx(-5.16e-6, abs=1e-8), (
        "the polarization correction the scalar equation cannot see"
    )


def test_a_core_that_is_not_there_is_two_layers() -> None:
    """Three layers with no step at the first interface must reproduce two exactly."""
    two = vector_modes([62.5e-6], [1.444, 1.0], WAVELENGTH, order=1, window=(1.43, 1.4439))
    three = vector_modes(
        [4.1e-6, 62.5e-6], [1.444, 1.444, 1.0], WAVELENGTH, order=1, window=(1.43, 1.4439)
    )
    assert len(two) == len(three) > 20
    for first, second in zip(two, three, strict=True):
        assert first.effective_index == pytest.approx(second.effective_index, rel=0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# The families, and what they are made of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("order", "names", "scalar"),
    [
        (0, ["TE01", "TM01", "TE02", "TM02"], [(1, 1), (1, 1), (1, 2), (1, 2)]),
        (1, ["HE11", "EH11", "HE12"], [(0, 1), (2, 1), (0, 2)]),
        (2, ["HE21", "EH21", "HE22"], [(1, 1), (3, 1), (1, 2)]),
        (3, ["HE31"], [(2, 1)]),
    ],
)
def test_the_families_are_the_lp_modes_taken_apart(
    order: int, names: list[str], scalar: list[tuple[int, int]]
) -> None:
    """At V = 6, each vector mode is the LP mode it belongs to, split off it by 2e-5.

    ``LP_lm`` is ``HE_{l+1,m}`` together with ``EH_{l-1,m}``, and for ``l = 1``
    with TE0m and TM0m as well. Counting them is what says the labels are right:
    get HE and EH the wrong way round and order three would have two modes where
    the fibre has one.
    """
    modes = vector_core_modes(WIDE, WAVELENGTH, order=order)
    assert [mode.name.split()[0] for mode in modes] == names
    for mode, (azimuthal, rank) in zip(modes, scalar, strict=True):
        partner = core_modes(WIDE, WAVELENGTH, order=azimuthal)[rank - 1]
        assert mode.effective_index == pytest.approx(partner.effective_index, abs=2e-5), mode.name


def test_weak_guidance_is_the_scalar_solver() -> None:
    """Take the glass-air step away and the two solvers must meet, and they do."""
    weak = StepIndexFibre(surrounding_index=1.43)
    modes = vector_cladding_modes(weak, WAVELENGTH, order=1, count=6)
    lp0 = cladding_modes(weak, WAVELENGTH, order=0, count=3)
    lp2 = cladding_modes(weak, WAVELENGTH, order=2, count=3)
    for mode in modes:
        partner = (lp0 if mode.family == "HE" else lp2)[mode.rank - 1]
        assert mode.effective_index == pytest.approx(partner.effective_index, abs=3e-7), mode.name

    (core,) = vector_core_modes(weak, WAVELENGTH)
    (scalar_core,) = core_modes(weak, WAVELENGTH)
    for mode in (m for m in modes if m.family == "HE"):
        vector = vector_coupling(core, mode, 3e-4)
        assert vector == pytest.approx(
            lpg_coupling(scalar_core, lp0[mode.rank - 1], 3e-4), rel=2e-3
        )
    assert vector_coupling(core, core, 3e-4) == pytest.approx(
        lpg_coupling(scalar_core, scalar_core, 3e-4), rel=3e-3
    )


def test_the_air_boundary_splits_what_the_lp_modes_merged() -> None:
    """Below their LP0m by 1.3e-7 at the top of the band and 4.4e-6 six modes down."""
    modes = vector_cladding_modes(FIBRE, WAVELENGTH, order=1, count=12)
    he = [mode for mode in modes if mode.family == "HE"]
    eh = [mode for mode in modes if mode.family == "EH"]
    lp0 = cladding_modes(FIBRE, WAVELENGTH, order=0, count=6)
    lp2 = cladding_modes(FIBRE, WAVELENGTH, order=2, count=6)
    splitting = [a.effective_index - b.effective_index for a, b in zip(he, lp0, strict=True)]
    assert splitting == pytest.approx(
        [-1.29e-7, -5.35e-7, -1.19e-6, -2.08e-6, -3.16e-6, -4.42e-6], rel=2e-2
    )
    assert all(a > b for a, b in pairwise(splitting)), "it grows down the band"
    for mode, partner in zip(eh, lp2, strict=True):
        assert mode.effective_index == pytest.approx(partner.effective_index, abs=5e-6)


def test_the_core_reaches_the_modes_the_scalar_solver_has_and_the_ones_it_does_not() -> None:
    """HE1m as the LP0m couple, to a part in a thousand; EH1m at a percent of that."""
    (core,) = vector_core_modes(FIBRE, WAVELENGTH)
    (scalar_core,) = core_modes(FIBRE, WAVELENGTH)
    modes = vector_cladding_modes(FIBRE, WAVELENGTH, order=1, count=8)
    lp0 = cladding_modes(FIBRE, WAVELENGTH, order=0, count=4)
    for mode in (m for m in modes if m.family == "HE"):
        assert vector_coupling(core, mode, 3e-4) == pytest.approx(
            lpg_coupling(scalar_core, lp0[mode.rank - 1], 3e-4), rel=2e-3
        )
    eh = [abs(vector_coupling(core, m, 3e-4)) for m in modes if m.family == "EH"]
    he = [abs(vector_coupling(core, m, 3e-4)) for m in modes if m.family == "HE"]
    assert eh == pytest.approx([0.140, 0.334, 0.581, 0.882], rel=2e-2)
    ratios = [small / large for small, large in zip(eh, he, strict=True)]
    assert max(ratios) < 0.02, "weak -- under two percent -- and not zero"
    assert min(ratios) > 0.005
    zero_order = vector_cladding_modes(FIBRE, WAVELENGTH, order=0, count=1)[0]
    assert vector_coupling(core, zero_order, 3e-4) == 0.0, "no coupling across azimuthal orders"


def test_the_modes_carry_orthogonal_power() -> None:
    """Distinct modes of one waveguide, so the cross term of their power flow must vanish."""
    modes = vector_cladding_modes(FIBRE, WAVELENGTH, order=1, count=5)
    for i, first in enumerate(modes):
        for second in modes[i + 1 :]:

            def cross(r: np.ndarray, a: object = first, b: object = second) -> np.ndarray:
                e_r, e_phi, _, _ = a.transverse(r)  # type: ignore[attr-defined]
                _, _, h_r, h_phi = b.transverse(r)  # type: ignore[attr-defined]
                return e_r * h_phi + e_phi * h_r

            overlap = first.integrate(cross) / math.sqrt(first.power() * second.power())
            assert abs(overlap) < 1e-10, (first.name, second.name, overlap)


def test_what_is_not_a_fibre_is_refused() -> None:
    with pytest.raises(ValueError, match="window must lie"):
        vector_modes([62.5e-6], [1.444, 1.0], WAVELENGTH, order=1, window=(1.43, 1.5))
    with pytest.raises(ValueError, match="higher index than the outside"):
        vector_modes([62.5e-6], [1.0, 1.444], WAVELENGTH, order=1, window=(1.1, 1.2))
    with pytest.raises(ValueError, match="one more index"):
        vector_modes([1e-6, 2e-6], [1.45, 1.0], WAVELENGTH, order=0, window=(1.1, 1.2))
    with pytest.raises(ValueError, match="increasing"):
        vector_modes([2e-6, 1e-6], [1.45, 1.44, 1.0], WAVELENGTH, order=0, window=(1.1, 1.2))
    with pytest.raises(ValueError, match="non-negative"):
        vector_modes([62.5e-6], [1.444, 1.0], WAVELENGTH, order=-1, window=(1.43, 1.44))
    with pytest.raises(ValueError, match="at least one cladding mode"):
        vector_cladding_modes(FIBRE, WAVELENGTH, count=0)
    with pytest.raises(ValueError, match="one wavelength"):
        (core,) = vector_core_modes(FIBRE, WAVELENGTH)
        vector_coupling(core, vector_core_modes(FIBRE, 1.3e-6)[0], 1e-4)


# ---------------------------------------------------------------------------
# What it moves: a long-period grating's notch
# ---------------------------------------------------------------------------

PERIOD = 500e-6
BAND = (1.575e-6, 1.595e-6)


def test_the_vector_notch_sits_a_nanometre_short_of_the_scalar_one() -> None:
    """The LP04 notch at 1584.1 nm is an HE14 notch at 1583.0, and as deep.

    The minimum sits about 0.17 nm short of the phase-matched wavelength either
    way, because the neighbouring modes' tails pull it; what the vector modes
    move is the pair of them together, by 1.10 nm.

    A nanometre is not a rounding error for a device whose whole use is reading
    where its notch went, and it is the size of the shift a real refractive index
    change is measured through -- which is the reason to have the vector modes.
    """
    scalar = long_period_resonances(FIBRE, period=PERIOD, count=4, band=BAND, grid_points=9)
    vector = long_period_resonances(
        FIBRE, period=PERIOD, count=8, band=BAND, grid_points=9, vector=True
    )
    assert len(scalar) == len(vector) == 1
    assert scalar[0][1] * 1e9 == pytest.approx(1584.12, abs=0.02)
    assert vector[0][1] * 1e9 == pytest.approx(1583.02, abs=0.02)

    frequencies = C_LIGHT / np.linspace(1.5815e-6, 1.5855e-6, 161)
    settings: dict[str, Any] = {
        "fibre": FIBRE,
        "period": PERIOD,
        "length": 25e-3,
        "index_modulation": 3e-4,
    }
    lp = long_period_grating(frequencies, cladding_modes=4, grid_points=5, **settings)
    he = long_period_grating(frequencies, cladding_modes=8, grid_points=5, vector=True, **settings)
    wavelengths = C_LIGHT / frequencies
    lp_notch = np.abs(lp.s[:, 1, 0]) ** 2
    he_notch = np.abs(he.s[:, 1, 0]) ** 2
    scalar_notch = wavelengths[lp_notch.argmin()] * 1e9
    vector_notch = wavelengths[he_notch.argmin()] * 1e9
    assert scalar_notch == pytest.approx(1583.95, abs=0.06)
    assert vector_notch == pytest.approx(1582.85, abs=0.06)
    assert scalar_notch - vector_notch == pytest.approx(1.10, abs=0.06)
    assert 10 * math.log10(he_notch.min()) == pytest.approx(
        10 * math.log10(lp_notch.min()), abs=0.3
    ), "as deep: the splitting moves the notch, it does not fill it"
