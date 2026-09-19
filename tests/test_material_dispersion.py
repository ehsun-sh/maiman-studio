"""The glass disperses: Sellmeier indices under the mode solvers.

The checks go from the glass up. Malitson's silica is held to its tabulated
indices and to the zero of its own material dispersion near 1.27 microns. Then
the fibre: with the glass dispersing, the default step-index fibre becomes the
standard single-mode fibre it was always meant to be -- dispersion zero inside
G.652's 1300 to 1324 nm, about 17 ps/nm/km at 1550 -- where with constant
indices it had only its waveguide's negative dispersion, and the total is the
sum of the two to within half a picosecond at 1550 nm. Last, what it does to
the gratings: a long-period grating's notches, set by the difference of two
indices that disperse differently, move by nanometres; a tilted grating's comb,
read against indices that hold at 1550 nm, by picometres.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman.components import LongPeriodGrating
from maiman.components.tilted import TiltedFiberBraggGrating
from maiman.modes import (
    StepIndexFibre,
    cladding_modes,
    core_modes,
    germania_fraction,
    glass_index,
)
from maiman.units import C_LIGHT

# ---------------------------------------------------------------------------
# The glass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wavelength", "index"),
    [(0.5876e-6, 1.4585), (1.0e-6, 1.4504), (1.55e-6, 1.4440)],
)
def test_silica_is_malitsons(wavelength: float, index: float) -> None:
    assert float(glass_index(wavelength)) == pytest.approx(index, abs=6e-5)


def test_silica_stops_dispersing_near_one_point_two_seven_microns() -> None:
    wavelengths = np.linspace(1.2e-6, 1.35e-6, 15001)
    curvature = np.gradient(np.gradient(glass_index(wavelengths), wavelengths), wavelengths)
    inner = slice(50, -50)
    zero = wavelengths[inner][np.argmin(np.abs(curvature[inner]))]
    assert zero == pytest.approx(1.2727e-6, abs=1e-9)


def test_germania_raises_the_index_and_its_fraction_is_found_again() -> None:
    assert float(glass_index(1.55e-6, 0.0)) == float(glass_index(1.55e-6))
    fraction = germania_fraction(0.0052)
    assert fraction == pytest.approx(0.0346, abs=5e-4), "3.5 mol %, a standard fibre's core"
    step = float(glass_index(1.55e-6, fraction) - glass_index(1.55e-6))
    assert step == pytest.approx(0.0052, abs=1e-12)
    with pytest.raises(ValueError, match="not reachable"):
        germania_fraction(0.5)
    with pytest.raises(ValueError, match="mole fraction"):
        glass_index(1.55e-6, 1.5)


# ---------------------------------------------------------------------------
# The fibre
# ---------------------------------------------------------------------------


def test_the_quoted_indices_hold_at_the_reference_and_nowhere_else() -> None:
    plain = StepIndexFibre()
    assert plain.at(1.3e-6) is plain
    fibre = StepIndexFibre(material_dispersion=True)
    there = fibre.at(1.55e-6)
    assert there.core_index == pytest.approx(1.4492, abs=1e-15)
    assert there.cladding_index == pytest.approx(1.4440, abs=1e-15)
    elsewhere = fibre.at(1.3e-6)
    assert elsewhere.cladding_index == pytest.approx(float(glass_index(1.3e-6)), abs=3e-5)
    step = elsewhere.core_index - elsewhere.cladding_index
    assert step == pytest.approx(0.00517, abs=1e-5), "the step itself disperses"
    assert core_modes(fibre, 1.55e-6)[0].effective_index == pytest.approx(
        core_modes(plain, 1.55e-6)[0].effective_index, abs=1e-14
    )
    assert cladding_modes(fibre, 1.55e-6, count=3)[2].effective_index == pytest.approx(
        cladding_modes(plain, 1.55e-6, count=3)[2].effective_index, abs=1e-14
    )
    with pytest.raises(ValueError, match="reference wavelength"):
        StepIndexFibre(material_dispersion=True, reference_wavelength=0.0)


def dispersion(fibre: StepIndexFibre, wavelength: float, step: float = 2e-9) -> float:
    """``D = -(lambda / c) d^2 n_eff / d lambda^2`` of the core mode [ps/(nm km)]."""

    def n(x: float) -> float:
        return core_modes(fibre, x)[0].effective_index

    curvature = (n(wavelength + step) - 2 * n(wavelength) + n(wavelength - step)) / step**2
    return -wavelength / C_LIGHT * curvature * 1e6


def material(wavelength: float, step: float = 2e-9) -> float:
    def n(x: float) -> float:
        return float(glass_index(x))

    curvature = (n(wavelength + step) - 2 * n(wavelength) + n(wavelength - step)) / step**2
    return -wavelength / C_LIGHT * curvature * 1e6


def test_the_default_fibre_becomes_a_standard_single_mode_fibre() -> None:
    fibre = StepIndexFibre(material_dispersion=True)
    at_1550 = dispersion(fibre, 1.55e-6)
    assert 15.0 < at_1550 < 18.0, "G.652 allows up to 18"
    assert at_1550 == pytest.approx(16.74, abs=0.05)

    lo, hi = 1.25e-6, 1.40e-6
    for _ in range(30):
        middle = 0.5 * (lo + hi)
        lo, hi = (middle, hi) if dispersion(fibre, middle) < 0.0 else (lo, middle)
    zero = 0.5 * (lo + hi)
    assert 1.300e-6 < zero < 1.324e-6, "inside G.652's window"
    assert zero == pytest.approx(1.3083e-6, abs=5e-10)


def test_without_the_glass_only_the_waveguide_disperses_and_the_two_add() -> None:
    plain = StepIndexFibre()
    fibre = StepIndexFibre(material_dispersion=True)
    for wavelength in (1.3e-6, 1.55e-6):
        assert dispersion(plain, wavelength) < 0.0
    # Away from the zero the sum is good to half a picosecond: 21.9 of silica's
    # own and -4.8 of the waveguide's against 16.7. What is left is the cross
    # term -- the core's germania and the group index the waveguide part carries.
    total = dispersion(fibre, 1.55e-6)
    assert total == pytest.approx(material(1.55e-6) + dispersion(plain, 1.55e-6), abs=0.5)


# ---------------------------------------------------------------------------
# The gratings
# ---------------------------------------------------------------------------


def test_a_long_period_grating_s_notches_move_by_nanometres() -> None:
    plain = dict(LongPeriodGrating(label="l").resonances())
    glass = LongPeriodGrating(label="l", material_dispersion=True)
    moved = dict(glass.resonances())
    assert sorted(moved) == sorted(plain)
    shifts = {rank: (moved[rank] - plain[rank]) * 1e9 for rank in plain}
    assert shifts[1] == pytest.approx(-4.17, abs=0.05)
    assert shifts[4] == pytest.approx(+1.37, abs=0.05)

    # And each one is where the dispersive fibre's own indices phase match.
    fibre = glass.fibre()
    period = glass.si("period")
    for rank, wavelength in moved.items():
        n0 = core_modes(fibre, wavelength)[0].effective_index
        nm = cladding_modes(fibre, wavelength, count=rank)[rank - 1].effective_index
        assert (n0 - nm) * period == pytest.approx(wavelength, rel=1e-9)


def test_a_tilted_grating_s_bragg_line_barely_moves() -> None:
    plain = TiltedFiberBraggGrating(label="t").bragg_wavelength()
    glass = TiltedFiberBraggGrating(label="t", material_dispersion=True).bragg_wavelength()
    assert abs(glass - plain) < 0.05e-9
    assert glass != plain


def test_the_flags_are_off_by_default() -> None:
    assert not LongPeriodGrating().material_dispersion
    assert not TiltedFiberBraggGrating().material_dispersion
    assert not StepIndexFibre().material_dispersion
