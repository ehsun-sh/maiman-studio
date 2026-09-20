"""What a grating coupler's teeth send back, and what the gap above it does.

Four things the stack model did not have, each checked against something that
does not share its algebra:

* the reflection into the waveguide, from the teeth's second order rather than
  from a quoted number -- against this module's own transfer-matrix grating;
* a bottom mirror, as a complex substrate index -- against the share going up
  with nothing but silicon under the oxide;
* the height the fibre sits at, which widens its beam, curves it, and rings
  between the facet and the chip -- against the etalon's own ``lambda^2 / 2 n h``;
* and apodization, a strength taper along the grating, which makes the emitted
  beam rounder and so easier for a fibre to catch.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from maiman.components import GratingCoupler
from maiman.photonics import (
    _fibre_gap_etalon,
    fiber_bragg_grating,
    fresnel_reflectance,
    grating_coupler_stack,
    grating_directionality,
    grating_harmonic,
    grating_tooth_reflection,
    stack_reflection,
)
from maiman.units import C_LIGHT

PERIOD = 611e-9
N_EFF = 2.71
LENGTH = 20e-6
REFERENCE = 1.55e-6


def teeth(
    wavelength: np.ndarray, *, contrast: float, duty: float, period: float = PERIOD
) -> np.ndarray:
    return grating_tooth_reflection(
        wavelength,
        period=period,
        effective_index=N_EFF,
        group_index=N_EFF,  # constant, so the comparison is against a constant grating
        reference_wavelength=REFERENCE,
        index_contrast=contrast,
        duty=duty,
        length=LENGTH,
    )


# ---------------------------------------------------------------------------
# The teeth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("contrast", "duty"), [(0.5, 0.4), (0.3, 0.3), (0.2, 0.65)])
def test_the_teeth_reflect_as_the_transfer_matrix_grating_does(
    contrast: float, duty: float
) -> None:
    """Its second order is a Bragg grating at ``n_eff Lambda``, and is solved as one."""
    wavelengths = np.linspace(1.50e-6, 1.75e-6, 2001)
    closed = np.abs(teeth(wavelengths, contrast=contrast, duty=duty))
    assembled = np.abs(
        fiber_bragg_grating(
            C_LIGHT / wavelengths,
            length=LENGTH,
            index_modulation=abs(grating_harmonic(contrast, duty, 2)),
            bragg_wavelength=N_EFF * PERIOD,
            n_eff=N_EFF,
            sections=400,
        ).s[:, 0, 0]
    )
    assert float(np.max(np.abs(closed - assembled))) < 1e-9
    assert float(closed.max()) > 0.9, "on its own resonance it is a mirror"


def test_a_fifty_fifty_grating_has_no_second_harmonic_and_reflects_nothing() -> None:
    assert grating_harmonic(0.5, 0.5, 2) == pytest.approx(0.0, abs=1e-15)
    assert grating_harmonic(0.5, 0.5, 1) == pytest.approx(2 * 0.5 / math.pi, rel=1e-12)
    nothing = teeth(np.linspace(1.5e-6, 1.7e-6, 101), contrast=0.5, duty=0.5)
    assert float(np.max(np.abs(nothing))) < 1e-14
    # Either side of half it grows again, and symmetrically.
    below = abs(complex(teeth(np.array([REFERENCE]), contrast=0.3, duty=0.45)[0]))
    above = abs(complex(teeth(np.array([REFERENCE]), contrast=0.3, duty=0.55)[0]))
    assert below == pytest.approx(above, rel=1e-12)
    assert 20 * math.log10(above) == pytest.approx(-21.5, abs=0.5)


def test_the_angle_is_what_keeps_the_reflection_down() -> None:
    """Straight up is on resonance; every degree of tilt detunes it further."""
    reflections = []
    for degrees in (0.0, 2.0, 5.0, 10.0, 15.0):
        # The period that radiates at this angle at the reference wavelength.
        period = REFERENCE / (N_EFF - math.sin(math.radians(degrees)))
        reflections.append(
            abs(complex(teeth(np.array([REFERENCE]), contrast=0.3, duty=0.55, period=period)[0]))
        )
    assert reflections[0] > 0.8, "a vertical coupler reflects its own Bragg line"
    assert all(a > b for a, b in pairwise(reflections))
    assert 20 * math.log10(reflections[3]) < -20.0


def test_what_is_not_a_grating_is_refused() -> None:
    with pytest.raises(ValueError, match="order"):
        grating_harmonic(0.3, 0.5, 0)
    with pytest.raises(ValueError, match="duty"):
        grating_harmonic(0.3, 1.5, 2)
    with pytest.raises(ValueError, match="period and length"):
        teeth(np.array([REFERENCE]), contrast=0.3, duty=0.5, period=-1.0)


# ---------------------------------------------------------------------------
# The stack, from above and below
# ---------------------------------------------------------------------------


def test_a_bare_interface_reflects_its_fresnel_share() -> None:
    """Silicon thick enough to hide what is under it is one interface."""
    reflected = abs(
        complex(
            stack_reflection(
                REFERENCE,
                emission_sine=0.0,
                silicon_thickness=1e-3,
                box_thickness=0.0,
            )
        )
    )
    assert reflected**2 == pytest.approx(fresnel_reflectance(1.0, 3.476), rel=1e-9)
    nothing = stack_reflection(
        REFERENCE,
        emission_sine=0.0,
        top_index=1.444,
        silicon_index=1.444,
        box_index=1.444,
        substrate_index=1.444,
    )
    assert abs(complex(nothing)) < 1e-12, "a stack with no steps reflects nothing"


def test_a_mirror_under_the_oxide_sends_almost_everything_up() -> None:
    sine = math.sin(math.radians(10.0))
    plain = float(grating_directionality(REFERENCE, emission_sine=sine))
    mirrored = float(
        grating_directionality(REFERENCE, emission_sine=sine, substrate_index=1.44 + 16j)
    )
    assert plain == pytest.approx(0.592, abs=0.002)
    assert mirrored > 0.98
    assert mirrored <= 1.0
    assert (
        abs(complex(stack_reflection(REFERENCE, emission_sine=sine, substrate_index=1.44 + 16j)))
        <= 1.0
    )


def test_a_gain_medium_is_refused_rather_than_quietly_flipped() -> None:
    for function in (grating_directionality, stack_reflection):
        with pytest.raises(ValueError, match="amplifies"):
            function(REFERENCE, emission_sine=0.0, substrate_index=1.44 - 16j)


# ---------------------------------------------------------------------------
# The gap above the chip
# ---------------------------------------------------------------------------


def etalon(wavelengths: np.ndarray, *, height: float, angle: float) -> np.ndarray:
    return _fibre_gap_etalon(
        wavelengths,
        emission_sine=np.full_like(wavelengths, 0.0),
        radius=5.2e-6,
        height=height,
        angle=angle,
        top_index=1.0,
        fibre_index=1.444,
        silicon_index=3.476,
        silicon_thickness=220e-9,
        box_index=1.444,
        box_thickness=2e-6,
        substrate_index=3.476,
    )


@pytest.mark.parametrize("height", [100e-6, 200e-6])
def test_the_gap_rings_at_its_own_free_spectral_range(height: float) -> None:
    angle = math.radians(10.0)
    wavelengths = np.linspace(1.50e-6, 1.60e-6, 40001)
    factor = etalon(wavelengths, height=height, angle=angle)
    peaks = np.flatnonzero((factor[1:-1] > factor[:-2]) & (factor[1:-1] > factor[2:])) + 1
    measured = float(np.mean(np.diff(wavelengths[peaks])))
    expected = REFERENCE**2 / (2.0 * height / math.cos(angle))
    assert measured == pytest.approx(expected, rel=0.05)


def test_tilting_the_fibre_damps_the_ring() -> None:
    """Each round trip lands ``2 h tan(theta)`` to the side, and misses the mode."""
    wavelengths = np.linspace(1.54e-6, 1.56e-6, 8001)
    ripples = [
        10 * math.log10(float(etalon(wavelengths, height=30e-6, angle=math.radians(a)).max()))
        - 10 * math.log10(float(etalon(wavelengths, height=30e-6, angle=math.radians(a)).min()))
        for a in (0.0, 12.0)
    ]
    assert ripples[0] > 0.5
    assert ripples[1] < 0.2
    assert np.all(etalon(wavelengths, height=0.0, angle=0.0) == 1.0), "no gap, no cavity"


def test_a_fibre_further_away_couples_less() -> None:
    """With the facet's reflection taken out, what is left is the beam spreading."""
    frequencies = C_LIGHT / np.linspace(1.50e-6, 1.60e-6, 401)
    losses = []
    for height in (0.0, 50e-6, 150e-6):
        s = grating_coupler_stack(
            frequencies,
            fibre_height=height,
            fibre_index=1.0,
            period=PERIOD,
            effective_index=N_EFF,
            group_index=4.0,
            reference_wavelength=REFERENCE,
            angle=math.radians(10.0),
        )
        losses.append(-10 * math.log10(float((np.abs(s.s[:, 1, 0]) ** 2).max())))
    assert losses[0] == pytest.approx(3.192, abs=0.01)
    assert all(a < b for a, b in pairwise(losses))
    assert losses[2] - losses[0] > 1.0


# ---------------------------------------------------------------------------
# Apodization
# ---------------------------------------------------------------------------


def test_a_tapered_grating_meets_the_fibre_better_than_a_uniform_one() -> None:
    frequencies = C_LIGHT / np.linspace(1.50e-6, 1.60e-6, 801)

    def loss(strength: float = 0.14e6, strength_end: float | None = None) -> float:
        s = grating_coupler_stack(
            frequencies,
            period=PERIOD,
            effective_index=N_EFF,
            group_index=4.0,
            reference_wavelength=REFERENCE,
            angle=math.radians(10.0),
            strength=strength,
            strength_end=strength_end,
        )
        return -10 * math.log10(float((np.abs(s.s[:, 1, 0]) ** 2).max()))

    uniform = loss()
    tapered = loss(strength=0.05e6, strength_end=0.4e6)
    assert uniform == pytest.approx(3.192, abs=0.01)
    assert uniform - tapered > 0.5, "an apodized grating is half a decibel better"
    assert loss(strength=0.14e6, strength_end=0.14e6) == pytest.approx(uniform, rel=1e-12)
    with pytest.raises(ValueError, match="strength"):
        loss(strength_end=-1.0)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def test_the_block_leaves_all_of_it_off_until_asked() -> None:
    coupler = GratingCoupler()
    assert not coupler.from_teeth
    assert coupler.si("fibre_height") == 0.0
    assert coupler.si("grating_strength_end") == 0.0
    assert coupler.substrate_extinction == 0.0
    specs = GratingCoupler.param_specs()
    assert specs["index_contrast"].applies_when == "from_teeth"
    assert specs["fibre_height"].applies_when == "from_stack"


def test_the_block_reflects_what_its_teeth_reflect() -> None:
    frequencies = C_LIGHT / np.linspace(1.53e-6, 1.57e-6, 201)
    quoted = GratingCoupler(label="g", from_stack=True, back_reflection=20.0)
    computed = GratingCoupler(label="g", from_stack=True, from_teeth=True, duty=0.55)
    flat = np.abs(quoted._matrix_factory("te")(frequencies).s[:, 0, 0])
    teeth_back = np.abs(computed._matrix_factory("te")(frequencies).s[:, 0, 0])
    assert np.allclose(flat, 10 ** (-1.0))
    middle = int(np.argmin(np.abs(C_LIGHT / frequencies - REFERENCE)))
    assert 20 * math.log10(float(teeth_back[middle])) == pytest.approx(-21.5, abs=1.0)
    assert teeth_back.max() > teeth_back.min(), "it rings with the grating's length"
    half = GratingCoupler(label="g", from_stack=True, from_teeth=True, duty=0.5)
    assert float(np.max(np.abs(half._matrix_factory("te")(frequencies).s[:, 0, 0]))) < 1e-14


def test_a_mirror_and_a_taper_make_the_coupler_a_good_one() -> None:
    frequencies = C_LIGHT / np.linspace(1.50e-6, 1.60e-6, 401)
    plain = GratingCoupler(label="g", from_stack=True)
    best = GratingCoupler(
        label="g",
        from_stack=True,
        substrate_extinction=16.0,
        grating_strength=0.05,
        grating_strength_end=0.4,
    )

    def loss(block: GratingCoupler) -> float:
        s = block._matrix_factory("te")(frequencies)
        return -10 * math.log10(float((np.abs(s.s[:, 1, 0]) ** 2).max()))

    assert loss(plain) == pytest.approx(3.19, abs=0.02)
    assert loss(best) < 1.0
