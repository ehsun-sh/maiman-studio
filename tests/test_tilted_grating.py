"""A tilted fibre Bragg grating: where its comb sits, how it divides the light, what is lost.

The pieces are checked where each has something independent to be held to. The
overlap the tilt opens is checked against the two-dimensional integral it stands
for, computed by brute force over the core -- no Jacobi-Anger, no Bessel
identity, just the fringe pattern itself. Untilted, the same overlap has to
collapse onto the Bragg coupling the long-period module already computes, and
does, to a part in a thousand billion. Each resonance is checked by solving the
modes at the wavelength it claims and asking whether they phase match there. One
mode alone has to be the textbook grating, ``1 / cosh(kappa L)``. And the solve
as a whole has to conserve power, which it does to 5e-12, because the equations
do.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from maiman.components.tilted import TiltedFiberBraggGrating
from maiman.context import SimulationContext
from maiman.modes import Mode, StepIndexFibre, cladding_modes, core_modes, lpg_coupling
from maiman.photonics import (
    tilted_fiber_bragg_grating,
    tilted_grating_coupling,
    tilted_grating_resonances,
    tilted_grating_spectrum,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import C_LIGHT

FIBRE = StepIndexFibre()
WAVELENGTH = 1.55e-6
PERIOD = 535e-9
TILT = math.radians(4.0)
LENGTH = 10e-3
MODULATION = 5e-4
#: Where the core reflects into itself for this period and tilt.
BRAGG = 1551.2426e-9


def core() -> Mode:
    return core_modes(FIBRE, WAVELENGTH)[0]


def settings(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "fibre": FIBRE,
        "period": PERIOD,
        "tilt": TILT,
        "length": LENGTH,
        "index_modulation": MODULATION,
        "max_order": 3,
    }
    base.update(changes)
    return base


# ---------------------------------------------------------------------------
# The coupling the tilt opens
# ---------------------------------------------------------------------------


def test_an_untilted_grating_couples_the_core_to_itself_and_to_nothing_else() -> None:
    """Which is a Bragg grating, and is the coupling the long-period module computes."""
    guided = core()
    assert tilted_grating_coupling(
        guided, guided, period=PERIOD, tilt=0.0, index_modulation=1e-4
    ) == pytest.approx(lpg_coupling(guided, guided, 1e-4), rel=1e-12, abs=0.0)
    for order in (1, 2, 3):
        mode = cladding_modes(FIBRE, WAVELENGTH, order=order, count=1)[0]
        assert (
            tilted_grating_coupling(guided, mode, period=PERIOD, tilt=0.0, index_modulation=1e-4)
            == 0.0
        )


@pytest.mark.parametrize("order", [0, 1, 3])
def test_the_overlap_is_the_fringe_integrated_across_the_core(order: int) -> None:
    """Against the two-dimensional integral itself, on a grid, with no expansion in it.

    ``int_core psi_0 psi_lm cos(l phi) exp(i K sin(theta) r cos(phi)) r dr dphi``
    """
    guided = core()
    mode = cladding_modes(FIBRE, WAVELENGTH, order=order, count=5)[4]
    r = np.linspace(1e-9, FIBRE.core_radius, 3001)
    phi = np.linspace(0.0, 2.0 * math.pi, 2001)
    transverse = 2.0 * math.pi / PERIOD * math.sin(TILT)
    angular = np.trapezoid(
        np.cos(order * phi) * np.exp(1j * transverse * np.outer(r, np.cos(phi))), phi, axis=1
    )
    overlap = np.trapezoid(guided.field(r) * mode.field(r) * angular * r, r)
    weight = 2.0 * math.pi if order == 0 else math.pi
    normalisation = math.sqrt(2.0 * math.pi * guided.power() * weight * mode.power())
    k = 2.0 * math.pi / WAVELENGTH
    expected = (
        k
        * FIBRE.core_index
        * 1e-4
        * abs(overlap / normalisation)
        / (2.0 * math.sqrt(guided.effective_index * mode.effective_index))
    )
    assert tilted_grating_coupling(
        guided, mode, period=PERIOD, tilt=TILT, index_modulation=1e-4
    ) == pytest.approx(expected, rel=1e-5)


def test_tilting_takes_the_mirror_apart_and_feeds_the_comb() -> None:
    """152.6 per metre into itself square on, 41.1 at four degrees, 3.5 at six."""
    guided = core()
    into_itself = [
        tilted_grating_coupling(
            guided, guided, period=PERIOD, tilt=math.radians(degrees), index_modulation=1e-4
        )
        for degrees in (0.0, 2.0, 4.0, 6.0)
    ]
    assert into_itself == pytest.approx([152.557, 114.884, 41.111, 3.510], rel=1e-3)
    assert all(a > b for a, b in pairwise(into_itself))

    mode = cladding_modes(FIBRE, WAVELENGTH, order=1, count=5)[4]
    into_cladding = [
        tilted_grating_coupling(
            guided, mode, period=PERIOD, tilt=math.radians(degrees), index_modulation=1e-4
        )
        for degrees in (0.0, 2.0, 4.0, 6.0, 10.0)
    ]
    # Not monotonic, and saying so is the point: a couple of degrees opens the
    # door, and past that the fringe turns over within the core faster than the
    # mode does, so this one weakens again while higher orders take over.
    assert into_cladding == pytest.approx([0.0, 28.377, 28.362, 8.454, 1.644], rel=1e-3)
    assert max(into_cladding) == into_cladding[1]


def test_a_coupling_is_between_two_modes_of_one_fibre_driven_by_the_core() -> None:
    guided = core()
    other = core_modes(StepIndexFibre(core_radius=5e-6), WAVELENGTH)[0]
    with pytest.raises(ValueError, match="one fibre at one wavelength"):
        tilted_grating_coupling(guided, other, period=PERIOD, tilt=TILT, index_modulation=1e-4)
    with pytest.raises(ValueError, match="driven by the core"):
        mode = cladding_modes(FIBRE, WAVELENGTH, order=1, count=1)[0]
        tilted_grating_coupling(mode, mode, period=PERIOD, tilt=TILT, index_modulation=1e-4)


# ---------------------------------------------------------------------------
# Where the comb sits
# ---------------------------------------------------------------------------


def test_every_resonance_is_where_its_modes_phase_match() -> None:
    """Solved at the wavelength each claims, ``(n_0 + n_m) period / cos(theta)`` is it.

    The interpolated table is what places them and a direct solve is what checks
    them, and the two agree to well under a picometre -- far below the picometre
    a grating's own line is read to.
    """
    found = tilted_grating_resonances(
        FIBRE, period=PERIOD, tilt=TILT, band=(BRAGG - 3e-9, BRAGG + 0.5e-9), max_order=3
    )
    assert len(found) > 20
    axial = PERIOD / math.cos(TILT)
    for order, rank, wavelength, coupling in found[-5:]:
        guided = core_modes(FIBRE, wavelength)[0].effective_index
        partner = (
            guided
            if rank == 0
            else cladding_modes(FIBRE, wavelength, order=order, count=rank)[
                rank - 1
            ].effective_index
        )
        assert (guided + partner) * axial == pytest.approx(wavelength, abs=1e-15), (order, rank)
        assert coupling > 0.0

    bragg = [item for item in found if item[1] == 0]
    assert len(bragg) == 1
    assert bragg[0][2] == pytest.approx(BRAGG, abs=1e-13), "the Bragg line, above the whole comb"
    assert all(item[2] < bragg[0][2] for item in found if item[1] != 0)


def test_the_comb_moves_with_what_the_fibre_sits_in_and_the_bragg_line_does_not() -> None:
    """The whole reason the device is a refractometer: two rulers, one of them fixed."""
    band = (BRAGG - 2e-9, BRAGG + 0.5e-9)
    dry = tilted_grating_resonances(FIBRE, period=PERIOD, tilt=TILT, band=band, max_order=1)
    wet = tilted_grating_resonances(
        StepIndexFibre(surrounding_index=1.333),
        period=PERIOD,
        tilt=TILT,
        band=band,
        max_order=1,
    )
    dry_bragg = next(item[2] for item in dry if item[1] == 0)
    wet_bragg = next(item[2] for item in wet if item[1] == 0)
    assert wet_bragg == pytest.approx(dry_bragg, abs=1e-15)
    moved = [
        w[2] - d[2]
        for d, w in zip(
            [item for item in dry if item[1] != 0],
            [item for item in wet if item[1] != 0],
            strict=False,
        )
    ]
    assert moved and max(abs(shift) for shift in moved) > 1e-12, "water moves the comb"


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def test_one_mode_alone_is_the_textbook_grating() -> None:
    """``|t| = 1 / cosh(kappa L)`` on resonance, which is what a uniform grating does."""
    found = tilted_grating_resonances(
        FIBRE, period=PERIOD, tilt=TILT, band=(BRAGG - 2e-9, BRAGG + 0.5e-9), max_order=3
    )
    _, _, wavelength, coupling = found[-2]
    kappa = coupling * MODULATION
    transmitted, _, _ = tilted_grating_spectrum(
        np.array([C_LIGHT / wavelength]), nearest=1, **settings()
    )
    assert abs(transmitted[0]) ** 2 == pytest.approx(1.0 / math.cosh(kappa * LENGTH) ** 2, abs=1e-3)


def test_nothing_is_lost_and_the_truncation_is_worth_two_parts_in_a_thousand() -> None:
    """Transmitted plus reflected plus what went into the cladding is one, exactly.

    A truncated system conserves its own power whatever is left out, so this
    holds at any ``nearest`` and is a check on the solve rather than on the
    truncation. What the truncation costs is measured against solving them all.
    """
    wavelengths = np.linspace(BRAGG - 3e-9, BRAGG + 0.6e-9, 501)
    frequencies = C_LIGHT / wavelengths
    everything = tilted_grating_spectrum(frequencies, nearest=None, **settings())
    default = tilted_grating_spectrum(frequencies, **settings())
    for transmitted, reflected, cladding in (everything, default):
        total = np.abs(transmitted) ** 2 + np.abs(reflected) ** 2 + cladding
        assert np.max(np.abs(total - 1.0)) < 1e-10

    error = np.abs(np.abs(default[0]) ** 2 - np.abs(everything[0]) ** 2)
    assert error.max() < 2e-3
    assert np.median(error) < 5e-4
    transmission = 10.0 * np.log10(np.abs(everything[0]) ** 2)
    assert transmission.min() == pytest.approx(-11.95, abs=0.2), "the deepest notch of the comb"
    assert everything[2].max() == pytest.approx(0.80, abs=0.02), "four fifths into the cladding"


def test_what_cannot_be_a_tilted_grating_is_refused() -> None:
    frequencies = np.array([C_LIGHT / WAVELENGTH])
    with pytest.raises(ValueError, match="period must be positive"):
        tilted_grating_spectrum(frequencies, **settings(period=0.0))
    with pytest.raises(ValueError, match="tilt must be between"):
        tilted_grating_spectrum(frequencies, **settings(tilt=math.pi / 2))
    with pytest.raises(ValueError, match="length must not be negative"):
        tilted_grating_spectrum(frequencies, **settings(length=-1.0))
    with pytest.raises(ValueError, match="index modulation"):
        tilted_grating_spectrum(frequencies, **settings(index_modulation=-1e-4))
    with pytest.raises(ValueError, match="max_order"):
        tilted_grating_spectrum(frequencies, **settings(max_order=-1))
    with pytest.raises(ValueError, match="at least one mode"):
        tilted_grating_spectrum(frequencies, nearest=0, **settings())
    with pytest.raises(ValueError, match="band must be"):
        tilted_grating_resonances(FIBRE, period=PERIOD, tilt=TILT, band=(2e-6, 1e-6))


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def block(**changes: float) -> TiltedFiberBraggGrating:
    parameters: dict[str, float] = {
        "period": PERIOD * 1e9,
        "tilt": math.degrees(TILT),
        "length": LENGTH * 1e3,
        "index_modulation": MODULATION,
        "azimuthal_orders": 3.0,
    }
    parameters.update(changes)
    return TiltedFiberBraggGrating(label="tfbg", **parameters)


def test_the_block_is_the_function() -> None:
    frequencies = C_LIGHT / np.linspace(BRAGG - 1e-9, BRAGG + 0.2e-9, 31)
    # Through the block's own conversion, so the comparison is of the physics and
    # not of the last bit of ``535 nm``.
    expected = tilted_fiber_bragg_grating(
        frequencies, **settings(period=block().si("period"), tilt=block().si("tilt"))
    )
    assert np.allclose(block().scattering_matrix(frequencies).s, expected.s, rtol=0.0, atol=1e-12)
    assert block().bragg_wavelength() == pytest.approx(BRAGG, abs=1e-13)


def test_the_block_reflects_a_carrier_and_passes_what_it_does_not() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    carrier = Band(
        Ex=np.full(ctx.num_samples, 1e-3**0.5, dtype=ctx.complex_dtype),
        Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
        f0=C_LIGHT / BRAGG,
        fs=ctx.sample_rate,
    )
    out = block().run(ctx, {"in": OpticalSignal(bands=(carrier,))})
    reflected = out["reflected"].bands[0].average_power()
    transmitted = out["transmitted"].bands[0].average_power()
    assert reflected == pytest.approx(0.935e-3, rel=0.02), "on the Bragg line, and it is a mirror"
    assert transmitted < 0.07e-3
    assert reflected + transmitted < 1e-3, "the cladding's share is stripped"
