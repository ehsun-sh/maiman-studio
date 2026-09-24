"""A long-period grating, on cladding modes the solver found.

The coupled equations are solved exactly for a uniform grating, so with one
cladding mode coupled the transmission has to *be* the textbook closed form, and
with none it has to be one. The rest pin what the device is used for: where its
notches land, how deep they cut, and how far a liquid moves them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import LongPeriodGrating
from maiman.context import SimulationContext
from maiman.modes import StepIndexFibre, cladding_modes, core_modes, lpg_coupling
from maiman.photonics import long_period_grating, long_period_resonances
from maiman.signals import Band, OpticalSignal
from maiman.units import C_LIGHT

FIBRE = StepIndexFibre()
PERIOD = 500e-6
LENGTH = 25e-3
MODULATION = 3e-4


@pytest.fixture(scope="module")
def resonances() -> list[tuple[int, float]]:
    return long_period_resonances(FIBRE, period=PERIOD, count=8)


def closed_form(wavelength: float, rank: int) -> float:
    """``cos^2(gamma L) + (Delta / 2 gamma)^2 sin^2(gamma L)``, for one cladding mode alone."""
    core = core_modes(FIBRE, wavelength)[0]
    mode = cladding_modes(FIBRE, wavelength, count=rank)[rank - 1]
    kappa = lpg_coupling(core, mode, MODULATION)
    delta = (
        2 * math.pi / wavelength * (mode.effective_index - core.effective_index)
        + 2 * math.pi / PERIOD
    )
    gamma = math.sqrt(kappa**2 + delta**2 / 4)
    return (
        math.cos(gamma * LENGTH) ** 2 + (delta / (2 * gamma)) ** 2 * math.sin(gamma * LENGTH) ** 2
    )


def test_the_notches_are_where_phase_matching_puts_them(
    resonances: list[tuple[int, float]],
) -> None:
    """Four cladding modes phase-match between 1.2 and 1.7 um at a 500 um period."""
    assert [rank for rank, _ in resonances] == [1, 2, 3, 4]
    assert [w * 1e9 for _, w in resonances] == pytest.approx(
        [1354.89, 1388.68, 1455.73, 1584.12], abs=0.01
    )
    for rank, wavelength in resonances:
        core = core_modes(FIBRE, wavelength)[0]
        mode = cladding_modes(FIBRE, wavelength, count=rank)[rank - 1]
        mismatch = core.effective_index - mode.effective_index - wavelength / PERIOD
        # 1e-11 of index is 5e-15 m of wavelength at this period: the polish has
        # converged to the solver's own precision, not to the interpolation's.
        assert abs(mismatch) < 1e-11, (rank, mismatch)


def test_one_mode_alone_is_the_closed_form(resonances: list[tuple[int, float]]) -> None:
    """Across a notch, and at its centre ``cos^2(kappa L)`` exactly."""
    centre = resonances[0][1]
    wavelengths = centre + np.array([-20e-9, -5e-9, 0.0, 3e-9, 15e-9])
    matrix = long_period_grating(
        C_LIGHT / wavelengths,
        fibre=FIBRE,
        period=PERIOD,
        length=LENGTH,
        index_modulation=MODULATION,
        cladding_modes=1,
    )
    measured = np.abs(matrix.s[:, 1, 0]) ** 2
    expected = [closed_form(float(w), 1) for w in wavelengths]
    assert measured == pytest.approx(expected, rel=0.0, abs=1e-10)

    core = core_modes(FIBRE, centre)[0]
    mode = cladding_modes(FIBRE, centre, count=1)[0]
    kappa = lpg_coupling(core, mode, MODULATION)
    assert measured[2] == pytest.approx(math.cos(kappa * LENGTH) ** 2, rel=0.0, abs=1e-10)


def test_without_modulation_the_light_goes_straight_through() -> None:
    wavelengths = np.linspace(1.3e-6, 1.6e-6, 7)
    matrix = long_period_grating(
        C_LIGHT / wavelengths, fibre=FIBRE, period=PERIOD, length=LENGTH, index_modulation=0.0
    )
    assert np.array_equal(matrix.s[:, 1, 0], np.ones(7, dtype=complex))
    assert np.array_equal(matrix.s[:, 0, 0], np.zeros(7, dtype=complex)), "nothing reflects"


@pytest.fixture(scope="module")
def spectrum() -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.linspace(1.2e-6, 1.7e-6, 2001)
    matrix = long_period_grating(
        C_LIGHT / wavelengths,
        fibre=FIBRE,
        period=PERIOD,
        length=LENGTH,
        index_modulation=MODULATION,
    )
    return wavelengths, matrix.s[:, 1, 0]


def test_a_spectrum_is_interpolated_to_a_direct_solve(
    spectrum: tuple[np.ndarray, np.ndarray],
) -> None:
    """Seventeen Chebyshev nodes stand in for 2001 mode solves, to a part in ten million."""
    wavelengths, through = spectrum
    picked = wavelengths[::400]
    direct = long_period_grating(
        C_LIGHT / picked, fibre=FIBRE, period=PERIOD, length=LENGTH, index_modulation=MODULATION
    )
    assert np.max(np.abs(direct.s[:, 1, 0] - through[::400])) < 1e-7


def test_the_cladding_takes_light_and_gives_none(spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    _, through = spectrum
    assert float(np.max(np.abs(through))) <= 1.0 + 1e-12


def test_a_liquid_pulls_the_notches_shortward(resonances: list[tuple[int, float]]) -> None:
    """Air to water to 1.43: LP04 moves 2.6 nm and then 12.1 nm, LP01 far less.

    A higher surrounding index lets a cladding mode's evanescent tail reach
    further out, which raises its effective index and shrinks the difference the
    phase matching is made of. The closer the liquid comes to the cladding's own
    index, the faster it moves -- which is the refractometer's sensitivity curve.
    """
    dry = dict(resonances)
    shifts: dict[float, dict[int, float]] = {}
    for liquid in (1.333, 1.43):
        wet = dict(
            long_period_resonances(StepIndexFibre(surrounding_index=liquid), period=PERIOD, count=8)
        )
        shifts[liquid] = {rank: (wet[rank] - dry[rank]) * 1e9 for rank in dry}

    assert shifts[1.333][4] == pytest.approx(-2.62, abs=0.05)
    assert shifts[1.43][4] == pytest.approx(-12.14, abs=0.05)
    assert all(shift < 0.0 for by_rank in shifts.values() for shift in by_rank.values())
    assert abs(shifts[1.43][1]) < abs(shifts[1.43][4]) / 10


def test_the_block_is_the_function_and_cuts_a_carrier_by_it(
    resonances: list[tuple[int, float]],
) -> None:
    block = LongPeriodGrating()
    centre = resonances[-1][1]
    frequencies = C_LIGHT / (centre + np.linspace(-10e-9, 10e-9, 9))
    expected = long_period_grating(
        frequencies, fibre=FIBRE, period=PERIOD, length=LENGTH, index_modulation=MODULATION
    )
    assert np.allclose(block.scattering_matrix(frequencies).s, expected.s, rtol=0.0, atol=1e-12)
    assert block.resonances() == pytest.approx(resonances, rel=1e-12)

    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=16)
    f0 = C_LIGHT / centre
    carrier = Band(
        Ex=np.full(ctx.num_samples, 1e-3**0.5, dtype=ctx.complex_dtype),
        Ey=np.zeros(ctx.num_samples, dtype=ctx.complex_dtype),
        f0=f0,
        fs=ctx.sample_rate,
    )
    out = block.run(ctx, {"in": OpticalSignal(bands=(carrier,))})["transmitted"]
    notch = abs(block.scattering_matrix(np.array([f0])).s[0, 1, 0]) ** 2
    assert out.bands[0].average_power() == pytest.approx(1e-3 * notch, rel=1e-5)
    # All eight cladding modes coupled; LP04 alone would be cos^2(kappa L), -19.6 dB.
    assert 10 * math.log10(notch) == pytest.approx(-19.23, abs=0.05)


def test_what_cannot_be_a_grating_is_refused() -> None:
    frequencies = np.array([C_LIGHT / 1.55e-6])
    with pytest.raises(ValueError, match="period"):
        long_period_grating(
            frequencies, fibre=FIBRE, period=0.0, length=LENGTH, index_modulation=MODULATION
        )
    with pytest.raises(ValueError, match="at least one cladding mode"):
        long_period_grating(
            frequencies,
            fibre=FIBRE,
            period=PERIOD,
            length=LENGTH,
            index_modulation=MODULATION,
            cladding_modes=0,
        )
    with pytest.raises(ValueError, match="fringe visibility is in"):
        long_period_grating(
            frequencies,
            fibre=FIBRE,
            period=PERIOD,
            length=LENGTH,
            index_modulation=MODULATION,
            visibility=0.0,
        )


# ---------------------------------------------------------------------------
# The pedestal the writing leaves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pedestal", [1e-4, 3e-4])
@pytest.mark.parametrize("rank", [2, 3])
def test_the_pedestal_moves_a_notch_as_raising_the_core_would(pedestal: float, rank: int) -> None:
    """Against a mode solve of the same fibre with its core index raised by the pedestal.

    Two independent routes to one number: here each mode is lifted by its
    self-coupling in the unwritten fibre's modes, first order in the pedestal;
    there the fibre is solved again with the glass itself changed. 20 and 60 nm
    of shift agree to half a percent -- the rest is second order, and the pull
    the other cladding modes put on a notch.
    """
    written = long_period_resonances(FIBRE, period=PERIOD, count=4)
    raised = long_period_resonances(
        StepIndexFibre(core_index=FIBRE.core_index + pedestal), period=PERIOD, count=4
    )
    before = dict(written)[rank]
    after = dict(raised)[rank]
    grid = np.linspace(after - 20e-9, after + 20e-9, 8001)
    through = np.abs(
        long_period_grating(
            C_LIGHT / grid,
            fibre=FIBRE,
            period=PERIOD,
            length=LENGTH,
            index_modulation=pedestal,
            visibility=1.0,
        ).s[:, 1, 0]
    )
    notch = float(grid[np.argmin(through)])
    assert notch - before == pytest.approx(after - before, rel=0.01)
    assert after - before > 15e-9, "tens of nanometres, which is why exposure is watched"


def test_a_compensated_grating_is_the_one_it_always_was() -> None:
    """``visibility=None`` is the default, and moves nothing."""
    frequencies = C_LIGHT / np.linspace(1.4e-6, 1.6e-6, 41)

    def solved(visibility: float | None) -> np.ndarray:
        return long_period_grating(
            frequencies,
            fibre=FIBRE,
            period=PERIOD,
            length=LENGTH,
            index_modulation=MODULATION,
            visibility=visibility,
        ).s

    plain = long_period_grating(
        frequencies, fibre=FIBRE, period=PERIOD, length=LENGTH, index_modulation=MODULATION
    ).s
    assert np.array_equal(plain, solved(None))
    assert not np.array_equal(plain, solved(1.0))


# ---------------------------------------------------------------------------
# A surrounding medium that disperses
# ---------------------------------------------------------------------------

#: A stand-in medium, one Sellmeier term, 1.333 at 1550 nm: it exercises the
#: machinery and stands for no liquid -- the library carries none.
STAND_IN = ((0.75, 0.10e-6),)


def test_the_surroundings_pass_through_the_quoted_index_and_disperse_from_there() -> None:
    fibre = StepIndexFibre(surrounding_index=1.333, surrounding_sellmeier=STAND_IN)
    assert fibre.surrounding_at(1.55e-6) == pytest.approx(1.333, abs=1e-15)
    assert fibre.surrounding_at(1.3e-6) > 1.333 > fibre.surrounding_at(1.6e-6)
    assert fibre.at(1.3e-6).surrounding_index == fibre.surrounding_at(1.3e-6)
    assert StepIndexFibre(surrounding_index=1.333).at(1.3e-6).surrounding_index == 1.333


def test_a_notch_sits_where_the_medium_s_own_index_there_would_put_it() -> None:
    """A fixed point: solved with the medium dispersing, every notch is where a constant
    medium at the index the dispersing one has *at that notch* puts it -- to 1e-14 m.

    And each moves the way a liquid moves a notch: short where the medium is
    denser than at 1550 nm, long where it is thinner.
    """
    fibre = StepIndexFibre(surrounding_index=1.333, surrounding_sellmeier=STAND_IN)
    plain = dict(
        long_period_resonances(StepIndexFibre(surrounding_index=1.333), period=PERIOD, count=6)
    )
    for rank, notch in long_period_resonances(fibre, period=PERIOD, count=6)[:4]:
        there = StepIndexFibre(surrounding_index=fibre.surrounding_at(notch))
        assert dict(long_period_resonances(there, period=PERIOD, count=6))[rank] == pytest.approx(
            notch, abs=1e-14
        )
        denser = fibre.surrounding_at(notch) > 1.333
        assert (notch < plain[rank]) if denser else (notch > plain[rank])


def test_surroundings_that_reach_the_cladding_are_refused_where_they_do() -> None:
    fibre = StepIndexFibre(surrounding_index=1.44, surrounding_sellmeier=((1.0, 0.3e-6),))
    with pytest.raises(ValueError, match="surroundings' index"):
        fibre.at(1.0e-6)
    with pytest.raises(ValueError, match="non-negative"):
        StepIndexFibre(surrounding_index=1.333, surrounding_sellmeier=((-1.0, 0.1e-6),))
