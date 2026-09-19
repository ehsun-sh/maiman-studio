"""Two long-period gratings on one fibre: the second hands back what the first took.

A single grating's cladding light is lost under the coating. Strip the coating
between two gratings and it is not: the first grating splits the core's light
between the core and a cladding mode, the two run at different speeds, and the
second grating recombines them -- a Mach-Zehnder interferometer inside one
fibre. What holds the model to that:

* no gap at all is one grating twice as long, to the last bit;
* at a 3 dB split the fringes run between ``(1 + g)^2 / 4`` and ``(1 - g)^2 / 4``,
  ``g`` the cladding's amplitude surviving the gap -- a perfect null with no loss;
* the fringes are ``lambda^2 / (dn_g d)`` apart, with ``dn_g`` the two modes'
  group-index difference from the mode solver, and closer to it the longer the
  gap is against the gratings;
* and with the cladding lossless between them nothing is lost: at the phase
  match, where one grating would have cut a notch, the pair passes everything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman.components import LongPeriodGrating
from maiman.modes import StepIndexFibre, cladding_modes, core_modes, lpg_coupling
from maiman.photonics import long_period_grating, long_period_resonances
from maiman.units import C_LIGHT

FIBRE = StepIndexFibre()
PERIOD = 500e-6
LENGTH = 25e-3


@pytest.fixture(scope="module")
def resonance() -> float:
    return dict(long_period_resonances(FIBRE, period=PERIOD, count=1))[1]


def three_db(wavelength: float) -> float:
    """The index modulation that hands half the core's power to LP01's cladding twin."""
    kappa = lpg_coupling(
        core_modes(FIBRE, wavelength)[0], cladding_modes(FIBRE, wavelength, count=1)[0], 1.0
    )
    return (math.pi / 4) / (kappa * LENGTH)


def through(
    wavelengths: np.ndarray, *, separation: float | None, loss: float = 0.0, **extra: object
) -> np.ndarray:
    settings: dict[str, object] = {
        "fibre": FIBRE,
        "period": PERIOD,
        "length": LENGTH,
        "index_modulation": 3e-4,
        "cladding_modes": 1,
    }
    settings.update(extra)
    s = long_period_grating(
        C_LIGHT / wavelengths,
        separation=separation,
        gap_loss_db=loss,
        **settings,  # type: ignore[arg-type]
    )
    return s.s[:, 1, 0]


def test_no_gap_is_one_grating_twice_as_long(resonance: float) -> None:
    wavelengths = np.linspace(resonance - 30e-9, resonance + 30e-9, 301)
    pair = through(wavelengths, separation=0.0, cladding_modes=4)
    single = through(wavelengths, separation=None, cladding_modes=4, length=2 * LENGTH)
    assert np.max(np.abs(pair - single)) < 1e-12


def local_extremes(values: np.ndarray, kind: str) -> np.ndarray:
    inner = values[1:-1]
    if kind == "max":
        found = (inner > values[:-2]) & (inner > values[2:])
    else:
        found = (inner < values[:-2]) & (inner < values[2:])
    return np.flatnonzero(found) + 1


@pytest.mark.parametrize("loss", [0.0, 3.0, 10.0])
def test_a_three_db_split_swings_between_the_two_arms_sum_and_difference(
    resonance: float, loss: float
) -> None:
    wavelengths = np.linspace(resonance - 3e-9, resonance + 3e-9, 6001)
    power = (
        np.abs(
            through(wavelengths, separation=0.2, loss=loss, index_modulation=three_db(resonance))
        )
        ** 2
    )
    g = 10 ** (-loss / 20)
    near = lambda idx: idx[np.argmin(np.abs(wavelengths[idx] - resonance))]  # noqa: E731
    assert power[near(local_extremes(power, "min"))] == pytest.approx((1 - g) ** 2 / 4, abs=1e-4)
    assert power[near(local_extremes(power, "max"))] == pytest.approx((1 + g) ** 2 / 4, rel=0.05)


def fringe_spacing(resonance: float, separation: float) -> float:
    wavelengths = np.linspace(resonance - 3e-9, resonance + 3e-9, 12001)
    power = (
        np.abs(through(wavelengths, separation=separation, index_modulation=three_db(resonance)))
        ** 2
    )
    return float(np.mean(np.diff(wavelengths[local_extremes(power, "min")])))


def group_index_difference(wavelength: float, step: float = 1e-10) -> float:
    def difference(x: float) -> float:
        return core_modes(FIBRE, x)[0].effective_index - (
            cladding_modes(FIBRE, x, count=1)[0].effective_index
        )

    slope = (difference(wavelength + step) - difference(wavelength - step)) / (2 * step)
    return difference(wavelength) - wavelength * slope


def test_the_fringes_are_the_group_index_difference_s(resonance: float) -> None:
    dn_g = group_index_difference(resonance)
    ratios = []
    for separation in (0.2, 0.8):
        predicted = resonance**2 / (dn_g * (separation + LENGTH))
        ratios.append(fringe_spacing(resonance, separation) / predicted)
    assert ratios[0] == pytest.approx(1.0, abs=0.05)
    assert ratios[1] == pytest.approx(1.0, abs=0.02)
    assert abs(ratios[1] - 1) < abs(ratios[0] - 1), "the gratings matter less as the gap grows"


def test_with_a_lossless_gap_the_pair_can_pass_what_one_grating_cuts(resonance: float) -> None:
    """At a full crossover each grating empties the core; the pair fills it again."""
    kappa = lpg_coupling(
        core_modes(FIBRE, resonance)[0], cladding_modes(FIBRE, resonance, count=1)[0], 1.0
    )
    full = (math.pi / 2) / (kappa * LENGTH)
    wavelengths = np.array([resonance])
    assert abs(through(wavelengths, separation=None, index_modulation=full)[0]) ** 2 < 1e-6
    wavelengths = np.linspace(resonance - 3e-9, resonance + 3e-9, 3001)
    pair = np.abs(through(wavelengths, separation=0.2, index_modulation=full)) ** 2
    assert float(pair.max()) == pytest.approx(1.0, abs=1e-3)


def test_the_block_writes_it_twice_only_when_asked() -> None:
    assert not LongPeriodGrating().pair
    specs = LongPeriodGrating.param_specs()
    assert specs["separation"].applies_when == "pair"
    assert specs["gap_loss"].applies_when == "pair"
    single = LongPeriodGrating(label="l", cladding_modes=1.0)
    pair = LongPeriodGrating(label="l", cladding_modes=1.0, pair=True, separation=0.0)
    longer = LongPeriodGrating(label="l", cladding_modes=1.0, length=50.0)
    grid = np.linspace(1.33e-6, 1.38e-6, 101)
    frequencies = C_LIGHT / grid
    one = single._matrix_factory()(frequencies).s[:, 1, 0]
    two = pair._matrix_factory()(frequencies).s[:, 1, 0]
    both = longer._matrix_factory()(frequencies).s[:, 1, 0]
    assert np.max(np.abs(two - both)) < 1e-12
    assert np.max(np.abs(two - one)) > 0.1
