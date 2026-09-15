"""An erbium transient across the spectrum: one inversion, a different swing everywhere.

The model adds nothing to the reservoir :func:`maiman.gain_transient` integrates;
it reads that reservoir's gain as an average inversion and spreads it across the
fibre's own curves. So the checks are the ones that must hold whatever the curves
are: at the centre wavelength it is the single-reservoir answer exactly, every
channel's swing is the tilt times that answer at every instant, each channel
settles where the spectrum says the new inversion puts it, and McCumber's
relation between the two curves is the relation it states.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import ErbiumSpectrum, gain_transient, spectral_gain_transient, step_schedule
from maiman.components import EDFA
from maiman.transient import BOLTZMANN
from maiman.units import C_LIGHT, H_PLANCK

CHANNEL_POWER = 10.0 ** (-6.0 / 10.0) / 1e3
CROSSOVER = 1531e-9
CHANNELS = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9])


def illustrative_spectrum() -> ErbiumSpectrum:
    """A shape, not a fibre: a 1530 nm peak on a broad shoulder, emission by McCumber."""
    nm = np.linspace(1500.0, 1600.0, 401)
    absorption = 30.0 * np.exp(-((nm - 1530.0) ** 2) / (2 * 8.0**2)) + 15.0 * np.exp(
        -((nm - 1548.0) ** 2) / (2 * 18.0**2)
    )
    return ErbiumSpectrum.from_mccumber(nm * 1e-9, absorption, crossover_wavelength=CROSSOVER)


def amplifier(*, saturate: bool = True, gain: float = 20.0) -> EDFA:
    return EDFA(gain=gain, saturate=saturate, saturation_power=17.0, label="edfa")


def drop(points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    return step_schedule(
        [(20e-3, 8 * CHANNEL_POWER), (60e-3, CHANNEL_POWER)], points_per_segment=points
    )


# ---------------------------------------------------------------------------
# The spectrum on its own
# ---------------------------------------------------------------------------


def test_mccumber_is_the_relation_it_states() -> None:
    spectrum = illustrative_spectrum()
    ratio = spectrum.full_inversion_gain_db / spectrum.absorption_db
    expected = np.exp(
        H_PLANCK * C_LIGHT * (1.0 / CROSSOVER - 1.0 / spectrum.wavelengths) / (BOLTZMANN * 295.0)
    )
    assert np.allclose(ratio, expected, rtol=1e-12, atol=0.0)
    at_crossover = np.interp(CROSSOVER, spectrum.wavelengths, ratio)
    assert at_crossover == pytest.approx(1.0, abs=2e-3), "emission equals absorption there"
    assert ratio[-1] > 1.0 > ratio[0], "emission wins longward, absorption shortward"


def test_gain_and_inversion_are_inverses() -> None:
    spectrum = illustrative_spectrum()
    inversions = np.linspace(0.0, 1.0, 11)
    for wavelength in CHANNELS:
        gains = spectrum.gain_db(wavelength, inversions)
        assert spectrum.inversion(wavelength, gains) == pytest.approx(inversions, abs=1e-12)
    assert spectrum.gain_db(1550e-9, 0.0)[0] == pytest.approx(
        -np.interp(1550e-9, spectrum.wavelengths, spectrum.absorption_db), rel=1e-12
    )


def test_the_tilt_is_the_ratio_of_slopes_and_does_not_depend_on_inversion() -> None:
    spectrum = illustrative_spectrum()
    tilt = spectrum.tilt(CHANNELS, 1550e-9)
    for low, high in ((0.3, 0.6), (0.5, 0.9)):
        moved = spectrum.gain_db(CHANNELS, high) - spectrum.gain_db(CHANNELS, low)
        reference = spectrum.gain_db(1550e-9, high) - spectrum.gain_db(1550e-9, low)
        assert moved / reference == pytest.approx(tilt, rel=1e-12)
    assert tilt[2] == pytest.approx(1.0, rel=1e-15)
    assert tilt[0] > tilt[1] > tilt[2] > tilt[3], "the absorption peak end swings hardest"


# ---------------------------------------------------------------------------
# The transient across the spectrum
# ---------------------------------------------------------------------------


def test_at_the_centre_wavelength_it_is_the_single_reservoir_answer() -> None:
    times, power = drop()
    spread = spectral_gain_transient(amplifier(), illustrative_spectrum(), times, power, CHANNELS)
    single = gain_transient(amplifier(), times, power)
    assert spread.gain_db[:, 2] == pytest.approx(single.gain_db, rel=0.0, abs=1e-10)
    assert spread.excursions[2] == pytest.approx(single.excursion, rel=0.0, abs=1e-10)


def test_every_channel_swings_by_the_tilt_times_the_reference_at_every_instant() -> None:
    spectrum = illustrative_spectrum()
    times, power = drop()
    spread = spectral_gain_transient(amplifier(), spectrum, times, power, CHANNELS)
    reference = spread.reference.gain_db - spread.reference.gain_db[0]
    moved = spread.gain_db - spread.gain_db[0]
    expected = reference[:, None] * spectrum.tilt(CHANNELS, 1550e-9)[None, :]
    assert np.allclose(moved, expected, rtol=0.0, atol=1e-9)
    assert spread.excursions == pytest.approx(
        spread.excursions[2] * spectrum.tilt(CHANNELS, 1550e-9), rel=1e-9
    )


def test_each_channel_settles_where_the_new_inversion_puts_it() -> None:
    spectrum = illustrative_spectrum()
    amp = amplifier()
    times, power = drop()
    spread = spectral_gain_transient(amp, spectrum, times, power, CHANNELS)
    for gain_before, gain_after, row in (
        (amp.effective_gain(8 * CHANNEL_POWER), None, 0),
        (amp.effective_gain(CHANNEL_POWER), None, -1),
    ):
        del gain_after
        inversion = spectrum.inversion(1550e-9, 10.0 * math.log10(gain_before))
        expected = spectrum.gain_db(CHANNELS, inversion)
        tolerance = 1e-10 if row == 0 else 5e-3
        assert spread.gain_db[row] == pytest.approx(expected, rel=0.0, abs=tolerance)


def test_an_unsaturated_amplifier_does_not_move_anywhere() -> None:
    times, power = drop(points=64)
    spread = spectral_gain_transient(
        amplifier(saturate=False), illustrative_spectrum(), times, power, CHANNELS
    )
    assert np.array_equal(spread.gain_db, np.repeat(spread.gain_db[:1], times.size, axis=0))
    assert np.all(spread.excursions == 0.0)


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_an_amplifier_the_fibre_cannot_be_is_refused() -> None:
    times, power = drop(points=16)
    with pytest.raises(ValueError, match="more than full inversion"):
        spectral_gain_transient(
            amplifier(gain=30.0), illustrative_spectrum(), times, power, CHANNELS
        )


def test_a_channel_outside_the_measured_band_is_refused() -> None:
    times, power = drop(points=16)
    with pytest.raises(ValueError, match="will not extrapolate"):
        spectral_gain_transient(amplifier(), illustrative_spectrum(), times, power, [1620e-9])


@pytest.mark.parametrize(
    ("wavelengths", "absorption", "emission", "message"),
    [
        ([1.55e-6], [1.0], [1.0], "at least two"),
        ([1.56e-6, 1.55e-6], [1.0, 1.0], [1.0, 1.0], "strictly increasing"),
        ([1.55e-6, 1.56e-6], [1.0, -1.0], [1.0, 1.0], "magnitudes"),
        ([1.55e-6, 1.56e-6], [1.0], [1.0, 1.0], "must match"),
    ],
)
def test_a_spectrum_that_is_not_one_is_refused(
    wavelengths: list[float], absorption: list[float], emission: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ErbiumSpectrum(
            wavelengths=np.array(wavelengths),
            absorption_db=np.array(absorption),
            full_inversion_gain_db=np.array(emission),
        )
