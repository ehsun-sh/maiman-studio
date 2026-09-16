"""The pump loop that answers an erbium transient, and which channels drain the reservoir.

Two things the uncontrolled model left out. A deployed amplifier measures its own
gain and moves the pump, so the excursion a channel drop causes is the loop's
failure to keep up rather than the erbium's full swing -- and the loop has a
bandwidth, a setpoint and a pump it can run out of. And a watt is not a watt:
stimulated emission empties the reservoir once per photon, so a channel drains it
by its own cross section and photon energy.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from maiman import (
    ErbiumSpectrum,
    PumpControl,
    controlled_gain_transient,
    gain_transient,
    saturating_power,
    saturating_weights,
    spectral_gain_transient,
    step_schedule,
)
from maiman.components import EDFA

CHANNEL = 10 ** (-6 / 10) / 1e3
REFERENCE = 1550e-9


def amplifier(gain: float = 20.0, *, saturate: bool = True) -> EDFA:
    return EDFA(gain=gain, saturate=saturate, saturation_power=17.0, label="edfa")


def drop(points: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    return step_schedule([(20e-3, 8 * CHANNEL), (60e-3, CHANNEL)], points_per_segment=points)


def spectrum() -> ErbiumSpectrum:
    nm = np.linspace(1500.0, 1600.0, 401)
    absorption = 30.0 * np.exp(-((nm - 1530.0) ** 2) / (2 * 8.0**2)) + 15.0 * np.exp(
        -((nm - 1548.0) ** 2) / (2 * 18.0**2)
    )
    return ErbiumSpectrum.from_mccumber(nm * 1e-9, absorption, crossover_wavelength=1531e-9)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_a_faster_loop_leaves_less_of_the_excursion() -> None:
    """3.21 dB uncontrolled, and less of it the faster the pump is allowed to move."""
    times, power = drop()
    uncontrolled = gain_transient(amplifier(), times, power)
    assert uncontrolled.excursion == pytest.approx(3.21, abs=0.02)

    excursions = [
        controlled_gain_transient(
            amplifier(), times, power, PumpControl(mode="gain", bandwidth=bandwidth)
        ).excursion
        for bandwidth in (10.0, 100.0, 1e3, 1e4)
    ]
    assert excursions == pytest.approx([2.15, 1.12, 0.43, 0.15], abs=0.05)
    assert all(a > b for a, b in pairwise(excursions))
    assert excursions[0] < uncontrolled.excursion


def test_integral_action_leaves_no_steady_state_error() -> None:
    """The gain comes back to the setpoint, not near it: that is what the integral is for."""
    times, power = drop()
    controlled = controlled_gain_transient(
        amplifier(), times, power, PumpControl(mode="gain", bandwidth=1e3)
    )
    assert controlled.gain_db[-1] == pytest.approx(controlled.setpoint, abs=0.02)
    assert controlled.pump_gain_db[-1] < 20.0, "less input needs less pump to hold the same gain"
    assert not controlled.pump_limited


def test_the_default_setpoint_is_where_the_amplifier_was_running() -> None:
    times, power = drop(points=400)
    controlled = controlled_gain_transient(
        amplifier(), times, power, PumpControl(mode="gain", bandwidth=1e3)
    )
    settled = 10.0 * math.log10(amplifier().effective_gain(8 * CHANNEL))
    assert controlled.setpoint == pytest.approx(settled, rel=1e-12)


def test_a_power_loop_holds_the_output_instead_of_the_gain() -> None:
    times, power = drop()
    controlled = controlled_gain_transient(
        amplifier(), times, power, PumpControl(mode="power", bandwidth=1e3)
    )
    output_dbm = 10.0 * np.log10(controlled.transient.output_power * 1e3)
    assert output_dbm[-1] == pytest.approx(controlled.setpoint, abs=0.05)
    assert controlled.pump_gain_db[-1] > 20.0, "nine decibels less input needs more pump"


def test_a_pump_that_runs_out_says_so() -> None:
    """Holding the output through a 9 dB drop needs 29 dB of pump; this one has 25."""
    times, power = drop()
    controlled = controlled_gain_transient(
        amplifier(), times, power, PumpControl(mode="power", bandwidth=1e3, max_pump_gain=25.0)
    )
    assert controlled.pump_limited
    assert controlled.pump_saturated
    assert controlled.pump_gain_db.max() == pytest.approx(25.0, abs=1e-6)
    output_dbm = 10.0 * np.log10(controlled.transient.output_power * 1e3)
    assert output_dbm[-1] == pytest.approx(16.38, abs=0.1)
    assert output_dbm[-1] < controlled.setpoint - 1.0, "it cannot reach the setpoint"


def test_saturating_on_the_way_is_not_the_same_as_running_out() -> None:
    """An integral loop slams the pump into its ceiling on a step and then comes back.

    With the pump's own ceiling out of reach the loop recovers and holds the
    setpoint, so ``pump_limited`` is false while ``pump_saturated`` is true.
    Reporting the first for the second would call a loop that worked a failure.
    """
    times, power = drop()
    controlled = controlled_gain_transient(
        amplifier(), times, power, PumpControl(mode="power", bandwidth=1e3, max_pump_gain=40.0)
    )
    assert controlled.pump_saturated, "9 dB of error drives it to the ceiling briefly"
    assert not controlled.pump_limited, "and it comes back off it"
    assert controlled.pump_gain_db[-1] == pytest.approx(29.15, abs=0.1)
    output_dbm = 10.0 * np.log10(controlled.transient.output_power * 1e3)
    assert output_dbm[-1] == pytest.approx(controlled.setpoint, abs=0.05)


def test_what_cannot_be_controlled_is_refused() -> None:
    times, power = drop(points=200)
    with pytest.raises(ValueError, match="pump ceiling"):
        controlled_gain_transient(
            amplifier(), times, power, PumpControl(bandwidth=1e3, max_pump_gain=17.0)
        )
    with pytest.raises(ValueError, match="amplifier that compresses"):
        controlled_gain_transient(
            amplifier(saturate=False), times, power, PumpControl(bandwidth=1e3)
        )
    with pytest.raises(ValueError, match="mode must be"):
        PumpControl(mode="current")
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        PumpControl(bandwidth=0.0)
    with pytest.raises(ValueError, match="ceiling must sit above"):
        PumpControl(max_pump_gain=5.0, min_pump_gain=10.0)


# ---------------------------------------------------------------------------
# Which channels drain it
# ---------------------------------------------------------------------------


def test_a_watt_at_the_reference_wavelength_weighs_exactly_one() -> None:
    """Which is what keeps every result taken before this existed where it was."""
    weights = saturating_weights(spectrum(), np.array([REFERENCE]), REFERENCE)
    assert weights[0] == pytest.approx(1.0, rel=1e-15)


def test_a_short_wavelength_drains_the_reservoir_harder() -> None:
    """Cross section and photon energy together: 1.90 at 1530 nm against 0.85 at 1560."""
    channels = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9])
    weights = saturating_weights(spectrum(), channels, REFERENCE)
    assert weights == pytest.approx([1.9002, 1.4887, 1.0, 0.8462], abs=0.001)
    assert all(a > b for a, b in pairwise(weights))

    tilt = spectrum().tilt(channels, REFERENCE)
    assert weights == pytest.approx(tilt * (channels / REFERENCE), rel=1e-12)


def test_the_weighted_power_is_the_total_when_everything_sits_at_the_reference() -> None:
    powers = np.array([[1e-3, 2e-3], [3e-4, 4e-4]])
    channels = np.array([REFERENCE, REFERENCE])
    weighted = saturating_power(spectrum(), channels, powers, REFERENCE)
    assert weighted == pytest.approx(powers.sum(axis=1), rel=1e-15)


def test_dropping_the_short_wavelengths_moves_the_gain_more() -> None:
    """The same watts removed, at different ends of the band, are not the same disturbance."""
    times, _ = drop(points=800)
    shape = (times.size, 2)
    before = np.full(shape, 4 * CHANNEL)

    def excursion(kept: float, dropped: float) -> float:
        powers = before.copy()
        powers[times >= 20e-3, 0] = 0.0  # the pair at `dropped` goes away
        channels = np.array([dropped, kept])
        spread = spectral_gain_transient(
            amplifier(),
            spectrum(),
            times,
            powers.sum(axis=1),
            channels,
            channel_powers=powers,
        )
        return float(spread.reference.excursion)

    short_end = excursion(kept=1560e-9, dropped=1530e-9)
    long_end = excursion(kept=1530e-9, dropped=1560e-9)
    assert short_end > long_end + 0.2, (short_end, long_end)


def test_channel_powers_have_to_match_the_channels() -> None:
    times, power = drop(points=200)
    with pytest.raises(ValueError, match="channels"):
        spectral_gain_transient(
            amplifier(),
            spectrum(),
            times,
            power,
            np.array([1550e-9, 1560e-9]),
            channel_powers=np.ones((times.size, 3)) * CHANNEL,
        )
