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
    effective_time_constant,
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


# ---------------------------------------------------------------------------
# The loop's three imperfections, each against the linearised loop
# ---------------------------------------------------------------------------
#
# About its rest point the reservoir is a first-order lag: a pump step of
# ``dp`` nepers moves the gain by ``k dp`` over ``tau_e``, with
# ``k = 1 / (1 + P_out / P_sat)`` and ``tau_e = k tau`` -- the effective time
# constant. The loop integrates the error at ``1 / tau_c``, so its crossover
# rate is ``a = k / tau_c``.

LOADED = 8 * CHANNEL


def linearised(control: PumpControl, lifetime: float) -> tuple[float, float, float]:
    """``(k, a, tau_e)`` for the gain loop about the amplifier's rest at ``LOADED``."""
    tau_e = effective_time_constant(amplifier(), LOADED, lifetime=lifetime)
    k = tau_e / lifetime
    return k, k / control.time_constant, tau_e


def settled_run(
    control: PumpControl, duration: float, points: int, *, lifetime: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``(times, gain error in dB)`` for a run that starts at rest and is only perturbed."""
    times, power = step_schedule([(duration, LOADED)], points_per_segment=points)
    run = controlled_gain_transient(
        amplifier(), times, power, control, lifetime=lifetime, seed=seed
    )
    return times, run.gain_db - run.setpoint


def test_the_imperfections_are_off_by_default_and_move_nothing() -> None:
    """A control built without them runs bit for bit as the ideal loop always did."""
    times, power = drop(points=300)
    plain = controlled_gain_transient(amplifier(), times, power, PumpControl(bandwidth=1e3))
    spelled = controlled_gain_transient(
        amplifier(),
        times,
        power,
        PumpControl(bandwidth=1e3, delay=0.0, detector_noise=0.0, dither_depth=0.0),
        seed=99,
    )
    assert np.array_equal(plain.gain_db, spelled.gain_db)
    assert np.array_equal(plain.pump_gain_db, spelled.pump_gain_db)


@pytest.mark.parametrize(("margin", "grows"), [(0.8, False), (1.2, True)])
def test_a_delayed_loop_is_stable_only_below_a_quarter_turn(margin: float, grows: bool) -> None:
    """``dx/dt = -a x(t - delay)`` is stable while ``a delay < pi/2`` (Hayes, 1950).

    The erbium is made fifty times faster than the loop so that it is the delay
    and not the reservoir's own lag that sets the phase. A 1 % input step kicks
    the loop; a fifth inside the boundary the ringing dies, a fifth outside it
    grows.
    """
    base = PumpControl(bandwidth=100.0)
    k, a, _ = linearised(base, 1e-3)
    lifetime = 1.0 / (50.0 * a * k)  # tau_e = k tau = 1 / (50 a)
    delay = margin * (math.pi / 2.0) / a
    control = PumpControl(bandwidth=100.0, delay=delay)
    times, power = step_schedule(
        [(delay, LOADED), (14 * delay, 0.99 * LOADED)], points_per_segment=700
    )
    run = controlled_gain_transient(amplifier(), times, power, control, lifetime=lifetime)
    error = run.gain_db - run.setpoint
    after = times - delay
    early = np.abs(error[(after > 2 * delay) & (after < 6 * delay)]).max()
    late = np.abs(error[(after > 10 * delay) & (after < 14 * delay)]).max()
    assert (late > 1.5 * early) if grows else (late < early / 1.5), (early, late)


def test_at_the_edge_a_delayed_loop_rings_at_four_delays() -> None:
    """On the boundary the root is ``s = j pi / (2 delay)``: a period of ``4 delay``."""
    base = PumpControl(bandwidth=100.0)
    k, a, _ = linearised(base, 1e-3)
    lifetime = 1.0 / (50.0 * a * k)
    delay = (math.pi / 2.0) / a
    control = PumpControl(bandwidth=100.0, delay=delay)
    times, power = step_schedule(
        [(delay, LOADED), (14 * delay, 0.99 * LOADED)], points_per_segment=1400
    )
    run = controlled_gain_transient(amplifier(), times, power, control, lifetime=lifetime)
    error = run.gain_db - run.setpoint
    window = times > 4 * delay
    crossings = times[window][1:][np.diff(np.sign(error[window])) != 0]
    period = 2.0 * float(np.mean(np.diff(crossings)))
    # The erbium's own lag, a fiftieth of 1/a, moves the phase by a few percent.
    assert period == pytest.approx(4.0 * delay, rel=0.05)


def test_detector_noise_wanders_the_gain_as_the_closed_loop_integral_says() -> None:
    """``var(g) = k N^2 / (4 tau_c)``, whatever the erbium's own lag.

    The loop and the lagging reservoir make a second-order system driven by
    white noise, and the variance of such a system does not depend on its
    ``s^2`` coefficient -- so the erbium runs at its real lifetime here. ``N``
    is a one-sided density, so the two-sided one the integral wants is
    ``N^2 / 2``.
    """
    control = PumpControl(bandwidth=100.0, detector_noise=4e-4)
    k, _, _ = linearised(control, 10e-3)
    expected = math.sqrt(k * control.detector_noise**2 / (4.0 * control.time_constant))
    spreads = []
    for seed in (1, 2):
        times, error = settled_run(control, 2.0, 4000, lifetime=10e-3, seed=seed)
        spreads.append(float(np.std(error[times > 0.1])))
    assert float(np.mean(spreads)) == pytest.approx(expected, rel=0.08)


def test_a_dither_reaches_the_gain_high_passed_by_the_loop() -> None:
    """``|g / d| = k w / |a - w^2 tau_e + j w|``: the loop fights it, the erbium lags it.

    A 0.01 dB tone on the pump at the loop's own crossover, with the erbium at
    its real lifetime -- its lag is the ``w^2 tau_e`` term and it is carried
    rather than assumed away.
    """
    base = PumpControl(bandwidth=100.0)
    k, a, tau_e = linearised(base, 10e-3)
    frequency = a / (2.0 * math.pi)
    control = PumpControl(bandwidth=100.0, dither_depth=0.01, dither_frequency=frequency)
    periods = 40
    times, error = settled_run(control, periods / frequency, 64 * periods, lifetime=10e-3)
    steady = times >= 20 / frequency
    swing = 2.0 * abs(np.mean(error[steady] * np.exp(-2j * math.pi * frequency * times[steady])))
    omega = 2.0 * math.pi * frequency
    expected = 0.01 * k * omega / abs(a - omega**2 * tau_e + 1j * omega)
    # Measured 0.9998 of it. Leaving out the erbium's lag would put it 20 % off,
    # so this tolerance is what makes the tau_e term something the test checks.
    assert swing == pytest.approx(expected, rel=0.005)


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
    with pytest.raises(ValueError, match="act before it measures"):
        PumpControl(delay=-1e-6)
    with pytest.raises(ValueError, match="not negative"):
        PumpControl(detector_noise=-1.0)
    with pytest.raises(ValueError, match="needs a frequency"):
        PumpControl(dither_depth=0.1)


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


# ---------------------------------------------------------------------------
# A proportional term and a loop filter (maiman-9ox)
# ---------------------------------------------------------------------------
#
# Linearised about a working point, the reservoir is ``tau d(dg)/dt = dp - a dg``
# with ``a = 1 + G (P / P_sat + b)`` and the loop's PI acts on ``r - dg``, all in
# nepers. A step in the setpoint therefore has the closed-form response of
#
#     (kp s + w_i) / (tau s^2 + (a + kp) s + w_i)                    (no filter)
#     (kp s + w_i) / (tau tau_f s^3 + (tau + a tau_f) s^2 + (a + kp) s + w_i)
#
# with ``w_i = 1 / tau_c`` -- a second-order system, then a third. The transient
# is integrated in full; a step small enough that the nonlinearity is invisible
# is compared to it.


def linearised_step(
    numerator: list[float], denominator: list[float], times: np.ndarray
) -> np.ndarray:
    """Unit-step response of ``N(s) / D(s)`` by partial fractions, from ``D``'s roots."""
    poles = np.roots(denominator)
    full = np.polymul(denominator, [1.0, 0.0])  # the step's own pole at zero
    slope = np.polyder(full)
    residues = [np.polyval(numerator, r) / np.polyval(slope, r) for r in poles]
    final = np.polyval(numerator, 0.0) / np.polyval(denominator, 0.0)
    response = final + sum(res * np.exp(r * times) for res, r in zip(residues, poles, strict=True))
    return np.asarray(np.real(response))


def loop_constants(control: PumpControl) -> tuple[float, float]:
    """``(a, tau)`` of the linearised reservoir at the run's settled point."""
    from maiman.transient import METASTABLE_LIFETIME

    edfa = amplifier()
    saturation = edfa.intrinsic_saturation_power(10 ** (edfa.gain / 10.0))
    own = edfa.self_saturation_load() / saturation
    settled = edfa.effective_gain(CHANNEL)
    return 1.0 + settled * (CHANNEL / saturation + own), METASTABLE_LIFETIME


def setpoint_step(control: PumpControl, size_db: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """The gain's move [dB] after a setpoint step of ``size_db``, and the times it is at."""
    edfa = amplifier()
    settled_db = 10.0 * math.log10(edfa.effective_gain(CHANNEL))
    times = np.linspace(0.0, 0.05, 2501)
    held = np.full_like(times, CHANNEL)
    run = controlled_gain_transient(
        edfa,
        times,
        held,
        PumpControl(**{**control.__dict__, "setpoint": settled_db + size_db}),
    )
    return times, run.gain_db - settled_db


@pytest.mark.parametrize("proportional", [0.0, 4.0, 20.0])
def test_the_pi_loop_follows_the_second_order_step_response(proportional: float) -> None:
    """Underdamped without the proportional term, critical near 16, overdamped past it.

    ``zeta = (a + kp) / (2 sqrt(tau / tau_c))``: 0.07 at ``kp = 0``, where the loop
    rings for tens of milliseconds, 0.3 at 4 and 1.3 at 20 -- where the PI's zero still
    leaves a 5 % overshoot. Integrated against the
    closed form to 2 % of the step.
    """
    control = PumpControl(mode="gain", bandwidth=1e3, proportional=proportional)
    a, tau = loop_constants(control)
    w_i = 1.0 / control.time_constant
    times, moved = setpoint_step(control)
    expected = 0.02 * linearised_step([proportional, w_i], [tau, a + proportional, w_i], times)
    assert np.max(np.abs(moved - expected)) < 0.02 * 0.02
    zeta = (a + proportional) / (2.0 * math.sqrt(tau / control.time_constant))
    if proportional == 0.0:
        assert zeta < 0.1 and moved.max() > 0.02 * 1.7, "and it overshoots by more than half"
    if proportional == 20.0:
        # Overdamped, but the PI's zero at ``-1 / (kp tau_c)`` still overshoots a little.
        assert zeta > 1.0 and 0.02 < moved.max() < 0.02 * 1.06


def test_the_proportional_term_takes_the_ringing_out_of_the_loop() -> None:
    """Overshoot falls monotonically as ``kp`` rises: 0.07 damping, then 0.3, then critical."""
    peaks = [
        setpoint_step(PumpControl(mode="gain", bandwidth=1e3, proportional=kp))[1].max() / 0.02
        for kp in (0.0, 2.0, 6.0, 16.0)
    ]
    assert peaks[0] > peaks[1] > peaks[2] > peaks[3]
    assert peaks[0] == pytest.approx(1.79, abs=0.03)


@pytest.mark.parametrize("filter_time", [2e-4, 1e-3])
def test_a_loop_filter_follows_the_third_order_step_response(filter_time: float) -> None:
    """A first-order low-pass on the error: the closed form is a cubic."""
    control = PumpControl(mode="gain", bandwidth=1e3, proportional=6.0, filter_time=filter_time)
    a, tau = loop_constants(control)
    w_i = 1.0 / control.time_constant
    kp = control.proportional
    times, moved = setpoint_step(control)
    expected = 0.02 * linearised_step(
        [kp, w_i],
        [
            tau * filter_time,
            tau + a * filter_time,
            a + kp,
            w_i,
        ],
        times,
    )
    assert np.max(np.abs(moved - expected)) < 0.02 * 0.03


def test_a_filter_costs_the_loop_phase_and_it_rings_more() -> None:
    """The same PI, a slower low-pass in front of it: more overshoot, not less."""
    peaks = [
        setpoint_step(PumpControl(mode="gain", bandwidth=1e3, proportional=6.0, filter_time=tf))[
            1
        ].max()
        / 0.02
        for tf in (0.0, 2e-4, 1e-3)
    ]
    assert peaks[0] < peaks[1] < peaks[2]


def test_neither_term_moves_anything_at_zero() -> None:
    """Both default to zero, and then the loop is the integrator it always was."""
    times, power = drop(300)
    plain = controlled_gain_transient(amplifier(), times, power, PumpControl(bandwidth=1e3))
    zeros = controlled_gain_transient(
        amplifier(), times, power, PumpControl(bandwidth=1e3, proportional=0.0, filter_time=0.0)
    )
    assert np.array_equal(plain.gain_db, zeros.gain_db)
    assert np.array_equal(plain.pump_gain_db, zeros.pump_gain_db)


def test_the_proportional_term_still_leaves_no_steady_state_error() -> None:
    times, power = drop()
    control = PumpControl(mode="gain", bandwidth=1e3, proportional=6.0, filter_time=2e-4)
    run = controlled_gain_transient(amplifier(), times, power, control)
    assert run.gain_db[-1] == pytest.approx(run.setpoint, abs=0.02)


def test_the_gains_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="proportional"):
        PumpControl(proportional=-1.0)
    with pytest.raises(ValueError, match="filter"):
        PumpControl(filter_time=-1e-3)


# ---------------------------------------------------------------------------
# A derivative term and back-calculation anti-windup (maiman-hmk)
# ---------------------------------------------------------------------------
#
# The derivative acts on the filtered error's slope, so with ``kd`` the controller
# is ``(kd s^2 + kp s + w_i) / (s (tau_f s + 1))`` and a setpoint step sees
#
#     (kd s^2 + kp s + w_i) / (tau tau_f s^3 + (tau + a tau_f + kd) s^2 + (a + kp) s + w_i)
#
# Back-calculation (Astrom and Hagglund 1995, sec. 3.5) pulls the integrator by
# ``(applied - wanted) / T_t`` while the pump is clamped; held there by an error
# ``e`` it settles where its slope is zero, ``limit - kp e + T_t e / tau_c``.


@pytest.mark.parametrize("derivative", [1e-3, 5e-3])
def test_a_derivative_term_follows_the_closed_form_step_response(derivative: float) -> None:
    """PID with the filter rolling off its derivative: the cubic above, to 3 % of the step."""
    control = PumpControl(
        mode="gain", bandwidth=1e3, proportional=6.0, filter_time=2e-4, derivative=derivative
    )
    a, tau = loop_constants(control)
    w_i = 1.0 / control.time_constant
    tf, kp = control.filter_time, control.proportional
    times, moved = setpoint_step(control)
    expected = 0.02 * linearised_step(
        [derivative, kp, w_i], [tau * tf, tau + a * tf + derivative, a + kp, w_i], times
    )
    assert np.max(np.abs(moved - expected)) < 0.02 * 0.03
    without = setpoint_step(PumpControl(**{**control.__dict__, "derivative": 0.0}))[1]
    assert np.max(np.abs(moved - without)) > 0.02 * 0.05, "and it moved the response"


def test_held_at_its_limit_the_integrator_settles_where_back_calculation_says() -> None:
    """An unreachable setpoint: ``limit - kp e + T_t e / tau_c``, to 1e-3 dB.

    Without tracking the integrator winds on to the ceiling behind the clamp and
    stays there; with it, it sits 10.6 dB under, where the pump is held at the
    limit by the proportional term rather than by the integrator's own clamp.
    """
    edfa = amplifier()
    settled = 10.0 * math.log10(edfa.effective_gain(CHANNEL))
    times = np.linspace(0.0, 0.05, 2501)
    held = np.full_like(times, CHANNEL)
    runs = {}
    for tracking in (0.0, 2e-4):
        control = PumpControl(
            bandwidth=1e3,
            proportional=6.0,
            max_pump_gain=21.0,
            setpoint=settled + 3.0,
            tracking_time=tracking,
        )
        runs[tracking] = (control, controlled_gain_transient(edfa, times, held, control))
    control, run = runs[2e-4]
    error = run.setpoint - run.gain_db[-1]
    predicted = 21.0 - control.proportional * error + 2e-4 * error / control.time_constant
    assert run.integrator_db[-1] == pytest.approx(predicted, abs=1e-3)
    assert run.pump_limited
    assert runs[0.0][1].integrator_db[-1] == pytest.approx(21.0, abs=1e-9)


def test_back_calculation_takes_the_windup_out_of_a_saturating_step() -> None:
    """A 1 dB step that runs the pump into its ceiling on the way: 0.050 dB over, then 0.035."""
    edfa = amplifier()
    settled = 10.0 * math.log10(edfa.effective_gain(CHANNEL))
    times = np.linspace(0.0, 0.05, 2501)
    held = np.full_like(times, CHANNEL)
    over = []
    for tracking in (0.0, 2e-4):
        run = controlled_gain_transient(
            edfa,
            times,
            held,
            PumpControl(
                bandwidth=1e3,
                proportional=6.0,
                max_pump_gain=22.0,
                setpoint=settled + 1.0,
                tracking_time=tracking,
            ),
        )
        assert run.pump_saturated and not run.pump_limited
        assert run.gain_db[-1] == pytest.approx(run.setpoint, abs=1e-3)
        over.append(float(run.gain_db.max() - run.setpoint))
    assert over[1] < 0.8 * over[0]


def test_derivative_and_tracking_move_nothing_at_zero() -> None:
    times, power = drop(300)
    plain = controlled_gain_transient(
        amplifier(), times, power, PumpControl(bandwidth=1e3, proportional=6.0, filter_time=2e-4)
    )
    zeros = controlled_gain_transient(
        amplifier(),
        times,
        power,
        PumpControl(
            bandwidth=1e3, proportional=6.0, filter_time=2e-4, derivative=0.0, tracking_time=0.0
        ),
    )
    assert np.array_equal(plain.gain_db, zeros.gain_db)
    assert np.array_equal(plain.pump_gain_db, zeros.pump_gain_db)


def test_a_derivative_needs_its_filter_and_neither_is_negative() -> None:
    with pytest.raises(ValueError, match="filter_time"):
        PumpControl(derivative=1e-3)
    with pytest.raises(ValueError, match="derivative"):
        PumpControl(derivative=-1e-3, filter_time=1e-4)
    with pytest.raises(ValueError, match="tracking"):
        PumpControl(tracking_time=-1.0)
    PumpControl(derivative=1e-3, filter_time=1e-4)
