"""Erbium gain dynamics, against the closed forms they have to reduce to.

The model in :mod:`maiman.transient` is the dynamic form of the steady state
:class:`~maiman.components.EDFA` already solves, so most of what is worth
asserting here is that the two agree: run the ODE to rest and it must land on
the component's own Newton solve, by a completely different route.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from maiman import (
    METASTABLE_LIFETIME,
    SimulationContext,
    effective_time_constant,
    gain_transient,
    step_schedule,
)
from maiman.components import EDFA
from maiman.units import db_to_linear

#: -6 dBm a channel, eight of them, which is a WDM comb this repository's own
#: examples could carry rather than a number chosen to make a curve look good.
CHANNEL_POWER = 10.0 ** (-6.0 / 10.0) / 1e3


def amplifier(gain: float = 20.0, *, saturate: bool = True) -> EDFA:
    return EDFA(gain=gain, saturate=saturate, saturation_power=17.0, label="edfa")


# ---------------------------------------------------------------------------
# the two closed forms


@pytest.mark.parametrize("power_dbm", [-30.0, -20.0, -10.0, -3.0, 0.0, 5.0, 10.0])
def test_the_ode_settles_onto_the_gain_the_component_solves(power_dbm: float) -> None:
    """The one test that makes this a model and not a second opinion.

    :meth:`EDFA.effective_gain` solves ``G = G_0 exp(-(G-1) P/P_sat)`` by Newton
    on ``ln G``. This integrates ``tau dg/dt = ln G_0 - g - (e^g - 1) P/P_sat``
    from a long way off and reads where it stops. Same two parameters, no shared
    arithmetic, and they have to meet — if they ever do not, one of them has been
    changed without the other.
    """
    amp = amplifier()
    power = 10.0 ** (power_dbm / 10.0) / 1e3
    # Start at the small-signal gain, which is wrong for every power here, so the
    # integrator has to do the work rather than being handed the answer.
    times = np.linspace(0.0, 40.0 * METASTABLE_LIFETIME, 4000)
    transient = gain_transient(
        amp,
        times,
        np.full(times.shape, power),
        initial_gain=db_to_linear(amp.gain),
    )

    assert transient.gain[-1] == pytest.approx(amp.effective_gain(power), rel=1e-9)


@pytest.mark.parametrize("power_dbm", [-20.0, -10.0, 0.0, 10.0])
def test_the_effective_time_constant_is_the_one_the_step_response_shows(
    power_dbm: float,
) -> None:
    """``tau_eff = tau / (1 + P_out / P_sat)``, measured rather than asserted.

    The reservoir drains at a rate proportional to how full it is, so an
    amplifier driven hard answers in a fraction of its own lifetime. The formula
    is the linearisation of the ODE about its rest point; what is checked is that
    a small step actually decays with that constant, to six figures.
    """
    amp = amplifier()
    power = 10.0 ** (power_dbm / 10.0) / 1e3
    predicted = effective_time_constant(amp, power)

    settled = math.log(amp.effective_gain(power))
    times = np.linspace(0.0, 4.0 * predicted, 20001)
    transient = gain_transient(
        amp, times, np.full(times.shape, power), initial_gain=math.exp(settled + 1e-4)
    )

    excess = np.log(transient.gain) - settled
    crossed = np.nonzero(excess <= 1e-4 / math.e)[0]
    assert crossed.size, "the perturbation should decay inside four time constants"
    measured = float(times[crossed[0]])
    assert measured == pytest.approx(predicted, rel=1e-3)


def test_the_time_constant_shortens_with_drive_and_never_exceeds_the_lifetime() -> None:
    """The direction is the physics, and it is the reason the effect is fast.

    An unsaturated amplifier relaxes at the bare metastable lifetime, 10 ms. Every
    watt of output shortens that, monotonically, which is why a deployed
    amplifier's transient is measured in a fraction of a lifetime.
    """
    amp = amplifier()
    constants = [
        effective_time_constant(amp, 10.0 ** (dbm / 10.0) / 1e3)
        for dbm in (-40.0, -20.0, -10.0, 0.0, 10.0, 15.0)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(constants))
    assert all(constant < METASTABLE_LIFETIME for constant in constants)
    assert effective_time_constant(amp, 0.0) == METASTABLE_LIFETIME


# ---------------------------------------------------------------------------
# why this is an analysis and not a block


def test_the_transient_is_far_longer_than_any_window_this_engine_runs() -> None:
    """The measurement the design decision rests on, kept so it stays true.

    There is no ``metastable_lifetime`` parameter on :class:`EDFA` because within
    one simulation window the gain is a constant — which is what the component
    already assumes. That is only right while the two timescales stay this far
    apart. If a window ever gets long enough, or an amplifier fast enough, that
    the ratio falls toward one, the gain inside a window stops being a constant
    and this decision needs making again rather than inheriting.
    """
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=16, sequence_length=4096, seed=1)
    window = ctx.sequence_length / ctx.bit_rate
    assert window < 200e-9, "a window here is nanoseconds"

    amp = amplifier()
    # The most saturated case in this repository's range: the fastest it gets.
    fastest = effective_time_constant(amp, 10.0 ** (10.0 / 10.0) / 1e3)
    assert fastest / window > 1e4, (
        f"the fastest transient is {fastest / window:.0f} windows long; below ~1e4 "
        f"the gain within a window is no longer a constant and EDFA has to know it"
    )


def test_an_unsaturated_amplifier_has_no_dynamics_to_integrate() -> None:
    """Its gain does not depend on its input, so nothing relaxes.

    Returned as the constant it is rather than as a flat line produced by
    integrating zero four thousand times, which would be the same numbers and a
    claim that something was computed.
    """
    amp = amplifier(saturate=False)
    times, power = step_schedule([(5e-3, 8 * CHANNEL_POWER), (5e-3, CHANNEL_POWER)])
    transient = gain_transient(amp, times, power)

    assert np.all(transient.gain == db_to_linear(amp.gain))
    assert transient.excursion == 0.0
    assert transient.substeps == 1


# ---------------------------------------------------------------------------
# the effect itself


def test_dropping_channels_raises_the_survivors_and_adding_them_lowers_it() -> None:
    """Both directions, because they are different failures in a deployed system.

    A drop raises the gain and can drive a receiver into overload; an add lowers
    it and can starve one. The excursion is signed for that reason, and asserting
    only the magnitude would let a sign error through that reverses which fault
    the model predicts.
    """
    amp = amplifier()
    times, dropped = step_schedule(
        [(20e-3, 8 * CHANNEL_POWER), (60e-3, CHANNEL_POWER)], points_per_segment=2000
    )
    _, added = step_schedule(
        [(20e-3, CHANNEL_POWER), (60e-3, 8 * CHANNEL_POWER)], points_per_segment=2000
    )

    drop = gain_transient(amp, times, dropped)
    add = gain_transient(amp, times, added)

    assert drop.excursion > 3.0, "seven channels of eight is a large disturbance"
    assert add.excursion < -3.0

    # The *steady states* are the same pair either way, so the swing between them
    # is one number and is asserted exactly. The two curves are not mirror images
    # of each other on a finite window, and that is not a discrepancy: they relax
    # at different rates, because tau_eff depends on the power each ends at. The
    # drop ends at one channel and settles in 7.9 ms, the add at eight and settles
    # in 4.9, so after the same 60 ms the add is the nearer of the two to rest.
    swing = 10.0 * math.log10(
        amp.effective_gain(CHANNEL_POWER) / amp.effective_gain(8 * CHANNEL_POWER)
    )
    assert drop.excursion == pytest.approx(swing, rel=2e-3)
    assert -add.excursion == pytest.approx(swing, rel=2e-3)
    assert abs(add.excursion) > abs(drop.excursion), "the add is further settled at 60 ms"

    # And each lands on the steady state for the power it ends at.
    assert drop.gain[-1] == pytest.approx(amp.effective_gain(CHANNEL_POWER), rel=1e-3)
    assert add.gain[-1] == pytest.approx(amp.effective_gain(8 * CHANNEL_POWER), rel=1e-3)


def test_the_relaxation_is_monotone_and_does_not_overshoot() -> None:
    """A first-order system cannot ring, so a curve that does is an integrator bug.

    Worth asserting because an RK step that is too long for the constant produces
    exactly that — an oscillation that looks like a physical overshoot and is not.
    """
    amp = amplifier()
    times, power = step_schedule(
        [(20e-3, 8 * CHANNEL_POWER), (60e-3, CHANNEL_POWER)], points_per_segment=1500
    )
    transient = gain_transient(amp, times, power)

    after = transient.gain_db[times >= 20e-3]
    assert np.all(np.diff(after) >= -1e-12), "the gain must climb without ringing"
    assert after[-1] <= 10.0 * math.log10(amp.effective_gain(CHANNEL_POWER)) + 1e-9


@pytest.mark.parametrize("points", [20, 100, 400, 4000])
def test_a_coarse_output_grid_still_gets_the_right_curve(points: int) -> None:
    """The substepping, which is the difference between a curve and a plausible one.

    A caller asking for twenty points across ninety milliseconds is asking where
    to draw, not how finely to integrate. The integrator takes its own steps from
    the fastest time constant in the run and reports how many, so the answer does
    not depend on the grid — measured here across a 200x range of spacing.
    """
    amp = amplifier()
    times, power = step_schedule(
        [(30e-3, 8 * CHANNEL_POWER), (60e-3, CHANNEL_POWER)], points_per_segment=points
    )
    transient = gain_transient(amp, times, power)

    assert transient.gain_db[-1] == pytest.approx(18.835760407, abs=1e-6)
    assert transient.excursion == pytest.approx(3.210448, abs=1e-5)
    if points == 20:
        assert transient.substeps > 1, "a grid this coarse has to be substepped"


def test_the_excursion_accumulates_down_a_chain_of_amplifiers() -> None:
    """Why one amplifier's 3 dB is not the number that takes a link out.

    There is no feedback from a later amplifier to an earlier one, so a chain
    integrates exactly in order: each takes the previous one's output on the same
    grid. The surviving channel's excursion is the *sum* of the gain excursions
    it passes through.

    It accumulates sub-linearly, not as eight times the first: each amplifier
    down the chain starts less saturated than the one before, so it has less
    compression to give back. Eight times 3.21 dB would be 25.7; the measured
    figure is 9.94.
    """
    times, power = step_schedule(
        [(20e-3, 8 * CHANNEL_POWER), (80e-3, CHANNEL_POWER)], points_per_segment=3000
    )
    # Span loss equal to the compressed operating gain, so the chain is level
    # before the drop rather than decaying into insignificance.
    loss = 10.0 ** (-15.63 / 10.0)

    drive = power
    survivor = np.zeros(times.size)
    first = 0.0
    for stage in range(8):
        transient = gain_transient(amplifier(), times, drive)
        if stage == 0:
            first = transient.excursion
        survivor = survivor + (transient.gain_db - transient.gain_db[0])
        drive = transient.output_power * loss

    peak = float(survivor[np.argmax(np.abs(survivor))])
    assert first == pytest.approx(3.21, abs=0.02)
    assert peak == pytest.approx(9.94, abs=0.01)
    assert peak > 2.5 * first, "a chain is materially worse than one amplifier"
    assert peak < 8.0 * first, "and materially better than eight times it"


# ---------------------------------------------------------------------------
# the schedule, and what is refused


def test_the_step_lands_on_a_sample_so_no_interval_straddles_it() -> None:
    """Which is what makes the integration of a step exact rather than nearly so.

    The input is right-continuous: the power over an interval is the value at its
    start. So the step has to land *on* a sample — the first of the segment it
    begins. Put it between two samples instead and the interval containing it
    gets integrated at one power for a span during which it was two, wrong by an
    amount nothing reports.
    """
    times, power = step_schedule([(10e-3, 2.0), (10e-3, 1.0)], points_per_segment=5)
    assert np.all(np.diff(times) > 0.0)

    step = np.nonzero(times == 10e-3)[0]
    assert step.size == 1, "the step time is a sample"
    assert power[int(step[0])] == 1.0, "and it already carries the new power"
    assert power[times < 10e-3].tolist() == [2.0] * 5
    assert power[times >= 10e-3].tolist() == [1.0] * 5
    assert times[-1] == pytest.approx(20e-3), "the last segment carries the final time"


def test_a_schedule_that_is_not_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        step_schedule([])
    with pytest.raises(ValueError, match="at least two points"):
        step_schedule([(1e-3, 1.0)], points_per_segment=1)
    with pytest.raises(ValueError, match="not a duration"):
        step_schedule([(0.0, 1.0)])
    with pytest.raises(ValueError, match="not a power"):
        step_schedule([(1e-3, -1.0)])


def test_a_grid_that_cannot_be_integrated_on_is_refused() -> None:
    """Each of these would otherwise produce a number, which is the problem."""
    amp = amplifier()
    times = np.linspace(0.0, 1e-3, 16)
    power = np.full(times.shape, CHANNEL_POWER)

    with pytest.raises(ValueError, match="must match times"):
        gain_transient(amp, times, power[:-1])
    with pytest.raises(ValueError, match="at least two times"):
        gain_transient(amp, times[:1], power[:1])
    with pytest.raises(ValueError, match="strictly increasing"):
        gain_transient(amp, times[::-1], power)
    with pytest.raises(ValueError, match="must not be negative"):
        gain_transient(amp, times, -power)
    with pytest.raises(ValueError, match="lifetime must be positive"):
        gain_transient(amp, times, power, lifetime=0.0)
    with pytest.raises(ValueError, match="must be 1-D"):
        gain_transient(amp, np.zeros((4, 4)), np.zeros((4, 4)))


def test_the_settling_time_is_measured_from_the_last_excursion_not_the_first() -> None:
    """A curve that crosses the band and comes back is not settled at the crossing."""
    times = np.linspace(0.0, 10.0, 11)
    amp = amplifier(saturate=False)
    transient = gain_transient(amp, times, np.full(times.shape, CHANNEL_POWER))
    # Flat: it is settled from the start, and the answer is zero rather than the
    # first index that happened to be inside the band.
    assert transient.settling_time() == 0.0

    schedule = step_schedule(
        [(20e-3, 8 * CHANNEL_POWER), (80e-3, CHANNEL_POWER)], points_per_segment=800
    )
    real = gain_transient(amplifier(), *schedule)
    settled = real.settling_time(tolerance_db=0.1)
    assert 20e-3 < settled < 100e-3
    assert abs(real.gain_db[-1] - real.gain_db[real.times >= settled][0]) <= 0.1
