"""A directly modulated laser, against the closed forms its rate equations imply.

The integration is checked where the equations have an analytic answer: the
threshold current, the slope of the light-current curve, the frequency a step
rings at, and the chirp -- which is measured from the field's own phase and
compared against the two-term expression it is supposed to produce, not declared
by it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import Graph, SimulationContext
from maiman.components import DirectlyModulatedLaser, Fiber, NRZDriver, PRBSGenerator
from maiman.laser import ELECTRON_CHARGE, LaserParameters, integrate_rate_equations
from maiman.signals import ElectricalSignal, OpticalSignal

LASER = LaserParameters()
SAMPLE_RATE = 640e9


def step(bias: float, top: float, samples: int = 8192, at: int = 1024) -> np.ndarray:
    return np.where(np.arange(samples) < at, bias, top)


# ---------------------------------------------------------------------------
# The steady state
# ---------------------------------------------------------------------------


def test_threshold_is_where_the_gain_balances_the_cavity() -> None:
    """``I_th = q V N_th / tau_n``, and the light-current curve kinks there."""
    threshold = LASER.threshold_current()
    assert threshold == pytest.approx(
        ELECTRON_CHARGE * LASER.active_volume * LASER.threshold_density / LASER.carrier_lifetime,
        rel=1e-12,
    )
    assert threshold * 1e3 == pytest.approx(23.575, abs=0.01)

    below = LASER.power_from_photons(LASER.steady_state(0.8 * threshold)[1])
    just_above = LASER.power_from_photons(LASER.steady_state(1.2 * threshold)[1])
    assert below < 1e-5, "below threshold there is only spontaneous emission"
    assert just_above > 100 * below


def test_the_light_current_slope_is_one_photon_per_electron() -> None:
    """``eta h nu / q``, which no rate in the model appears in."""
    low, high = 40e-3, 60e-3
    power_low = float(LASER.power_from_photons(LASER.steady_state(low)[1]))
    power_high = float(LASER.power_from_photons(LASER.steady_state(high)[1]))
    assert (power_high - power_low) / (high - low) == pytest.approx(
        LASER.slope_efficiency(), rel=0.01
    )


def test_a_settled_integration_is_the_steady_state_it_started_from() -> None:
    """Integrating a constant current must not move: the two solvers have to agree."""
    constant = np.full(2048, 45e-3)
    solved = integrate_rate_equations(LASER, constant, SAMPLE_RATE)
    expected = float(LASER.power_from_photons(LASER.steady_state(45e-3)[1]))
    assert solved.power == pytest.approx(expected, rel=1e-9)
    assert float(np.ptp(solved.instantaneous_frequency())) < 1e3, "nothing chirps at rest"


def test_the_photon_density_and_the_power_are_inverses() -> None:
    photons = LASER.steady_state(50e-3)[1]
    power = float(LASER.power_from_photons(photons))
    assert LASER.photons_from_power(power) == pytest.approx(photons, rel=1e-12)


# ---------------------------------------------------------------------------
# The transient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bias", "top"), [(30e-3, 50e-3), (35e-3, 60e-3)])
def test_a_step_rings_at_the_relaxation_frequency(bias: float, top: float) -> None:
    """Read off a spectrum of the step response, against ``sqrt(g0 S0 / tau_p) / 2 pi``.

    The measured peak sits a little below the undamped formula because gain
    compression damps the ringing, which is the direction it must err in.
    """
    solved = integrate_rate_equations(LASER, step(bias, top), SAMPLE_RATE)
    transient = solved.power[1024:5120]
    transient = transient - transient[-1024:].mean()
    spectrum = np.abs(np.fft.rfft(transient * np.hanning(transient.size)))
    frequencies = np.fft.rfftfreq(transient.size, 1.0 / SAMPLE_RATE)
    measured = float(frequencies[int(np.argmax(spectrum[1:])) + 1])

    undamped = LASER.relaxation_frequency(top)
    assert measured == pytest.approx(undamped, rel=0.05)
    assert measured < undamped, "gain compression damps it, which lowers the peak"
    assert solved.power.max() > 1.3 * solved.power[-1], "a step overshoots before it settles"


def test_the_confinement_factor_is_not_in_the_relaxation_frequency() -> None:
    """``Gamma`` cancels between the two equations; keeping it costs a factor of 1.8 here."""
    photons = LASER.steady_state(50e-3)[1]
    assert LASER.relaxation_frequency(50e-3) == pytest.approx(
        math.sqrt(LASER.gain_slope * photons / LASER.photon_lifetime) / (2 * math.pi), rel=1e-12
    )


def test_the_chirp_is_the_transient_and_adiabatic_terms_together() -> None:
    """Measured from the phase, against ``alpha/4pi d(lnP)/dt + kappa P``.

    Neither term is applied anywhere in the integration: the phase follows the
    carrier density, and this expression is what that is known to produce. The
    correlation across the transient is the strong form of the claim.
    """
    solved = integrate_rate_equations(LASER, step(30e-3, 50e-3), SAMPLE_RATE)
    power = solved.power
    measured = solved.instantaneous_frequency()
    kappa = LASER.adiabatic_chirp_factor()
    model = (
        LASER.linewidth_enhancement / (4 * math.pi) * np.gradient(np.log(power), 1.0 / SAMPLE_RATE)
        + kappa * power
    )

    settled = slice(7000, 8000)
    biased = slice(400, 1000)
    assert measured[settled].mean() == pytest.approx(model[settled].mean(), rel=0.02)
    assert measured[biased].mean() == pytest.approx(model[biased].mean(), rel=0.1)
    assert np.corrcoef(measured[1024:2048], model[1024:2048])[0, 1] > 0.999
    assert measured[1024:1200].max() > 10e9, "the transient chirp is tens of gigahertz"


def test_the_adiabatic_chirp_is_proportional_to_power() -> None:
    """The steady offset between two levels, against kappa times their power difference."""
    solved = integrate_rate_equations(LASER, step(30e-3, 50e-3), SAMPLE_RATE)
    measured = solved.instantaneous_frequency()
    power = solved.power
    low, high = slice(400, 1000), slice(7000, 8000)
    moved = measured[high].mean() - measured[low].mean()
    assert moved == pytest.approx(
        LASER.adiabatic_chirp_factor() * (power[high].mean() - power[low].mean()), rel=0.03
    )


def test_a_laser_that_cannot_be_driven_is_refused() -> None:
    with pytest.raises(ValueError, match="negative current"):
        integrate_rate_equations(LASER, np.array([-1e-3, 1e-3]), SAMPLE_RATE)
    with pytest.raises(ValueError, match="at least two samples"):
        integrate_rate_equations(LASER, np.array([1e-3]), SAMPLE_RATE)
    with pytest.raises(ValueError, match="output_coupling"):
        LaserParameters(output_coupling=1.5)
    with pytest.raises(ValueError, match="photon_lifetime must be positive"):
        LaserParameters(photon_lifetime=0.0)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def drive(ctx: SimulationContext, volts: np.ndarray) -> ElectricalSignal:
    return ElectricalSignal(samples=volts, fs=ctx.sample_rate, unit="V")


def test_the_block_is_the_rate_equations_and_its_datasheet_numbers_agree() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=32, sequence_length=64)
    laser = DirectlyModulatedLaser(bias_current=35.0, transconductance=25.0, label="dml")
    volts = np.tile(np.repeat([0.0, 1.0], ctx.samples_per_symbol), ctx.sequence_length // 2)
    waveform = drive(ctx, volts)

    assert laser.threshold_current() == pytest.approx(LASER.threshold_current(), rel=1e-12)
    assert laser.slope_efficiency() == pytest.approx(LASER.slope_efficiency(), rel=1e-12)
    assert laser.drive_current(waveform).max() == pytest.approx(60e-3, rel=1e-12)
    assert laser.drive_current(waveform).min() == pytest.approx(35e-3, rel=1e-12)

    (band,) = laser.run(ctx, {"in": waveform})["out"].bands
    solved = laser.solve(waveform)
    assert np.allclose(np.abs(band.Ex) ** 2, solved.power, rtol=1e-6, atol=0.0)
    assert band.f0 == pytest.approx(299792458.0 / 1310e-9, rel=1e-12)


def test_the_block_chirps_a_data_pattern() -> None:
    """A modulated laser's line is tens of gigahertz wide; a CW one's is not."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=32, sequence_length=128, seed=5)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, label="prbs"))
    driver = graph.add(NRZDriver(v_low=0.0, v_high=1.0, label="drv"))
    laser = graph.add(DirectlyModulatedLaser(label="dml"))
    span = graph.add(Fiber(length=0.0, attenuation=0.0, label="fibre"))
    graph.connect(prbs, driver["in"])
    graph.connect(driver, laser["in"])
    graph.connect(laser, span["in"])
    out = graph.run().port(span, "out")
    assert isinstance(out, OpticalSignal)

    (band,) = out.bands
    phase = np.unwrap(np.angle(band.Ex.astype(np.complex128)))
    frequency = np.gradient(phase, 1.0 / ctx.sample_rate) / (2 * math.pi)
    assert float(np.ptp(frequency)) > 5e9, "modulating the current moves the line"
    power = np.abs(band.Ex) ** 2
    assert 10 * math.log10(power.max() / power.min()) > 3.0, "and modulates the power"
