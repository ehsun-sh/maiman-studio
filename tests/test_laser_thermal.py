"""A laser's junction warms, and its line moves with it.

Most of a laser's drive becomes heat, the junction warms by ``R_th P_diss``, and
a warm junction emits longer: ``0.09 nm/K`` for a DFB's grating. Two things
follow, on two time axes, and the model keeps them apart.

* **The bias sets where the line sits.** Settled, at 35 mA, the 1310 nm laser
  here runs 1.6 K warm and emits 0.14 nm long, which is 25 GHz. That belongs to
  the band's own centre frequency: a window of a few hundred nanoseconds cannot
  carry it as a phase ramp.
* **The pattern moves it, slowly.** The junction follows its dissipation with a
  time constant of microseconds, so the line drifts red through a long mark.
  That is thermal chirp, and it is carried in the phase, where it belongs.

Checked against the closed forms the integration does not share: the settled
rise, the exponential the pole implies, and the wavelength each gives.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from maiman.components.dml import DirectlyModulatedLaser
from maiman.laser import LaserParameters, ThermalModel, integrate_rate_equations
from maiman.units import C_LIGHT, from_si, to_si

WARM = ThermalModel()


def settled(model: ThermalModel, parameters: LaserParameters, current: float) -> float:
    """The rise the junction comes to rest at, with the laser's own light taken off."""
    _, photons = parameters.steady_state(current)
    return float(model.rise(current, parameters.power_from_photons(photons)))


# ---------------------------------------------------------------------------
# The model itself
# ---------------------------------------------------------------------------


def test_what_goes_in_and_does_not_leave_as_light_is_the_heat() -> None:
    model = ThermalModel(resistance=45.0, junction_voltage=0.9, series_resistance=5.0)
    assert float(model.dissipation(0.035)) == pytest.approx(0.035 * (0.9 + 0.035 * 5.0), rel=1e-12)
    assert float(model.dissipation(0.035, 0.005)) == pytest.approx(
        0.035 * (0.9 + 0.035 * 5.0) - 0.005, rel=1e-12
    )
    assert float(model.rise(0.035)) == pytest.approx(45.0 * float(model.dissipation(0.035)))
    # A laser cannot be cooled by its own light.
    assert float(model.dissipation(0.001, 1.0)) == 0.0


def test_a_warm_junction_emits_longer_and_lower() -> None:
    model = ThermalModel(drift=0.09e-9)
    assert float(model.wavelength_at(1310e-9, 2.0)) == pytest.approx(1310e-9 + 0.18e-9, rel=1e-12)
    shift = float(model.frequency_shift(1310e-9, 2.0))
    assert shift == pytest.approx(-C_LIGHT * 0.18e-9 / 1310e-9**2, rel=1e-12)
    assert shift < 0.0, "warmer is longer, and longer is lower"


def test_what_is_not_a_junction_is_refused() -> None:
    with pytest.raises(ValueError, match="thermal time constant"):
        ThermalModel(time_constant=0.0)
    with pytest.raises(ValueError, match="resistance"):
        ThermalModel(resistance=-1.0)


def test_the_units_it_is_quoted_in_are_known() -> None:
    assert to_si(45.0, "K/W") == 45.0
    assert to_si(0.09, "nm/K") == pytest.approx(0.09e-9, rel=1e-12)
    assert to_si(1.5, "us") == pytest.approx(1.5e-6, rel=1e-12)
    assert from_si(2e-6, "us") == pytest.approx(2.0, rel=1e-12)
    assert to_si(5.0, "ohm") == 5.0


# ---------------------------------------------------------------------------
# The integration
# ---------------------------------------------------------------------------


def test_the_junction_comes_to_rest_where_the_closed_form_says() -> None:
    """Two code paths: a one-pole ODE integrated, and ``R_th P_diss``."""
    parameters = LaserParameters()
    model = ThermalModel(time_constant=20e-9)
    fs = 2e10
    current = np.full(4000, 0.060)
    run = integrate_rate_equations(
        parameters, current, fs, substeps=8, thermal=model, reference_rise=0.0
    )
    assert run.temperature is not None
    assert run.temperature[-1] == pytest.approx(settled(model, parameters, 0.060), rel=1e-3)
    assert run.wavelength()[-1] == pytest.approx(
        float(model.wavelength_at(parameters.wavelength, run.temperature[-1])), rel=1e-12
    )


def test_it_follows_its_drive_with_the_time_constant_it_was_given() -> None:
    parameters = LaserParameters()
    model = ThermalModel(time_constant=50e-9)
    fs = 2e10
    half = 6000
    current = np.concatenate([np.full(half, 0.020), np.full(6 * half, 0.060)])
    run = integrate_rate_equations(
        parameters, current, fs, substeps=8, thermal=model, reference_rise=0.0
    )
    assert run.temperature is not None
    rise = run.temperature[half:]
    start, final = float(rise[0]), float(rise[-1])
    assert start == pytest.approx(settled(model, parameters, 0.020), rel=1e-3)
    assert final == pytest.approx(settled(model, parameters, 0.060), rel=1e-3)
    crossing = start + (final - start) * (1.0 - math.exp(-1.0))
    reached = int(np.argmin(np.abs(rise - crossing)))
    assert float(run.times[half + reached] - run.times[half]) == pytest.approx(
        model.time_constant, rel=0.05
    )


def test_the_line_drifts_red_through_a_long_mark() -> None:
    """Thermal chirp: the slow tail under the carriers' own adiabatic chirp."""
    parameters = LaserParameters()
    model = ThermalModel(time_constant=50e-9)
    fs = 2e10
    half = 4000
    current = np.concatenate([np.full(half, 0.020), np.full(3 * half, 0.060)])
    run = integrate_rate_equations(
        parameters, current, fs, substeps=8, thermal=model, reference_rise=0.0
    )
    frequency = run.instantaneous_frequency()
    # Well clear of the switching edge, where the carriers' chirp has settled.
    through_the_mark = frequency[half + 200 :]
    assert through_the_mark[-1] < through_the_mark[0], "it runs red as the junction warms"
    assert run.temperature is not None
    drift = float(
        model.frequency_shift(
            parameters.wavelength, run.temperature[-1] - run.temperature[half + 200]
        )
    )
    assert float(through_the_mark[-1] - through_the_mark[0]) == pytest.approx(drift, rel=0.05)


def test_a_hotter_drive_drifts_further() -> None:
    parameters = LaserParameters()
    model = ThermalModel(time_constant=50e-9)
    drifts = []
    for high in (0.040, 0.060, 0.090):
        current = np.concatenate([np.full(2000, 0.020), np.full(6000, high)])
        run = integrate_rate_equations(
            parameters, current, 2e10, substeps=8, thermal=model, reference_rise=0.0
        )
        frequency = run.instantaneous_frequency()
        drifts.append(float(frequency[2200] - frequency[-1]))
    assert all(a < b for a, b in pairwise(drifts))


def test_with_no_thermal_model_nothing_moves_at_all() -> None:
    parameters = LaserParameters()
    current = np.concatenate([np.full(500, 0.020), np.full(500, 0.060)])
    plain = integrate_rate_equations(parameters, current, 2e10, substeps=8)
    assert plain.temperature is None
    assert np.all(plain.wavelength() == parameters.wavelength)
    warm = integrate_rate_equations(
        parameters, current, 2e10, substeps=8, thermal=ThermalModel(resistance=0.0)
    )
    assert np.allclose(warm.phase, plain.phase, rtol=0.0, atol=0.0), "no heat, no drift"


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def test_the_block_puts_the_bias_s_own_shift_in_the_carrier() -> None:
    cold = DirectlyModulatedLaser(label="l")
    warm = DirectlyModulatedLaser(label="l", thermal=True)
    assert cold.thermal_model() is None
    assert cold.junction_rise() == 0.0
    assert cold.emission_wavelength() == cold.si("wavelength")

    assert warm.junction_rise() == pytest.approx(1.6, abs=0.1)
    assert warm.emission_wavelength() - warm.si("wavelength") == pytest.approx(
        WARM.drift * warm.junction_rise(), rel=1e-12
    )
    offset = C_LIGHT / warm.emission_wavelength() - C_LIGHT / warm.si("wavelength")
    assert offset == pytest.approx(-25e9, abs=2e9), "a quarter of a 100 GHz channel"


def test_more_bias_is_more_heat_and_a_longer_line() -> None:
    wavelengths = []
    for bias in (20.0, 35.0, 50.0, 80.0):
        laser = DirectlyModulatedLaser(label="l", thermal=True, bias_current=bias)
        wavelengths.append(laser.emission_wavelength())
    assert all(a < b for a, b in pairwise(wavelengths))
    per_milliamp = (wavelengths[1] - wavelengths[0]) / 15.0
    assert per_milliamp == pytest.approx(4.6e-12, abs=1e-12), "a few picometres a milliamp"


def test_the_flag_is_off_by_default_and_the_knobs_hang_off_it() -> None:
    assert not DirectlyModulatedLaser().thermal
    specs = DirectlyModulatedLaser.param_specs()
    for name in (
        "thermal_resistance",
        "thermal_time_constant",
        "wavelength_drift",
        "junction_voltage",
        "series_resistance",
    ):
        assert specs[name].applies_when == "thermal", name
