"""Pump depletion: the power four-wave mixing and Raman scattering move comes from somewhere.

Four-wave mixing used to create its products out of nothing -- the pumps came
out of a span exactly as bright as if they had made no products at all. Each
product now takes its photons from the pumps that made it and gives one to the
idler, which is checked here against energy conservation and against the full
nonlinear Schrodinger equation, solved by split-step with both tones in one band.

Raman scattering already moved power rather than making it. What it lacked was
a gain shape past 13.2 THz, where a straight line over-predicts the transfer,
and photons rather than watts as the conserved quantity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components.fiber import Fiber
from maiman.kernels import (
    RAMAN_TRIANGLE_LIMIT,
    attenuation_db_per_m_to_alpha,
    effective_length,
    fwm_power_transfer,
    raman_gain,
    raman_tilt,
    raman_transfer,
)
from maiman.signals import Band, OpticalSignal
from maiman.units import to_si, wavelength_to_frequency

ANCHOR = 193.4e12
FS = 160e9
SAMPLES = 1024
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=3, precision="double"
)

#: Deliberately off a uniform grid, so no product lands on a channel: every
#: product is a band of its own and the energy books close exactly.
UNEVEN = (0.0, 100e9, 250e9)


def cw(power: float, f0: float, samples: int = SAMPLES) -> Band:
    return Band(
        Ex=np.full(samples, math.sqrt(power), dtype=np.complex128),
        Ey=np.zeros(samples, dtype=np.complex128),
        f0=f0,
        fs=FS,
    )


def total(signal: OpticalSignal) -> float:
    return sum(band.average_power() for band in signal.bands)


def kerr_span(**settings: float | bool) -> Fiber:
    base: dict[str, float | bool] = {
        "length": 20.0,
        "attenuation": 0.0,
        "dispersion": 0.0,
        "nonlinearity": 1.3,
        "mixing_floor": 200.0,
    }
    base.update(settings)
    return Fiber(label="span", **base)


def uneven_comb(power: float) -> OpticalSignal:
    return OpticalSignal(bands=tuple(cw(power, ANCHOR + offset) for offset in UNEVEN))


# ---------------------------------------------------------------------------
# Four-wave mixing: who pays
# ---------------------------------------------------------------------------


def test_the_transfer_balances_energy_and_feeds_the_idler() -> None:
    """Two photons destroyed, two created, and the frequencies add up."""
    f_i, f_j, f_k = ANCHOR, ANCHOR + 100e9, ANCHOR + 250e9
    product = 3.7e-6
    lost_i, lost_j, gained_k = fwm_power_transfer(f_i, f_j, f_k, product)
    assert lost_i + lost_j == pytest.approx(gained_k + product, rel=1e-12, abs=0.0)
    assert gained_k > 0.0, "the idler is amplified, not depleted"


def test_a_degenerate_pump_pays_both_shares() -> None:
    """When ``i`` and ``j`` are one channel, it gives up two photons per product photon."""
    f_pump, f_idler = ANCHOR, ANCHOR + 100e9
    product = 1e-6
    lost_i, lost_j, gained_k = fwm_power_transfer(f_pump, f_pump, f_idler, product)
    product_frequency = 2 * f_pump - f_idler
    assert lost_i + lost_j == pytest.approx(
        2 * product * f_pump / product_frequency, rel=1e-12, abs=0.0
    )
    assert lost_i + lost_j - gained_k == pytest.approx(product, rel=1e-9, abs=0.0)


def test_a_lossless_span_comes_out_with_the_power_it_was_given() -> None:
    """The test the undepleted model failed: products used to be extra power."""
    out = kerr_span().run(CTX, {"in": uneven_comb(1e-3)})
    signal, diagnostics = out["out"], out["diagnostics"]
    assert len(signal.bands) > len(UNEVEN), "products were generated"
    assert total(signal) == pytest.approx(3e-3, rel=1e-12, abs=0.0)
    assert diagnostics.fwm_depletion > 1e-3


def test_with_loss_mixing_moves_power_and_adds_none() -> None:
    """Against the same span with mixing switched off, the totals are the same."""
    lossy = {"length": 80.0, "attenuation": 0.2}
    mixed = kerr_span(**lossy).run(CTX, {"in": uneven_comb(3e-3)})["out"]
    unmixed = kerr_span(four_wave_mixing=False, **lossy).run(CTX, {"in": uneven_comb(3e-3)})["out"]
    assert total(mixed) == pytest.approx(total(unmixed), rel=1e-12, abs=0.0)


def test_the_diagnostic_is_the_fraction_the_products_hold() -> None:
    out = kerr_span().run(CTX, {"in": uneven_comb(1e-3)})
    signal = out["out"]
    carriers = {ANCHOR + offset for offset in UNEVEN}
    products = sum(b.average_power() for b in signal.bands if b.f0 not in carriers)
    assert out["diagnostics"].fwm_depletion == pytest.approx(
        products / total(signal), rel=1e-9, abs=0.0
    )


def test_at_link_powers_the_pumps_barely_notice() -> None:
    """Four channels at 0 dBm through 80 km of standard fibre: parts per million."""
    signal = OpticalSignal(bands=tuple(cw(1e-3, ANCHOR + i * 100e9) for i in range(4)))
    span = kerr_span(length=80.0, attenuation=0.2, dispersion=17.0)
    depletion = span.run(CTX, {"in": signal})["diagnostics"].fwm_depletion
    assert 0.0 < depletion < 1e-4


def test_a_product_bigger_than_its_pump_is_refused() -> None:
    """Outside the regime the formula holds in, balancing the books would be a fiction."""
    with pytest.raises(ValueError, match="more than its pump holds"):
        kerr_span().run(CTX, {"in": uneven_comb(100e-3)})


# ---------------------------------------------------------------------------
# Four-wave mixing: against the nonlinear Schrodinger equation
# ---------------------------------------------------------------------------

TONE = 10e9


def two_tones(power: float, km: float) -> tuple[float, float, float, float]:
    """Pump power lost and product power made, from the NLSE and from the model.

    The reference puts both tones in *one* band, where the split-step solves the
    Kerr term exactly -- mixing, depletion and all. The model puts them in two
    bands, where mixing is the undepleted formula and depletion is the transfer
    above. Lossless and without dispersion, so nothing else moves the pumps.
    """
    t = np.arange(SAMPLES) / FS
    field = math.sqrt(power) * (np.exp(-2j * np.pi * TONE * t) + np.exp(2j * np.pi * TONE * t))
    one = Band(Ex=field, Ey=np.zeros(SAMPLES, dtype=np.complex128), f0=ANCHOR, fs=FS)
    exact = kerr_span(length=km, max_nonlinear_phase=1e-4)
    (reference,) = exact.run(CTX, {"in": OpticalSignal(bands=(one,))})["out"].bands
    spectrum = np.fft.fft(reference.Ex) / SAMPLES
    bins = np.fft.fftfreq(SAMPLES, 1.0 / FS)

    def at(frequency: float) -> float:
        return float(abs(spectrum[int(np.argmin(abs(bins - frequency)))]) ** 2)

    exact_loss = 2 * power - at(TONE) - at(-TONE)
    exact_products = at(3 * TONE) + at(-3 * TONE)

    split = OpticalSignal(bands=(cw(power, ANCHOR - TONE), cw(power, ANCHOR + TONE)))
    model = kerr_span(length=km).run(CTX, {"in": split})["out"]
    pumps = [b for b in model.bands if abs(abs(b.f0 - ANCHOR) - TONE) < 1e3]
    products = [b for b in model.bands if abs(abs(b.f0 - ANCHOR) - 3 * TONE) < 1e3]
    model_loss = 2 * power - sum(b.average_power() for b in pumps)
    model_products = sum(b.average_power() for b in products)
    return exact_loss, exact_products, model_loss, model_products


def test_the_pumps_lose_what_the_full_equation_says_they_lose() -> None:
    """10 mW over 5 km, gamma P L = 0.065: pump loss and products both within a percent."""
    exact_loss, exact_products, model_loss, model_products = two_tones(10e-3, 5.0)
    assert model_loss == pytest.approx(exact_loss, rel=0.01, abs=0.0)
    assert model_products == pytest.approx(exact_products, rel=0.01, abs=0.0)


def test_the_agreement_fades_where_the_formula_stops_holding() -> None:
    """The product is still the undepleted formula, and it says so as power rises.

    0.2 % at gamma P L = 0.065, 2 % at 0.195, 8 % at 0.39: the model over-predicts,
    because the real products are held back by the depletion and nonlinear phase
    the formula leaves out. This pins that the error grows, rather than claiming
    it is not there.
    """
    errors = []
    for power in (10e-3, 30e-3, 60e-3):
        exact_loss, _, model_loss, _ = two_tones(power, 5.0)
        errors.append(model_loss / exact_loss - 1.0)
    assert errors[0] < 0.005
    assert errors[0] < errors[1] < errors[2]
    assert errors[2] > 0.05


# ---------------------------------------------------------------------------
# Raman: the integrated model
# ---------------------------------------------------------------------------

C_R = to_si(0.028, "1/W/km/THz")
L_EFF = effective_length(attenuation_db_per_m_to_alpha(0.2e-3), 80e3)


def c_band() -> tuple[list[float], list[float]]:
    return [193.1e12 + i * 50e9 for i in range(80)], [1e-3] * 80


def s_c_and_l() -> tuple[list[float], list[float]]:
    frequencies = list(
        np.arange(wavelength_to_frequency(1625e-9), wavelength_to_frequency(1460e-9), 100e9)
    )
    return frequencies, [1e-3] * len(frequencies)


def tilt_db(ratios: list[float]) -> float:
    return 10.0 * math.log10(ratios[0] / ratios[-1])


def test_on_a_straight_line_the_integrator_is_the_closed_form() -> None:
    frequencies, powers = c_band()
    closed = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    integrated = raman_transfer(
        frequencies,
        powers,
        gain_slope=C_R,
        effective_length=L_EFF,
        profile="triangle",
        photon_conserving=False,
    )
    assert integrated == pytest.approx(closed, rel=1e-12, abs=0.0)


def test_below_the_peak_the_silica_shape_is_the_straight_line() -> None:
    inside = np.linspace(0.0, RAMAN_TRIANGLE_LIMIT, 50)
    assert np.allclose(
        raman_gain(inside, gain_slope=C_R, profile="silica"),
        raman_gain(inside, gain_slope=C_R, profile="triangle"),
        rtol=1e-12,
        atol=0.0,
    )
    past = np.array([20e12])
    assert (
        raman_gain(past, gain_slope=C_R, profile="silica")[0]
        < raman_gain(past, gain_slope=C_R, profile="triangle")[0] / 3
    )


def test_photons_are_conserved_and_the_lattice_keeps_the_defect() -> None:
    frequencies, powers = c_band()
    ratios = raman_transfer(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    photons_in = sum(p / f for p, f in zip(powers, frequencies, strict=True))
    photons_out = sum(p * r / f for p, r, f in zip(powers, ratios, frequencies, strict=True))
    assert photons_out == pytest.approx(photons_in, rel=1e-12, abs=0.0)
    change = sum(p * r for p, r in zip(powers, ratios, strict=True)) / sum(powers) - 1.0
    assert -1e-3 < change < 0.0, "a few parts in ten thousand of the power go to the glass"


def test_c_and_l_together_do_not_reach_the_peak() -> None:
    """Which the notes used to say they did."""
    width = wavelength_to_frequency(1530e-9) - wavelength_to_frequency(1610e-9)
    assert width == pytest.approx(9.74e12, rel=1e-3)
    assert width < RAMAN_TRIANGLE_LIMIT


def test_s_c_and_l_tilt_far_less_than_a_straight_line_says() -> None:
    """21 THz at 0 dBm per 100 GHz over one span: 11.19 dB on the line, 6.39 dB on glass."""
    frequencies, powers = s_c_and_l()
    assert frequencies[-1] - frequencies[0] > RAMAN_TRIANGLE_LIMIT
    line = raman_tilt(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    glass = raman_transfer(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    assert tilt_db(line) == pytest.approx(11.19, abs=0.01)
    assert tilt_db(glass) == pytest.approx(6.39, abs=0.01)


def test_the_block_switches_models_past_the_peak() -> None:
    """Inside the peak the closed form, exactly; past it the integrated shape."""
    frequencies, powers = s_c_and_l()
    span = Fiber(length=80.0, attenuation=0.2, dispersion=0.0, raman_gain_slope=0.028, label="wide")
    bands = tuple(cw(p, f, samples=64) for p, f in zip(powers, frequencies, strict=True))
    diagnostics = span.run(CTX, {"in": OpticalSignal(bands=bands)})["diagnostics"]
    expected = raman_transfer(frequencies, powers, gain_slope=C_R, effective_length=L_EFF)
    assert diagnostics.raman_tilt == pytest.approx(tilt_db(expected), rel=1e-9, abs=0.0)


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown Raman profile"):
        raman_gain(np.array([1e12]), gain_slope=C_R, profile="lorentzian")
