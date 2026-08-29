"""Cross-phase modulation, walk-off, and four-wave mixing between channels.

The two effects are validated differently because they are computed differently.
Cross-phase modulation is checked against closed-form nonlinear phase — where a
step size can be tightened until the model reaches the analytic answer — and
four-wave mixing against the undepleted-pump solution evaluated here in the test
rather than imported from the module under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oosim import SimulationContext
from oosim.components.fiber import Fiber
from oosim.kernels import (
    attenuation_db_per_m_to_alpha,
    dispersion_to_beta2,
    effective_length,
    fwm_efficiency,
    fwm_phase_mismatch,
    fwm_product_power,
    propagate_coupled_ssfm,
    propagate_ssfm,
    walkoff_from_dispersion,
)
from oosim.signals import Band, OpticalSignal
from oosim.units import C_LIGHT, wavelength_to_frequency

WAVELENGTH = 1550e-9
GAMMA = 1.3e-3  # 1/W/m
ALPHA = attenuation_db_per_m_to_alpha(0.2e-3)
SPAN = 80e3
FS = 160e9
SAMPLES = 1024
POWER = 1e-3
L_EFF = effective_length(ALPHA, SPAN)
ANCHOR = wavelength_to_frequency(WAVELENGTH)
SPACING = 100e9


def cw(power: float = POWER, samples: int = SAMPLES) -> np.ndarray:
    return np.full(samples, math.sqrt(power), dtype=np.complex128)


def mean_nonlinear_phase(fields: list[np.ndarray], **kwargs: object) -> float:
    """Mean phase the first field picks up, relative to its launch phase."""
    count = len(fields)
    out, _ = propagate_coupled_ssfm(
        fields,
        FS,
        beta2=[0.0] * count,
        walkoff=[0.0] * count,
        gamma=GAMMA,
        alpha=ALPHA,
        distance=SPAN,
        **kwargs,  # type: ignore[arg-type]
    )
    return float(np.mean(np.angle(out[0] / fields[0])))


# ---------------------------------------------------------------------------
# Cross-phase modulation
# ---------------------------------------------------------------------------


def test_one_field_reproduces_the_scalar_propagator_exactly() -> None:
    """The scalar SSFM is written as a call to the coupled one; prove it collapses.

    Not a tautology worth skipping: ``2*total - |A|**2`` reducing to ``|A|**2``
    is the identity the whole cross-phase term rests on, and if it ever stopped
    holding every single-channel result in the suite would move at once.
    """
    rng = np.random.default_rng(11)
    field = (rng.normal(size=SAMPLES) + 1j * rng.normal(size=SAMPLES)) * 0.02
    beta2 = dispersion_to_beta2(17e-6, WAVELENGTH)

    scalar, scalar_diag = propagate_ssfm(
        field, FS, beta2=beta2, gamma=GAMMA, alpha=ALPHA, distance=SPAN
    )
    coupled, coupled_diag = propagate_coupled_ssfm(
        [field], FS, beta2=[beta2], walkoff=[0.0], gamma=GAMMA, alpha=ALPHA, distance=SPAN
    )
    assert np.array_equal(scalar, coupled[0])
    assert scalar_diag.steps == coupled_diag.steps


def test_self_phase_modulation_converges_to_gamma_p_leff() -> None:
    """A CW channel accumulates ``gamma * P * L_eff``, and tightening steps reaches it."""
    expected = GAMMA * POWER * L_EFF
    errors = []
    for cap in (0.02, 0.001, 0.00005):
        phase = mean_nonlinear_phase([cw()], max_nonlinear_phase=cap)
        errors.append(abs(phase / expected - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-4


def test_a_neighbour_rotates_the_phase_twice_as_hard_as_the_channel_itself() -> None:
    """The factor of two, which is the entire content of cross-phase modulation.

    Two equal CW channels: each sees ``gamma*(P + 2P)*L_eff``, three times what
    it would see alone. Nothing here depends on getting ``L_eff`` or ``gamma``
    right — the ratio cancels both — so this isolates the coefficient.
    """
    alone = mean_nonlinear_phase([cw()], max_nonlinear_phase=0.0002)
    paired = mean_nonlinear_phase([cw(), cw()], max_nonlinear_phase=0.0002)
    assert paired / alone == pytest.approx(3.0, rel=1e-3)


def test_three_channels_give_five_times_the_solo_phase() -> None:
    """``P + 2P + 2P = 5P``: the coefficient is per neighbour, not per link."""
    alone = mean_nonlinear_phase([cw()], max_nonlinear_phase=0.0002)
    tripled = mean_nonlinear_phase([cw(), cw(), cw()], max_nonlinear_phase=0.0002)
    assert tripled / alone == pytest.approx(5.0, rel=1e-3)


def test_a_dark_neighbour_contributes_nothing() -> None:
    """Guards the sum: an empty band must not add power just by being present.

    Compared against the closed form rather than against the solo run, because
    the two runs do not take the same steps. The bound is on the largest
    rotation applied to *any* field, and a dark field sees the largest of all —
    twice the total power, with none of its own to subtract — so adding one
    halves the step size while changing the answer not at all.
    """
    expected = GAMMA * POWER * L_EFF
    with_dark = mean_nonlinear_phase(
        [cw(), np.zeros(SAMPLES, dtype=np.complex128)], max_nonlinear_phase=2e-5
    )
    assert with_dark == pytest.approx(expected, rel=1e-4)


def probe_beside_pump(
    spacing: float, dispersion: float, *, duty: float = 0.5, **kwargs: object
) -> tuple[np.ndarray, object]:
    """XPM phase on a CW probe next to an exactly balanced on/off pump."""
    rng = np.random.default_rng(7)
    bits = np.zeros(SAMPLES // 16)
    bits[: round(duty * bits.size)] = 1.0
    rng.shuffle(bits)
    pump = np.sqrt(POWER * np.repeat(bits, 16)).astype(np.complex128)
    beta2 = dispersion_to_beta2(dispersion, WAVELENGTH)

    options: dict[str, object] = {
        "gamma": GAMMA,
        "alpha": ALPHA,
        "distance": SPAN,
        **kwargs,
    }
    pair, diagnostics = propagate_coupled_ssfm(
        [cw(), pump],
        FS,
        beta2=[beta2, beta2],
        walkoff=[0.0, walkoff_from_dispersion(beta2, spacing)],
        **options,  # type: ignore[arg-type]
    )
    solo, _ = propagate_coupled_ssfm(
        [cw()],
        FS,
        beta2=[beta2],
        walkoff=[0.0],
        **options,  # type: ignore[arg-type]
    )
    return np.unwrap(np.angle(pair[0]) - np.angle(solo[0])), diagnostics


def test_peak_to_peak_xpm_matches_the_closed_form() -> None:
    """With nothing sliding past, the swing is ``2 * gamma * P * L_eff`` exactly."""
    phase, _ = probe_beside_pump(0.0, 0.0, max_nonlinear_phase=0.0002)
    swing = float(phase.max() - phase.min())
    assert swing == pytest.approx(2.0 * GAMMA * POWER * L_EFF, rel=2e-3)


def test_walkoff_conserves_the_mean_phase_it_removes_the_modulation() -> None:
    """The invariant that says walk-off redistributes rather than destroys.

    Sliding a neighbour past a channel cannot change how much of its energy the
    channel integrates — only when. So the *mean* cross-phase shift is fixed by
    the neighbour's average power however fast it slides, while the *variation*
    around that mean collapses. That split is the whole reason dispersion helps:
    a constant phase offset is absorbed by carrier recovery and costs nothing,
    and it is the modulation that closes an eye.
    """
    means, deviations = [], []
    for spacing_ghz in (0.0, 25.0, 100.0, 400.0):
        phase, _ = probe_beside_pump(spacing_ghz * 1e9, 17e-6)
        means.append(float(phase.mean()))
        deviations.append(float(phase.std()))

    for value in means:
        assert value == pytest.approx(2.0 * GAMMA * (POWER / 2) * L_EFF, rel=0.02)
    assert deviations == sorted(deviations, reverse=True)
    assert deviations[-1] < deviations[0] / 4.0


def test_walkoff_matches_d_times_delta_lambda() -> None:
    """The engineering form: channels separate by ``D * delta_lambda`` per km."""
    dispersion = 17e-6  # s/m^2, i.e. 17 ps/nm/km
    beta2 = dispersion_to_beta2(dispersion, WAVELENGTH)
    for spacing in (50e9, 100e9, 200e9):
        delta_lambda = WAVELENGTH**2 * spacing / C_LIGHT
        # abs=0 is not decoration. Walk-off in SI units is ~1e-14 s/m and
        # pytest.approx carries a default absolute tolerance of 1e-12, which
        # would swallow the whole quantity and make this assertion vacuous —
        # it did, and a doubled walk-off passed it.
        assert abs(walkoff_from_dispersion(beta2, spacing)) == pytest.approx(
            dispersion * delta_lambda, rel=1e-12, abs=0.0
        )


def test_zero_dispersion_means_no_walkoff() -> None:
    """Walk-off is derived from the dispersion, not declared beside it."""
    assert walkoff_from_dispersion(dispersion_to_beta2(0.0, WAVELENGTH), 100e9) == 0.0


def test_each_field_comes_back_in_its_own_retarded_frame() -> None:
    """A pure group delay is divided out, so a linear run is untouched by walk-off."""
    rng = np.random.default_rng(5)
    field = (rng.normal(size=SAMPLES) + 1j * rng.normal(size=SAMPLES)) * 0.01
    beta2 = dispersion_to_beta2(17e-6, WAVELENGTH)
    out, _ = propagate_coupled_ssfm(
        [field, field],
        FS,
        beta2=[beta2, beta2],
        walkoff=[0.0, walkoff_from_dispersion(beta2, 400e9)],
        gamma=0.0,
        alpha=0.0,
        distance=SPAN,
    )
    # Same beta2, same input, no nonlinearity: the walk-off is the only thing
    # that differs between them, and it must leave nothing behind.
    assert np.allclose(out[0], out[1], atol=1e-12)


def test_walkoff_slip_stays_inside_its_bound() -> None:
    """The reported slip is the quantity the bound claims to cap."""
    for slip in (0.25, 1.0):
        _, diagnostics = probe_beside_pump(400e9, 17e-6, max_walkoff_slip=slip)
        assert diagnostics.peak_walkoff_slip <= slip * (1.0 + 1e-9)  # type: ignore[attr-defined]
        assert diagnostics.walkoff_span > 0.0  # type: ignore[attr-defined]


def test_tightening_the_slip_bound_converges() -> None:
    """Halving the allowed slip must shrink the answer's movement, not shift it."""
    reference, _ = probe_beside_pump(200e9, 17e-6, max_walkoff_slip=0.05, max_nonlinear_phase=5e-4)
    errors = []
    for slip in (2.0, 0.5, 0.125):
        phase, _ = probe_beside_pump(200e9, 17e-6, max_walkoff_slip=slip, max_nonlinear_phase=5e-4)
        errors.append(float(np.max(np.abs(phase - reference))))
    assert errors == sorted(errors, reverse=True)


def test_mismatched_time_grids_are_refused() -> None:
    """Better to stop than to silently propagate the channels independently."""
    with pytest.raises(ValueError, match="common time grid"):
        propagate_coupled_ssfm(
            [cw(samples=SAMPLES), cw(samples=SAMPLES // 2)],
            FS,
            beta2=[0.0, 0.0],
            walkoff=[0.0, 0.0],
            gamma=GAMMA,
            alpha=ALPHA,
            distance=SPAN,
        )


@pytest.mark.parametrize("bad", [{"max_nonlinear_phase": 0.0}, {"max_walkoff_slip": -1.0}])
def test_step_bounds_must_be_positive(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        propagate_coupled_ssfm(
            [cw()],
            FS,
            beta2=[0.0],
            walkoff=[0.0],
            gamma=GAMMA,
            alpha=ALPHA,
            distance=SPAN,
            **bad,  # type: ignore[arg-type]
        )


def test_per_field_parameters_must_match_the_field_count() -> None:
    with pytest.raises(ValueError, match="one entry per field"):
        propagate_coupled_ssfm(
            [cw(), cw()],
            FS,
            beta2=[0.0],
            walkoff=[0.0, 0.0],
            gamma=GAMMA,
            alpha=ALPHA,
            distance=SPAN,
        )


# ---------------------------------------------------------------------------
# Four-wave mixing, as arithmetic
# ---------------------------------------------------------------------------


def test_effective_length_saturates_at_one_over_alpha() -> None:
    assert effective_length(ALPHA, 1e9) == pytest.approx(1.0 / ALPHA, rel=1e-9)
    assert effective_length(0.0, SPAN) == SPAN
    assert effective_length(ALPHA, SPAN) == pytest.approx(21169.27, rel=1e-6)


def test_efficiency_is_one_when_phase_matched() -> None:
    assert fwm_efficiency(0.0, ALPHA, SPAN) == pytest.approx(1.0, rel=1e-12)
    assert fwm_efficiency(0.0, 0.0, SPAN) == pytest.approx(1.0, rel=1e-12)


def test_lossless_efficiency_is_the_sinc_squared_of_the_mismatch() -> None:
    """The limit the closed form is hardest to get right, so it is checked directly."""
    for mismatch in (1e-5, 5e-5, 2e-4):
        argument = mismatch * SPAN / 2.0
        expected = (math.sin(argument) / argument) ** 2
        assert fwm_efficiency(mismatch, 0.0, SPAN) == pytest.approx(expected, rel=1e-9)


def test_efficiency_is_even_in_the_mismatch() -> None:
    for mismatch in (1e-5, 3e-4):
        assert fwm_efficiency(mismatch, ALPHA, SPAN) == pytest.approx(
            fwm_efficiency(-mismatch, ALPHA, SPAN), rel=1e-12
        )


def test_phase_mismatch_grows_as_the_square_of_the_spacing() -> None:
    """Which is why widening the grid suppresses mixing quadratically."""
    beta2 = dispersion_to_beta2(17e-6, WAVELENGTH)
    base = abs(fwm_phase_mismatch(beta2, 0.0, 0.0, 50e9))
    assert abs(fwm_phase_mismatch(beta2, 0.0, 0.0, 100e9)) == pytest.approx(4.0 * base, rel=1e-12)
    assert abs(fwm_phase_mismatch(beta2, 0.0, 0.0, 200e9)) == pytest.approx(16.0 * base, rel=1e-12)


def test_phase_mismatch_vanishes_at_zero_dispersion() -> None:
    assert fwm_phase_mismatch(0.0, 0.0, 0.0, 100e9) == 0.0


def test_dispersion_suppresses_the_product_by_orders_of_magnitude() -> None:
    """17 ps/nm/km against a 100 GHz grid is the difference the parameter buys."""
    beta2 = dispersion_to_beta2(17e-6, WAVELENGTH)
    matched = fwm_efficiency(0.0, ALPHA, SPAN)
    mismatched = fwm_efficiency(fwm_phase_mismatch(beta2, 0.0, 0.0, SPACING), ALPHA, SPAN)
    assert 10.0 * math.log10(matched / mismatched) == pytest.approx(45.4, abs=0.5)


def test_non_degenerate_products_are_six_db_stronger() -> None:
    """d = 2 against d = 1, from counting terms rather than from a fit."""
    common = {
        "gamma": GAMMA,
        "alpha": ALPHA,
        "distance": SPAN,
        "phase_mismatch": 0.0,
    }
    degenerate = fwm_product_power(POWER, POWER, POWER, degenerate=True, **common)  # type: ignore[arg-type]
    mixed = fwm_product_power(POWER, POWER, POWER, degenerate=False, **common)  # type: ignore[arg-type]
    assert 10.0 * math.log10(mixed / degenerate) == pytest.approx(6.0206, rel=1e-6)


def test_product_power_is_cubic_in_the_pump_power() -> None:
    common = {
        "gamma": GAMMA,
        "alpha": ALPHA,
        "distance": SPAN,
        "phase_mismatch": 0.0,
        "degenerate": True,
    }
    weak = fwm_product_power(POWER, POWER, POWER, **common)  # type: ignore[arg-type]
    strong = fwm_product_power(2 * POWER, 2 * POWER, 2 * POWER, **common)  # type: ignore[arg-type]
    assert strong / weak == pytest.approx(8.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Four-wave mixing, through the component
# ---------------------------------------------------------------------------


def comb(channels: int, power: float = POWER) -> tuple[SimulationContext, OpticalSignal]:
    """``channels`` unmodulated carriers on a uniform grid, X-polarized."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=3)
    bands = tuple(
        Band(
            Ex=cw(power),
            Ey=np.zeros(SAMPLES, dtype=np.complex128),
            f0=ANCHOR + index * SPACING,
            fs=FS,
        )
        for index in range(channels)
    )
    return ctx, OpticalSignal(bands=bands, noise=())


def propagate(channels: int, dispersion: float = 0.0, **kwargs: object) -> dict[str, object]:
    ctx, signal = comb(channels)
    kwargs.setdefault("mixing_floor", 200.0)
    fiber = Fiber(
        length=80.0,
        attenuation=0.2,
        dispersion=dispersion,
        nonlinearity=1.3,
        **kwargs,  # type: ignore[arg-type]
    )
    return fiber.run(ctx, {"in": signal})


def band_at(signal: OpticalSignal, offset_ghz: float) -> Band | None:
    target = ANCHOR + offset_ghz * 1e9
    return next((b for b in signal.bands if abs(b.f0 - target) < 1e3), None)


def test_the_component_reproduces_the_closed_form_it_is_built_on() -> None:
    """One pump pair, one product, and the analytic answer recomputed here.

    The degenerate product at ``2*f0 - f1`` has exactly one contributing triplet,
    so no random relative phase enters and the comparison can be exact.
    """
    result = propagate(2)
    product = band_at(result["out"], -100.0)  # type: ignore[arg-type]
    assert product is not None

    expected = fwm_product_power(
        POWER,
        POWER,
        POWER,
        gamma=GAMMA,
        alpha=ALPHA,
        distance=SPAN,
        phase_mismatch=0.0,
        degenerate=True,
    )
    # 1e-7 rather than 1e-9: the component reaches this number through a
    # square root, a unit phasor and the mean of 1024 samples, which costs a
    # few parts in 1e9. abs=0.0 because a product power of 2e-8 W sits well
    # under pytest.approx's default absolute tolerance, which would otherwise
    # accept any answer at all.
    assert product.average_power() == pytest.approx(expected, rel=1e-7, abs=0.0)


def test_dispersion_suppresses_the_product_in_the_graph_too() -> None:
    flat = band_at(propagate(2, dispersion=0.0)["out"], -100.0)  # type: ignore[arg-type]
    dispersive = band_at(propagate(2, dispersion=17.0)["out"], -100.0)  # type: ignore[arg-type]
    assert flat is not None and dispersive is not None
    suppression = 10.0 * math.log10(flat.average_power() / dispersive.average_power())
    assert suppression == pytest.approx(45.4, abs=0.5)


def test_products_land_where_the_frequencies_say_they_do() -> None:
    """Four channels on a uniform grid: six products outside, four folded in."""
    result = propagate(4)
    signal: OpticalSignal = result["out"]  # type: ignore[assignment]
    offsets = sorted(round((band.f0 - ANCHOR) / 1e9) for band in signal.bands)
    assert offsets == [-300, -200, -100, 0, 100, 200, 300, 400, 500, 600]
    assert result["diagnostics"].mixing_products == 10  # type: ignore[attr-defined]


def test_on_a_uniform_grid_the_in_band_products_are_not_separable() -> None:
    """The reason equal spacing is the worst case, stated as a measurement.

    Every product whose frequency coincides with a channel is folded into that
    channel's own band. No filter downstream can remove it, and the channel's
    power moves — in either direction, because the product arrives with a phase
    nobody controls and can subtract as easily as add.
    """
    result = propagate(4)
    signal: OpticalSignal = result["out"]  # type: ignore[assignment]
    channels = [band_at(signal, offset) for offset in (0.0, 100.0, 200.0, 300.0)]
    assert all(band is not None for band in channels)

    undisturbed = POWER * 10.0 ** (-16.0 / 10.0)
    shifts = [
        10.0 * math.log10(band.average_power() / undisturbed)  # type: ignore[union-attr]
        for band in channels
    ]
    assert any(shift > 0.05 for shift in shifts)
    assert any(shift < -0.05 for shift in shifts)
    assert all(abs(shift) < 3.0 for shift in shifts)


def test_a_product_frequency_that_is_a_pump_is_not_counted_twice() -> None:
    """``k`` equal to ``i`` or ``j`` is cross-phase modulation, already applied.

    With two channels the only triplets left after that exclusion are the two
    degenerate ones, so exactly two products appear — not four.
    """
    assert propagate(2)["diagnostics"].mixing_products == 2  # type: ignore[attr-defined]


def test_one_channel_mixes_with_nothing() -> None:
    assert propagate(1)["diagnostics"].mixing_products == 0  # type: ignore[attr-defined]


def test_the_floor_discards_products_nobody_could_measure() -> None:
    """Default settings must not carry a forest of 90 dB-down tones forward."""
    kept = propagate(4, dispersion=17.0)
    dropped = propagate(4, dispersion=17.0, mixing_floor=70.0)  # the default
    assert kept["diagnostics"].mixing_products == 10  # type: ignore[attr-defined]
    assert dropped["diagnostics"].mixing_products == 0  # type: ignore[attr-defined]
    assert len(dropped["out"].bands) == 4  # type: ignore[attr-defined]


def test_mixing_can_be_switched_off_without_switching_off_the_kerr_effect() -> None:
    result = propagate(4, four_wave_mixing=False)
    assert result["diagnostics"].mixing_products == 0  # type: ignore[attr-defined]
    assert len(result["out"].bands) == 4  # type: ignore[attr-defined]
    # The channels still saw each other's power, which is the other half.
    assert result["diagnostics"].steps > 0  # type: ignore[attr-defined]


def test_coupling_can_be_switched_off_and_it_changes_the_answer() -> None:
    """A flag that makes no difference is a flag that is not wired up."""
    coupled = propagate(2, four_wave_mixing=False)
    independent = propagate(2, four_wave_mixing=False, cross_phase_modulation=False)
    a = band_at(coupled["out"], 0.0)  # type: ignore[arg-type]
    b = band_at(independent["out"], 0.0)  # type: ignore[arg-type]
    assert a is not None and b is not None
    phase_difference = float(np.mean(np.angle(a.Ex / b.Ex)))
    # Two equal channels: coupling triples the nonlinear phase, so the gap is
    # twice what one channel accumulates on its own.
    assert phase_difference == pytest.approx(2.0 * GAMMA * POWER * L_EFF, rel=0.05)


def test_a_run_is_reproducible_and_the_seed_actually_moves_it() -> None:
    """Product phases are drawn, so pin both halves of that: same seed, same answer."""
    first = band_at(propagate(3)["out"], -100.0)  # type: ignore[arg-type]
    again = band_at(propagate(3)["out"], -100.0)  # type: ignore[arg-type]
    assert first is not None and again is not None
    assert first.average_power() == pytest.approx(again.average_power(), rel=1e-12, abs=0.0)

    labelled = band_at(propagate(3, label="other")["out"], -100.0)  # type: ignore[arg-type]
    assert labelled is not None
    assert labelled.average_power() != pytest.approx(first.average_power(), rel=1e-6, abs=0.0)


def test_channels_on_different_grids_are_refused_by_the_component() -> None:
    """The kernel's rule, surfaced with the fix in the message."""
    ctx, signal = comb(2)
    odd = Band(
        Ex=cw(samples=SAMPLES // 2),
        Ey=np.zeros(SAMPLES // 2, dtype=np.complex128),
        f0=ANCHOR + 2 * SPACING,
        fs=FS,
    )
    mixed = OpticalSignal(bands=(*signal.bands, odd), noise=())
    fiber = Fiber(length=80.0, nonlinearity=1.3, label="span")
    with pytest.raises(ValueError, match="cross_phase_modulation=False"):
        fiber.run(ctx, {"in": mixed})


def test_a_linear_span_is_untouched_by_either_effect() -> None:
    """gamma = 0 must stay exactly linear: no products, no coupling, no stepping."""
    ctx, signal = comb(4)
    fiber = Fiber(length=80.0, attenuation=0.2, dispersion=17.0, nonlinearity=0.0)
    result = fiber.run(ctx, {"in": signal})
    assert result["diagnostics"].mixing_products == 0  # type: ignore[attr-defined]
    assert result["diagnostics"].steps == 0  # type: ignore[attr-defined]
    assert len(result["out"].bands) == 4  # type: ignore[attr-defined]
