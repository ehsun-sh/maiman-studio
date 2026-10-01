"""The tone solver: constant tones through a span, every mixing order at once.

``tone_solver`` integrates the coupled-mode equations of the launched tones and
every product above the floor (:func:`maiman.kernels.fwm_tone_solve`) instead of
summing the perturbative series :meth:`Fiber._mix` does, which stops at second
order and assumes undepleted pumps (maiman-ufk). What the series leaves out grows
as ``gamma P L``, so the tests that matter are the ones at powers where it fails:
the split-step is the reference, and the kernel is checked against three
closed-form limits besides.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from maiman.components import EDFA, Fiber
from maiman.context import SimulationContext
from maiman.kernels import (
    PMDSection,
    apply_pmd,
    attenuation_db_per_m_to_alpha,
    effective_length,
    fwm_cascade_amplitude,
    fwm_phase_mismatch,
    fwm_product_power,
    fwm_tone_solve,
    propagate_ssfm,
    raman_coupling,
    raman_tilt,
    raman_transfer,
    random_pmd_sections,
)
from maiman.signals import Band, OpticalSignal

ANCHOR = 193.1e12
SPACING = 100e9
GAMMA = 1.3  # 1/W/km
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


def two_pumps(pump: float) -> OpticalSignal:
    return OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(256, np.sqrt(pump), dtype=np.complex128),
                Ey=np.zeros(256, dtype=np.complex128),
                f0=ANCHOR + index * SPACING,
                fs=160e9,
            )
            for index in range(2)
        )
    )


def beta2_of(dispersion: float) -> float:
    return Fiber(dispersion=dispersion).reference_beta2(two_pumps(1e-3))


def split_step(
    pump: float, dispersion: float, span: float, spans: int, attenuation: float, gain: float
) -> dict[int, complex]:
    """The one-band split-step's amplitude ``m`` spacings below the lower pump, m = 1, 2."""
    n, fs = 4096, 3.2e12
    t = np.arange(n) / fs
    field = np.sqrt(pump) * (1.0 + np.exp(2j * np.pi * SPACING * t))
    beta2 = beta2_of(dispersion)
    mismatch = abs(fwm_phase_mismatch(beta2, 0.0, 0.0, SPACING))
    alpha = attenuation_db_per_m_to_alpha(attenuation * 1e-3)
    for _ in range(spans):
        field, _ = propagate_ssfm(
            field,
            fs,
            beta2=beta2,
            gamma=GAMMA * 1e-3,
            alpha=alpha,
            distance=span,
            max_nonlinear_phase=1e-4,
            max_step=0.01 / mismatch,
        )
        field = field * 10 ** (gain / 20)
    spectrum = np.fft.fft(field) / n
    return {m: complex(spectrum[-round(m * SPACING / (fs / n))]) for m in (1, 2)}


def tone_link(
    pump: float,
    dispersion: float,
    span: float,
    spans: int,
    attenuation: float = 0.0,
    gain: float = 0.0,
    **settings: float | bool,
) -> tuple[dict[int, complex], OpticalSignal]:
    signal = two_pumps(pump)
    for index in range(spans):
        signal = Fiber(
            length=span / 1e3,
            attenuation=attenuation,
            dispersion=dispersion,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            tone_solver=True,
            label=f"span{index}",
            **settings,
        ).run(CTX, {"in": signal})["out"]
        if gain:
            signal = EDFA(gain=gain, noise_figure=0.0, label=f"amp{index}").run(
                CTX, {"in": signal}
            )["out"]
    found = {}
    for m in (1, 2):
        band = [b for b in signal.bands if abs(b.f0 - (ANCHOR - m * SPACING)) < 1e3]
        found[m] = complex(np.mean(band[0].Ex)) if band else 0j
    return found, signal


# -- fwm_tone_solve: closed-form limits ---------------------------------------


def tones(pump: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.array([k * SPACING for k in range(-4, 6)], dtype=float)
    start = np.zeros(offsets.size, dtype=np.complex128)
    start[4] = start[5] = math.sqrt(pump)
    return offsets, start


@pytest.mark.parametrize("pump", [1e-3, 3e-2])
def test_total_power_is_conserved_without_loss(pump: float) -> None:
    """Every term of ``|A|^2 A`` moves power and none makes it: the sum holds to the integrator."""
    offsets, start = tones(pump)
    end, _ = fwm_tone_solve(
        offsets, start, beta2=beta2_of(2.0), gamma=GAMMA * 1e-3, alpha=0.0, distance=20e3
    )
    assert np.sum(np.abs(end) ** 2) == pytest.approx(np.sum(np.abs(start) ** 2), rel=1e-9)


def test_loss_takes_exactly_its_share() -> None:
    """With nothing to mix or turn (gamma = 0) each tone falls by ``exp(-alpha L / 2)``."""
    offsets, start = tones(1e-3)
    alpha = attenuation_db_per_m_to_alpha(0.2e-3)
    end, _ = fwm_tone_solve(
        offsets, start, beta2=beta2_of(2.0), gamma=0.0, alpha=alpha, distance=20e3
    )
    assert np.abs(end[4]) == pytest.approx(math.sqrt(1e-3) * math.exp(-alpha * 20e3 / 2), rel=1e-9)


def test_the_undepleted_limit_is_the_first_order_product_power() -> None:
    """At 1 microwatt the first product is ``fwm_product_power``, to the terms it leaves out.

    ``gamma P L`` is 3e-5 here, so the mixing series' next terms are 1e-9 of it.
    """
    pump = 1e-6
    offsets, start = tones(pump)
    beta2 = beta2_of(2.0)
    end, _ = fwm_tone_solve(
        offsets, start, beta2=beta2, gamma=GAMMA * 1e-3, alpha=0.0, distance=20e3
    )
    closed = fwm_product_power(
        pump,
        pump,
        pump,
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=20e3,
        phase_mismatch=fwm_phase_mismatch(beta2, SPACING, SPACING, 0.0),
        degenerate=True,
        nonlinear_rate=0.0,
    )
    assert abs(end[3]) ** 2 == pytest.approx(closed, rel=1e-3)


def test_phase_matched_the_second_order_grows_as_the_cascade_integral() -> None:
    """No dispersion, 1 microwatt: the second product is the cascade integral's ``L^2 / 2``.

    Thompson and Roy's second order at zero mismatch, the same closed form
    :func:`fwm_cascade_amplitude` gives; the solver reaches it with no series.
    """
    pump = 1e-6
    offsets, start = tones(pump)
    end, _ = fwm_tone_solve(offsets, start, beta2=0.0, gamma=GAMMA * 1e-3, alpha=0.0, distance=20e3)
    # Two routes into 3 f_low - 2 f_high: (P1, low, high*), d = 2, and (low, low, P2*), d = 1.
    route_a = fwm_cascade_amplitude(
        pump,
        pump,
        pump,
        pump,
        pump,
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=20e3,
        phase_mismatch_1=0.0,
        phase_mismatch_2=0.0,
        degenerate_1=True,
        degenerate_2=False,
    )
    route_b = fwm_cascade_amplitude(
        pump,
        pump,
        pump,
        pump,
        pump,
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=20e3,
        phase_mismatch_1=0.0,
        phase_mismatch_2=0.0,
        degenerate_1=True,
        degenerate_2=True,
        conjugate_first=True,
    )
    assert abs(end[2]) == pytest.approx(abs(route_a + route_b), rel=2e-3)


# -- against the split-step, where the series fails ---------------------------


@pytest.mark.parametrize(
    ("pump", "dispersion", "span", "spans"),
    [(30e-3, 1.0, 20e3, 1), (10e-3, 2.0, 10e3, 2), (20e-3, 4.0, 5e3, 4), (10e-3, -2.0, 5e3, 4)],
)
def test_lossless_spans_land_on_the_split_step_at_powers_the_series_misses(
    pump: float, dispersion: float, span: float, spans: int
) -> None:
    """The cases the perturbative series is 25 % (30 mW, one 20 km span) to 2.3 times off.

    Every order shown: the first product, the second and, in the tone set,
    everything above the floor. Agreement is to the split-step's own accuracy.
    """
    reference = split_step(pump, dispersion, span, spans, 0.0, 0.0)
    found, _ = tone_link(pump, dispersion, span, spans)
    for m in (1, 2):
        assert abs(found[m]) == pytest.approx(abs(reference[m]), rel=2e-3)


def test_amplified_lossy_spans_land_on_the_split_step() -> None:
    """Three 40 km spans at 0.2 dB/km, each followed by the gain that puts the power back."""
    reference = split_step(20e-3, 2.0, 40e3, 3, 0.2, 8.0)
    found, _ = tone_link(20e-3, 2.0, 40e3, 3, attenuation=0.2, gain=8.0)
    for m in (1, 2):
        assert abs(found[m]) == pytest.approx(abs(reference[m]), rel=2e-3)


def test_a_lossless_span_keeps_the_launched_power() -> None:
    _, signal = tone_link(30e-3, 1.0, 20e3, 1)
    total = sum(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in signal.bands)
    assert total == pytest.approx(60e-3, rel=1e-9)


def test_tone_order_one_keeps_only_the_first_products() -> None:
    _, one = tone_link(10e-3, 2.0, 10e3, 1, tone_order=1.0)
    _, three = tone_link(10e-3, 2.0, 10e3, 1, tone_order=3.0)
    assert len(one.bands) == 4  # two pumps and the two first-order products
    assert len(three.bands) > len(one.bands)


# -- what it refuses ------------------------------------------------------------


def test_a_modulated_band_is_refused_by_name() -> None:
    signal = two_pumps(1e-3)
    wobble = np.exp(1j * np.linspace(0.0, 3.0, 256))
    bands = (
        signal.bands[0],
        Band(
            Ex=signal.bands[1].Ex * wobble, Ey=signal.bands[1].Ey, f0=signal.bands[1].f0, fs=160e9
        ),
    )
    with pytest.raises(ValueError, match="modulated"):
        Fiber(length=10.0, dispersion=2.0, nonlinearity=GAMMA, tone_solver=True).run(
            CTX, {"in": OpticalSignal(bands=bands)}
        )


@pytest.mark.parametrize(
    "other",
    [
        {"pump_phase": True},
        {"cascaded_fwm": True},
        {"mixing_steps": 4.0},
    ],
)
def test_it_is_refused_beside_what_it_replaces(other: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="tone_solver"):
        Fiber(tone_solver=True, **other).validate()
    Fiber(tone_solver=True).validate()


# -- two polarizations ------------------------------------------------------------


def jones_pumps(power: float, jones: tuple[complex, complex]) -> OpticalSignal:
    return OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(256, jones[0] * np.sqrt(power), dtype=np.complex128),
                Ey=np.full(256, jones[1] * np.sqrt(power), dtype=np.complex128),
                f0=ANCHOR + index * SPACING,
                fs=160e9,
            )
            for index in range(2)
        )
    )


ONE_TONE = np.array([0.0])
RATE_LENGTH = 5e3


def turned(amplitude: np.ndarray, power: float, *, coherent: bool = True) -> np.ndarray:
    return fwm_tone_solve(
        ONE_TONE,
        amplitude,
        beta2=0.0,
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=RATE_LENGTH,
        coherent=coherent,
    )[0]


@pytest.mark.parametrize(
    ("name", "jones", "coherent", "rate"),
    [
        ("linear", (1.0, 0.0), True, 1.0),
        ("diagonal", (1 / math.sqrt(2), 1 / math.sqrt(2)), True, 1.0),
        ("circular", (1 / math.sqrt(2), 1j / math.sqrt(2)), True, 2.0 / 3.0),
        ("circular, phase only", (1 / math.sqrt(2), 1j / math.sqrt(2)), False, 5.0 / 6.0),
        ("diagonal, phase only", (1 / math.sqrt(2), 1 / math.sqrt(2)), False, 5.0 / 6.0),
    ],
)
def test_one_tone_turns_at_the_kerr_rate_its_polarization_has(
    name: str, jones: tuple[complex, complex], coherent: bool, rate: float
) -> None:
    """``gamma P L`` for linear light, two thirds of it for circular, in the isotropic tensor.

    The isotropic form of ``(2/3)(E . E*) E + (1/3)(E . E) E*`` has ``(2/3 + |u^T u|^2 / 3)``
    for a state ``u``: one for linear light, ``2/3`` for circular. Phase-only, the
    coherent term is gone and diagonal light turns at ``(x^2 + y^2 + 4/3 x y)`` --
    five sixths, with ``x = y = 1/2`` -- and circular the same, since the phase-only
    form does not see the relative phase. The tone stays in its state, and power is
    untouched.
    """
    power = 0.02
    start = np.array([jones]) * math.sqrt(power)
    end = turned(start, power, coherent=coherent)
    assert end.shape == start.shape
    assert np.angle(end[0, 0] / start[0, 0]) == pytest.approx(
        rate * GAMMA * 1e-3 * power * RATE_LENGTH, abs=1e-6
    )
    assert abs(end[0, 1] / end[0, 0]) == pytest.approx(abs(start[0, 1] / start[0, 0]), abs=1e-9)


def test_a_single_polarization_is_the_scalar_problem_exactly() -> None:
    """Jones vectors with nothing on y are the scalar solve: the extra axis changes no number."""
    offsets, start = tones(20e-3)
    beta2 = beta2_of(2.0)
    scalar, _ = fwm_tone_solve(
        offsets, start, beta2=beta2, gamma=GAMMA * 1e-3, alpha=0.0, distance=10e3
    )
    vector, _ = fwm_tone_solve(
        offsets,
        np.stack([start, np.zeros_like(start)], axis=1),
        beta2=beta2,
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=10e3,
    )
    assert np.allclose(vector[:, 0], scalar, rtol=0, atol=1e-12)
    assert np.allclose(vector[:, 1], 0.0, atol=1e-15)


def test_an_orthogonal_tone_cross_phase_modulates_at_two_thirds() -> None:
    """A tone on x beside one on y, far apart: it turns by ``gamma (P_x + (2/3) P_y) L``.

    Isotropic silica: a neighbour polarized the other way still turns a tone, at two
    thirds of what a parallel one does -- the ``(2/3)[tr J + J + J*]`` matrix's
    entry. The tones are a terahertz apart, so nothing mixes, and there is nothing
    else in the tone set for it to.
    """
    p_x, p_y = 0.02, 0.05
    offsets = np.array([0.0, 1e12])
    start = np.array([[math.sqrt(p_x), 0.0], [0.0, math.sqrt(p_y)]], dtype=np.complex128)
    end, _ = fwm_tone_solve(
        offsets, start, beta2=beta2_of(2.0), gamma=GAMMA * 1e-3, alpha=0.0, distance=RATE_LENGTH
    )
    wanted = GAMMA * 1e-3 * (p_x + (2.0 / 3.0) * p_y) * RATE_LENGTH
    assert np.angle(end[0, 0] / start[0, 0]) == pytest.approx(wanted, abs=2e-4)


@pytest.mark.parametrize("coherent", [True, False])
def test_power_over_both_polarizations_is_conserved(coherent: bool) -> None:
    offsets, _ = tones(30e-3)
    start = np.zeros((offsets.size, 2), dtype=np.complex128)
    start[4] = [math.cos(0.3), 1j * math.sin(0.3)]
    start[5] = [math.cos(1.1), math.sin(1.1)]
    start *= math.sqrt(30e-3)
    end, _ = fwm_tone_solve(
        offsets,
        start,
        beta2=beta2_of(2.0),
        gamma=GAMMA * 1e-3,
        alpha=0.0,
        distance=20e3,
        coherent=coherent,
    )
    assert np.sum(np.abs(end) ** 2) == pytest.approx(np.sum(np.abs(start) ** 2), rel=1e-9)


@pytest.mark.parametrize(
    "jones",
    [(math.cos(0.3), 1j * math.sin(0.3)), (1 / math.sqrt(2), 1 / math.sqrt(2)), (1, 0)],
)
@pytest.mark.parametrize("coherent", [True, False])
def test_the_vector_tone_solver_agrees_with_the_perturbative_path_at_low_power(
    jones: tuple[complex, complex], coherent: bool
) -> None:
    """1 mW over 10 km: the first product's Jones vector against ``_mix``'s vector drive.

    The perturbative path has its own tensor, its own pump phases and its own
    mixing integral; at 1 mW the terms it leaves out are half a percent, and the two
    agree to that: total power within 1 %, each axis within 2 %.
    """

    def product(**settings: object) -> tuple[complex, complex]:
        out = Fiber(
            length=10.0,
            attenuation=0.0,
            dispersion=2.0,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            cross_polarization=True,
            coherent_polarization=coherent,
            label="f",
            **settings,  # type: ignore[arg-type]
        ).run(CTX, {"in": jones_pumps(1e-3, jones)})["out"]
        assert isinstance(out, OpticalSignal)
        band = next(b for b in out.bands if abs(b.f0 - (ANCHOR - SPACING)) < 1e3)
        return complex(np.mean(band.Ex)), complex(np.mean(band.Ey))

    series = product(pump_phase=True)
    solved = product(tone_solver=True)
    power_series = abs(series[0]) ** 2 + abs(series[1]) ** 2
    power_solved = abs(solved[0]) ** 2 + abs(solved[1]) ** 2
    assert power_solved / power_series == pytest.approx(1.0, abs=0.01)
    for axis in (0, 1):
        if abs(series[axis]) > 1e-9:
            assert abs(solved[axis]) / abs(series[axis]) == pytest.approx(1.0, abs=0.02)


def test_the_tone_solver_takes_cross_polarization_now() -> None:
    Fiber(tone_solver=True, cross_polarization=True, coherent_polarization=True).validate()


# -- stimulated Raman scattering ---------------------------------------------------


def comb(frequencies: list[float], powers: list[float]) -> OpticalSignal:
    return OpticalSignal(
        bands=tuple(
            Band(
                Ex=np.full(256, math.sqrt(p), dtype=np.complex128),
                Ey=np.zeros(256, dtype=np.complex128),
                f0=f,
                fs=160e9,
            )
            for f, p in zip(frequencies, powers, strict=True)
        )
    )


SLOPE = 0.028  # 1/W/km/THz, silica's


@pytest.mark.parametrize(
    "frequencies",
    [[193.6e12, 194.6e12, 195.6e12], [190e12, 200e12, 210e12]],
    ids=["inside the gain peak's line", "across the peak"],
)
def test_raman_in_the_tone_solver_is_the_kernels_redistribution(frequencies: list[float]) -> None:
    """Three 5 mW tones over 80 km: each keeps what ``raman_tilt`` / ``raman_transfer`` leave it.

    The closed form inside the straight line of the gain (13.2 THz), the integrated
    silica shape with photons conserved past it -- the rule the perturbative path
    uses -- against kernels that share no code with the amplitude equations. Mixing
    is off (``mixing_floor`` zero keeps no products), so it is scattering alone, to
    the kernels' own tolerance; the loss is divided back out, as they do.
    """
    powers = [5e-3, 5e-3, 5e-3]
    out = Fiber(
        length=80.0,
        attenuation=0.2,
        dispersion=17.0,
        nonlinearity=GAMMA,
        mixing_floor=0.0,
        raman_gain_slope=SLOPE,
        tone_solver=True,
        label="span",
    ).run(CTX, {"in": comb(frequencies, powers)})["out"]
    assert isinstance(out, OpticalSignal)
    assert len(out.bands) == 3
    leff = effective_length(attenuation_db_per_m_to_alpha(0.2e-3), 80e3)
    slope = SLOPE * 1e-3 / 1e12
    if max(frequencies) - min(frequencies) <= 13.2e12:
        expected = raman_tilt(frequencies, powers, gain_slope=slope, effective_length=leff)
    else:
        expected = raman_transfer(frequencies, powers, gain_slope=slope, effective_length=leff)
    kept = [
        float(np.mean(np.abs(b.Ex) ** 2)) / p * 10 ** (0.2 * 80 / 10)
        for b, p in zip(out.bands, powers, strict=True)
    ]
    assert kept == pytest.approx(expected, rel=2e-4)
    assert kept[0] > 1.0 > kept[2], "the short-wavelength end loses to the long"


def test_without_photons_the_power_is_conserved_and_with_them_it_pays_the_quantum_defect() -> None:
    """The coupling matrix's own property: watts move without loss, photons cost ``f_n / f_m``."""
    frequencies = [190e12, 200e12, 210e12]
    offsets = np.array(frequencies) - frequencies[0]
    start = np.sqrt(np.array([5e-3, 5e-3, 5e-3]))
    for photons in (False, True):
        coupling = raman_coupling(
            frequencies,
            gain_slope=SLOPE * 1e-3 / 1e12,
            profile="triangle",
            photon_conserving=photons,
        )
        end, _ = fwm_tone_solve(
            offsets,
            start.astype(np.complex128),
            beta2=beta2_of(17.0),
            gamma=0.0,
            alpha=0.0,
            distance=80e3,
            raman=coupling,
        )
        total = float(np.sum(np.abs(end) ** 2)) / float(np.sum(start**2))
        if photons:
            assert total < 1.0 - 1e-3
        else:
            assert total == pytest.approx(1.0, abs=1e-9)


def test_raman_moves_no_phase() -> None:
    """Scattering is incoherent: it moves power and leaves each tone's phase to the Kerr terms."""
    offsets, start = tones(20e-3)
    beta2 = beta2_of(2.0)
    coupling = raman_coupling(
        list(ANCHOR + offsets),
        gain_slope=SLOPE * 1e-3 / 1e12,
        profile="triangle",
        photon_conserving=False,
    )
    plain, _ = fwm_tone_solve(
        offsets, start, beta2=beta2, gamma=GAMMA * 1e-3, alpha=0.0, distance=10e3
    )
    scattered, _ = fwm_tone_solve(
        offsets, start, beta2=beta2, gamma=GAMMA * 1e-3, alpha=0.0, distance=10e3, raman=coupling
    )
    # A 100 GHz comb sits well inside the gain's straight line, so what scattering
    # moves is a fraction of a percent of the power, and no phase at all.
    assert np.angle(scattered[4] / plain[4]) == pytest.approx(0.0, abs=5e-3)


def test_raman_is_no_longer_refused_by_the_tone_solver() -> None:
    Fiber(tone_solver=True, raman_gain_slope=SLOPE).validate()


# -- PMD ------------------------------------------------------------------------------


def pmd_chain(
    label: str, coefficient: float, length_km: float, sections: int
) -> tuple[PMDSection, ...]:
    """The waveplate chain a span of this label draws: the same stream the split-step reads."""
    mean = Fiber(length=length_km, pmd_coefficient=coefficient).mean_dgd()
    return random_pmd_sections(mean, sections, CTX.rng("Fiber", label, "pmd"))


def pmd_span(nonlinearity: float = 1e-9, **settings: object) -> Fiber:
    return Fiber(
        label="pmd-span",
        length=20.0,
        attenuation=0.0,
        dispersion=2.0,
        nonlinearity=nonlinearity,  # the default: nothing to mix, what is left is the chain
        four_wave_mixing=True,
        mixing_floor=0.0,
        tone_solver=True,
        cross_polarization=True,
        pmd_coefficient=0.5,
        pmd_sections=12.0,
        **settings,  # type: ignore[arg-type]
    )


JONES = (math.cos(0.4), 1j * math.sin(0.4))


def test_a_tone_meets_pmd_at_its_own_frequency() -> None:
    """With nothing to mix, the first tone leaves the chain as ``apply_pmd`` leaves it at rest.

    The chain is measured from the first band's carrier, so that tone sits where a
    section's delay is a phase of one and only its Jones rotation is left;
    ``apply_pmd`` does that through the frequency domain, sharing no arithmetic with
    the tone solver. The second, 100 GHz on, meets the chain 100 GHz on and is
    turned otherwise -- by tenths, from a 2 ps chain -- which the rest-frame rotation
    every tone used to get could not say. The chain's Jones matrix at that offset,
    from :func:`pmd_jones_matrix`, is where it lands.
    """
    from maiman.kernels import pmd_jones_matrix

    signal = jones_pumps(1e-6, JONES)
    out = pmd_span().run(CTX, {"in": signal})["out"]
    assert isinstance(out, OpticalSignal)
    chain = pmd_chain("pmd-span", 0.5, 20.0, 12)
    gaps = []
    for band, launched in zip(out.bands, signal.bands, strict=False):
        omega = 2.0 * math.pi * (launched.f0 - signal.bands[0].f0)
        jones = np.array([np.mean(launched.Ex), np.mean(launched.Ey)])
        wanted = pmd_jones_matrix(chain, omega) @ jones
        got = np.array([np.mean(band.Ex), np.mean(band.Ey)])
        assert np.linalg.norm(got - wanted) < 1e-9 * np.linalg.norm(wanted)
        at_rest = apply_pmd(launched.Ex, launched.Ey, launched.fs, chain)
        rest = np.array([np.mean(at_rest[0]), np.mean(at_rest[1])])
        gaps.append(np.linalg.norm(got - rest) / np.linalg.norm(rest))
    assert gaps[0] < 1e-9
    assert gaps[1] > 0.05


def test_the_diagnostics_report_the_dgd_of_the_chain_that_was_drawn() -> None:
    from maiman.kernels import differential_group_delay

    diagnostics = pmd_span().run(CTX, {"in": jones_pumps(1e-6, JONES)})["diagnostics"]
    assert diagnostics.differential_group_delay == pytest.approx(
        differential_group_delay(pmd_chain("pmd-span", 0.5, 20.0, 12)), rel=1e-12
    )
    assert diagnostics.differential_group_delay > 0.0


def test_interleaved_sections_are_the_same_chain_when_nothing_is_nonlinear() -> None:
    """Cut into twelve solves with a rotation between each, the tone is where it was."""
    signal = jones_pumps(1e-6, JONES)
    end = pmd_span().run(CTX, {"in": signal})["out"]
    between = pmd_span(interleave_pmd=True).run(CTX, {"in": signal})["out"]
    assert isinstance(end, OpticalSignal) and isinstance(between, OpticalSignal)
    for a, b in zip(end.bands, between.bands, strict=False):
        assert np.mean(a.Ex) == pytest.approx(np.mean(b.Ex), rel=1e-6, abs=1e-12)
        assert np.mean(a.Ey) == pytest.approx(np.mean(b.Ey), rel=1e-6, abs=1e-12)


def test_where_the_chain_sits_matters_once_light_is_strong() -> None:
    """At 30 mW the Kerr effect is not covariant under a general rotation, so order shows.

    The isotropic tensor's coherent term turns with the polarization it acts on: a
    chain of rotations before the Kerr steps and one after do not commute with it. The
    two differ visibly here and agree at a microwatt (the test above).
    """
    signal = jones_pumps(30e-3, JONES)
    end = pmd_span(GAMMA).run(CTX, {"in": signal})["out"]
    between = pmd_span(GAMMA, interleave_pmd=True).run(CTX, {"in": signal})["out"]
    assert isinstance(end, OpticalSignal) and isinstance(between, OpticalSignal)
    gap = abs(np.mean(end.bands[0].Ex) - np.mean(between.bands[0].Ex))
    assert gap > 1e-3 * abs(np.mean(end.bands[0].Ex))


def test_pmd_conserves_power_through_the_chain_and_the_kerr_effect() -> None:
    signal = jones_pumps(30e-3, JONES)
    for interleave in (False, True):
        out = pmd_span(GAMMA, interleave_pmd=interleave).run(CTX, {"in": signal})["out"]
        assert isinstance(out, OpticalSignal)
        total = sum(float(np.mean(np.abs(b.Ex) ** 2 + np.abs(b.Ey) ** 2)) for b in out.bands)
        assert total == pytest.approx(60e-3, rel=1e-8)


def test_pmd_needs_the_two_polarizations_together() -> None:
    with pytest.raises(ValueError, match="cross_polarization"):
        Fiber(tone_solver=True, pmd_coefficient=0.1).validate()
    Fiber(tone_solver=True, pmd_coefficient=0.1, cross_polarization=True).validate()
