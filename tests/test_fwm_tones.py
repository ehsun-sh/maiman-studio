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
    attenuation_db_per_m_to_alpha,
    fwm_cascade_amplitude,
    fwm_phase_mismatch,
    fwm_product_power,
    fwm_tone_solve,
    propagate_ssfm,
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
        {"cross_polarization": True},
        {"mixing_steps": 4.0},
        {"pmd_coefficient": 0.1},
        {"raman_gain_slope": 0.028},
    ],
)
def test_it_is_refused_beside_what_it_replaces(other: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="tone_solver"):
        Fiber(tone_solver=True, **other).validate()
    Fiber(tone_solver=True).validate()
