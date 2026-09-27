"""Mixing products that mix again: what the multi-band model does and does not carry.

Pinned, not tuned (maiman-z8j). Two equal pumps 100 GHz apart make first-order
products one spacing out, and those, mixing with the pumps, make second-order
products two spacings out. The one-band split-step carries all of it; the
multi-band model computes each span's products from the bands present at its
start and adds them at its end.

**Within a span**, ``cascaded_fwm`` now lets a first-order product drive a
further one before the span ends -- :func:`~maiman.kernels.fwm_cascade_integral`
integrates both triads' build-up in closed form, in both the role a first-order
product can play (un-conjugated beside a launched pump, or conjugated beside
two). Off by default, so every span's result is exactly what it was.

**Across spans** the gap is still open: a product mixes again as a pump in the
next span, but with a phase drawn on its frequencies rather than the one it was
made with, so the spans do not add the way the split-step's do. What that costs
is written down here too, as numbers a change has to explain.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.context import SimulationContext
from maiman.kernels import fwm_cascade_integral, fwm_phase_mismatch, propagate_ssfm
from maiman.signals import Band, OpticalSignal

ANCHOR = 193.1e12
SPACING = 100e9
GAMMA = 1.3  # 1/W/km
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


def products(
    pump: float, dispersion: float, span: float, spans: int, cascaded: bool = False
) -> tuple[dict[int, float], dict[int, float]]:
    """``(split-step, model)`` power at ``m`` spacings below the lower pump [W], for m = 1, 2."""
    probe = Band(
        Ex=np.ones(256, dtype=np.complex128),
        Ey=np.zeros(256, dtype=np.complex128),
        f0=ANCHOR,
        fs=160e9,
    )
    beta2 = Fiber(dispersion=dispersion).reference_beta2(OpticalSignal(bands=(probe,)))
    n, fs = 4096, 3.2e12
    t = np.arange(n) / fs
    field = np.sqrt(pump) * (1.0 + np.exp(2j * np.pi * SPACING * t))
    mismatch = abs(fwm_phase_mismatch(beta2, 0.0, 0.0, SPACING))
    for _ in range(spans):
        field, _ = propagate_ssfm(
            field,
            fs,
            beta2=beta2,
            gamma=GAMMA * 1e-3,
            alpha=0.0,
            distance=span,
            max_nonlinear_phase=1e-3,
            max_step=0.05 / mismatch,
        )
    spectrum = np.fft.fft(field) / n
    reference = {m: float(abs(spectrum[-round(m * SPACING / (fs / n))]) ** 2) for m in (1, 2)}

    signal = OpticalSignal(
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
    for index in range(spans):
        signal = Fiber(
            length=span / 1e3,
            attenuation=0.0,
            dispersion=dispersion,
            nonlinearity=GAMMA,
            mixing_floor=250.0,
            pump_phase=True,
            cascaded_fwm=cascaded,
            label=f"span{index}",
        ).run(CTX, {"in": signal})["out"]
    model = {}
    for m in (1, 2):
        found = [b for b in signal.bands if abs(b.f0 - (ANCHOR - m * SPACING)) < 1e3]
        model[m] = float(np.mean(np.abs(found[0].Ex) ** 2)) if found else 0.0
    return reference, model


def test_within_a_span_a_product_does_not_mix_again_by_default() -> None:
    """10 mW, 20 km, ``cascaded_fwm`` off: the first order within 2.5 %, the second absent.

    The split-step's second-order product is 2.2e-10 W, 39 dB under the first:
    what a product makes with the pumps during the span it was made in. Off by
    default, the multi-band model still does not compute it -- this is the
    behaviour every earlier result was taken with, and it must not move.
    """
    reference, model = products(10e-3, 2.0, 20e3, 1, cascaded=False)
    assert model[1] / reference[1] == pytest.approx(0.976, abs=0.005)
    assert reference[2] == pytest.approx(2.16e-10, rel=0.02)
    assert model[2] == 0.0


def test_within_a_span_cascaded_fwm_recovers_the_second_order_product() -> None:
    """The same case with ``cascaded_fwm`` on: the second order appears, within 6 %.

    Before: 0.0 W, absent. After: 2.05e-10 W against the split-step's 2.16e-10 W
    -- 5 % low, where the truncation at second order in gamma is expected to
    leave something: the next term is third order, of the same relative size as
    the nonlinear phase this span puts on a pump, about a quarter of a radian at
    30 mW and smaller here. The first-order product is untouched, to the last
    digit -- the new term is barred from landing back on it (see
    :meth:`~maiman.components.fiber.Fiber._mix_cascade`), so the two orders are
    additive, not entangled.
    """
    reference, model = products(10e-3, 2.0, 20e3, 1, cascaded=True)
    assert model[1] / reference[1] == pytest.approx(0.976, abs=0.005)
    assert model[2] / reference[2] == pytest.approx(0.948, abs=0.01)


def test_within_a_span_cascaded_fwm_converges_as_power_falls() -> None:
    """Deep in the perturbative regime the truncated second order should track the reference.

    At 10 mW the two triads' own nonlinear phase is not negligible and the
    third-order term this stops at leaves 5 %. Run the same case at a tenth the
    power -- an order of magnitude further from where the pumps' own phase
    matters -- and the shortfall drops by more than half, the signature of a
    correction that is missing a smaller, not a wrong, term.
    """
    reference, model = products(1e-3, 2.0, 20e3, 1, cascaded=True)
    assert model[2] / reference[2] == pytest.approx(0.994, abs=0.01)


def test_across_spans_it_mixes_again_with_a_phase_it_was_not_made_with() -> None:
    """The same 20 km as four 5 km spans, ``cascaded_fwm`` off: the second order 4.8x too strong.

    After each span the products are bands and mix as pumps in the next. But a
    triplet's phase is drawn on its frequencies -- right for a data channel,
    whose phase nobody knows -- and a product's phase is known: it was made from
    the pumps beside it. Drawn instead, the cascade adds where it should cancel.
    This is the part (b) of maiman-z8j that is not fixed here; see the module
    docstring and the next test for what turning ``cascaded_fwm`` on does and
    does not do about it.
    """
    reference, model = products(10e-3, 2.0, 5e3, 4, cascaded=False)
    assert model[1] / reference[1] == pytest.approx(0.970, abs=0.005)
    assert model[2] / reference[2] == pytest.approx(4.79, rel=0.02)


def test_across_spans_cascaded_fwm_does_not_fix_the_drawn_phase() -> None:
    """The same four spans with ``cascaded_fwm`` on too: worse, not better -- 9.7x, not 4.8x.

    ``cascaded_fwm`` is a within-span correction: it lets each span generate its
    own second-order product honestly, but every span still hands the *next*
    span a first-order product with a drawn phase rather than the one physics
    gave it, and the next span's ordinary triad loop treats that product as an
    ordinary pump. So each span now contributes a second-order term twice --
    once honestly, within itself, and once again through the drawn-phase route
    the next span takes it through -- and the two do not cancel the way the
    split-step's would. Fixing this is part (b) of maiman-z8j: carrying a
    product's own phase across spans instead of drawing one, which is a larger
    change than this pass makes (see the report for what it would take).
    """
    reference, model = products(10e-3, 2.0, 5e3, 4, cascaded=True)
    assert model[1] / reference[1] == pytest.approx(0.969, abs=0.005)
    assert model[2] / reference[2] == pytest.approx(9.69, rel=0.02)


def test_at_high_power_the_first_order_falls_short_too() -> None:
    """30 mW: 3.7 % of a pump converted, and the undepleted, uncascaded product 25 % low."""
    reference, model = products(30e-3, 1.0, 20e3, 1)
    assert reference[1] / 30e-3 == pytest.approx(0.037, abs=0.002)
    assert model[1] / reference[1] == pytest.approx(0.745, abs=0.01)


# -- fwm_cascade_integral: closed-form and numerical cross-checks -----------


def test_cascade_integral_is_exact_at_phase_matching() -> None:
    """Both triads phase matched, lossless: the double integral is ``integral_0^L z dz = L^2/2``.

    ``M1(z) = z`` there -- the first product's build-up is unbounded linear
    growth, not oscillation -- and the outer integral of ``z`` against a second
    phase-matched triad is textbook.
    """
    length = 15_000.0
    value = fwm_cascade_integral(0.0, 0.0, 0.0, length)
    assert value == pytest.approx(complex(length**2 / 2.0), rel=1e-12)


def test_cascade_integral_matches_direct_numerical_quadrature() -> None:
    """A handful of mismatched, lossy cases against the double integral done by brute force.

    Composite Simpson's rule on a fine grid, nested: the inner integral built up
    at every outer node, then the outer integral over those. Not how the fibre
    block computes it -- that is the closed form this checks -- but independent
    of it, which is the point.
    """
    rng = np.random.default_rng(0)
    length = 20_000.0
    for _ in range(6):
        delta_beta1 = float(rng.uniform(-2e-3, 2e-3))
        delta_beta2 = float(rng.uniform(-2e-3, 2e-3))
        alpha = float(rng.uniform(0.0, 3e-5))
        z = np.linspace(0.0, length, 4001)
        r1 = complex(-alpha, delta_beta1)
        inner = np.where(
            np.abs(z) < 1e-9,
            0.0,
            (np.exp(r1 * z) - 1.0) / r1 if abs(r1) > 0 else z,
        )
        outer_integrand = inner * np.exp(complex(-alpha, delta_beta2) * z)
        numerical = np.trapezoid(outer_integrand, z)
        closed_form = fwm_cascade_integral(delta_beta1, delta_beta2, alpha, length)
        assert closed_form == pytest.approx(complex(numerical), rel=1e-4, abs=1e-6)
