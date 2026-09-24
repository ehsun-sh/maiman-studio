"""Mixing products that mix again: what the multi-band model does and does not carry.

Pinned, not tuned (maiman-z8j). Two equal pumps 100 GHz apart make first-order
products one spacing out, and those, mixing with the pumps, make second-order
products two spacings out. The one-band split-step carries all of it; the
multi-band model computes each span's products from the bands present at its
start and adds them at its end. So within a span a product never mixes again,
and across spans it mixes as a pump whose phase is drawn afresh rather than the
one it was made with. What that costs is written down here, as numbers a change
has to explain.
"""

from __future__ import annotations

import numpy as np
import pytest

from maiman.components import Fiber
from maiman.context import SimulationContext
from maiman.kernels import fwm_phase_mismatch, propagate_ssfm
from maiman.signals import Band, OpticalSignal

ANCHOR = 193.1e12
SPACING = 100e9
GAMMA = 1.3  # 1/W/km
CTX = SimulationContext(
    bit_rate=10e9, samples_per_symbol=16, sequence_length=16, seed=3, precision="double"
)


def products(
    pump: float, dispersion: float, span: float, spans: int
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
            label=f"span{index}",
        ).run(CTX, {"in": signal})["out"]
    model = {}
    for m in (1, 2):
        found = [b for b in signal.bands if abs(b.f0 - (ANCHOR - m * SPACING)) < 1e3]
        model[m] = float(np.mean(np.abs(found[0].Ex) ** 2)) if found else 0.0
    return reference, model


def test_within_a_span_a_product_does_not_mix_again() -> None:
    """10 mW, 20 km: the first order within 2.5 %, the second not there at all.

    The split-step's second-order product is 2.2e-10 W, 39 dB under the first:
    what a product makes with the pumps during the span it was made in.
    """
    reference, model = products(10e-3, 2.0, 20e3, 1)
    assert model[1] / reference[1] == pytest.approx(0.976, abs=0.005)
    assert reference[2] == pytest.approx(2.16e-10, rel=0.02)
    assert model[2] == 0.0


def test_across_spans_it_mixes_again_with_a_phase_it_was_not_made_with() -> None:
    """The same 20 km as four 5 km spans: the second order appears, 4.8 times too strong.

    After each span the products are bands and mix as pumps in the next. But a
    triplet's phase is drawn on its frequencies -- right for a data channel,
    whose phase nobody knows -- and a product's phase is known: it was made from
    the pumps beside it. Drawn instead, the cascade adds where it should cancel.
    """
    reference, model = products(10e-3, 2.0, 5e3, 4)
    assert model[1] / reference[1] == pytest.approx(0.970, abs=0.005)
    assert model[2] / reference[2] == pytest.approx(4.79, rel=0.02)


def test_at_high_power_the_first_order_falls_short_too() -> None:
    """30 mW: 3.7 % of a pump converted, and the undepleted, uncascaded product 25 % low."""
    reference, model = products(30e-3, 1.0, 20e3, 1)
    assert reference[1] / 30e-3 == pytest.approx(0.037, abs=0.002)
    assert model[1] / reference[1] == pytest.approx(0.745, abs=0.01)
