"""Bands close enough to beat do, against closed forms.

A square-law detector sees ``|E_a + E_b|^2``, which is two powers and a cross
term ``2 Re(E_a E_b* exp(2j pi df t))`` per polarization. On a channel grid the
cross term sits hundreds of gigahertz out and is rejected; two carriers inside
half the sample rate put it in the photocurrent. Each relation below is the
textbook one for two optical fields on one diode (Agrawal, *Fiber-Optic
Communication Systems*, 4th ed., sec. 10.1 for the heterodyne tone), and the
delayed self-heterodyne's coherence ``exp(-pi linewidth |delay|)`` is the one
Okoshi, Kikuchi and Nakayama measured a linewidth by (Electron. Lett. 16(16),
1980).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components import CoherentReceiver, CWLaser, PINPhotodiode
from maiman.signals import Band, OpticalSignal, WalkoffHistory

#: 1550 nm, give or take: only the offsets between carriers matter here.
F0 = 193.1e12


def cw(ctx: SimulationContext, power: float, f0: float, *, axis: str = "x") -> Band:
    field = np.full(ctx.num_samples, math.sqrt(power), dtype=np.complex128)
    empty = np.zeros(ctx.num_samples, dtype=np.complex128)
    ex, ey = (field, empty) if axis == "x" else (empty, field)
    return Band(Ex=ex, Ey=ey, f0=f0, fs=ctx.sample_rate)


def quiet_pin() -> PINPhotodiode:
    return PINPhotodiode(responsivity=0.8, shot_noise=False, thermal_noise=False, label="pin")


def detect(ctx: SimulationContext, signal: OpticalSignal) -> np.ndarray:
    return np.asarray(quiet_pin().run(ctx, {"in": signal})["out"].samples, dtype=np.float64)


def tone(ctx: SimulationContext, samples: np.ndarray, offset: float) -> float:
    """Amplitude of the cosine at ``offset`` [Hz] in a real waveform."""
    return 2.0 * abs(np.mean(samples * np.exp(-2j * np.pi * offset * ctx.time_axis())))


@pytest.fixture
def ctx() -> SimulationContext:
    # 6.4 ns: a 10 GHz tone is exactly 64 cycles of the window.
    return SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=1)


def test_two_carriers_beat_at_their_offset_with_the_geometric_mean(
    ctx: SimulationContext,
) -> None:
    """``i = R (P_a + P_b) + 2 R sqrt(P_a P_b) cos(2 pi df t)``."""
    p_a, p_b, offset = 1e-3, 0.25e-3, 10e9
    current = detect(ctx, OpticalSignal(bands=(cw(ctx, p_a, F0), cw(ctx, p_b, F0 + offset))))
    assert float(np.mean(current)) == pytest.approx(0.8 * (p_a + p_b), rel=1e-6)
    assert tone(ctx, current, offset) == pytest.approx(2 * 0.8 * math.sqrt(p_a * p_b), rel=1e-6)


def test_orthogonal_carriers_do_not_beat(ctx: SimulationContext) -> None:
    """The cross term is per polarization, and X against Y has none."""
    p_a, p_b, offset = 1e-3, 0.25e-3, 10e9
    signal = OpticalSignal(bands=(cw(ctx, p_a, F0), cw(ctx, p_b, F0 + offset, axis="y")))
    current = detect(ctx, signal)
    assert tone(ctx, current, offset) < 1e-9 * 0.8 * p_a
    assert float(np.mean(current)) == pytest.approx(0.8 * (p_a + p_b), rel=1e-6)


def test_carriers_past_half_the_sample_rate_still_add_as_powers(ctx: SimulationContext) -> None:
    """A 100 GHz neighbour on a 160 GHz grid: the two detected apart, summed."""
    offset = 100e9
    assert offset > ctx.sample_rate / 2.0
    a, b = cw(ctx, 1e-3, F0), cw(ctx, 0.25e-3, F0 + offset)
    together = detect(ctx, OpticalSignal(bands=(a, b)))
    apart = detect(ctx, OpticalSignal(bands=(a,))) + detect(ctx, OpticalSignal(bands=(b,)))
    np.testing.assert_allclose(together, apart, rtol=1e-6)
    assert np.ptp(together) < 1e-6 * float(np.mean(together)), "and nothing rides on it"


@pytest.mark.parametrize(
    ("delay", "coherence", "tolerance"),
    [
        # One field against itself: exact.
        (0.0, 1.0, 1e-6),
        # Measured 0.505; four seeds spread by 0.005 about it.
        (math.log(2.0) / (math.pi * 50e6), 0.5, 0.02),
        # Nothing shared, so what is left is the estimator's own floor:
        # sqrt(coherence time / window) = sqrt(6.4 ns / 6.55 us) = 0.03, and
        # twice that is the bound.
        (10 / (math.pi * 50e6), 0.0, 0.06),
    ],
)
def test_a_laser_beaten_against_its_own_delayed_copy_loses_coherence_as_its_linewidth_says(
    delay: float, coherence: float, tolerance: float
) -> None:
    """Delayed self-heterodyne: the tone keeps ``exp(-pi linewidth |delay|)`` of its amplitude.

    One 50 MHz laser, a copy of its field 5 GHz up and late by ``delay`` through
    the walk-off the detector applies. Undelayed the two are one field and the
    tone is whole; at ``ln2 / (pi linewidth)`` half of it survives; ten
    coherence times out, none. Averaged over seeds, because a window of a
    thousand coherence times estimates it to a few percent and no better.
    """
    linewidth, offset = 50e6, 5e9
    ratios = []
    for seed in (1, 2, 3, 4):
        # 6.55 us at 40 GHz: 5 GHz is a whole number of cycles of it.
        ctx = SimulationContext(
            bit_rate=10e9, samples_per_symbol=4, sequence_length=65536, seed=seed
        )
        laser = CWLaser(power=0.0, linewidth=linewidth / 1e3, label="laser").run(ctx, {})["out"]
        source = laser.bands[0]
        power = source.average_power()
        copy = Band(Ex=source.Ex, Ey=source.Ey, f0=source.f0 + offset, fs=source.fs)
        signal = OpticalSignal(
            bands=(source, copy), walkoff=WalkoffHistory(carriers=((copy.f0, delay),))
        )
        current = detect(ctx, signal)
        # Each field carries half the tone's amplitude: 2 R P at full coherence.
        ratios.append(tone(ctx, current, offset) / (2 * 0.8 * power))
    assert float(np.mean(ratios)) == pytest.approx(coherence, abs=tolerance)


def test_a_coherent_receiver_hears_a_second_carrier_inside_its_band(
    ctx: SimulationContext,
) -> None:
    """``i + jq = R sum_b E_b E_lo* exp(2j pi (f_b - f_lo) t)`` over every band within reach.

    Two carriers 2.5 and 6.25 GHz above the LO. The nearest was always detected; the
    other used to be dropped, and it is inside the receiver's band.
    """
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64, seed=1)
    p_lo, p_a, p_b = 1e-3, 1e-4, 4e-5
    receiver = CoherentReceiver(responsivity=0.8, shot_noise=False, thermal_noise=False)
    out = receiver.run(
        ctx,
        {
            "in": OpticalSignal(bands=(cw(ctx, p_a, F0 + 2.5e9), cw(ctx, p_b, F0 + 6.25e9))),
            "lo": OpticalSignal(bands=(cw(ctx, p_lo, F0),)),
        },
    )
    mix = np.asarray(out["i"].samples, dtype=np.float64) + 1j * np.asarray(
        out["q"].samples, dtype=np.float64
    )
    for offset, power in ((2.5e9, p_a), (6.25e9, p_b)):
        amplitude = abs(np.mean(mix * np.exp(-2j * np.pi * offset * ctx.time_axis())))
        assert amplitude == pytest.approx(0.8 * math.sqrt(power * p_lo), rel=1e-6)
