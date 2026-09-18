"""A laser's own noise: its linewidth, its intensity noise, and the partition between its modes.

None of the famous results is written into the integration. The Langevin forces
are the textbook set and nothing more; what the tests check is that the numbers
the textbooks derive from them come *out*:

* **Schawlow and Townes.** With ``alpha = 0`` the phase diffuses at
  ``R_sp / (4 pi P)``: measured 3.13 MHz against 3.18.
* **Henry.** With ``alpha = 4`` the carriers' answer to each kick adds a phase
  kick of its own, and the line is ``1 + alpha^2 = 17`` times wider: measured
  55.7 MHz against 54.1. Nowhere does the code multiply by seventeen.
* **Mode partition.** A Fabry-Perot laser's modes fight over one reservoir, so
  their total is quieter than any one of them; delay each by its own amount, as
  dispersion does, and the cancellation undoes itself.
"""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from maiman import SimulationContext
from maiman.components.dml import DirectlyModulatedLaser
from maiman.laser import (
    LaserEnsemble,
    LaserParameters,
    MultimodeEnsemble,
    dispersed_power,
    integrate_multimode,
    integrate_rate_equations,
    laser_ensemble,
    multimode_steady_state,
)
from maiman.signals import ElectricalSignal

BASE = LaserParameters()
CURRENT = 2.0 * BASE.threshold_current()


def ensemble(parameters: LaserParameters, current: float = CURRENT, seed: int = 1) -> LaserEnsemble:
    return laser_ensemble(
        parameters, current, 100e9, 1500, realizations=400, rng=np.random.default_rng(seed)
    )


# ---------------------------------------------------------------------------
# The linewidth
# ---------------------------------------------------------------------------


def test_without_alpha_the_line_is_schawlow_and_townes() -> None:
    quiet = dataclasses.replace(BASE, linewidth_enhancement=0.0, gain_compression=0.0)
    expected = quiet.schawlow_townes_linewidth(CURRENT)
    assert expected == pytest.approx(3.18e6, rel=0.01)
    measured = ensemble(quiet).measured_linewidth(500)
    assert measured == pytest.approx(expected, rel=0.05)


def test_alpha_widens_it_by_one_plus_alpha_squared_and_nothing_says_so() -> None:
    """Henry's factor of seventeen, measured rather than multiplied in."""
    laser = dataclasses.replace(BASE, gain_compression=0.0)
    assert laser.linewidth(CURRENT) == pytest.approx(
        17.0 * laser.schawlow_townes_linewidth(CURRENT), rel=1e-12
    )
    measured = ensemble(laser).measured_linewidth(1000)
    assert measured == pytest.approx(laser.linewidth(CURRENT), rel=0.05)
    assert measured > 12.0 * laser.schawlow_townes_linewidth(CURRENT)


def test_gain_compression_moves_the_line_and_does_not_widen_it() -> None:
    """The steady frequency offset compression holds is taken out before the width is read."""
    measured = ensemble(BASE).measured_linewidth(1000)
    assert measured == pytest.approx(BASE.linewidth(CURRENT), rel=0.05)


def test_a_brighter_laser_is_a_narrower_one() -> None:
    """``1 / P``: twice as far above threshold, very nearly half the width."""
    laser = dataclasses.replace(BASE, gain_compression=0.0)
    high = 3.0 * BASE.threshold_current()
    expected = laser.linewidth(CURRENT) / laser.linewidth(high)
    assert expected == pytest.approx(2.0, rel=0.02)
    low_width = ensemble(laser, CURRENT).measured_linewidth(1000)
    high_width = ensemble(laser, high, seed=2).measured_linewidth(1000)
    assert low_width / high_width == pytest.approx(expected, rel=0.08)


def test_the_intensity_noise_peaks_at_the_relaxation_oscillation() -> None:
    """The same forces, read in amplitude: RIN peaks where the laser rings."""
    frequencies, rin = ensemble(BASE).relative_intensity_noise()
    band = frequencies > 0.5e9
    peak = float(frequencies[band][np.argmax(rin[band])])
    assert peak == pytest.approx(BASE.relaxation_frequency(CURRENT), rel=0.1)


def test_without_noise_nothing_changed() -> None:
    drive = np.full(64, CURRENT)
    drive[32:] *= 1.2
    plain = integrate_rate_equations(BASE, drive, 50e9)
    asked = integrate_rate_equations(BASE, drive, 50e9, noise=False)
    assert np.array_equal(plain.phase, asked.phase)
    assert np.array_equal(plain.photons, asked.photons)
    with pytest.raises(ValueError, match="rng"):
        integrate_rate_equations(BASE, drive, 50e9, noise=True)


def test_the_block_carries_the_noise_and_repeats_it() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=32, seed=4)
    drive = ElectricalSignal(samples=np.zeros(ctx.num_samples), fs=ctx.sample_rate, unit="V")
    quiet = DirectlyModulatedLaser(bias_current=50.0, label="dml")
    noisy = DirectlyModulatedLaser(bias_current=50.0, noise=True, label="dml")
    first = noisy.run(ctx, {"in": drive})["out"].bands[0].Ex
    again = noisy.run(ctx, {"in": drive})["out"].bands[0].Ex
    still = quiet.run(ctx, {"in": drive})["out"].bands[0].Ex
    assert np.array_equal(first, again), "seeded from the run, so it repeats"
    assert not np.allclose(first, still)
    assert noisy.linewidth() == pytest.approx(
        noisy.parameters().linewidth(noisy.si("bias_current")), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Mode partition
# ---------------------------------------------------------------------------

MODES: dict[str, Any] = {"modes": 7, "spacing": 1e-9}


def test_one_mode_is_the_single_mode_laser() -> None:
    """The multimode steady state, reduced to one mode, is the single-mode one."""
    carriers, photons = multimode_steady_state(
        BASE, CURRENT, modes=1, spacing=1e-9, gain_bandwidth=20e-9
    )
    single = BASE.steady_state(CURRENT)
    assert carriers == pytest.approx(single[0], rel=1e-9)
    assert float(photons[0]) == pytest.approx(single[1], rel=1e-7)


def partitioned(gain_bandwidth: float) -> MultimodeEnsemble:
    return integrate_multimode(
        BASE,
        CURRENT,
        50e9,
        1000,
        gain_bandwidth=gain_bandwidth,
        realizations=100,
        rng=np.random.default_rng(1),
        settle=500,
        **MODES,
    )


def test_the_modes_sit_where_the_gain_curve_puts_them() -> None:
    _, steady = multimode_steady_state(BASE, CURRENT, gain_bandwidth=20e-9, **MODES)
    run = partitioned(20e-9)
    mean = run.mode_power.mean(axis=(0, 2))
    assert mean / mean.sum() == pytest.approx(steady / steady.sum(), abs=0.01)
    assert (steady / steady.sum())[3] == pytest.approx(0.893, abs=0.002)


@pytest.mark.parametrize(("gain_bandwidth", "ratio"), [(20e-9, 0.383), (40e-9, 0.046)])
def test_the_total_is_quieter_than_its_parts(gain_bandwidth: float, ratio: float) -> None:
    """What one mode gains the reservoir takes from the others: the parts cancel in the sum."""
    run = partitioned(gain_bandwidth)
    power = run.mode_power
    parts = sum(float(power[:, m].var()) for m in range(power.shape[1]))
    total = float(run.total_power.var())
    assert total / parts == pytest.approx(ratio, rel=0.1)


def test_dispersion_undoes_the_cancellation() -> None:
    """Undelayed it is the quiet total; delayed apart, the noisy sum of the parts."""
    run = partitioned(20e-9)
    power = run.mode_power
    total = run.total_power
    none = dispersed_power(power, run.offsets, 50e9, dispersion=0.0)
    assert np.allclose(none, total, rtol=1e-12, atol=0.0)

    parts = sum(float(power[:, m].var()) for m in range(power.shape[1]))
    noise = [
        float(dispersed_power(power, run.offsets, 50e9, dispersion=amount).var())
        for amount in (0.1, 0.3, 1.0, 3.0)
    ]
    assert all(a < b for a, b in pairwise(noise)), "more dispersion, more noise"
    assert noise[-1] / parts == pytest.approx(1.0, abs=0.06)
    assert noise[-1] > 2.0 * float(total.var())


def test_what_is_not_a_laser_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one mode"):
        multimode_steady_state(BASE, CURRENT, modes=0, spacing=1e-9, gain_bandwidth=20e-9)
    with pytest.raises(ValueError, match="positive"):
        multimode_steady_state(BASE, CURRENT, modes=3, spacing=0.0, gain_bandwidth=20e-9)
    with pytest.raises(ValueError, match="realization"):
        laser_ensemble(BASE, CURRENT, 1e10, 100, realizations=0, rng=np.random.default_rng())
    assert math.isfinite(BASE.linewidth(CURRENT))
