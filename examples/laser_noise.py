"""A laser's own noise: why its line is wide, and why a Fabry-Perot laser does not reach far.

Spontaneous emission is random. Every photon it drops into the lasing mode jolts
the field's phase and its intensity, and a semiconductor laser answers each jolt
by moving its carrier density -- which moves its index, which jolts the phase
again. Three consequences, each measured here from the rate equations with their
Langevin forces and nothing written in by hand:

1. **The linewidth.** Schawlow and Townes' ``R_sp / 4 pi P`` from the phase
   noise alone, and Henry's ``1 + alpha^2`` on top from the carriers' answer --
   seventeen times wider for ``alpha = 4``, which is why a DFB laser's line is
   tens of megahertz and not the few that phase diffusion alone would draw.
2. **The intensity noise**, which peaks where the laser rings.
3. **Mode partition.** A Fabry-Perot laser has several longitudinal modes fed by
   one carrier reservoir. What one gains the others lose, so the total is quiet
   and each mode alone is not -- and fibre dispersion delays each mode by its own
   amount, so the cancellation that kept the total quiet undoes itself.

Run: ``python examples/laser_noise.py``
"""

from __future__ import annotations

import dataclasses

import numpy as np

from maiman.laser import (
    LaserParameters,
    dispersed_power,
    integrate_multimode,
    laser_ensemble,
    multimode_steady_state,
)

LASER = LaserParameters()
CURRENT = 2.0 * LASER.threshold_current()


def linewidth() -> None:
    print(f"1. The linewidth at twice threshold ({CURRENT * 1e3:.1f} mA)")
    print(f"     {'alpha':>6} {'Schawlow-Townes':>16} {'Henry':>10} {'measured':>10}")
    for alpha in (0.0, 2.0, 4.0):
        laser = dataclasses.replace(LASER, linewidth_enhancement=alpha)
        run = laser_ensemble(
            laser, CURRENT, 100e9, 1500, realizations=400, rng=np.random.default_rng(1)
        )
        print(
            f"     {alpha:6.1f} {laser.schawlow_townes_linewidth(CURRENT) / 1e6:13.2f}MHz "
            f"{laser.linewidth(CURRENT) / 1e6:7.2f}MHz {run.measured_linewidth(1000) / 1e6:7.2f}MHz"
        )
    print(
        "     The measured column is phase diffusion over 10 ns, averaged over 400 runs.\n"
        "     Nothing in the integration multiplies by 1 + alpha^2; the carriers do.\n"
    )


def intensity() -> None:
    run = laser_ensemble(
        LASER, CURRENT, 100e9, 1500, realizations=400, rng=np.random.default_rng(2)
    )
    frequencies, rin = run.relative_intensity_noise()
    band = frequencies > 0.5e9
    peak = float(frequencies[band][np.argmax(rin[band])])
    print("2. The intensity noise")
    print(
        f"     peaks at {peak / 1e9:.2f} GHz; the relaxation frequency is "
        f"{LASER.relaxation_frequency(CURRENT) / 1e9:.2f} GHz"
    )
    print(f"     {10 * np.log10(rin[band].max()):.1f} dB/Hz at the peak\n")


def partition() -> None:
    modes, spacing, gain_width = 7, 1e-9, 40e-9
    _, steady = multimode_steady_state(
        LASER, CURRENT, modes=modes, spacing=spacing, gain_bandwidth=gain_width
    )
    run = integrate_multimode(
        LASER,
        CURRENT,
        50e9,
        1000,
        modes=modes,
        spacing=spacing,
        gain_bandwidth=gain_width,
        realizations=100,
        rng=np.random.default_rng(3),
        settle=500,
    )
    power = run.mode_power
    total = run.total_power
    parts = sum(float(power[:, m].var()) for m in range(modes))
    print(f"3. A Fabry-Perot laser, {modes} modes {spacing * 1e9:.0f} nm apart")
    print(
        "     share of the power, mode by mode: "
        + " ".join(f"{s:.2f}" for s in steady / steady.sum())
    )
    mean = float(total.mean())
    print(f"     noise of the total   {float(total.var()) / mean**2:.2e} (relative variance)")
    print(f"     sum of the modes'    {parts / mean**2:.2e}")
    print(f"     {'D L':>10} {'received noise':>15}   17 ps/nm/km is 0.017 s/m per km")
    for amount in (0.0, 0.1, 0.3, 1.0, 3.0):
        received = dispersed_power(power, run.offsets, 50e9, dispersion=amount)
        print(f"     {amount:7.2f}s/m {float(received.var()) / float(received.mean()) ** 2:15.2e}")
    print(
        "     At the laser the partition cancels in the sum. Each kilometre of standard\n"
        "     fibre at 1550 nm delays neighbouring modes 17 ps further apart: sixty\n"
        "     kilometres (1 s/m) and the receiver sees seven times the laser's own noise,\n"
        "     a hundred and eighty and it sees most of every mode's. A floor no amount of\n"
        "     received power lifts."
    )


def main() -> None:
    linewidth()
    intensity()
    partition()


if __name__ == "__main__":
    main()
