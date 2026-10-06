"""An MMI, a Mach-Zehnder, and the two knobs that are not the same knob.

The devices the roadmap named and Phase 4 did not have. Both are assembled from
what was already here — an MMI is a scattering matrix with the self-imaging phase
relations in it, and a Mach-Zehnder is two couplers with two paths between them,
put in a :class:`~maiman.circuit.Circuit` and solved. Neither transfer function
is written down anywhere in the engine.

Four things are worth reading off the output.

**A 2 x 2 MMI is a 3 dB directional coupler.** Not approximately: the same
matrix, factor of j included. That is a real result rather than a coincidence of
this implementation — self-imaging at ``N = 2`` produces exactly the quadrature
relation a codirectional coupler does — and it is why an MMI is a drop-in
anywhere a coupler is. The reason a foundry reaches for one is not what happens
at the nominal ratio but what happens when lithography moves off it.

**What distinguishes an MMI from a box that divides power is the phase.** The
amplitudes are the easy half: self-imaging splits evenly, so every path is
``1/sqrt(N)``, and any model gets that right. The phases between the images are
the content, and the instrument that detects a wrong one is unitarity — a matrix
with even amplitudes and invented phases still divides power correctly and stops
conserving it.

**Path imbalance and a heater are different knobs.** ``length_difference`` is a
delay, so its phase grows with frequency and the device becomes a *filter* with a
free spectral range of ``c / (n_g dL)`` — the interleaver in every WDM
transmitter. ``phase_shift`` is flat with frequency and slides that comb sideways
without changing its period. A balanced MZI has no period at all and is a pure
switch. Confusing the two is how an interleaver ends up with the right extinction
at the wrong channel spacing.

**Energy is conserved at every setting, and that is not automatic.** An
interferometer is precisely where a coupler modelled without its quadrature term
starts manufacturing power at one arm phase and losing it at another. The switch
below sums to one across a full turn, to twelve digits.
"""

from __future__ import annotations

import math

import numpy as np

from maiman import Graph, SimulationContext
from maiman.components import MMI, CWLaser, MachZehnderInterferometer, PowerMeter
from maiman.photonics import (
    SILICON_STRIP_NGROUP,
    directional_coupler,
    mach_zehnder,
    mmi_coupler,
    mmi_phase_relations,
)
from maiman.units import C_LIGHT, wavelength_to_frequency

F0 = wavelength_to_frequency(1550e-9)
ARM = 100e-6  # m
DELTA_L = 200e-6  # m — sets the interleaver's period


def main() -> None:
    grid = np.array([F0])

    print("The MMI, port count by port count")
    print(f"  {'N':>3}  {'per path':>10}  {'unitary to':>12}  {'== 3 dB coupler':>16}")
    print("  " + "-" * 48)
    for ports in (1, 2, 3, 4, 8):
        matrix = mmi_coupler(grid, ports=ports)
        share = matrix.power("out1", "in1")[0]
        s = matrix.s[0]
        error = float(np.abs(s.conj().T @ s - np.eye(2 * ports)).max())
        same = (
            "yes"
            if ports == 2
            and np.allclose(s, directional_coupler(grid, coupling=0.5).s[0], atol=1e-15)
            else "—"
        )
        print(f"  {ports:3d}  {10 * math.log10(share):9.3f} dB  {error:12.1e}  {same:>16}")
    print("  the unitarity column is the one testing the phase relations;")
    print("  even amplitudes with invented phases pass every other check.\n")

    print("A 4 x 4 MMI's phase matrix, in units of pi")
    for row in mmi_phase_relations(4) / math.pi:
        print("   " + "  ".join(f"{v:6.2f}" for v in row))
    print("  not literally symmetric above N = 3 — the pairs differ by whole")
    print("  turns of 2 pi — but exp(i phi) is, which is the part that is real.\n")

    print("The balanced interferometer is a switch")
    print(f"  {'heater phase':>14}  {'bar':>8}  {'cross':>8}  {'sum':>10}")
    print("  " + "-" * 44)
    worst = 0.0
    for eighth in range(9):
        phase = eighth * math.pi / 4.0
        matrix = mach_zehnder(grid, arm_length=ARM, phase_shift=phase, reference_frequency=F0)
        bar = matrix.power("out1", "in1")[0]
        cross = matrix.power("out2", "in1")[0]
        worst = max(worst, abs(bar + cross - 1.0))
        print(f"  {phase / math.pi:11.3f} pi  {bar:8.5f}  {cross:8.5f}  {bar + cross:10.7f}")
    print(f"  power conserved to {worst:.1e} across the turn — which is what the")
    print("  coupler's factor of j is for, and an interferometer is where it shows.\n")

    print("The unbalanced interferometer is a filter")
    span = np.linspace(F0 - 2e12, F0 + 2e12, 40001)
    matrix = mach_zehnder(span, arm_length=ARM, length_difference=DELTA_L, reference_frequency=F0)
    power = matrix.power("out1", "in1")
    peaks = span[1:-1][
        (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:]) & (power[1:-1] > 0.9)
    ]
    measured = float(np.mean(np.diff(peaks)))
    expected = C_LIGHT / (SILICON_STRIP_NGROUP * DELTA_L)
    print(f"  dL = {DELTA_L * 1e6:.0f} um over a {ARM * 1e6:.0f} um arm")
    print(f"  measured FSR  {measured / 1e9:8.3f} GHz   from {len(peaks)} peaks")
    print(f"  c / (n_g dL)  {expected / 1e9:8.3f} GHz   n_g = {SILICON_STRIP_NGROUP}")
    print(f"  agreement     {abs(measured / expected - 1):8.1e}")
    print(f"  n_eff would have given {C_LIGHT / (2.44 * DELTA_L) / 1e9:.0f} GHz — nearly")
    print("  double, and every plot would still look like an interleaver.\n")

    print("And the same two devices as blocks, in a link")
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=8, sequence_length=64, seed=1)
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
    splitter = graph.add(MMI(4, excess_loss=0.0, label="mmi"))
    meters = [graph.add(PowerMeter(label=f"pm{k}")) for k in range(4)]
    graph.connect(laser, splitter["in1"])
    for index, meter in enumerate(meters):
        graph.connect(splitter[f"out{index + 1}"], meter["in"])
    results = graph.run()
    levels = " ".join(f"{results[m].power_dbm:+.3f}" for m in meters)
    print(f"  1 x 4 MMI from 0 dBm:  {levels} dBm   (10 log10 1/4 = -6.021)")

    for phase, name in ((0.0, "cross"), (math.pi, "bar")):
        graph = Graph(ctx)
        laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, label="tx"))
        mzi = graph.add(
            MachZehnderInterferometer(
                phase_shift=phase, propagation_loss=0.0, arm_length=ARM * 1e6, label="mzi"
            )
        )
        bar_meter = graph.add(PowerMeter(label="bar"))
        cross_meter = graph.add(PowerMeter(label="cross"))
        graph.connect(laser, mzi["in1"])
        graph.connect(mzi["out1"], bar_meter["in"])
        graph.connect(mzi["out2"], cross_meter["in"])
        run = graph.run()
        bar_mw = 10.0 ** (run[bar_meter].power_dbm / 10.0)
        cross_mw = 10.0 ** (run[cross_meter].power_dbm / 10.0)
        print(
            f"  MZI at {phase / math.pi:.0f} pi:  bar {bar_mw:.6f} mW, "
            f"cross {cross_mw:.6f} mW  -> {name}"
        )


if __name__ == "__main__":
    main()
