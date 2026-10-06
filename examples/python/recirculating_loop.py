"""A pulse going round a fibre loop, one lap at a time.

A recirculating loop is how a laboratory tests a transoceanic link without owning
one: a few spans closed on themselves through a coupler, a pulse let in once, and
the same hardware crossed again on every lap. What makes it a loop in *time*
rather than a cavity is that each lap arrives one loop-time after the last.

Two blocks together say that. ``Feedback`` closes the cycle and sets how many
laps are run; ``DelayLine`` makes each of them later. Without the delay the same
graph is a fixed-point iteration and every lap lands on top of the first -- the
second table below is that contrast, and it is the reason the block exists.
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.components import (
    Attenuator,
    DelayLine,
    DirectionalCoupler,
    Feedback,
    GaussianPulse,
)
from maiman.signals import OpticalSignal

LOOP_PS = 800.0
LOOP_LOSS_DB = 1.0

#: Four laps from a pulse centred in a 6.4 ns window: a fifth, 3.2 ns after the
#: first, would leave the end of the window and come back in at the start.
LAPS = 4


def run_loop(delay_ps: float) -> np.ndarray:
    """Power on the loop's output port [W], sample by sample."""
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    graph = Graph(ctx)
    pulse = graph.add(GaussianPulse(width=10.0, label="pulse"))
    coupler = graph.add(DirectionalCoupler(True, coupling=0.5, label="coupler"))
    delay = graph.add(DelayLine(delay=delay_ps, label="delay"))
    loss = graph.add(Attenuator(attenuation=LOOP_LOSS_DB, label="loss"))
    loop = graph.add(Feedback(passes=float(LAPS), label="loop"))
    graph.connect(pulse, coupler["in1"])
    graph.connect(coupler["out2"], delay["in"])
    graph.connect(delay["out"], loss["in"])
    graph.connect(loss["out"], loop["in"])
    graph.connect(loop["out"], coupler["in2"])

    signal = graph.run(keep=[coupler]).port(coupler, "out1")
    assert isinstance(signal, OpticalSignal)
    (band,) = signal.bands
    return np.abs(band.Ex.astype(np.complex128)) ** 2


def main() -> None:
    ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
    start = ctx.num_samples // 2
    step = round(LOOP_PS * 1e-12 * ctx.sample_rate)
    gain = 10.0 ** (-LOOP_LOSS_DB / 10.0)

    print(f"1. A 3 dB coupler closed through {LOOP_PS:.0f} ps and {LOOP_LOSS_DB:.0f} dB")
    centre_ns = start / ctx.sample_rate * 1e9
    print(f"     window {ctx.time_window * 1e9:.1f} ns, pulse centred at {centre_ns:.1f} ns\n")
    print(f"     {'lap':>3}  {'arrives':>8}  {'peak':>9}  {'expected':>9}")
    power = run_loop(LOOP_PS)
    expected = 0.5e-3
    for lap in range(LAPS):
        if lap == 1:
            expected = 0.25e-3 * gain
        elif lap > 1:
            expected *= 0.5 * gain
        peak = power[start + lap * step]
        print(
            f"     {lap:3d}  {lap * LOOP_PS:6.0f}ps  {peak * 1e3:7.4f}mW  {expected * 1e3:7.4f}mW"
        )
    print(
        "     lap 0 is the coupler's straight half; lap 1 both crossings and one pass\n"
        "     of loss; every lap after is another straight half and another pass.\n"
    )

    print("2. The same graph with no delay")
    flat = run_loop(0.0)
    first, second = flat[start] * 1e3, flat[start + step] * 1e3
    print(f"     at 0 ps     {first:7.4f}mW   every lap added on the first")
    print(f"     at {LOOP_PS:.0f} ps  {second:7.4f}mW   and nothing where the second should be")
    print(
        "     without a delay a pass is not a lap: it is a step towards a fixed point,\n"
        "     which is right for a short cavity and wrong for this."
    )


if __name__ == "__main__":
    main()
