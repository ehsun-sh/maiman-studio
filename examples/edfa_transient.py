"""EDFA gain dynamics: what a channel drop does, and how far down a chain it goes.

An erbium amplifier shares one inversion between everything passing through it.
Take channels away and the survivors get the gain those channels were using —
not instantly, but over the milliseconds the metastable level takes to refill.
That excursion is the reason a deployed line system has a transient control loop,
and it is what this prints.

Three tables. The first is the timescale, which is what makes this an analysis
on its own time axis rather than a block in a link: the relaxation is four to
five orders of magnitude slower than any waveform window this engine runs. The
second is a single amplifier losing seven channels of eight. The third is the
same disturbance down a chain, which is where the number stops being academic.

Run: ``python examples/edfa_transient.py``
"""

from __future__ import annotations

import numpy as np

from maiman import (
    METASTABLE_LIFETIME,
    SimulationContext,
    effective_time_constant,
    gain_transient,
    step_schedule,
)
from maiman.components import EDFA

#: A -6 dBm channel, eight of them. Ordinary numbers for a WDM comb rather than
#: ones chosen to make the curve dramatic.
CHANNEL_DBM = -6.0
CHANNELS_BEFORE = 8
CHANNELS_AFTER = 1

#: Span loss equal to the amplifier's *compressed* operating gain, so the chain
#: is level before the drop. Setting it to the small-signal gain instead would
#: leave every amplifier weaker than the last, and the excursion would fall down
#: the chain for a reason that has nothing to do with dynamics.
SPAN_LOSS_DB = 15.63
CHAIN_LENGTH = 8


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) / 1e3


def amplifier() -> EDFA:
    return EDFA(gain=20.0, noise_figure=5.0, saturate=True, saturation_power=17.0, label="edfa")


def timescales() -> None:
    print("Why this is not a block in the link")
    ctx = SimulationContext(bit_rate=32e9, samples_per_symbol=16, sequence_length=4096, seed=1)
    window = ctx.sequence_length / ctx.bit_rate
    print(f"  a simulation window: 4096 symbols at 32 GBd = {window * 1e9:.0f} ns")
    print(f"  erbium metastable lifetime                  = {METASTABLE_LIFETIME * 1e3:.0f} ms\n")

    amp = amplifier()
    print(f"{'input':>10} {'gain':>9} {'output':>10} {'tau_eff':>11} {'in windows':>13}")
    for dbm in (-20.0, -10.0, 0.0, 5.0, 10.0):
        power = dbm_to_watt(dbm)
        gain = amp.effective_gain(power)
        constant = effective_time_constant(amp, power)
        print(
            f"{dbm:+9.0f}d {10 * np.log10(gain):8.2f}d "
            f"{10 * np.log10(gain * power * 1e3):+9.2f}d {constant * 1e3:10.2f}m "
            f"{constant / window:13,.0f}"
        )
    print("  (d = dB or dBm, m = ms)\n")


def single_amplifier() -> None:
    print(
        f"One amplifier, {CHANNELS_BEFORE} channels at {CHANNEL_DBM:+.0f} dBm -> {CHANNELS_AFTER}"
    )
    amp = amplifier()
    channel = dbm_to_watt(CHANNEL_DBM)
    times, power = step_schedule(
        [(20e-3, CHANNELS_BEFORE * channel), (80e-3, CHANNELS_AFTER * channel)],
        points_per_segment=3000,
    )
    transient = gain_transient(amp, times, power)
    print(f"  {transient!r}")
    print(f"  integration substeps per output interval: {transient.substeps}\n")

    print(f"{'time':>10} {'gain':>9} {'survivor out':>14}")
    survivor = transient.gain * channel
    for target in (0.0, 19.9e-3, 20.1e-3, 21e-3, 25e-3, 30e-3, 50e-3, 100e-3):
        index = int(np.argmin(np.abs(times - target)))
        print(
            f"{times[index] * 1e3:9.1f}m {transient.gain_db[index]:8.2f}d "
            f"{10 * np.log10(survivor[index] * 1e3):+13.2f}d"
        )
    print()


def chain() -> None:
    print(f"The same drop down {CHAIN_LENGTH} amplifiers, span loss {SPAN_LOSS_DB} dB")
    channel = dbm_to_watt(CHANNEL_DBM)
    times, power = step_schedule(
        [(20e-3, CHANNELS_BEFORE * channel), (80e-3, CHANNELS_AFTER * channel)],
        points_per_segment=3000,
    )
    loss = 10.0 ** (-SPAN_LOSS_DB / 10.0)

    # No feedback from a later amplifier to an earlier one, so the chain
    # integrates exactly in order rather than needing to be solved together.
    drive = power
    accumulated = np.zeros(times.size)
    single = 0.0
    print(f"{'after amp':>10} {'this one':>10} {'accumulated':>13} {'of a single':>13}")
    for stage in range(CHAIN_LENGTH):
        transient = gain_transient(amplifier(), times, drive)
        accumulated = accumulated + (transient.gain_db - transient.gain_db[0])
        peak = float(accumulated[np.argmax(np.abs(accumulated))])
        if stage == 0:
            single = transient.excursion
        print(f"{stage + 1:>10} {transient.excursion:+9.2f}d {peak:+12.2f}d {peak / single:12.2f}x")
        drive = transient.output_power * loss

    print(
        "\n  It accumulates sub-linearly, not as eight times the first: each amplifier"
        "\n  down the chain starts less saturated than the one before it, so it has"
        "\n  less compression left to give back."
    )


def main() -> None:
    timescales()
    single_amplifier()
    chain()
    print(
        "\nWhat is absent: one reservoir means one gain for every wavelength, and a"
        "\nreal erbium transient tilts the gain spectrum as the inversion changes, so"
        "\na survivor's excursion depends on where in the band it sits. The pump is"
        "\nimplicit in the small-signal gain and does not respond -- this is the"
        "\nuncontrolled case, which is the one a control loop is sized against."
    )


if __name__ == "__main__":
    main()
