"""The loop a deployed amplifier answers a channel drop with, and what it costs to be slow.

``edfa_transient.py`` computes the uncontrolled excursion: seven channels of eight
disappear and the survivors rise by more than three decibels over milliseconds. No
line system is left to do that. It measures its own gain and moves the pump, and
what is left of the excursion is the loop's failure to keep up.

Three tables. The loop's bandwidth against what it leaves behind. What happens
when the pump runs out, which a booster holding output power through a nine
decibel drop does. And which channels drain the reservoir hardest, because a watt
at 1530 nm is not a watt at 1560.

Run: ``python examples/edfa_pump_control.py``
"""

from __future__ import annotations

import numpy as np

from maiman import (
    ErbiumSpectrum,
    PumpControl,
    controlled_gain_transient,
    gain_transient,
    saturating_weights,
    step_schedule,
)
from maiman.components import EDFA

CHANNEL_DBM = -6.0
CHANNELS_BEFORE = 8
CHANNELS_AFTER = 1
REFERENCE = 1550e-9


def amplifier() -> EDFA:
    return EDFA(gain=20.0, noise_figure=5.0, saturate=True, saturation_power=17.0, label="edfa")


def schedule() -> tuple[np.ndarray, np.ndarray]:
    channel = 10.0 ** (CHANNEL_DBM / 10.0) / 1e3
    return step_schedule(
        [(20e-3, CHANNELS_BEFORE * channel), (60e-3, CHANNELS_AFTER * channel)],
        points_per_segment=1500,
    )


def illustrative_spectrum() -> ErbiumSpectrum:
    nm = np.linspace(1500.0, 1600.0, 401)
    absorption = 30.0 * np.exp(-((nm - 1530.0) ** 2) / (2 * 8.0**2)) + 15.0 * np.exp(
        -((nm - 1548.0) ** 2) / (2 * 18.0**2)
    )
    return ErbiumSpectrum.from_mccumber(nm * 1e-9, absorption, crossover_wavelength=1531e-9)


def bandwidths() -> None:
    times, power = schedule()
    free = gain_transient(amplifier(), times, power)
    print(f"1. {CHANNELS_BEFORE - CHANNELS_AFTER} channels of {CHANNELS_BEFORE} dropped")
    print(
        f"     no loop at all: excursion {free.excursion:+.2f} dB, "
        f"settles in {free.settling_time() * 1e3:.1f} ms\n"
    )
    print(f"     {'loop':>10} {'excursion':>11} {'left of it':>11} {'pump after':>11}")
    for bandwidth in (10.0, 100.0, 1e3, 1e4):
        held = controlled_gain_transient(
            amplifier(), times, power, PumpControl(mode="gain", bandwidth=bandwidth)
        )
        share = held.excursion / free.excursion
        print(
            f"     {bandwidth / 1e3:8.2f}kHz {held.excursion:+10.3f}d {share * 100:10.0f}% "
            f"{held.pump_gain_db[-1]:10.2f}d"
        )
    print(
        "     (d = dB) The erbium answers in milliseconds, so a loop a decade faster\n"
        "     than that takes most of the excursion out and one a decade slower does not.\n"
    )


def limits() -> None:
    times, power = schedule()
    print("2. A booster holding output power instead of gain")
    for ceiling in (40.0, 25.0):
        held = controlled_gain_transient(
            amplifier(),
            times,
            power,
            PumpControl(mode="power", bandwidth=1e3, max_pump_gain=ceiling),
        )
        output = 10.0 * np.log10(held.transient.output_power * 1e3)
        if held.pump_limited:
            verdict = "pump limited"
        elif held.pump_saturated:
            verdict = "holds it, after saturating on the way"
        else:
            verdict = "holds it"
        print(
            f"     ceiling {ceiling:5.1f} dB: pump reaches {held.pump_gain_db.max():5.2f} dB, "
            f"output ends {output[-1]:6.2f} dBm against {held.setpoint:6.2f} -- {verdict}"
        )
    print(
        "     nine decibels less input needs nine more decibels of gain to hold the\n"
        "     output, and a pump that cannot deliver them says so rather than pretending.\n"
    )


def weights() -> None:
    spectrum = illustrative_spectrum()
    channels = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9])
    print("3. What a watt is worth, by where it sits (an illustrative spectrum)")
    print(f"     {'channel':>9} {'drains at':>10}")
    for wavelength, weight in zip(
        channels, saturating_weights(spectrum, channels, REFERENCE), strict=True
    ):
        print(f"     {wavelength * 1e9:7.1f}nm {weight:9.3f}x")
    print(
        "     a photon empties the reservoir once, so the rate is the photon flux times\n"
        "     the cross section: dropping the short-wavelength half of a comb is a larger\n"
        "     disturbance than dropping the long-wavelength half of the same power."
    )


def main() -> None:
    bandwidths()
    limits()
    weights()


if __name__ == "__main__":
    main()
