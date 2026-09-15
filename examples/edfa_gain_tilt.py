"""A channel drop, seen at four wavelengths instead of one.

``edfa_transient.py`` reports one excursion for a surviving channel, because its
reservoir has one gain. An erbium amplifier's reservoir is its inversion, and a
change in the inversion moves every wavelength's gain -- by an amount set by the
fibre's absorption and emission there. So the same drop raises a channel at
1530 nm by far more than one at 1560 nm, and a line system designed around the
centre wavelength's number under-protects its short-wavelength receivers.

**The spectrum here is a shape, not a fibre.** A 1530 nm absorption peak on a
broad shoulder, with emission derived by McCumber's relation. The ratios this
prints are what that shape gives; bring a datasheet's Giles parameters for a real
coil and :class:`~maiman.ErbiumSpectrum` takes them as they are.

Run: ``python examples/edfa_gain_tilt.py``
"""

from __future__ import annotations

import numpy as np

from maiman import ErbiumSpectrum, spectral_gain_transient, step_schedule
from maiman.components import EDFA

CHANNEL_DBM = -6.0
CHANNELS_BEFORE = 8
CHANNELS_AFTER = 1
WAVELENGTHS_NM = (1530.0, 1540.0, 1550.0, 1560.0)


def illustrative_spectrum() -> ErbiumSpectrum:
    nm = np.linspace(1500.0, 1600.0, 401)
    absorption = 30.0 * np.exp(-((nm - 1530.0) ** 2) / (2 * 8.0**2)) + 15.0 * np.exp(
        -((nm - 1548.0) ** 2) / (2 * 18.0**2)
    )
    return ErbiumSpectrum.from_mccumber(nm * 1e-9, absorption, crossover_wavelength=1531e-9)


def main() -> None:
    spectrum = illustrative_spectrum()
    amplifier = EDFA(gain=20.0, noise_figure=5.0, saturate=True, saturation_power=17.0)
    channel = 10.0 ** (CHANNEL_DBM / 10.0) / 1e3
    times, power = step_schedule(
        [(20e-3, CHANNELS_BEFORE * channel), (60e-3, CHANNELS_AFTER * channel)],
        points_per_segment=2000,
    )
    wavelengths = np.array(WAVELENGTHS_NM) * 1e-9
    spread = spectral_gain_transient(amplifier, spectrum, times, power, wavelengths)
    tilt = spectrum.tilt(wavelengths, spread.reference_wavelength)

    dropped = CHANNELS_BEFORE - CHANNELS_AFTER
    print(f"{CHANNELS_BEFORE} channels at {CHANNEL_DBM:.0f} dBm, {dropped} dropped")
    print(f"inversion {spread.inversion[0]:.3f} -> {spread.inversion[-1]:.3f}\n")
    print(f"{'channel':>9} {'tilt':>6} {'excursion':>11} {'gain before':>12} {'after':>8}")
    for index, nm in enumerate(WAVELENGTHS_NM):
        print(
            f"{nm:7.1f}nm {tilt[index]:6.2f} {spread.excursions[index]:+10.2f}d "
            f"{spread.gain_db[0, index]:11.2f}d {spread.gain_db[-1, index]:7.2f}d"
        )
    print(
        "\n(d = dB) The centre channel's excursion is the single-reservoir one; the\n"
        "rest are it times the tilt, which does not depend on the inversion and so\n"
        "is one curve per amplifier, measured once."
    )


if __name__ == "__main__":
    main()
