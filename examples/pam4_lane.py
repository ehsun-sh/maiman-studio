"""A 53 Gb/s PAM4 lane, through a receiver too narrow for it, and what equalisation buys back.

26.5625 GBd carries two bits a symbol -- one lane of a 200G data-centre link --
on a Mach-Zehnder into a photodiode whose electrical bandwidth is a third of the
symbol rate. That is the regime short-reach optics are built in: the cheap
receiver smears each symbol into its neighbours, and the DSP undoes it.

Two tables. The first is the transmitter's linearity: evenly spaced volts on a
``cos^2`` modulator are not evenly spaced powers, and the outer eyes pay. The
second is the receiver, from no equalisation at all to feed-forward taps and
decision feedback.

Run: ``python examples/pam4_lane.py``
"""

from __future__ import annotations

import numpy as np

from maiman import Graph, SimulationContext
from maiman.components import (
    CWLaser,
    ElectricalFilter,
    FFEDFEEqualizer,
    MachZehnderModulator,
    PAM4Driver,
    PINPhotodiode,
    PRBSGenerator,
)
from maiman.signals import PAMMeasurement

SYMBOL_RATE = 26.5625e9
RECEIVER_GHZ = 7.0


def lane(*, predistort: bool, ffe_taps: int, dfe_taps: int) -> PAMMeasurement:
    ctx = SimulationContext(bit_rate=SYMBOL_RATE, samples_per_symbol=8, sequence_length=4096)
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=15.0, bits_per_symbol=2.0, label="prbs"))
    driver = graph.add(PAM4Driver(v_low=3.2, v_high=0.8, predistort=predistort, label="pam4"))
    laser = graph.add(CWLaser(power=0.0, wavelength=1310.0, label="tx"))
    modulator = graph.add(MachZehnderModulator(v_pi=4.0, label="mzm"))
    pin = graph.add(PINPhotodiode(label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=RECEIVER_GHZ, label="lpf"))
    eq = graph.add(FFEDFEEqualizer(ffe_taps=float(ffe_taps), dfe_taps=float(dfe_taps), label="eq"))
    graph.connect(prbs, driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver, modulator["electrical_in"])
    graph.connect(modulator, pin["in"])
    graph.connect(pin, lpf["in"])
    graph.connect(lpf, eq["in"])
    graph.connect(prbs, eq["reference"])
    result = graph.run()[eq]
    assert isinstance(result, PAMMeasurement)
    return result


def linearity() -> None:
    print("1. The transmitter: four levels on a cos^2 modulator")
    modulator = MachZehnderModulator(v_pi=4.0)
    for predistort in (False, True):
        driver = PAM4Driver(v_low=3.2, v_high=0.8, predistort=predistort)
        powers = modulator.power_transmission(driver.level_voltages())
        steps = np.diff(powers)
        name = "predistorted" if predistort else "even volts"
        print(
            f"     {name:>12}: level powers {np.round(powers, 3)}, "
            f"largest step / smallest {steps.max() / steps.min():.2f}"
        )
    print()


def receiver() -> None:
    print(f"2. The receiver: {RECEIVER_GHZ:.0f} GHz at {SYMBOL_RATE / 1e9:.4f} GBd, 4096 symbols")
    print(f"     {'equaliser':>16} {'SNR':>8} {'symbol errors':>14} {'Gaussian SER':>13}")
    for label, ffe, dfe in (
        ("none", 1, 0),
        ("FFE 5", 5, 0),
        ("FFE 9", 9, 0),
        ("FFE 9 + DFE 2", 9, 2),
    ):
        result = lane(predistort=True, ffe_taps=ffe, dfe_taps=dfe)
        print(
            f"     {label:>16} {result.snr_db:6.2f}dB "
            f"{result.symbol_errors:>8d}/{result.symbols_evaluated} "
            f"{result.ser_expected:13.2e}"
        )
    print(
        "     the Gaussian SER is what the equalised SNR would cost if what remained were\n"
        "     noise; a count far above it is residual ISI or a DFE propagating its errors."
    )


def main() -> None:
    linearity()
    receiver()


if __name__ == "__main__":
    main()
