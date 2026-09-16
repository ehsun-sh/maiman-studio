"""What a directly modulated laser costs: its own chirp, and the reach that takes.

A DML is one chip where an externally modulated transmitter is two, and it draws
a fraction of the power. What it gives up is in its output. The carrier density
has to move for the power to move, the index moves with the carriers, and the
line moves with the index -- so every edge is chirped, and the pulses arrive at
the far end of a span spread by dispersion acting on a signal whose instantaneous
frequency is not constant.

Three tables. The chirp itself, measured from the phase the rate equations
produce. The laser's own numbers, which are the datasheet's. And the reach: the
same pattern, the same receiver, from this laser and from a Mach-Zehnder that
does not chirp.

Run: ``python examples/dml_reach.py``
"""

from __future__ import annotations

import math

import numpy as np

from maiman import Graph, SimulationContext
from maiman.component import Component
from maiman.components import (
    BERAnalyzer,
    CWLaser,
    DirectlyModulatedLaser,
    ElectricalFilter,
    Fiber,
    MachZehnderModulator,
    NRZDriver,
    PINPhotodiode,
    PRBSGenerator,
)
from maiman.signals import EyeMeasurement, OpticalSignal

WAVELENGTH_NM = 1550.0
#: Biased well above threshold and swung hard, because a laser rings at
#: ``sqrt(g0 S0 / tau_p)`` and a 10 Gb/s pattern into one ringing at 3.5 GHz is
#: closed by its own relaxation oscillation before dispersion touches it. At
#: 90 mA this one rings at 8.5 GHz, which is what a 10G DML is specified to do.
BIAS_MA = 90.0
TRANSCONDUCTANCE = 60.0
SPANS_KM = (0.0, 5.0, 10.0, 20.0, 40.0)


def context() -> SimulationContext:
    return SimulationContext(bit_rate=10e9, samples_per_symbol=32, sequence_length=512, seed=11)


def laser() -> DirectlyModulatedLaser:
    return DirectlyModulatedLaser(
        bias_current=BIAS_MA,
        transconductance=TRANSCONDUCTANCE,
        wavelength=WAVELENGTH_NM,
        label="dml",
    )


def link(*, direct: bool, length_km: float) -> EyeMeasurement:
    """One link, modulated at the laser or after it, into the same receiver."""
    ctx = context()
    graph = Graph(ctx)
    prbs = graph.add(PRBSGenerator(order=7.0, label="prbs"))

    source: Component
    if direct:
        # More current is more light, so a one is the high voltage.
        driver = graph.add(NRZDriver(v_low=0.0, v_high=1.0, label="drv"))
        source = graph.add(laser())
        graph.connect(driver, source["in"])
    else:
        # The same average power and the same extinction, with no chirp: a
        # push-pull Mach-Zehnder's field transmission is real. Its curve runs the
        # other way -- full transmission at zero volts -- so a one is the *low*
        # voltage, which is how the OOK link this repository ships drives one.
        driver = graph.add(NRZDriver(v_low=1.0, v_high=0.0, label="drv"))
        continuous = graph.add(
            CWLaser(power=external_power_dbm(), wavelength=WAVELENGTH_NM, label="cw")
        )
        source = graph.add(
            MachZehnderModulator(v_pi=1.0, extinction_ratio=external_extinction(), label="mzm")
        )
        graph.connect(continuous, source["optical_in"])
        graph.connect(driver, source["electrical_in"])
    graph.connect(prbs, driver["in"])

    span = graph.add(Fiber(length=length_km, attenuation=0.2, dispersion=17.0, label="fibre"))
    pin = graph.add(PINPhotodiode(label="pin"))
    lpf = graph.add(ElectricalFilter(bandwidth=7.5, label="lpf"))
    ber = graph.add(BERAnalyzer(label="ber"))
    graph.connect(source, span["in"])
    graph.connect(span["out"], pin["in"])
    graph.connect(pin, lpf["in"])
    graph.connect(lpf, ber["in"])
    graph.connect(prbs, ber["reference"])
    measured = graph.run()[ber]
    assert isinstance(measured, EyeMeasurement)
    return measured


def solved() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The laser's power, frequency offset and time base on a square drive."""
    ctx = context()
    block = laser()
    pattern = np.tile(np.repeat([0.0, 1.0], ctx.samples_per_symbol), ctx.sequence_length // 2)
    from maiman.signals import ElectricalSignal

    wave = block.solve(ElectricalSignal(samples=pattern, fs=ctx.sample_rate, unit="V"))
    return wave.times, wave.power, wave.instantaneous_frequency()


def external_power_dbm() -> float:
    """The average power the DML delivers, so the two transmitters are compared fairly."""
    _, power, _ = solved()
    return 10.0 * math.log10(float(np.mean(power)) * 1e3) + 3.0


def external_extinction() -> float:
    """The DML's own extinction ratio, given to the modulator it is compared against."""
    _, power, _ = solved()
    settled = power[power.size // 2 :]
    return 10.0 * math.log10(float(np.max(settled)) / float(np.min(settled)))


def chirp() -> None:
    _, power, frequency = solved()
    high = power > 0.5 * (power.max() + power.min())
    print("1. The chirp, measured from the phase the rate equations produced")
    print(f"     extinction ratio          {external_extinction():6.2f} dB")
    print(f"     average power             {10 * math.log10(power.mean() * 1e3):6.2f} dBm")
    print(f"     adiabatic offset, ones    {frequency[high].mean() / 1e9:+6.2f} GHz")
    print(f"     adiabatic offset, zeros   {frequency[~high].mean() / 1e9:+6.2f} GHz")
    print(f"     transient peak            {frequency.max() / 1e9:+6.2f} GHz")
    print(f"     total excursion           {np.ptp(frequency) / 1e9:6.2f} GHz\n")


def datasheet() -> None:
    block = laser()
    print("2. What the active region implies, in the numbers a datasheet quotes")
    print(f"     threshold current         {block.threshold_current() * 1e3:6.2f} mA")
    print(f"     slope efficiency          {block.slope_efficiency():6.3f} W/A")
    print(f"     relaxation frequency      {block.relaxation_frequency() / 1e9:6.2f} GHz at bias")
    print(f"     drive                     {BIAS_MA:.0f} to {BIAS_MA + TRANSCONDUCTANCE:.0f} mA\n")


def reach() -> None:
    print("3. The same pattern, chirped and unchirped, into the same receiver")
    print(f"     {'span':>8} {'DML Q':>8} {'external Q':>12} {'penalty':>9}")
    for length in SPANS_KM:
        direct = link(direct=True, length_km=length)
        external = link(direct=False, length_km=length)
        penalty = 20.0 * math.log10(external.q_factor / direct.q_factor)
        print(
            f"     {length:6.0f}km {direct.q_factor:8.2f} {external.q_factor:12.2f} "
            f"{penalty:8.2f}dB"
        )
    print(
        "     both transmitters carry the same average power and the same extinction;\n"
        "     what separates them down the span is that one of them chirps.\n"
        "\n"
        "     The penalty is not monotone, and that is the interesting part: the eye at\n"
        "     5 km is better than back to back. Dispersion acts on a signal whose\n"
        "     instantaneous frequency moves with its power, and over a short span that\n"
        "     partly undoes the laser's own ringing before it starts spreading the\n"
        "     pulses. Past 10 km there is nothing left to undo."
    )


def main() -> None:
    chirp()
    datasheet()
    reach()
    _ = OpticalSignal


if __name__ == "__main__":
    main()
