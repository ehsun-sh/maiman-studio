"""A coherent link over real dispersive fibre.

Every other coherent example in this directory is back to back: transmitter
straight into receiver, with nothing between them. That was a real gap. Chromatic
dispersion is the impairment that dominates any span longer than a patch cord,
and until :class:`~maiman.components.dsp.DispersionCompensator` existed the
transceiver had never met it.

Run this and the first table shows why. Five kilometres — a metro hop — costs
twenty decibels of SNR. Eighty kilometres is not degraded, it is destroyed:
symbols land at chance.

The second table shows the recovery, and it is the whole argument for coherent
detection. Dispersion is an all-pass phase: it rearranges the field in time
without removing anything, so a receiver that measures the field rather than its
square still holds every bit of it. One static filter puts it back. A link that
is stone dead at the photodiode returns to back-to-back quality, out to a
thousand kilometres, with no penalty that grows with distance.

The last table takes the number away. A receiver is never told how long its fibre
is, and the third table shows the compensator measuring it — to within nine ps/nm
from back to back out to a thousand kilometres, which leaves a link as good as
one that was handed the answer.

Loss and nonlinearity are switched off here on purpose. This example is about one
mechanism, and leaving the others in would make it about a power budget instead.
"""

from __future__ import annotations

from maiman import Graph, SimulationContext
from maiman.components import (
    CarrierRecovery,
    CoherentReceiver,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    PRBSGenerator,
    QAMMapper,
)
from maiman.components.dsp import DispersionDiagnostics
from maiman.dsp import dispersive_spread
from maiman.signals import ConstellationMeasurement

DISPERSION = 17.0  # ps/nm/km — standard single-mode fibre at 1550 nm
V_PI = 4.0
SYMBOL_RATE = 32e9
BITS_PER_SYMBOL = 4
WAVELENGTH = 1550.0


def build(
    length_km: float, *, compensate_km: float | None
) -> tuple[Graph, ConstellationAnalyzer, DispersionCompensator]:
    """Build the link. ``compensate_km=None`` leaves the dispersion in place."""
    ctx = SimulationContext(
        bit_rate=SYMBOL_RATE,
        samples_per_symbol=16,
        sequence_length=4096,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    prbs = graph.add(
        PRBSGenerator(order=23.0, bits_per_symbol=float(BITS_PER_SYMBOL), label="prbs")
    )
    mapper = graph.add(QAMMapper(bits_per_symbol=float(BITS_PER_SYMBOL), label="map"))
    driver = graph.add(
        IQDriver(
            v_pi=V_PI,
            predistort=True,
            pulse_shaping=True,
            roll_off=0.2,
            drive_ratio=0.4,
            label="drv",
        )
    )
    # Ideal lasers, so the only thing acting is the fibre. Phase noise has its own
    # example; mixing the two would make neither table readable.
    laser = graph.add(CWLaser(power=0.0, wavelength=WAVELENGTH, linewidth=0.0, label="tx"))
    modulator = graph.add(IQModulator(v_pi=V_PI, label="mod"))
    lo = graph.add(CWLaser(power=13.0, wavelength=WAVELENGTH, linewidth=0.0, label="lo"))
    receiver = graph.add(CoherentReceiver(responsivity=0.8, label="rx"))
    compensator = graph.add(
        DispersionCompensator(
            accumulated_dispersion=DISPERSION * (compensate_km or 0.0),
            wavelength=WAVELENGTH,
            label="cdc",
        )
    )
    sampler = graph.add(IQSampler(matched_filter=True, roll_off=0.2, label="smp"))
    recovery = graph.add(CarrierRecovery(label="cr"))
    analyzer = graph.add(ConstellationAnalyzer(ignore_edges=64.0, label="vsa"))

    graph.connect(prbs["out"], mapper["in"])
    graph.connect(mapper["out"], driver["in"])
    graph.connect(laser, modulator["optical_in"])
    graph.connect(driver["i"], modulator["i"])
    graph.connect(driver["q"], modulator["q"])
    if length_km:
        fiber = graph.add(
            Fiber(
                length=length_km,
                attenuation=0.0,
                dispersion=DISPERSION,
                nonlinearity=0.0,
                label="fib",
            )
        )
        graph.connect(modulator, fiber["in"])
        graph.connect(fiber, receiver["in"])
    else:
        graph.connect(modulator, receiver["in"])
    graph.connect(lo, receiver["lo"])
    graph.connect(receiver["i"], compensator["i"])
    graph.connect(receiver["q"], compensator["q"])
    graph.connect(compensator["i"], sampler["i"])
    graph.connect(compensator["q"], sampler["q"])
    graph.connect(mapper["out"], sampler["reference"])
    graph.connect(sampler["out"], recovery["in"])
    graph.connect(recovery["out"], analyzer["in"])
    graph.connect(mapper["out"], analyzer["reference"])
    return graph, analyzer, compensator


def measure(length_km: float, *, compensate_km: float | None) -> ConstellationMeasurement:
    graph, analyzer, _ = build(length_km, compensate_km=compensate_km)
    return graph.run()[analyzer]


def measure_blind(length_km: float) -> tuple[ConstellationMeasurement, DispersionDiagnostics]:
    """The same link, with the compensator told nothing and made to find it.

    The declared parameter is set to zero so that nothing correct is left in the
    graph for the search to fall back on: whatever the filter removes, it worked
    out from the signal.
    """
    graph, analyzer, compensator = build(length_km, compensate_km=0.0)
    results = graph.run(overrides={(compensator, "estimate"): True}, keep=[compensator])
    return results[analyzer], results.port(compensator, "diagnostics")


def main() -> None:
    spans = (0.0, 5.0, 20.0, 80.0, 400.0, 1000.0)
    occupied = SYMBOL_RATE * 1.2

    print(f"16-QAM at {SYMBOL_RATE * BITS_PER_SYMBOL / 1e9:.0f} Gb/s over standard fibre")
    print(f"D = {DISPERSION:.0f} ps/nm/km, loss and nonlinearity off\n")

    print("            accumulated   spread          uncompensated              compensated")
    print("  span          [ps/nm]  [symbols]      EVM       SNR    errors      EVM       SNR")
    print("  " + "-" * 84)
    for length in spans:
        accumulated = DISPERSION * length
        spread = dispersive_spread(accumulated * 1e-3, occupied, WAVELENGTH * 1e-9, SYMBOL_RATE)
        raw = measure(length, compensate_km=None)
        fixed = measure(length, compensate_km=length)
        errors = f"{raw.symbol_errors}/{raw.symbols_evaluated}"
        print(
            f"  {length:6.0f} km  {accumulated:9.0f}  {spread:8.1f}   "
            f"{raw.evm * 100:7.2f}% {raw.snr_db:8.2f} dB {errors:>11}   "
            f"{fixed.evm * 100:6.2f}% {fixed.snr_db:8.2f} dB"
        )

    # How sharp the setting is. A compensator is not a knob to be roughly right
    # about: the residual after a mismatch is the mismatch, so being four
    # kilometres out costs what four kilometres of uncompensated span costs.
    print("\n  80 km span, compensator mis-set:")
    print("     set to      error        EVM       SNR")
    print("  " + "-" * 42)
    for setting in (72.0, 76.0, 79.0, 80.0, 81.0, 84.0, 88.0):
        result = measure(80.0, compensate_km=setting)
        print(
            f"  {setting:8.0f} km {(setting - 80.0) * DISPERSION:+8.0f} ps/nm "
            f"{result.evm * 100:8.2f}% {result.snr_db:8.2f} dB"
        )

    # And then: not told at all.
    #
    # The table above is the reason this one matters. A compensator has to be set
    # to within a few ps/nm or it is worse than useless, and no deployed receiver
    # is ever told the number — it measures it during acquisition, from the
    # signal, before the equaliser or the carrier loop have converged. Switching
    # `estimate` on does that here: a clock-tone scan across the whole search
    # range, then a refinement that walks in on the shape of the intensity.
    #
    # The last two columns are the ones to read together. Being right in ps/nm is
    # not the claim; leaving behind a link as good as one that was told the
    # answer is.
    print("\n  Blind: the compensator is given nothing and finds it")
    print("            accumulated   estimated     error   contrast        EVM       EVM")
    print("  span          [ps/nm]     [ps/nm]   [ps/nm]              declared     blind")
    print("  " + "-" * 78)
    for length in spans:
        accumulated = DISPERSION * length
        blind, diagnostics = measure_blind(length)
        declared = measure(length, compensate_km=length)
        found = diagnostics.accumulated_dispersion * 1e3
        print(
            f"  {length:6.0f} km  {accumulated:9.0f}   {found:9.1f} {found - accumulated:+9.1f}   "
            f"{diagnostics.contrast:8.1f}   {declared.evm * 100:7.2f}% {blind.evm * 100:8.2f}%"
        )


if __name__ == "__main__":
    main()
