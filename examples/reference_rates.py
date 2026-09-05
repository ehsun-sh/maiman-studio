"""400G and 800G reference transceivers, and what they cost in optical SNR.

Three configurations, all dual-polarization coherent, all built from one number
and arithmetic. 400G is DP-16QAM at 59.84 GBd — the shape a 400ZR module has.
800G is the same payload doubled, and there are two ways to double it: twice the
symbol rate at the same format, or the same-ish symbol rate at a denser one.
Which one a line system picks is a trade between optical SNR and spectrum, and
this prints both sides of it.

Nothing here is quoted from a standard. The symbol rates for 800G are derived
from the 400G one, the line rates are ``baud * bits * 2``, and the required OSNR
is *measured* — a noise-loaded link, bisected until the counted bit error rate
sits on the threshold — and then compared against
:func:`maiman.analysis.required_osnr`, which knows nothing about any of it.

Run: ``python examples/reference_rates.py``
"""

from __future__ import annotations

from pathlib import Path

from maiman import Component, Graph, SimulationContext
from maiman.analysis import required_osnr
from maiman.component import Port
from maiman.components import (
    EDFA,
    Attenuator,
    ButterflyEqualizer,
    CarrierRecovery,
    ConstellationAnalyzer,
    CWLaser,
    DispersionCompensator,
    DualPolarizationReceiver,
    Fiber,
    IQDriver,
    IQModulator,
    IQSampler,
    OSNRMeter,
    PolarizationCombiner,
    PRBSGenerator,
    QAMMapper,
    Splitter,
)
from maiman.project import save

#: The one measured number everything else follows from: DP-16QAM at this baud
#: carries a 400 Gb/s payload with room for the FEC, which is the 400ZR shape.
BAUD_400G = 59.84e9

#: Pre-FEC bit error rate the required-OSNR figures are quoted at. Soft-decision
#: FEC in this class corrects from somewhere near here; the exact threshold is a
#: property of the code, so it is a parameter of the table rather than a fact
#: about the link.
THRESHOLD = 2e-2

ROLL_OFF = 0.1
SPAN_KM = 80.0

#: name -> (symbol rate, bits per symbol per polarization, grid slot)
CONFIGURATIONS: dict[str, tuple[float, int, float]] = {
    # Twice the payload needs twice the baud at the same format...
    "400G DP-16QAM": (BAUD_400G, 4, 75e9),
    "800G DP-16QAM": (2.0 * BAUD_400G, 4, 150e9),
    # ...or two thirds of that baud at a format carrying half again as many bits.
    "800G DP-64QAM": (2.0 * BAUD_400G * 4 / 6, 6, 100e9),
}


def build(
    symbol_rate: float,
    bits: int,
    *,
    pad_db: float = 0.0,
    span_km: float = 0.0,
    equalize: bool = True,
    sequence_length: int = 4096,
) -> tuple[Graph, dict[str, ConstellationAnalyzer], OSNRMeter]:
    """One transceiver, optionally over a span, optionally noise loaded.

    ``pad_db`` is a variable optical attenuator followed by an amplifier that
    exactly undoes it: the power comes back and the noise does not, which is how
    a required-OSNR measurement is made on a bench.

    ``equalize=False`` wires the carrier recovery straight onto the samplers.
    Nothing rotates the polarization here, so an ideal separator would be the
    identity and leaving it out costs nothing — which makes the difference
    between the two a measurement of the blind equaliser itself.
    """
    ctx = SimulationContext(
        bit_rate=symbol_rate,
        samples_per_symbol=4,
        sequence_length=sequence_length,
        seed=2026,
        precision="double",
    )
    graph = Graph(ctx)
    laser = graph.add(CWLaser(power=0.0, wavelength=1550.0, linewidth=100.0, label="tx"))
    splitter = graph.add(Splitter(2, label="pbs"))
    graph.connect(laser, splitter["in"])

    mappers: dict[str, QAMMapper] = {}
    modulators: dict[str, IQModulator] = {}
    for index, axis in enumerate(("x", "y")):
        prbs = graph.add(
            PRBSGenerator(
                order=23.0 if axis == "x" else 15.0,
                bits_per_symbol=float(bits),
                label=f"prbs_{axis}",
            )
        )
        mapper = graph.add(QAMMapper(bits_per_symbol=float(bits), label=f"map_{axis}"))
        driver = graph.add(
            IQDriver(
                v_pi=4.0,
                predistort=True,
                drive_ratio=0.4,
                pulse_shaping=True,
                roll_off=ROLL_OFF,
                label=f"drv_{axis}",
            )
        )
        modulator = graph.add(IQModulator(v_pi=4.0, label=f"mod_{axis}"))
        graph.chain(prbs, mapper, driver)
        graph.connect(splitter[f"out{index}"], modulator["optical_in"])
        graph.connect(driver["i"], modulator["i"])
        graph.connect(driver["q"], modulator["q"])
        mappers[axis], modulators[axis] = mapper, modulator

    combiner = graph.add(PolarizationCombiner(label="pbc"))
    graph.connect(modulators["x"], combiner["x"])
    graph.connect(modulators["y"], combiner["y"])

    tail: Component = combiner
    if span_km:
        fiber = graph.add(Fiber(length=span_km, attenuation=0.2, dispersion=17.0, label="fib"))
        booster = graph.add(EDFA(gain=0.2 * span_km, noise_figure=5.0, label="edfa"))
        graph.connect(combiner, fiber["in"])
        graph.connect(fiber, booster["in"])
        tail = booster
    if pad_db:
        pad = graph.add(Attenuator(attenuation=pad_db, label="voa"))
        loader = graph.add(EDFA(gain=pad_db, noise_figure=5.0, label="ase"))
        graph.connect(tail, pad["in"])
        graph.connect(pad, loader["in"])
        tail = loader

    meter = graph.add(OSNRMeter(label="osnr"))
    graph.connect(tail, meter["in"])

    lo = graph.add(CWLaser(power=13.0, wavelength=1550.0, linewidth=100.0, label="lo"))
    receiver = graph.add(DualPolarizationReceiver(responsivity=0.8, label="rx"))
    graph.connect(tail, receiver["in"])
    graph.connect(lo, receiver["lo"])

    samplers: dict[str, IQSampler] = {}
    for axis in ("x", "y"):
        source: Component = receiver
        if span_km:
            compensator = graph.add(
                DispersionCompensator(
                    accumulated_dispersion=17.0 * span_km, wavelength=1550.0, label=f"cdc_{axis}"
                )
            )
            graph.connect(receiver[f"{axis}i"], compensator["i"])
            graph.connect(receiver[f"{axis}q"], compensator["q"])
            source = compensator
            ports = ("i", "q")
        else:
            ports = (f"{axis}i", f"{axis}q")
        sampler = graph.add(IQSampler(matched_filter=True, roll_off=ROLL_OFF, label=f"smp_{axis}"))
        graph.connect(source[ports[0]], sampler["i"])
        graph.connect(source[ports[1]], sampler["q"])
        graph.connect(mappers[axis]["out"], sampler["reference"])
        samplers[axis] = sampler

    sources: dict[str, Port] = {axis: samplers[axis]["out"] for axis in ("x", "y")}
    if equalize:
        equalizer = graph.add(ButterflyEqualizer(label="eq"))
        graph.connect(samplers["x"]["out"], equalizer["x"])
        graph.connect(samplers["y"]["out"], equalizer["y"])
        sources = {axis: equalizer[f"{axis}_out"] for axis in ("x", "y")}

    analyzers: dict[str, ConstellationAnalyzer] = {}
    for axis in ("x", "y"):
        recovery = graph.add(CarrierRecovery(label=f"cr_{axis}"))
        graph.connect(sources[axis], recovery["in"])
        analyzer = graph.add(ConstellationAnalyzer(ignore_edges=128.0, label=f"vsa_{axis}"))
        graph.connect(recovery["out"], analyzer["in"])
        graph.connect(mappers[axis]["out"], analyzer["reference"])
        analyzers[axis] = analyzer
    return graph, analyzers, meter


def measure(
    symbol_rate: float, bits: int, pad_db: float, *, equalize: bool = True
) -> tuple[float, float]:
    """OSNR and counted BER, averaged over the two tributaries."""
    graph, analyzers, meter = build(symbol_rate, bits, pad_db=pad_db, equalize=equalize)
    results = graph.run()
    counted = [results[analyzer].ber_counted for analyzer in analyzers.values()]
    return float(results[meter]), sum(counted) / len(counted)


def measured_required_osnr(
    symbol_rate: float, bits: int, *, equalize: bool = True, threshold: float = THRESHOLD
) -> float:
    """Bisect the attenuator until the counted BER lands on the threshold.

    Counted, not estimated: an estimate derived from the measured SNR through the
    same closed form the answer is compared against would be circular, and the
    window carries enough symbols for a few hundred errors at this rate.
    """
    low, high = 0.0, 40.0
    for _ in range(14):
        middle = 0.5 * (low + high)
        if measure(symbol_rate, bits, middle, equalize=equalize)[1] < threshold:
            low = middle
        else:
            high = middle
    return measure(symbol_rate, bits, 0.5 * (low + high), equalize=equalize)[0]


def layout(graph: Graph) -> dict[str, dict[str, float]]:
    """Positions for the studio: a column per stage, in dependency order."""
    stages = [
        ("prbs", "map", "drv"),
        ("tx", "pbs", "mod", "pbc"),
        ("fib", "edfa", "voa", "ase", "lo"),
        ("osnr", "rx", "cdc"),
        ("smp", "eq"),
        ("cr", "vsa"),
    ]
    positions: dict[str, dict[str, float]] = {}
    for column, prefixes in enumerate(stages):
        row = 0
        for component in graph.components:
            if not component.label.split("_")[0].startswith(tuple(prefixes)):
                continue
            if component.label in positions:
                continue
            positions[component.label] = {"x": 40.0 + column * 150.0, "y": 40.0 + row * 90.0}
            row += 1
    for component in graph.components:
        positions.setdefault(component.label, {"x": 40.0, "y": 40.0})
    return positions


def main() -> None:
    print("Reference transceivers, dual-polarization coherent.\n")
    print(
        f"{'configuration':16} {'GBd':>8} {'line rate':>11} {'payload':>9} "
        f"{'slot':>7} {'b/s/Hz':>7}"
    )
    print("-" * 64)
    for name, (rate, bits, slot) in CONFIGURATIONS.items():
        line = rate * bits * 2
        payload = 400e9 if name.startswith("400G") else 800e9
        print(
            f"{name:16} {rate / 1e9:8.2f} {line / 1e9:8.0f} Gb/s {payload / 1e9:6.0f} G "
            f"{slot / 1e9:5.0f} GHz {payload / slot:7.2f}"
        )
    print(
        f"\n  Line rate is baud x bits x 2 polarizations. A 400 Gb/s payload inside\n"
        f"  {BAUD_400G * 4 * 2 / 1e9:.0f} Gb/s leaves "
        f"{100 * (1 - 400e9 / (BAUD_400G * 4 * 2)):.1f} % for forward error correction and\n"
        f"  framing, and the 800G rows carry the same fraction."
    )

    print(f"\nRequired OSNR at a pre-FEC BER of {THRESHOLD:.0e}:\n")
    print(
        f"{'configuration':16} {'closed form':>12} {'ideal DSP':>11} {'penalty':>8} "
        f"{'blind equaliser':>16}"
    )
    print("-" * 68)
    ideal: dict[str, float] = {}
    for name, (rate, bits, _) in CONFIGURATIONS.items():
        want = required_osnr(THRESHOLD, bits, symbol_rate=rate)
        clean = measured_required_osnr(rate, bits, equalize=False)
        blind = measured_required_osnr(rate, bits, equalize=True)
        ideal[name] = clean
        print(f"{name:16} {want:9.2f} dB {clean:8.2f} dB {clean - want:+8.2f} {blind:13.2f} dB")
    print(
        "\n  The closed form assumes a perfect transmitter, perfect DSP and a\n"
        "  noiseless receiver, so the gap in the fourth column is the transmitter's\n"
        "  implementation penalty.\n"
        "\n  The fifth is what the DSP costs on top. Nothing rotates the polarization\n"
        "  on this bench, so a perfect separator would be the identity and the blind\n"
        "  butterfly equaliser should be free — half a decibel at 16-QAM and 1.3 at\n"
        "  64-QAM is what it actually is.\n"
        "\n  It used to be eleven decibels at 64-QAM. A polarization rotation is\n"
        "  memoryless, so fitting it with seven taps per path leaves twenty-four\n"
        "  parameters holding nothing but gradient noise — which the single-radius\n"
        "  first stage has no per-sample truth to pin down, and which at 64-QAM is\n"
        "  the whole margin. Settling the centre tap first fixes that and breaks the\n"
        "  case where the eye is closed by dispersion instead, so the block runs both\n"
        "  and keeps whichever lands closer to the constellation's rings."
    )

    print("\nTwo ways to carry 800 Gb/s, with the DSP out of the way:\n")
    doubled = ideal["800G DP-16QAM"] - ideal["400G DP-16QAM"]
    denser = ideal["800G DP-64QAM"] - ideal["800G DP-16QAM"]
    print(f"  twice the baud, same format   {doubled:+6.2f} dB of OSNR, twice the spectrum")
    print(f"  denser format, less baud      {denser:+6.2f} dB of OSNR, two thirds of it")
    print(
        "\n  The first is the price of bandwidth and is 3 dB in theory: twice the\n"
        "  symbol rate collects twice the noise and nothing else changes. The second\n"
        "  buys a third of the spectrum back, and is what a link with filled fibre\n"
        "  and optical SNR to spare pays for it."
    )

    here = Path(__file__).parent
    for name, filename in (("400G DP-16QAM", "zr400.maiman"), ("800G DP-16QAM", "zr800.maiman")):
        rate, bits, _ = CONFIGURATIONS[name]
        graph, _, _ = build(rate, bits, span_km=SPAN_KM, sequence_length=1024)
        save(graph, here / filename, ui=layout(graph))
        print(f"\nwrote {filename}: {name} over {SPAN_KM:.0f} km, {len(graph.components)} blocks")


if __name__ == "__main__":
    main()
